import os
import time
import streamlit as st
import google.generativeai as genai


class GeminiSQLChatInterface:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp"):
        # Configure API
        genai.configure(api_key=api_key)

        # Generation configuration
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Create the model with system instruction
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            system_instruction=(
                "You are an expert SQL developer working with complex database schemas. Your role is to develop precise, "
                "efficient SQL queries for user requests. Provide clear explanations, optimize queries, and offer insights "
                "into the SQL logic. Handle uploaded files for data analysis tasks."
            ),
        )

        # Initialize chat session
        self.chat_session = None
        self.files = []
        self.file_uris = []

    def upload_files(self, file_paths):
        """
        Upload files to Gemini and prepare their URIs for context.

        :param file_paths: List of file paths to upload
        :return: List of uploaded file URIs
        """
        self.files = []
        self.file_uris = []

        for path in file_paths:
            # Determine MIME type
            mime_type = "text/csv" if path.endswith('.csv') else None

            # Upload file to Gemini
            uploaded_file = genai.upload_file(path, mime_type=mime_type)
            self.files.append(uploaded_file)
            self.file_uris.append(uploaded_file.uri)  # Collect URI for context

            st.sidebar.success(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")

        # Wait for file processing
        self._wait_for_files_active()

        return self.file_uris

    def _wait_for_files_active(self):
        """
        Wait for uploaded files to be processed by Gemini.
        """
        st.sidebar.info("Waiting for file processing...")
        for name in (file.name for file in self.files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING":
                st.sidebar.text("Processing files...")
                time.sleep(10)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                st.sidebar.error(f"File {file.name} failed to process")
                raise Exception(f"File {file.name} failed to process")
        st.sidebar.success("All files processed successfully!")

    def start_chat(self):
        """
        Start a new chat session with the uploaded files as context.
        """
        # Prepare initial context
        initial_context = [
            {
                "role": "user",
                "parts": [
                    "I have uploaded some files. Please analyze these files to help me with SQL queries.",
                    "The file URIs are:",
                    *self.file_uris  # Include URIs directly
                ],
            }
        ]

        # Start chat session
        self.chat_session = self.model.start_chat(history=initial_context)
        st.sidebar.info("Chat session started with file context.")

    def send_message(self, user_input):
        """
        Send a message to Gemini and get a response.

        :param user_input: User's query/message
        :return: Response from Gemini
        """
        # Ensure the chat session is initialized
        if not self.chat_session:
            self.start_chat()

        # Send the user's message and get response
        response = self.chat_session.send_message(user_input)
        return response.text

    def clear_files(self):
        """
        Clear uploaded files from Gemini.
        """
        for file in self.files:
            try:
                genai.delete_file(file.name)
                st.sidebar.info(f"Deleted file: {file.name}")
            except Exception as e:
                st.sidebar.error(f"Error deleting file {file.name}: {e}")
        self.files = []
        self.file_uris = []


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Gemini SQL Query Assistant",
        page_icon=":robot_face:",
        layout="wide",
    )

    # Title
    st.title("🤖 Gemini SQL Query Assistant")
    st.markdown("Develop SQL queries with AI assistance using file data!")

    # Sidebar for API key and file upload
    st.sidebar.header("Configuration")

    # API Key Input
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.sidebar.text_input(
        "Google AI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your Google AI API key to use the Gemini model"
    )

    # Store API key in session state
    if api_key:
        st.session_state.api_key = api_key

    # File Upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Files (CSV or other supported types)",
        type=["csv", "json", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Initialize session state for chat history and Gemini interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'gemini_interface' not in st.session_state:
        st.session_state.gemini_interface = None

    # Check if API key is provided
    if not api_key:
        st.warning("Please enter a Google AI API key to proceed.")
        return

    # Main chat interface
    try:
        # Initialize or retrieve Gemini Chat Interface
        if st.session_state.gemini_interface is None:
            st.session_state.gemini_interface = GeminiSQLChatInterface(api_key)

        # Upload and process files if provided
        if uploaded_files:
            # Save uploaded files temporarily
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_path)

            # Upload files to Gemini
            st.session_state.gemini_interface.upload_files(temp_files)

        # Chat input
        user_input = st.chat_input("Enter your SQL query request...")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Display user message
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Generating SQL query..."):
                    response = st.session_state.gemini_interface.send_message(user_input)
                st.write(response)

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chat history
        st.sidebar.header("Chat History")
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.sidebar.text(f"👤 {message['content']}")
            else:
                st.sidebar.text(f"🤖 {message['content']}")

        # Reset button
        if st.sidebar.button("Start New Conversation"):
            st.session_state.chat_history = []
            st.session_state.gemini_interface = None
            st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up temporary files and Gemini files if interface exists
        if st.session_state.gemini_interface:
            st.session_state.gemini_interface.clear_files()


if __name__ == "__main__":
    main()
