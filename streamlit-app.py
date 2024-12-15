import os
import time
import streamlit as st
import google.generativeai as genai
import pandas as pd

class GeminiSQLChatInterface:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp"):
        # Configure API 
        genai.configure(api_key=api_key)
        
        # Generation configuration
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Create the model with system instruction
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            system_instruction="You are an expert SQL developer and you are given some table definitions. Your role is to develop SQL queries for a given request by the user. Provide clear, concise, and optimized SQL queries."
        )
        
        # Initialize chat session 
        self.chat_session = None
        self.files = []
    
    def upload_files(self, file_paths):
        """
        Upload files to Gemini and wait for processing
        
        :param file_paths: List of file paths to upload
        :return: List of uploaded file objects
        """
        self.files = []
        for path in file_paths:
            mime_type = "text/csv" if path.endswith('.csv') else None
            uploaded_file = genai.upload_file(path, mime_type=mime_type)
            self.files.append(uploaded_file)
            st.sidebar.success(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")
        
        # Wait for files to be active
        self._wait_for_files_active()
        
        return self.files
    
    def _wait_for_files_active(self):
        """
        Wait for uploaded files to be processed
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
    
    def start_chat(self, initial_files=None):
        """
        Start a new chat session
        
        :param initial_files: Optional list of files to include in initial context
        """
        initial_context = []
        if initial_files or self.files:
            files_to_use = initial_files or self.files
            initial_context = [
                {
                    "role": "user",
                    "parts": files_to_use + ["Analyze these files and provide context for SQL queries."],
                }
            ]
        
        self.chat_session = self.model.start_chat(history=initial_context)
    
    def send_message(self, user_input):
        """
        Send a message and get response
        
        :param user_input: User's input message
        :return: Model's response
        """
        if not self.chat_session:
            self.start_chat()
        
        response = self.chat_session.send_message(user_input)
        return response.text
    
    def clear_files(self):
        """
        Clear uploaded files
        """
        for file in self.files:
            try:
                genai.delete_file(file.name)
                st.sidebar.info(f"Deleted file: {file.name}")
            except Exception as e:
                st.sidebar.error(f"Error deleting file {file.name}: {e}")
        self.files = []

def load_csv_data(uploaded_files):
    """
    Load CSV files and display their contents
    
    :param uploaded_files: List of uploaded Streamlit file objects
    :return: List of pandas DataFrames
    """
    dataframes = []
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            dataframes.append(df)
            st.sidebar.subheader(f"Contents of {uploaded_file.name}")
            st.sidebar.dataframe(df.head())
        except Exception as e:
            st.sidebar.error(f"Error reading {uploaded_file.name}: {e}")
    return dataframes

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Gemini SQL Query Assistant", 
        page_icon=":robot_face:", 
        layout="wide"
    )
    
    # Title
    st.title("🤖 Gemini SQL Query Assistant")
    st.markdown("Develop SQL queries with AI assistance!")
    
    # Sidebar for API key and file upload
    st.sidebar.header("Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input(
        "Google AI API Key", 
        type="password", 
        help="Enter your Google AI API key to use the Gemini model"
    )
    
    # File Upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV Files", 
        type=["csv"], 
        accept_multiple_files=True
    )
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Check if API key is provided
    if not api_key:
        st.warning("Please enter a Google AI API key to proceed.")
        return
    
    # Main chat interface
    try:
        # Initialize Gemini Chat Interface
        chat_interface = GeminiSQLChatInterface(api_key)
        
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
            chat_interface.upload_files(temp_files)
            
            # Load and display CSV data
            load_csv_data(uploaded_files)
        
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
                    response = chat_interface.send_message(user_input)
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
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        # Clean up temporary files and Gemini files
        if 'chat_interface' in locals():
            chat_interface.clear_files()
        
        # Remove temporary CSV files
        if 'temp_files' in locals():
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
