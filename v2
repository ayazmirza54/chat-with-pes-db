import os
import json
import time
import streamlit as st
import google.generativeai as genai
import pandas as pd

class JSONColumnProcessor:
    @staticmethod
    def process_json_column(df):
        """
        Process columns that might contain JSON data
        
        :param df: Input pandas DataFrame
        :return: Processed DataFrame with JSON columns expanded
        """
        # Identify potential JSON columns
        json_columns = []
        for column in df.columns:
            try:
                # Check if the column contains JSON-like strings
                sample_values = df[column].dropna().head()
                if sample_values.apply(JSONColumnProcessor._is_json_like).any():
                    json_columns.append(column)
            except Exception:
                pass
        
        # Expand JSON columns
        for column in json_columns:
            try:
                # Attempt to parse JSON and expand
                df[column] = df[column].apply(JSONColumnProcessor._safe_json_parse)
                
                # If successful, try to flatten the JSON
                expanded_df = pd.json_normalize(df[column])
                
                # Rename columns to avoid conflicts
                expanded_df.columns = [f"{column}_{subcol}" for subcol in expanded_df.columns]
                
                # Merge expanded columns back to original DataFrame
                df = pd.concat([df, expanded_df], axis=1)
                
                # Drop original JSON column
                df = df.drop(columns=[column])
            
            except Exception as e:
                st.sidebar.warning(f"Could not fully process JSON column {column}: {e}")
        
        return df
    
    @staticmethod
    def _is_json_like(value):
        """
        Check if a value is JSON-like
        
        :param value: Value to check
        :return: Boolean indicating if value is JSON-like
        """
        if not isinstance(value, str):
            return False
        
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def _safe_json_parse(value):
        """
        Safely parse JSON values
        
        :param value: JSON string to parse
        :return: Parsed JSON or original value
        """
        if pd.isna(value):
            return None
        
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

class FileManager:
    @staticmethod
    def ensure_upload_dir():
        """
        Ensure the upload directory exists
        """
        upload_dir = os.path.join(os.getcwd(), 'uploaded_files')
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir
    
    @staticmethod
    def save_uploaded_file(uploaded_file):
        """
        Save an uploaded file to a persistent location
        
        :param uploaded_file: Streamlit uploaded file
        :return: Path to the saved file
        """
        upload_dir = FileManager.ensure_upload_dir()
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    @staticmethod
    def read_csv_with_robust_parsing(file_path):
        """
        Read CSV with more robust parsing
        
        :param file_path: Path to the CSV file
        :return: Processed DataFrame
        """
        try:
            # Try reading with default settings
            df = pd.read_csv(file_path, engine='python')
            return df
        except Exception as e:
            st.sidebar.warning(f"Default CSV parsing failed: {e}")
            
            # Try alternative parsing methods
            try:
                # Try reading with explicit separator detection
                df = pd.read_csv(file_path, engine='python', sep=None)
                return df
            except Exception as alt_e:
                st.sidebar.error(f"Advanced CSV parsing failed: {alt_e}")
                
                # Read as plain text to debug
                with open(file_path, 'r') as f:
                    sample_lines = f.readlines()[:5]
                    st.sidebar.text("Sample file contents:")
                    for line in sample_lines:
                        st.sidebar.text(line.strip())
                
                raise ValueError(f"Unable to parse CSV file: {file_path}")

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
                "You are an expert SQL developer working with complex database schemas, "
                "including tables with JSON-expanded columns. Your role is to develop precise, "
                "efficient SQL queries for user requests. Provide clear explanations, "
                "optimize queries, and offer insights into the SQL logic. If a query is complex, "
                "break down your approach step by step. Be prepared to handle nested or expanded JSON data."
            )
        )
        
        # Initialize chat session 
        self.chat_session = None
        self.files = []
        self.processed_dataframes = []
    
    def upload_files(self, file_paths):
        """
        Upload files to Gemini and process them
        
        :param file_paths: List of file paths to upload
        :return: List of uploaded file objects
        """
        self.files = []
        self.processed_dataframes = []
        
        for path in file_paths:
            # Determine mime type
            mime_type = "text/csv" if path.endswith('.csv') else None
            
            # Upload file
            uploaded_file = genai.upload_file(path, mime_type=mime_type)
            self.files.append(uploaded_file)
            st.sidebar.success(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")
            
            # Process CSV files
            if mime_type == "text/csv":
                try:
                    # Read CSV with robust parsing
                    df = FileManager.read_csv_with_robust_parsing(path)
                    
                    # Process JSON columns
                    processed_df = JSONColumnProcessor.process_json_column(df)
                    
                    self.processed_dataframes.append(processed_df)
                except Exception as e:
                    st.sidebar.error(f"Error processing {path}: {e}")
        
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
        Start a new chat session with initial context
        
        :param initial_files: Optional list of files to include in initial context
        """
        initial_context = []
        
        # Prepare context with file information
        if initial_files or self.files:
            files_to_use = initial_files or self.files
            
            # Generate context description
            context_parts = [
                "These are the table definitions and sample data. "
                "I've processed the files, including handling any JSON columns. "
                "Here's a summary of the processed data:"
            ]
            
            # Add DataFrame summaries
            for i, df in enumerate(self.processed_dataframes, 1):
                context_parts.append(f"\nDataset {i}:")
                context_parts.append(f"- Columns: {', '.join(df.columns)}")
                context_parts.append(f"- Number of rows: {len(df)}")
                
                # Add a few sample rows description
                if not df.empty:
                    context_parts.append("- Sample data structure:")
                    sample_row = df.iloc[0].to_dict()
                    for key, value in list(sample_row.items())[:5]:  # Limit to first 5 columns
                        context_parts.append(f"  * {key}: {type(value).__name__}")
            
            initial_context = [
                {
                    "role": "user",
                    "parts": files_to_use + context_parts,
                }
            ]
        
        # Start chat session
        self.chat_session = self.model.start_chat(history=initial_context)
    
    def send_message(self, user_input):
        """
        Send a message and get response while maintaining context
        
        :param user_input: User's input message
        :return: Model's response
        """
        # Ensure chat session is initialized
        if not self.chat_session:
            self.start_chat()
        
        # Send message and get response
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
        self.processed_dataframes = []

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Gemini SQL Query Assistant", 
        page_icon=":robot_face:", 
        layout="wide"
    )
    
    # Title
    st.title("🤖 Gemini SQL Query Assistant")
    st.markdown("Develop SQL queries with AI assistance, including JSON column support!")
    
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
        "Upload CSV Files (including JSON columns)", 
        type=["csv"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Initialize session state for chat history, Gemini interface, and uploaded file paths
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'gemini_interface' not in st.session_state:
        st.session_state.gemini_interface = None
    
    if 'uploaded_file_paths' not in st.session_state:
        st.session_state.uploaded_file_paths = []
    
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
            # Save uploaded files to a persistent location
            temp_files = []
            for uploaded_file in uploaded_files:
                # Save file and get persistent path
                file_path = FileManager.save_uploaded_file(uploaded_file)
                temp_files.append(file_path)
            
            # Store file paths in session state for persistence
            st.session_state.uploaded_file_paths.extend(temp_files)
            
            # Upload files to Gemini
            st.session_state.gemini_interface.upload_files(temp_files)
            
            # Display processed DataFrames
            if st.session_state.gemini_interface.processed_dataframes:
                st.sidebar.header("Processed Datasets")
                for i, df in enumerate(st.session_state.gemini_interface.processed_dataframes, 1):
                    with st.sidebar.expander(f"Dataset {i}"):
                        st.dataframe(df.head())
        
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
                    # Use the persistent Gemini interface from session state
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
            # Clear chat history and Gemini interface
            st.session_state.chat_history = []
            st.session_state.gemini_interface = None
            
            # Keep uploaded files, just reset the interface
            st.experimental_rerun()
        
        # Display uploaded files
        st.sidebar.header("Uploaded Files")
        for file_path in st.session_state.uploaded_file_paths:
            st.sidebar.text(os.path.basename(file_path))
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        # Clean up Gemini files if interface exists
        if st.session_state.gemini_interface:
            st.session_state.gemini_interface.clear_files()

if __name__ == "__main__":
    main()
