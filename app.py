import streamlit as st
import os
import tempfile
from rag_engine import initialize_settings, load_and_index_file, get_query_engine, format_sources
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Paper Chat", layout="wide")

st.title("📚 RAG Paper Chat")
st.markdown("Upload an academic paper (PDF) and ask questions about it!")

# Sidebar for configuration and file upload
with st.sidebar:
    st.header("Configuration")
    
    # API Keys (if not in .env)
    if not os.getenv("OPENAI_API_KEY"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        llama_key = st.text_input("LlamaCloud API Key", type="password")
        if llama_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_key

    st.divider()
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Initialize settings (once)
initialize_settings()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Handle file upload and indexing
if uploaded_file:
    # Check if a new file is uploaded
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF... This may take a while for large papers."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load and index
                index = load_and_index_file(tmp_path, file_name=uploaded_file.name)
                st.session_state.query_engine = get_query_engine(index)
                st.session_state.current_file = uploaded_file.name
                
                # Clear chat history for new document
                st.session_state.messages = []
                st.success(f"Indexed {uploaded_file.name} successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Clean up temp file
                os.remove(tmp_path)

# Display chat interface
if st.session_state.query_engine:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the paper..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                response = st.session_state.query_engine.query(prompt)
                
                # Format response with sources
                answer = response.response
                sources = format_sources(response.source_nodes)
                full_response = answer + sources
                
                st.markdown(full_response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    if not uploaded_file:
        st.info("Please upload a PDF file to start chatting.")
    else:
        pass

