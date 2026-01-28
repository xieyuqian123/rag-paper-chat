import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import our new modules
from rag_engine import initialize_settings, PaperManager
from query_classifier import QueryClassifier, QueryType
from conversation_memory import PaperConversationManager, Message
from hybrid_generator import HybridAnswerGenerator

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Paper Chat Pro", layout="wide")

# Initialize Session State
if "paper_manager" not in st.session_state:
    initialize_settings()
    st.session_state.paper_manager = PaperManager()

if "query_classifier" not in st.session_state:
    st.session_state.query_classifier = QueryClassifier()

if "hybrid_generator" not in st.session_state:
    st.session_state.hybrid_generator = HybridAnswerGenerator()

if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = PaperConversationManager()

if "current_paper_id" not in st.session_state:
    st.session_state.current_paper_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Configuration and Paper Management
with st.sidebar:
    st.title("📚 Paper Manager")
    
    # API Keys
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OpenAI API Key missing")
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        st.warning("LlamaCloud API Key missing")
        
    st.divider()
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload New Paper (PDF)", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF... (This may take a minute)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Load paper
                collection_name = st.session_state.paper_manager.load_paper(
                    tmp_path, uploaded_file.name
                )
                st.session_state.current_paper_id = collection_name
                st.success(f"Indexed: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    st.divider()
    
    # Paper List
    st.subheader("Your Papers")
    papers = st.session_state.paper_manager.list_papers()
    
    if not papers:
        st.info("No papers indexed yet.")
    else:
        for paper_id in papers:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                # Clean up display name
                display_name = paper_id.replace("paper_", "").replace("_", " ")
                if st.button(display_name, key=f"select_{paper_id}", 
                           use_container_width=True,
                           type="primary" if st.session_state.current_paper_id == paper_id else "secondary"):
                    st.session_state.current_paper_id = paper_id
                    st.session_state.messages = st.session_state.conversation_manager.get_memory(paper_id).get_history()
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{paper_id}"):
                    st.session_state.paper_manager.delete_paper(paper_id)
                    if st.session_state.current_paper_id == paper_id:
                        st.session_state.current_paper_id = None
                        st.session_state.messages = []
                    st.rerun()

# Main Chat Interface
st.title("🎓 RAG Paper Chat Pro")

if not st.session_state.current_paper_id:
    st.info("👈 Please upload or select a paper from the sidebar to start chatting.")
else:
    # Display active paper
    display_name = st.session_state.current_paper_id.replace("paper_", "").replace("_", " ")
    st.caption(f"Current Paper: **{display_name}**")
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            
    # Chat Input
    if prompt := st.chat_input("Ask a question about the paper..."):
        # 1. Add User Message
        user_msg = Message(role="user", content=prompt)
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get memory for current paper
        memory = st.session_state.conversation_manager.get_memory(st.session_state.current_paper_id)
        memory.add_message("user", prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # 2. Classify Query
                    classification = st.session_state.query_classifier.classify(prompt)
                    
                    # 3. Condense Query (if needed for RAG)
                    search_query = prompt
                    if classification.query_type not in [QueryType.BACKGROUND, QueryType.FACTUAL]:
                        search_query = memory.condense_query(prompt)
                    
                    # 4. Retrieve Nodes (if strategy uses RAG)
                    strategy = st.session_state.query_classifier.get_strategy(classification.query_type)
                    source_nodes = []
                    
                    if strategy["use_rag"]:
                        retriever = st.session_state.paper_manager.get_retriever(
                            st.session_state.current_paper_id,
                            similarity_top_k=strategy["retrieval_k"]
                        )
                        source_nodes = retriever.retrieve(search_query)
                    
                    # 5. Generate Hybrid Answer
                    # Get conversation context summary
                    context_summary = memory.get_context_summary()
                    
                    result = st.session_state.hybrid_generator.generate(
                        question=prompt,
                        classification=classification,
                        source_nodes=source_nodes,
                        conversation_context=context_summary
                    )
                    
                    # 6. Display Response
                    formatted_response = st.session_state.hybrid_generator.format_answer_with_sources(result)
                    st.markdown(formatted_response)
                    
                    # Save to history
                    assistant_msg = Message(role="assistant", content=formatted_response)
                    st.session_state.messages.append(assistant_msg)
                    memory.add_message("assistant", formatted_response)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
