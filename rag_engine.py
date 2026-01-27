import os
import re
import streamlit as st
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv

load_dotenv()


# Initialize Settings
def initialize_settings():
    if not os.getenv("OPENAI_API_KEY"):
        pass
    
    # LLM Configuration
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")
    llm_base_url = os.getenv("LLM_BASE_URL")  # None by default (uses OpenAI)
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Embedding Configuration
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_base_url = os.getenv("EMBED_BASE_URL")  # None by default
    
    # Create LLM
    llm_kwargs = {"model": llm_model, "temperature": llm_temperature}
    if llm_base_url:
        llm_kwargs["api_base"] = llm_base_url
    Settings.llm = OpenAI(**llm_kwargs)
    
    # Create Embedding Model
    embed_kwargs = {"model": embed_model}
    if embed_base_url:
        embed_kwargs["api_base"] = embed_base_url
    Settings.embed_model = OpenAIEmbedding(**embed_kwargs)


def clean_document_text(text: str) -> str:
    """
    Cleans the document text by:
    1. Removing References/Bibliography section
    2. Removing page numbers
    """
    # Remove References section (common patterns)
    patterns = [
        r'\n#{1,3}\s*References?\s*\n.*',
        r'\n#{1,3}\s*Bibliography\s*\n.*',
        r'\n\*{0,2}References?\*{0,2}\s*\n.*',
        r'\nREFERENCES\s*\n.*',
        r'\nBIBLIOGRAPHY\s*\n.*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_section_from_text(text: str) -> str:
    """
    Extracts the most recent section header from the text.
    Looks for Markdown headers (# ## ###) or bold text patterns.
    """
    # Find all section headers in the text
    header_patterns = [
        r'^#{1,3}\s+(.+?)$',  # Markdown headers
        r'^\*\*(.+?)\*\*$',   # Bold text as headers
        r'^([A-Z][A-Z\s]{2,})$',  # ALL CAPS headers
    ]
    
    headers = []
    for pattern in header_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        headers.extend(matches)
    
    if headers:
        # Return the first (most relevant) header, cleaned up
        return headers[0].strip().replace('#', '').strip()
    return "Unknown Section"


@st.cache_resource(show_spinner="Parsing PDF with LlamaParse...")
def load_and_index_file(file_path, file_name=None):
    """
    Loads a PDF file using LlamaParse, cleans it, and creates a hierarchical index.
    Uses ChromaDB for persistence.
    """
    
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables.")

    if not file_name:
        file_name = os.path.basename(file_path)
    
    # Clean filename for ChromaDB collection name
    collection_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', file_name)
    collection_name = f"paper_{collection_name}"[:63]
    if collection_name.endswith('_'):
        collection_name = collection_name[:-1]

    print(f"Using ChromaDB collection: {collection_name}")

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Check if collection already has data
    if chroma_collection.count() > 0:
        print(f"Loading existing index from collection: {collection_name}")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
        )
    else:
        print(f"Parsing and indexing file: {file_path}")
        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            language="en", 
        )

        file_extractor = {".pdf": parser}
        
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        ).load_data()
        
        print(f"Loaded {len(documents)} raw documents.")
        
        # Clean documents and add page metadata
        cleaned_docs = []
        for idx, doc in enumerate(documents):
            cleaned_text = clean_document_text(doc.text)
            if cleaned_text:
                # Extract section from text
                section = extract_section_from_text(doc.text)
                
                # Create enhanced metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata['page_number'] = idx + 1  # 1-indexed page number
                metadata['section'] = section
                metadata['file_name'] = file_name
                
                cleaned_docs.append(Document(
                    text=cleaned_text,
                    metadata=metadata
                ))
        
        print(f"After cleaning: {len(cleaned_docs)} documents.")
        
        # Hierarchical Node Parser
        # Creates nodes at multiple granularities: 2048, 512, 128 tokens
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        
        nodes = node_parser.get_nodes_from_documents(cleaned_docs)
        print(f"Created {len(nodes)} hierarchical nodes.")
        
        # Get leaf nodes (smallest chunks) for indexing
        leaf_nodes = get_leaf_nodes(nodes)
        print(f"Leaf nodes for indexing: {len(leaf_nodes)}")
        
        # Create docstore to store all nodes (for auto-merging retrieval)
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )
        
        # Index only leaf nodes
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
        )
        
    return index


def get_query_engine(index):
    """
    Creates a query engine that returns source nodes for citation.
    """
    try:
        base_retriever = index.as_retriever(similarity_top_k=6)
        retriever = AutoMergingRetriever(
            base_retriever,
            index.storage_context,
            verbose=True
        )
        query_engine = RetrieverQueryEngine.from_args(retriever)
        return query_engine
    except Exception:
        # Fallback to simple query engine
        return index.as_query_engine(similarity_top_k=6)


def format_sources(source_nodes) -> str:
    """
    Formats source nodes into a readable citation string.
    """
    if not source_nodes:
        return ""
    
    sources = []
    seen = set()  # Avoid duplicate sources
    
    for i, node in enumerate(source_nodes, 1):
        metadata = node.node.metadata if hasattr(node, 'node') else node.metadata
        
        page = metadata.get('page_number', metadata.get('page_label', '?'))
        section = metadata.get('section', 'Unknown')
        file_name = metadata.get('file_name', 'Document')
        
        # Create unique key to avoid duplicates
        key = f"{page}-{section}"
        if key in seen:
            continue
        seen.add(key)
        
        sources.append(f"- **Page {page}** | Section: *{section}*")
    
    if sources:
        return "\n\n---\n📚 **Sources:**\n" + "\n".join(sources[:5])  # Limit to 5 sources
    return ""
