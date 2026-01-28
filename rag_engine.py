import os
import re
import shutil
import chromadb
from typing import List, Optional, Dict
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize global settings
def initialize_settings():
    if not os.getenv("OPENAI_API_KEY"):
        # Expect API key to be set in environment
        pass
    
    # LLM Configuration
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Embedding Configuration
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_base_url = os.getenv("EMBED_BASE_URL")
    
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


class PaperManager:
    """
    Manages multiple academic papers, including parsing, indexing, and retrieval.
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.indices: Dict[str, VectorStoreIndex] = {}
        
        # Initialize LlamaParse
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables.")

    def _clean_text(self, text: str) -> str:
        """
        Cleans document text by removing references and page numbers.
        """
        # Remove References section
        patterns = [
            r'\n#{1,3}\s*References?\s*\n.*',
            r'\n#{1,3}\s*Bibliography\s*\n.*',
            r'\n\*{0,2}References?\*{0,2}\s*\n.*',
            r'\nREFERENCES\s*\n.*',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()

    def _extract_section(self, text: str) -> str:
        """Extracts the most recent section header."""
        header_patterns = [
            r'^#{1,3}\s+(.+?)$',
            r'^\*\*(.+?)\*\*$',
            r'^([A-Z][A-Z\s]{2,})$',
        ]
        headers = []
        for pattern in header_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            headers.extend(matches)
        
        if headers:
            return headers[0].strip().replace('#', '').strip()
        return "Unknown Section"

    def _get_collection_name(self, filename: str) -> str:
        """Generates a valid ChromaDB collection name from filename."""
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
        name = f"paper_{name}"[:63]
        if name.endswith('_'):
            name = name[:-1]
        return name

    def load_paper(self, file_path: str, file_name: str) -> str:
        """
        Loads and indexes a paper. Returns the collection name.
        """
        collection_name = self._get_collection_name(file_name)
        print(f"Processing paper: {file_name} -> {collection_name}")
        
        collection = self.chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Check if already indexed
        if collection.count() > 0:
            print(f"Loading existing index for {file_name}")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
            self.indices[collection_name] = index
            return collection_name
            
        print(f"Parsing new file: {file_path}")
        # Enhanced parsing specifically for tables and figures
        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            language="en",
            premium_mode=True, # Improved table extraction
        )
        
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".pdf": parser}
        ).load_data()
        
        # Clean and enrich documents
        cleaned_docs = []
        for idx, doc in enumerate(documents):
            cleaned_text = self._clean_text(doc.text)
            if cleaned_text:
                section = self._extract_section(doc.text)
                metadata = doc.metadata.copy()
                metadata.update({
                    'page_number': idx + 1,
                    'section': section,
                    'file_name': file_name
                })
                cleaned_docs.append(Document(text=cleaned_text, metadata=metadata))
        
        # Hierarchical Indexing
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        nodes = node_parser.get_nodes_from_documents(cleaned_docs)
        leaf_nodes = get_leaf_nodes(nodes)
        
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )
        
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
        )
        self.indices[collection_name] = index
        return collection_name

    def get_retriever(self, collection_name: str, similarity_top_k: int = 6):
        """Gets an auto-merging retriever for a specific paper."""
        if collection_name not in self.indices:
            # Try to load if not in memory but exists in DB
            try:
                collection = self.chroma_client.get_collection(collection_name)
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.indices[collection_name] = VectorStoreIndex.from_vector_store(
                    vector_store, storage_context=storage_context
                )
            except Exception:
                raise ValueError(f"Paper {collection_name} not found or not indexed.")
                
        index = self.indices[collection_name]
        base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return AutoMergingRetriever(
            base_retriever,
            index.storage_context,
            verbose=True
        )
        
    def list_papers(self) -> List[str]:
        """Lists all indexed paper collection names."""
        collections = self.chroma_client.list_collections()
        return [c.name for c in collections if c.name.startswith("paper_")]

    def delete_paper(self, collection_name: str):
        """Deletes a paper from the index."""
        try:
            self.chroma_client.delete_collection(collection_name)
            if collection_name in self.indices:
                del self.indices[collection_name]
        except Exception as e:
            print(f"Error deleting paper {collection_name}: {e}")
