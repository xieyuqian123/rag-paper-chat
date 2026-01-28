# 🎓 RAG Paper Chat Pro

An advanced academic paper assistant that combines **RAG (Retrieval-Augmented Generation)** with **LLM reasoning** to provide accurate, deep, and context-aware answers.

## 🚀 Key Features

### 🧠 Hybrid AI Intelligence
- **Intelligent Query Classification**: Automatically detects if your question is factual, analytical, comparative, or background knowledge.
- **Hybrid Answering**: Combines precise citations from the paper with the general knowledge of large language models.
- **Strategy Selection**: Dynamically chooses the best answering strategy (e.g., pure RAG for facts, RAG+LLM for analysis).

### 📚 Multi-Paper Management
- **Paper Library**: Upload and manage multiple PDF papers.
- **Instant Switching**: Switch between different papers without losing context.
- **Persistent Indexing**: Papers are indexed once and stored locally for instant access later.

### 💬 Advanced Conversation
- **Context Awareness**: Remembers your conversation history and understands follow-up questions.
- **Smart Rewriting**: Automatically refines vague queries based on previous context.
- **Citation Tracking**: Every factual answer includes precise page numbers and section headers.

### 🔍 Enhanced Parsing
- **LlamaParse Integration**: State-of-the-art PDF parsing for complex academic layouts.
- **Table & Figure Support**: Optimized extraction for key data in tables and charts.
- **Hierarchical Indexing**: Retrieves information at multiple levels of granularity/context.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-paper-chat
   ```

2. **Install dependencies** using uv:
   ```bash
   uv sync
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Optional: Custom LLM Settings
# LLM_MODEL=gpt-4o
# LLM_TEMPERATURE=0.2
```

## 🚦 Usage

Run the application:

```bash
uv run streamlit run app.py
```

## 🏗️ Architecture

- **Frontend**: Streamlit
- **Vector Store**: ChromaDB
- **RAG Engine**: LlamaIndex + LlamaParse
- **LLM**: OpenAI GPT-4o
- **Modules**:
  - `rag_engine.py`: Core paper processing & indexing
  - `query_classifier.py`: Intent recognition
  - `conversation_memory.py`: History & context management
  - `hybrid_generator.py`: Answer synthesis