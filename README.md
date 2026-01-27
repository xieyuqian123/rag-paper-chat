   git clone <repository-url>
   cd rag-paper-chat
   ```

2. **Install dependencies** using uv:
   ```bash
   uv sync
   ```

## Configuration

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
```

## Usage

Run the application using uv:

```bash
uv run streamlit run app.py