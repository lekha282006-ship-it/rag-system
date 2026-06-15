# Local RAG Chatbot

A completely local Retrieval-Augmented Generation (RAG) chatbot that runs on your machine without any external API keys. Uses Ollama for LLM inference and HuggingFace embeddings for semantic search.

## Features

- **100% Local**: No OpenAI, Anthropic, or other cloud API keys required
- **PDF Ingestion**: Upload and process PDF documents
- **Semantic Search**: Uses sentence-transformers for intelligent document retrieval
- **Vector Storage**: ChromaDB for efficient similarity search
- **Streamlit UI**: Clean, modern web interface
- **Multiple Models**: Support for various Ollama models (llama2, mistral, codellama, phi)

## Prerequisites

1. **Ollama**: Install and run Ollama locally
   - Download from [ollama.ai](https://ollama.ai)
   - Pull a model: `ollama pull llama2` (or mistral, codellama, etc.)
   - Ensure Ollama is running: `ollama serve`

2. **Python 3.8+**: Make sure you have Python installed

## Installation

1. Clone or navigate to the project directory

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. **Ingest Documents**:
   - Click "Browse files" in the sidebar
   - Select a PDF file
   - Click "Ingest PDF" to process it

4. **Chat**:
   - Ask questions about your documents in the chat interface
   - View answers with source references

## Project Structure

```
rag system/
├── rag_core.py          # Core RAG logic (ingestion, retrieval, QA)
├── app.py               # Streamlit UI
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── chroma_db/          # Vector database (created automatically)
└── uploads/            # Uploaded PDFs (created automatically)
```

## How It Works

1. **Ingestion**: PDFs are loaded, split into chunks, and embedded using sentence-transformers
2. **Storage**: Embeddings are stored in ChromaDB for fast similarity search
3. **Retrieval**: When you ask a question, relevant document chunks are retrieved
4. **Generation**: Ollama LLM generates answers using retrieved context

## Supported Ollama Models

- llama2 (default)
- mistral
- codellama
- phi

You can pull additional models with: `ollama pull <model-name>`

## Troubleshooting

**Ollama connection error**: Ensure Ollama is running with `ollama serve`

**Model not found**: Pull the model first with `ollama pull <model-name>`

**Memory issues**: Use a smaller model or reduce chunk size in `rag_core.py`

**Slow responses**: First query may be slow as embeddings load; subsequent queries are faster

## Customization

- **Embedding model**: Change in `rag_core.py` (line 12)
- **Chunk size**: Modify in `rag_core.py` (line 28-30)
- **Retrieval count**: Change `k` parameter in `rag_core.py` (line 68)
- **Prompt template**: Edit in `rag_core.py` (line 55-62)

## License

MIT License - Feel free to use and modify for your needs
