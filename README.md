# 📄 Document Intelligence Assistant(RAG CHATBOT)

An AI-powered **Document Question Answering (Q&A) system** featuring a multi-stage hybrid retrieval pipeline, cross-encoder reranking, conversational flexibility, and source transparency. The application dynamically ingests local PDF documents and extracts live web content from user-provided URLs to build an on-the-fly searchable knowledge base.

---

## 🚀 Key Features

### 🌐 Multi-Source Ingestion Engine
Seamlessly processes unstructured content from uploaded PDF documents and live web pages simultaneously, enabling unified information retrieval across multiple data sources.

### 🧠 Advanced Hybrid Retrieval & Reranking Architecture
Combines a multi-tiered retrieval network to ensure pinpoint context accuracy:
* **Dense Retrieval:** Semantic vector search using HuggingFace sentence embeddings stored in **ChromaDB**.
* **Sparse Retrieval:** Keyword-matched index tracking utilizing **Rank-BM25**.
* **Ensemble Blending:** Combines sparse and dense outputs using an **EnsembleRetriever** (50/50 weighted split).
* **Deep Reranking:** Filters and optimizes the top retrieved contexts using a **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) before passing payloads to the LLM.

### 🛡️ Dynamic Guardrails & Chat Flexibility
Implements an intelligent hybrid prompt design. When documents are active, strict boundaries prevent hallucinations. If the knowledge base is cleared, the system automatically falls back to an open, conversational general-knowledge assistant.

### 💬 Contextual Chat Memory
Maintains active dialogue history to resolve pronoun references and support coherent multi-turn conversations.

### 📖 Transparent Source Verification
Displays expandable source sections beneath responses, allowing users to inspect the original text chunks, page numbers, or live web links used to generate answers.



<img width="1901" height="918" alt="Screenshot 2026-06-22 144317" src="https://github.com/user-attachments/assets/b020cd98-8769-433f-8d87-b18b2624ea04" />
<img width="1893" height="838" alt="Screenshot 2026-06-22 144034" src="https://github.com/user-attachments/assets/d5e104d5-b827-4ebd-a926-26b54726699f" />

---

# 🛠️ System Architecture & Tech Stack

| Component                | Technology                                                            |
| ------------------------ | --------------------------------------------------------------------- |
| Frontend Dashboard       | Streamlit                                                             |
| RAG Framework            | LangChain (`langchain-core`, `langchain-community`, `langchain-groq`) |
| Vector Database          | ChromaDB                                                              |
| Keyword Retrieval Engine | Rank-BM25                                                             |
| Deep Reranking Node      | Sentence-Transformers Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)        |
| LLM Backend              | Llama-3.3-70B-Versatile via Groq API                                  |
| PDF Processing           | PyPDF Loader                                                          |
| Web Content Extraction   | BeautifulSoup4, Requests                                              |
| URL Validation           | Validators                                                            |
| Persistent Storage       | SQLite3 (Chroma Core backend Engine)                                  |

---
# 📁 Project Structure

```text
├── .venv/               # Isolated Python virtual environment
├── chroma_db/           # Persistent vector database (Git Ignored)
│   └── chroma.sqlite3
├── uploads/             # Temporary PDF storage (Git Ignored)
├── .env                 # Environment variables / API Keys (Git Ignored)
├── .gitignore           # Repository file filter configuration
├── app.py               # Streamlit user interface & state coordinator
├── cloud_llm_handler.py # Groq LLM client & token tracking configuration
├── rag_core.py          # Hybrid retrieval, reranking, and orchestrator logic
├── url_ingestion.py     # Web scraping and URL content extraction
└── requirements.txt     # Project dependencies


⚙️ Installation & Setup
Follow these steps to set up and run the application locally.

1. Clone the Repository
Bash
git clone <your-repository-url>
cd rag-system
2. Create a Virtual Environment
Bash
python -m venv .venv
Activate the environment
Windows

Bash
.venv\Scripts\activate
macOS/Linux

Bash
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a file named .env in the project root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
Note: The .gitignore file completely prevents sensitive credentials from being pushed to GitHub.

5. Launch the Application
Start the Streamlit server:

Bash
streamlit run app.py
The application will launch on your local host and automatically open in your default web browser.

PDF Documents                  Web Content URLs
             │                               │
             ▼                               ▼
       PyPDF Loader                  BeautifulSoup4
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
               Recursive Character Splitter
                     (1000 size / 200 overlap)
                             │
                             ▼
               ┌───────────────────────────┐
               │  Hybrid Parallel Search   │
               ├───────────────────────────┤
               │ Dense: ChromaDB Vector    │
               │ Sparse: BM25 Keywords     │
               └─────────────┬─────────────┘
                             │
                             ▼
                     Ensemble Retriever
                        (50/50 Blend)
                             │
                             ▼
                 Cross-Encoder Reranker
               (ms-marco-MiniLM Top-4 Selection)
                             │
                             ▼
                Groq Llama-3.3-70B Inference
               (Context-Constrained Prompting)
                             │
                             ▼
           Verified Response + Source Attribution



This project is intended for educational and research purposes.
