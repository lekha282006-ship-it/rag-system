# Document Intelligence Assistant

An AI-powered document Q&A application featuring a dual-engine hybrid search pipeline, real-time context streaming, conversational memory, and precise source verification. The application dynamically ingests local PDF documents and extracts live web text from user-provided URLs to form an on-the-fly searchable knowledge base.

---

##  Key Features

* **Multi-Source Ingestion Engine**: Seamlessly processes unstructured content from uploaded local PDF documents and web scraped URLs simultaneously.
* **NLP Hybrid Search Architecture**: Combines vector density embeddings (Dense Retrieval via ChromaDB) with classic statistical keyword indexing (Sparse Retrieval via Rank-BM25) for accurate context matching.
* **Hallucination Guardrails**: Implements strict system prompting constraints to ensure answers are strictly derived from the parsed context.
* **Contextual Chat Memory**: Tracks active dialogue windows to resolve pronoun references and multi-turn follow-up questions cleanly.
* **Transparent Source Auditor**: Includes built-in expanders underneath chat bubbles revealing the origin documents or web links utilized to fulfill responses.

---

##  System Architecture & Tech Stack

* **Frontend Dashboard**: Streamlit
* **RAG Orchestration Framework**: LangChain Suite (`langchain-core`, `langchain-community`, `langchain-groq`)
* **Vector Storage Core**: ChromaDB (Backed by an isolated, persistent SQLite3 instance)
* **Keyword Matching Core**: Rank-BM25
* **Cloud LLM Engine**: Llama-3.3-70b-versatile (Dispatched ultra-fast via Groq hardware acceleration API)
* **Extraction Utilities**: PyPDF, BeautifulSoup4, Requests, and Validators

---

##  Project Directory Layout

```text
├── .venv/               # Isolated Python virtual environment dependencies
├── chroma_db/           # Local persistent SQLite vector payload storage binaries
│   ├── 9bf92cde-.../    # Generated system index tables and structural link lists
│   └── chroma.sqlite3   # Relational lookup index tracking document mappings
├── uploads/             # Temporary server directory staging uploaded PDF streams
├── .env                 # Local security credential parameters (GROQ_API_KEY)
├── .gitignore           # Safeguards workspace metadata from git tracking
├── app.py               # Main UI rendering layout and tracking script
├── cloud_llm.py         # Configures Groq Cloud LLM client instantiation handles
├── rag_core.py          # Script processing document parsing and Hybrid Retrieval
├── url_ingestion.py     # Clean web scraping script for live target URLs
└── requirements.txt     # Complete system dependency tree map

⚙️ Installation & Setup
Clone the Project Repository

Bash
git clone <your-repository-url>
cd rag-system
Initialize a Virtual Environment

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Core System Dependencies

Bash
pip install -r requirements.txt
Add API Token Parameters
Create a .env file directly inside the workspace root and apply your Groq cloud endpoint parameters:

Code snippet
GROQ_API_KEY=your_actual_groq_api_key_here
Boot the Streamlit Server Application Instance

Bash
streamlit run app.py