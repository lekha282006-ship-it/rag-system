import os
from typing import List, Optional, Any, Dict
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from sentence_transformers import CrossEncoder

# Note: Ensure url_ingestion.py exists in your workspace containing these functions
try:
    from url_ingestion import ingest_urls, validate_url
except ImportError:
    # Fallback placeholders if file isn't created yet to prevent complete crash
    def validate_url(url: str) -> bool: return True
    def ingest_urls(urls: list, processed: set) -> dict: 
        return {'success': [], 'failed': [], 'skipped': [], 'total_chunks': 0, 'all_documents': []}


@st.cache_resource
def get_embeddings():
    """Cached embedding model initialization."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore(persist_directory: str, _embeddings):
    """Cached vectorstore loading."""
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=_embeddings
        )
    return None


@st.cache_resource
def get_cross_encoder():
    """Cached CrossEncoder initialization for reranking."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_hybrid_retriever(vectorstore, documents):
    """Create a hybrid retriever combining vector similarity and BM25."""
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever


def rerank_results(query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
    """Rerank documents using CrossEncoder."""
    if not documents:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


class HybridRerankRetriever(BaseRetriever):
    """Custom retriever that combines hybrid search with reranking."""
    
    hybrid_retriever: Any
    top_k: int = 4

    # Correct Pydantic v2 configuration strategy
    model_config = {
        "arbitrary_types_allowed": True
    }

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve and rerank documents."""
        initial_docs = self.hybrid_retriever.invoke(query, **kwargs)
        reranked_docs = rerank_results(query, initial_docs, top_k=self.top_k)
        return reranked_docs


class RAGCore:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.llm_provider = "groq"
        
        self.embeddings = get_embeddings()
        self.vectorstore = load_vectorstore(self.persist_directory, self.embeddings)
        self.qa_chain = None

        self.llm = ChatGroq(
            model=self.model_name,
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        self.document_chunks = []
        self.processed_urls = set()

    def set_llm(self, provider=None, model_name=None, groq_api_key=None):
        """Placeholder for dynamic alterations."""
        pass

    def ingest_pdf(self, pdf_path: str, vectorstore=None) -> None:
        """Load and ingest a PDF document into the vector store."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        source_filename = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata['source'] = source_filename

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        target_vectorstore = vectorstore if vectorstore is not None else self.vectorstore

        if target_vectorstore is None:
            target_vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore = target_vectorstore
        else:
            target_vectorstore.add_documents(texts)

        # Vectorstore state handles tracking persistence across newer variants
        if hasattr(target_vectorstore, 'persist'):
            target_vectorstore.persist()
            
        self.document_chunks.extend(texts)
        self.qa_chain = None
        print(f"Successfully ingested {len(texts)} chunks from {pdf_path}")

    def load_vectorstore(self) -> None:
        """Load existing vector store from disk using cached function."""
        self.vectorstore = load_vectorstore(self.persist_directory, self.embeddings)
        if self.vectorstore:
            print("Loaded existing vector store from disk")
            try:
                # --- FIX: Fetch raw data to rebuild chunks for BM25 ---
                raw_data = self.vectorstore.get()
                if raw_data and 'documents' in raw_data and 'metadatas' in raw_data:
                    self.document_chunks = [
                        Document(page_content=content, metadata=meta)
                        for content, meta in zip(raw_data['documents'], raw_data['metadatas'])
                    ]
                    print(f"Successfully restored {len(self.document_chunks)} chunks for Hybrid Search.")
            except Exception as e:
                print(f"Could not reconstruct document chunks for BM25: {e}")
        else:
            print("No existing vector store found. Please ingest documents first.")

    def create_qa_chain(self) -> None:
        """Create the RAG question-answering chain with hybrid search and reranking."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")

        if not self.document_chunks:
            raise ValueError("No document chunks available. Please ingest documents first.")

        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        hybrid_retriever = get_hybrid_retriever(self.vectorstore, self.document_chunks)
        rerank_retriever = HybridRerankRetriever(hybrid_retriever=hybrid_retriever, top_k=4)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=rerank_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def query(self, question: str) -> dict:
        """Query the RAG system with a question."""
        # Fallback to normal Groq chat if no documents exist
        if not self.document_chunks:
            response = self.llm.invoke(question)
            return {
                "answer": response.content,
                "sources": []
            }

        if self.qa_chain is None:
            self.create_qa_chain()

        result = self.qa_chain.invoke({"query": question})
        sources = []

        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                })

        return {
            "answer": result["result"],
            "sources": sources
        }

    def ingest_urls(self, urls_list: List[str]) -> dict:
        """Ingest web URLs into the vector store."""
        if not urls_list:
            return {'success': [], 'failed': [], 'skipped': [], 'total_chunks': 0}

        valid_urls = []
        for url in urls_list:
            if validate_url(url):
                valid_urls.append(url)
            else:
                print(f"Invalid URL skipped: {url}")

        if not valid_urls:
            return {'success': [], 'failed': [], 'skipped': urls_list, 'total_chunks': 0}

        results = ingest_urls(valid_urls, self.processed_urls)

        if results.get('all_documents'):
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=results['all_documents'],
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(results['all_documents'])

            self.document_chunks.extend(results['all_documents'])
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
                
            self.qa_chain = None
            print(f"Successfully ingested {results['total_chunks']} chunks from {len(results['success'])} URLs")

        return results

    def clear_vectorstore(self) -> None:
        """Clear the vector store by wiping database contents internally to bypass Windows file locks."""
        self.qa_chain = None
        
        if self.vectorstore is not None:
            try:
                # Get all document IDs currently in the Chroma database
                existing_data = self.vectorstore.get()
                if existing_data and 'ids' in existing_data and existing_data['ids']:
                    # Delete all IDs from the collection internally
                    self.vectorstore.delete(ids=existing_data['ids'])
                    
                # For newer Chroma versions, persist the empty state
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
                    
                print("Chroma collection cleared successfully via internal delete.")
            except Exception as e:
                print(f"Internal database clear failed, attempting fallback: {e}")
                # Fallback: if it's completely empty or errors out, reset the reference
                self.vectorstore = None

        # Reset internal application states
        self.document_chunks = []
        self.processed_urls = set()
        
        # Clear the Streamlit resource cache so it doesn't hold stale connections
        load_vectorstore.clear()
        print("Knowledge base states reset.")