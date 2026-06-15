import os
from typing import List, Optional, Any
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from sentence_transformers import CrossEncoder


# Cached functions for expensive operations
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
    """Create a hybrid retriever combining vector similarity and BM25.
    
    Args:
        vectorstore: Chroma vectorstore
        documents: List of document chunks for BM25
        
    Returns:
        EnsembleRetriever combining vector and BM25 retrievers
    """
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6
    
    # Create vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Create ensemble retriever with equal weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever


def rerank_results(query, documents, top_k=4):
    """Rerank documents using CrossEncoder.
    
    Args:
        query: Search query
        documents: List of documents to rerank
        top_k: Number of top results to return
        
    Returns:
        List of top_k documents sorted by relevance score
    """
    if not documents:
        return []
    
    cross_encoder = get_cross_encoder()
    
    # Create pairs for scoring
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Score documents
    scores = cross_encoder.predict(pairs)
    
    # Sort documents by score (highest first)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k documents
    return [doc for doc, score in scored_docs[:top_k]]


class HybridRerankRetriever(BaseRetriever):
    """Custom retriever that combines hybrid search with reranking."""
    
    hybrid_retriever: Any
    top_k: int = 4
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query, **kwargs):
        """Retrieve and rerank documents."""
        # Get initial results from hybrid retriever
        initial_docs = self.hybrid_retriever.get_relevant_documents(query, **kwargs)
        
        # Rerank results
        reranked_docs = rerank_results(query, initial_docs, top_k=self.top_k)
        
        return reranked_docs


class RAGCore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vectorstore = load_vectorstore(persist_directory, self.embeddings)
        self.qa_chain = None
        self.llm = Ollama(model="llama2")  # or "mistral", "codellama", etc.
        self.document_chunks = []  # Track all document chunks for BM25
        
    def ingest_pdf(self, pdf_path: str, vectorstore=None) -> None:
        """Load and ingest a PDF document into the vector store.
        
        Args:
            pdf_path: Path to the PDF file to ingest
            vectorstore: Optional existing vectorstore to add documents to.
                        If None, creates or uses self.vectorstore.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add source filename to metadata
        source_filename = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata['source'] = source_filename
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Use provided vectorstore or fall back to instance vectorstore
        target_vectorstore = vectorstore if vectorstore is not None else self.vectorstore
        
        # Create or update vector store
        if target_vectorstore is None:
            target_vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore = target_vectorstore
        else:
            target_vectorstore.add_documents(texts)
        
        # Persist to disk
        target_vectorstore.persist()
        # Track chunks for BM25
        self.document_chunks.extend(texts)
        # Invalidate QA chain cache since documents changed
        self.qa_chain = None
        print(f"Successfully ingested {len(texts)} chunks from {pdf_path}")
    
    def load_vectorstore(self) -> None:
        """Load existing vector store from disk using cached function."""
        self.vectorstore = load_vectorstore(self.persist_directory, self.embeddings)
        if self.vectorstore:
            print("Loaded existing vector store from disk")
        else:
            print("No existing vector store found. Please ingest documents first.")
    
    def create_qa_chain(self) -> None:
        """Create the RAG question-answering chain with hybrid search and reranking."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")
        
        if not self.document_chunks:
            raise ValueError("No document chunks available. Please ingest documents first.")
        
        # Define custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create hybrid retriever with BM25 and vector similarity
        hybrid_retriever = get_hybrid_retriever(self.vectorstore, self.document_chunks)
        
        # Wrap with reranking
        rerank_retriever = HybridRerankRetriever(hybrid_retriever=hybrid_retriever, top_k=4)
        
        # Create retrieval chain with hybrid reranking retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=rerank_retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG system with a question."""
        if self.qa_chain is None:
            self.create_qa_chain()
        
        result = self.qa_chain.invoke({"query": question})
        
        # Extract source documents
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def clear_vectorstore(self) -> None:
        """Clear the vector store and delete persisted data."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("Vector store cleared")
        self.vectorstore = None
        self.qa_chain = None
        self.document_chunks = []
        # Clear the cached vectorstore
        load_vectorstore.clear()
