import os
from typing import List, Optional
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


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


class RAGCore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vectorstore = load_vectorstore(persist_directory, self.embeddings)
        self.qa_chain = None
        self.llm = Ollama(model="llama2")  # or "mistral", "codellama", etc.
        
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
        """Create the RAG question-answering chain."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")
        
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
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            ),
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
        # Clear the cached vectorstore
        load_vectorstore.clear()
