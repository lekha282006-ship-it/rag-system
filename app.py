import streamlit as st
from rag_core import RAGCore
import os

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "rag_core" not in st.session_state:
    st.session_state.rag_core = RAGCore()
    st.session_state.rag_core.load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain_needs_rebuild" not in st.session_state:
    st.session_state.qa_chain_needs_rebuild = False

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []


def main():
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    /* Primary accent color */
    :root {
        --primary-color: #2563EB;
    }
    
    /* Chat user messages */
    .stChatMessage[data-testid="chat-message-0"] {
        background-color: #EFF6FF;
        border-radius: 12px;
    }
    
    /* Chat assistant messages */
    .stChatMessage[data-testid="chat-message-1"] {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E293B;
    }
    
    .css-1d391kg .stHeader {
        color: #2563EB;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    
    /* Page badge styling */
    .page-badge {
        display: inline-block;
        background-color: #2563EB;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("# 📄 Document Intelligence Assistant")
    st.markdown("**AI-powered document Q&A with conversational memory and source verification — runs entirely on local LLMs**")
    st.markdown("<hr style='border: 2px solid #2563EB; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # PDF upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files to add to the knowledge base"
        )
        
        if uploaded_files:
            # Save uploaded files
            os.makedirs("./uploads", exist_ok=True)
            
            if st.button("Ingest PDF"):
                with st.spinner("Ingesting documents..."):
                    newly_processed = []
                    for uploaded_file in uploaded_files:
                        file_path = f"./uploads/{uploaded_file.name}"
                        
                        # Skip if already processed
                        if uploaded_file.name in st.session_state.processed_documents:
                            st.info(f"Skipping {uploaded_file.name} (already processed)")
                            continue
                        
                        try:
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            st.session_state.rag_core.ingest_pdf(file_path)
                            st.session_state.processed_documents.append(uploaded_file.name)
                            newly_processed.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error ingesting {uploaded_file.name}: {str(e)}")
                    
                    if newly_processed:
                        st.session_state.qa_chain_needs_rebuild = True
                        st.success(f"Successfully ingested: {', '.join(newly_processed)}")
                    else:
                        st.info("No new documents to ingest")
        
        st.divider()
        
        # Clear vector store
        if st.button("🗑️ Clear Knowledge Base"):
            st.session_state.rag_core.clear_vectorstore()
            st.session_state.messages = []
            st.session_state.processed_documents = []
            st.session_state.qa_chain_needs_rebuild = True
            st.success("Knowledge base cleared")
        
        st.divider()
        
        # Show loaded documents
        if st.session_state.processed_documents:
            st.header("📄 Loaded Documents")
            for doc_name in st.session_state.processed_documents:
                st.markdown(f"📄 {doc_name}")
        else:
            st.header("📄 Loaded Documents")
            st.caption("No documents loaded")
        
        # Clear chat history
        if st.button("💬 Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared")
        
        st.divider()
        
        # System Status section
        st.header("📊 System Status")
        
        # Get document chunk count
        chunk_count = "No documents loaded"
        status_color = "gray"
        if st.session_state.rag_core.vectorstore is not None:
            try:
                chunk_count = str(len(st.session_state.rag_core.vectorstore.get()))
                status_color = "green"
            except:
                chunk_count = "Error loading count"
        
        # Status indicator
        st.markdown(f"<span style='display:inline-block; width:10px; height:10px; background-color:{status_color}; border-radius:50%; margin-right:8px;'></span>**Status:** {'Documents Loaded' if status_color == 'green' else 'No Documents'}", unsafe_allow_html=True)
        st.markdown(f"**Document Chunks:** {chunk_count}")
        st.markdown(f"**Current Model:** {st.session_state.rag_core.llm.model}")
        
        st.divider()
        
        # Model selection
        st.header("⚙️ Settings")
        model_options = ["llama2", "mistral", "codellama", "phi"]
        selected_model = st.selectbox(
            "Select Ollama Model",
            model_options,
            index=0,
            help="Choose the local LLM model to use"
        )
        
        if selected_model != st.session_state.rag_core.llm.model:
            st.session_state.rag_core.llm.model = selected_model
            st.session_state.qa_chain_needs_rebuild = True
            st.info(f"Model changed to {selected_model}")
        
        st.divider()
        
        # Instructions
        st.markdown("""
        ### Instructions:
        1. Upload PDF documents
        2. Click "Ingest PDF" to add to knowledge base
        3. Ask questions in the chat
        4. View answers with source references
        
        ### Requirements:
        - Ollama must be running locally
        - Install models: `ollama pull llama2`
        """)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        # Extract page number from metadata if available
                        page_num = source['metadata'].get('page', 'N/A')
                        st.markdown(f"<span class='page-badge'>Page {page_num}</span> **Source {i}:**", unsafe_allow_html=True)
                        st.text(source["content"])
                        st.caption(f"Metadata: {source['metadata']}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Rebuild QA chain if needed (model changed or documents ingested)
                    if st.session_state.qa_chain_needs_rebuild:
                        st.session_state.rag_core.qa_chain = None
                        st.session_state.qa_chain_needs_rebuild = False
                    
                    result = st.session_state.rag_core.query(prompt)
                    
                    st.markdown(result["answer"])
                    
                    # Add sources to message
                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    }
                    
                    # Display sources in expander
                    if result["sources"]:
                        with st.expander("📖 View Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                # Extract page number from metadata if available
                                page_num = source['metadata'].get('page', 'N/A')
                                st.markdown(f"<span class='page-badge'>Page {page_num}</span> **Source {i}:**", unsafe_allow_html=True)
                                st.text(source["content"])
                                st.caption(f"Metadata: {source['metadata']}")
                                st.divider()
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    assistant_message = {
                        "role": "assistant",
                        "content": error_message
                    }
        
        # Add assistant message to chat history
        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
