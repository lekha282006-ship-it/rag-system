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

if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []


def main():
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    :root {
        --primary-color: #2563EB;
    }
    .stChatMessage[data-testid="chat-message-0"] {
        background-color: #EFF6FF;
        border-radius: 12px;
    }
    .stChatMessage[data-testid="chat-message-1"] {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] { 
        background-color: #1E293B; 
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] .stMarkdown h3 { 
        color: #2563EB; 
    }
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
    st.markdown("**AI-powered document Q&A with conversational memory and source verification**")
    st.markdown("<hr style='border: 2px solid #2563EB; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files to add to the knowledge base"
        )

        if uploaded_files:
            os.makedirs("./uploads", exist_ok=True)

            if st.button("Ingest PDF"):
                with st.spinner("Ingesting documents..."):
                    newly_processed = []
                    for uploaded_file in uploaded_files:
                        file_path = f"./uploads/{uploaded_file.name}"

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

        st.header("🌐 URL Ingestion")
        url_input = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            help="Enter web URLs to ingest content from"
        )

        if st.button("Ingest URLs"):
            if url_input.strip():
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                with st.spinner("Fetching and processing URLs..."):
                    try:
                        results = st.session_state.rag_core.ingest_urls(urls)

                        for success_item in results['success']:
                            if success_item['url'] not in st.session_state.processed_urls:
                                st.session_state.processed_urls.append(success_item['url'])

                        if results['success']:
                            st.session_state.qa_chain_needs_rebuild = True
                            success_urls = [item['url'] for item in results['success']]
                            st.success(f"Successfully ingested {len(success_urls)} URLs ({results['total_chunks']} chunks)")

                        if results['failed']:
                            failed_info = [f"{item['url']}: {item['error']}" for item in results['failed']]
                            st.error(f"Failed URLs:\n" + "\n".join(failed_info))

                        if results['skipped']:
                            st.info(f"Skipped {len(results['skipped'])} URLs (already processed)")

                    except Exception as e:
                        st.error(f"Error processing URLs: {str(e)}")
            else:
                st.warning("Please enter at least one URL")

        st.divider()

        if st.button("🗑️ Clear Knowledge Base"):
            st.session_state.rag_core.clear_vectorstore()
            st.session_state.messages = []
            st.session_state.processed_documents = []
            st.session_state.processed_urls = []
            st.session_state.qa_chain_needs_rebuild = True
            st.success("Knowledge base cleared")

        st.divider()

        st.header("📄 Loaded Sources")

        if st.session_state.processed_documents or st.session_state.processed_urls:
            if st.session_state.processed_documents:
                st.markdown("**Documents:**")
                for doc_name in st.session_state.processed_documents:
                    st.markdown(f"📄 {doc_name}")

            if st.session_state.processed_urls:
                st.markdown("**URLs:**")
                for url in st.session_state.processed_urls:
                    short_url = url[:50] + "..." if len(url) > 50 else url
                    st.markdown(f"🌐 {short_url}")
        else:
            st.caption("No sources loaded")

        if st.button("💬 Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared")

        st.divider()

        st.header("📊 System Status")

        chunk_count = "No documents loaded"
        status_color = "gray"
        if st.session_state.rag_core.vectorstore is not None:
            try:
                chunk_count = str(len(st.session_state.rag_core.vectorstore.get()))
                status_color = "green"
            except:
                chunk_count = "Error loading count"

        st.markdown(f"<span style='display:inline-block; width:10px; height:10px; background-color:{status_color}; border-radius:50%; margin-right:8px;'></span>**Status:** {'Documents Loaded' if status_color == 'green' else 'No Documents'}", unsafe_allow_html=True)
        st.markdown(f"**Document Chunks:** {chunk_count}")
        st.markdown(f"**Current Model:** {st.session_state.rag_core.model_name} (Groq)")

        st.divider()

        st.header("⚙️ Settings")


        st.success("🚀 Using Groq Cloud")
        st.caption("Model: llama-3.3-70b-versatile")

        st.divider()

        st.markdown("""
        ### Instructions:
        1. Upload PDF documents
        2. Click "Ingest PDF" to add to knowledge base
        3. Ask questions in the chat
        4. View answers with source references

        ### Requirements:
- Internet connection required
- GROQ_API_KEY configured in `.env`
        """)

    # Main chat interface
    st.header("💬 Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        metadata = source['metadata']
                        source_type = metadata.get('source_type', 'document')

                        if source_type == 'url':
                            url = metadata.get('url', 'Unknown URL')
                            domain = metadata.get('domain', 'Unknown')
                            title = metadata.get('original_title', domain)
                            short_url = url[:60] + "..." if len(url) > 60 else url

                            st.markdown(f"<span class='page-badge'>🌐 URL</span> **Source {i}:**", unsafe_allow_html=True)
                            st.markdown(f"**{title}**")
                            st.caption(f"From: {short_url}")
                             # Shows a clean preview snippet inside the UI expander element
                            preview_text = source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"]
                            st.text(preview_text)
                            st.caption(f"Domain: {domain} | Fetched: {metadata.get('fetch_time', 'N/A')[:10]}")
                        else:
                            page_num = metadata.get('page', 'N/A')
                            source_name = metadata.get('source', 'Unknown')

                            st.markdown(f"<span class='page-badge'>📄 Doc</span> **Source {i}:**", unsafe_allow_html=True)
                            st.markdown(f"**{source_name}**")
                            st.text(source["content"])
                            st.caption(f"Page: {page_num}")

                        st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.qa_chain_needs_rebuild:
                        st.session_state.rag_core.qa_chain = None
                        st.session_state.qa_chain_needs_rebuild = False
                        
                    result = st.session_state.rag_core.query(prompt)
                
                    st.markdown(result["answer"])

                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    }

                    if result["sources"]:
                        with st.expander("📖 View Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                metadata = source['metadata']
                                source_type = metadata.get('source_type', 'document')

                                if source_type == 'url':
                                    url = metadata.get('url', 'Unknown URL')
                                    domain = metadata.get('domain', 'Unknown')
                                    title = metadata.get('original_title', domain)
                                    short_url = url[:60] + "..." if len(url) > 60 else url

                                    st.markdown(f"<span class='page-badge'>🌐 URL</span> **Source {i}:**", unsafe_allow_html=True)
                                    st.markdown(f"**{title}**")
                                    st.caption(f"From: {short_url}")
                                    st.text(source["content"])
                                    st.caption(f"Domain: {domain} | Fetched: {metadata.get('fetch_time', 'N/A')[:10]}")
                                else:
                                    page_num = metadata.get('page', 'N/A')
                                    source_name = metadata.get('source', 'Unknown')

                                    st.markdown(f"<span class='page-badge'>📄 Doc</span> **Source {i}:**", unsafe_allow_html=True)
                                    st.markdown(f"**{source_name}**")
                                    st.text(source["content"])
                                    st.caption(f"Page: {page_num}")

                                st.divider()

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    assistant_message = {
                        "role": "assistant",
                        "content": error_message
                    }

        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()