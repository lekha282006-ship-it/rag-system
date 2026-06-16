import os
import streamlit as st
from rag_core import RAGCore

def main():
    # 1. Native Streamlit page configuration setup
    st.set_page_config(
        page_title="Document Intelligence Assistant",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Document Intelligence Assistant")
    st.caption("AI-powered document Q&A with conversational memory and source verification")
    st.write("---")

    # 2. Initialize the backend engine core inside the session state
    if "rag_core" not in st.session_state:
        st.session_state.rag_core = RAGCore()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ------------------------------------------------------------------
    # 📚 SIDEBAR: Document Ingestion Control Center
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("📚 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("🚀 Ingest PDFs", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing and indexing documents..."):
                    os.makedirs("./uploads", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("./uploads", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Ingest via backend core
                        st.session_state.rag_core.ingest_pdf(file_path)
                st.success("All documents processed successfully!")
                st.rerun()
            else:
                st.warning("Please upload at least one PDF file.")

        st.write("---")
        st.header("🌐 URL Ingestion")
        urls_input = st.text_area("Enter URLs (one per line)", height=100)
        
        if st.button("🔗 Ingest URLs", use_container_width=True):
            if urls_input.strip():
                urls_list = [url.strip() for url in urls_input.split("\n") if url.strip()]
                with st.spinner("Scraping and analyzing web content..."):
                    st.session_state.rag_core.ingest_urls(urls_list)
                st.success("URLs processed successfully!")
                st.rerun()
            else:
                st.warning("Please enter at least one valid URL.")

        st.write("---")
        st.header("📊 System Status")
        
        # Calculate dynamic text metrics safely
        chunk_count = len(st.session_state.rag_core.document_chunks)
        if chunk_count > 0:
            st.metric(label="Document Chunks Loaded", value=f"{chunk_count}")
            st.info("✅ System active and armed with context.")
        else:
            st.metric(label="Document Chunks Loaded", value="0")
            st.warning("⚠️ No data loaded. Operating on baseline intelligence.")

        st.write("---")
        st.header("⚙️ Settings")
        if st.button("🗑️ Clear Knowledge Base", type="primary", use_container_width=True):
            with st.spinner("Wiping environment tables..."):
                st.session_state.rag_core.clear_vectorstore()
                st.session_state.messages = []
            st.success("Knowledge base completely reset!")
            st.rerun()

    # ------------------------------------------------------------------
    # 💬 MAIN INTERFACE: Conversational Chat Pipeline
    # ------------------------------------------------------------------
    st.subheader("💬 Chat")

    # Display historical session messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 View Verification Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**📄 Source {idx + 1}**: {src['metadata'].get('source', 'Unknown')}")
                        st.caption(src["content"])

    # Accept active chat inputs
    if user_query := st.chat_input("Ask a question about your files..."):
        # Display user bubble
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context indexes..."):
                try:
                    result = st.session_state.rag_core.query(user_query)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.write(answer)
                    if sources:
                        with st.expander("📖 View Verification Sources"):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**📄 Source {idx + 1}**: {src['metadata'].get('source', 'Unknown')}")
                                st.caption(src["content"])

                    # Save to state historical log
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")


# 3. Clean runtime method entry point validation
if __name__ == "__main__":
    main()