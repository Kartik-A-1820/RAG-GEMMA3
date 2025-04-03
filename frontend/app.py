import streamlit as st
import requests

st.set_page_config(page_title="RAG with Gemma-3", layout="wide")
st.title("📄🧠 RAG Project — LangChain + Gemma-3")

tab1, tab2 = st.tabs(["🔍 Ask Questions (Gemma3)", "📝 Summarize PDF"])

# ---------- TAB 1: Query Mode ----------
with tab1:
    st.subheader("1. Upload Document for Question Answering")
    uploaded_file = st.file_uploader("Upload (PDF, TXT, DOC/DOCX, CSV)", type=["pdf", "txt", "doc", "docx", "csv"])

    if uploaded_file:
        with st.spinner("Uploading and indexing..."):
            files = {"file": uploaded_file}
            response = requests.post("http://localhost:8000/ingest", files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(result.get("message", "File ingested successfully."))
                st.write("📄 Chunks Created:", result.get("num_documents"))
            else:
                st.error("❌ Failed to ingest document.")

    st.markdown("---")
    st.subheader("2. Ask Your Query (Powered by Gemma3)")

    query = st.text_input("Type your question here:")

    if st.button("Ask Gemma3"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response = requests.post("http://localhost:8000/query/local", json={"query": query})
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Answer Generated!")
                    st.subheader("Answer")
                    st.markdown(result["answer"])

                    st.subheader("🔗 Source References")
                    for i, ref in enumerate(result.get("references", []), start=1):
                        st.markdown(f"**[{i}] Source:** {ref['source']}")
                        preview = " ".join(ref["content"].split()[:100])
                        st.markdown(f"`{preview}...`")
                        st.markdown("---")
                else:
                    st.error("❌ Error fetching answer. Try again.")

# ---------- TAB 2: Summarization Mode ----------
with tab2:
    st.subheader("Upload a PDF to Generate Summary")
    summary_file = st.file_uploader("Upload (PDF, TXT, DOC/DOCX, CSV)", type=["pdf", "txt", "doc", "docx", "csv"], key="summary")

    if summary_file:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                files = {"file": summary_file}
                response = requests.post("http://localhost:8000/summarize_pdf", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("📘 Final Summary")
                    st.markdown(result.get("summary", "No summary returned."))
                else:
                    st.error("❌ Failed to summarize the document.")
