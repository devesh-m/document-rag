import os

import httpx
import streamlit as st


st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")

API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000")


def upload_document(uploaded_file) -> dict:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = httpx.post(f"{API_BASE_URL}/upload", files=files, timeout=240.0)
    response.raise_for_status()
    return response.json()


def query_documents(question: str, top_k: int) -> dict:
    response = httpx.post(
        f"{API_BASE_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=240.0,
    )
    response.raise_for_status()
    return response.json()


st.title("Document Intelligence Assistant")
st.caption("Backend-first RAG demo with FastAPI, SQLite, FAISS, sentence-transformers, and Ollama.")

st.subheader("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None and st.button("Index document", use_container_width=True):
    try:
        payload = upload_document(uploaded_file)
        st.success(payload["message"])
        st.json(payload)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

st.subheader("Ask a question")
question = st.text_area(
    "Question",
    placeholder="Example: What does the uploaded contract say about termination notice?",
    height=140,
)
top_k = st.slider(
    "Retrieved chunks",
    min_value=1,
    max_value=10,
    value=5,
    help="This is the number of document chunks retrieved from FAISS before the prompt is sent to Ollama.",
)

if st.button("Run query", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            payload = query_documents(question.strip(), top_k)
            st.markdown("### Answer")
            st.write(payload["answer"])
            st.caption(f"Retrieved chunks used: {payload['retrieved_chunks']}")
            st.markdown("### Citations")
            for citation in payload["citations"]:
                with st.expander(
                    f"{citation['filename']} | page {citation.get('page_number') or 'unknown'} | score {citation.get('similarity_score')}"
                ):
                    st.write(citation["snippet"])
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
