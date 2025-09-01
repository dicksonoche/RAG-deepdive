"""
Streamlit frontend for the RAG-deepdive system.

This module provides a user-friendly web interface for interacting with the RAG system.
Users can upload files for indexing, configure the backend, and engage in a chat session
with a conversational AI powered by the FastAPI backend. The interface supports streaming
responses and maintains a chat history within the session.

Key features:
- File upload and indexing with a unique `chat_uid`.
- Health check for the backend server.
- Interactive chat with model selection and customizable chatbot name.
- Support for streaming responses from the backend.
"""
import os
import uuid
import mimetypes
import requests
import streamlit as st

# Accepted file types must match `FileUtils.ALLOWED_FILES` in `src/helpers.py`
ACCEPTED_FILE_TYPES = [
    "txt", "csv", "htm", "html", "pdf", "json", "doc", "docx", "pptx"
]

GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

DEFAULT_BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "http://localhost:5000")

def ensure_chat_uid() -> str:
    """
    Ensure a unique chat session ID exists in the Streamlit session state.

    Generates a new UUID if none exists, ensuring each session has a unique identifier
    for indexing and querying.

    Returns:
        str: The chat session ID (UUID).
    """
    if "chat_uid" not in st.session_state:
        st.session_state.chat_uid = str(uuid.uuid4())
    return st.session_state.chat_uid

def check_health(backend_url: str) -> tuple[bool, str]:
    """
    Check the health of the FastAPI backend server.

    Sends a GET request to the `/health` endpoint and returns the status and response details.

    Args:
        backend_url (str): The base URL of the FastAPI backend.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating success and the response details.
    """
    try:
        r = requests.get(f"{backend_url.rstrip('/')}/health", timeout=10)
        if r.status_code == 200:
            return True, r.text
        return False, f"Non-200: {r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)

def index_files(backend_url: str, chat_uid: str, uploaded_files: list) -> requests.Response:
    """
    Upload and index files to the FastAPI backend.

    Prepares uploaded files as multipart form data and sends them to the `/index` endpoint
    for embedding generation and storage.

    Args:
        backend_url (str): The base URL of the FastAPI backend.
        chat_uid (str): The unique identifier for the chat session.
        uploaded_files (list): List of uploaded files from Streamlit.

    Returns:
        requests.Response: The HTTP response from the backend.
    """
    url = f"{backend_url.rstrip('/')}/index"
    files = []
    for upl in uploaded_files:
        file_bytes = upl.getvalue()
        guessed_type, _ = mimetypes.guess_type(upl.name)
        content_type = guessed_type or "application/octet-stream"
        files.append(("files", (upl.name, file_bytes, content_type)))
    data = {"chat_uid": chat_uid}
    return requests.post(url, data=data, files=files, timeout=300)

def stream_chat(backend_url: str, query_payload: dict):
    """
    Stream a chat response from the FastAPI backend.

    Sends a POST request to the `/chat` endpoint and yields response tokens as they are received.
    Handles encoding issues and raises errors for non-200 responses.

    Args:
        backend_url (str): The base URL of the FastAPI backend.
        query_payload (dict): The JSON payload containing the query, model, chat_uid, and chatbot_name.

    Yields:
        str: Token chunks from the backend response.

    Raises:
        RuntimeError: If the chat request fails with a non-200 status code.
    """
    url = f"{backend_url.rstrip('/')}/chat"
    with requests.post(url, json=query_payload, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            text = resp.text
            raise RuntimeError(f"Chat request failed ({resp.status_code}): {text}")
        for chunk in resp.iter_content(chunk_size=1, decode_unicode=False):
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                try:
                    yield chunk.decode("utf-8")
                except UnicodeDecodeError:
                    yield chunk.decode("latin-1", errors="ignore")
            else:
                yield chunk

# --------------------------- UI ---------------------------
st.set_page_config(page_title="DSIMON Chat", page_icon="💬", layout="centered")
st.title("DSIMON Chat UI")

# Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL, help="FastAPI server base URL")
    if st.button("Check backend health"):
        ok, details = check_health(backend_url)
        if ok:
            st.success("Backend healthy")
            st.code(details)
        else:
            st.error("Backend not reachable")
            st.code(details)
    model = st.selectbox("Model", GROQ_MODELS, index=3)
    chatbot_name = st.text_input("Chatbot name (optional)", value="DSIMONAssistant")
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.experimental_rerun()
    st.caption("Ensure the backend is running and ChromaDB is reachable.")

st.subheader("1) Session and Documents")
chat_uid = st.text_input("Chat UID", value=ensure_chat_uid(), help="A unique identifier for this conversation")
uploaded_files = st.file_uploader(
    "Upload files to index", type=ACCEPTED_FILE_TYPES, accept_multiple_files=True, help=f"Accepted: {', '.join(ACCEPTED_FILE_TYPES)}"
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Index files", type="primary", disabled=not uploaded_files):
        try:
            with st.spinner("Indexing files and generating embeddings..."):
                resp = index_files(backend_url, chat_uid, uploaded_files)
            if resp.status_code == 200:
                st.success("Embeddings generated and stored successfully.")
            else:
                st.error(f"Indexing failed ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"Indexing error: {exc}")
with col2:
    if st.button("New Chat UID"):
        st.session_state.chat_uid = str(uuid.uuid4())
        st.experimental_rerun()

st.markdown("---")
st.subheader("2) Chat")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom
user_prompt = st.chat_input("Your message")
if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""
        payload = {
            "query": user_prompt.strip(),
            "model": model,
            "chat_uid": chat_uid,
            "chatbot_name": chatbot_name.strip(),
        }
        try:
            with st.spinner("Generating answer..."):
                for token in stream_chat(backend_url, payload):
                    accumulated += token
                    placeholder.markdown(accumulated)
            st.session_state.messages.append({"role": "assistant", "content": accumulated})
        except Exception as exc:
            st.error(f"Chat error: {exc}")

st.markdown("---")
st.caption("Tip: If you see connection errors, confirm the FastAPI server is running and that CHROMADB_HOST/PORT are set.")