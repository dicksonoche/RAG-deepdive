import os
import uuid
import mimetypes
import requests
import streamlit as st

# Simple Streamlit frontend for RAG-deepdive
# - Upload accepted files and index them under a chat_uid
# - Send a query and stream the answer from the FastAPI backend

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
    if "chat_uid" not in st.session_state:
        st.session_state.chat_uid = str(uuid.uuid4())
    return st.session_state.chat_uid


def check_health(backend_url: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{backend_url.rstrip('/')}/health", timeout=10)
        if r.status_code == 200:
            return True, r.text
        return False, f"Non-200: {r.status_code} {r.text}"
    except Exception as e:
        return False, str(e)


def index_files(backend_url: str, chat_uid: str, uploaded_files: list) -> requests.Response:
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
    url = f"{backend_url.rstrip('/')}/chat"
    with requests.post(url, json=query_payload, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            # Surface server error
            text = resp.text
            raise RuntimeError(f"Chat request failed ({resp.status_code}): {text}")

        # Stream token chunks from the backend; ensure text
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
    # add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # stream assistant response
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