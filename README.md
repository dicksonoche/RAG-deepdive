# DSimon RAG Engine

A simple yet complete Retrieval-Augmented Generation (RAG) system with:
- **Upload endpoint**: accepts multiple file formats and extracts text
- **ChromaDB**: stores vector embeddings per conversation (`chat_uid`)
- **Chat endpoint**: streams grounded answers using LlamaIndex chat engine + Groq LLMs
- **Structured logging**: colorized console logs and file logs

## Architecture Overview
- **Embeddings**: `HuggingFaceEmbedding` (default model `BAAI/bge-small-en`) is used to embed chunked text.
- **Vector store**: ChromaDB HTTP server holds per-conversation collections named `dsimon-{chat_uid}-embeddings`.
- **RAG engine**: LlamaIndex builds a `VectorStoreIndex` on top of Chroma and exposes a chat engine.
- **LLM**: Groq models via either the raw Groq SDK (base) or the LlamaIndex Groq wrapper (rag); see `src/models.py`.
- **Memory (prototype)**: A `ChatMemoryBuffer` is stored in FastAPI `app.state` (single-process only; not production-safe).

## Key Components
- `app.py`: FastAPI app exposing `/health`, `/index`, `/chat`.
- `src/helpers.py`:
  - `FileUtils`: validates uploads and writes them to a temp dir
  - `EmbeddingUtils`: splits docs, embeds, and persists to Chroma; also retrieves `VectorStoreIndex`
  - `init_chroma`: connects to Chroma HTTP server and returns a collection
  - `generate_response`: creates a LlamaIndex chat engine and streams tokens
- `src/models.py`: model constants and `LLMClient` factory for Groq and LlamaIndex clients
- `src/config.py`: loads environment variables via `python-dotenv`
- `src/prompts.py`: default system prompt (format string with `{chatbot_desc}`)
- `src/loghandler.py`: colorized logging and file/console handlers
- `main.py`: local CLI demo to generate and query embeddings from `./docs`

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11.9 (recommended)
- Virtual environment activated
- ChromaDB server running
- GROQ API key

### **1. Setup Environment**
```bash
# Activate virtual environment
source rag.venv/bin/activate

# Install dependencies (using uv pip install)
uv pip install -r requirements.txt

# Set environment variables
source .env
```

### **2. Start ChromaDB Server**
```bash
# Option 1: Using Docker
docker run -p 8000:8000 chromadb/chroma

# Option 2: Using ChromaDB CLI (if installed)
chroma run --port 8000
```

### **3. Start RAG Backend**
```bash
# In one terminal
source rag.venv/bin/activate
source .env
uvicorn app:app --port 5000 --reload
```

### **4. Add Your Documents**
```bash
# Using the helper script
python3.11 add_documents.py /path/to/your/document.pdf

# Or using Streamlit frontend
streamlit run streamlit_app.py
```

### **5. Run Evaluation**
```bash
# Set the chat_uid from your indexed documents
export EVAL_CHAT_UID='your-documents-chat-uid'

# Run evaluation
python3.11 run_evaluation.py
```

## Requirements
- Python 3.10+
- A running ChromaDB HTTP server (or set environment to your instance)
- A Groq API key

Install Python deps:
```bash
python -m venv rag.venv && source rag.venv/bin/activate
pip install -r requirements.txt
```

Required environment variables (use a local `.env` file or export):
```bash
# Groq
GROQ_API_KEY=your_groq_api_key

# Chroma HTTP server
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
# Optional, default false
CHROMADB_SSL=
```

## Running the API
Start the FastAPI server:
```bash
python app.py
```
This runs Uvicorn on `uvicorn app:app --port 5000 --reload`.
Run chroma `chroma run --port 8000`

Health check:
```bash
curl -s http://localhost:5000/health | jq
```

### Index documents
Send a multipart form with one or more files and a unique `chat_uid`:
```bash
curl -X POST http://localhost:5000/index \
  -F "chat_uid=my-session-1" \
  -F "files=@./docs/sample.pdf" \
  -F "files=@./docs/notes.txt"
```
On success, embeddings are stored in a Chroma collection `dsimon-my-session-1-embeddings`.

### Chat with context
Use the same `chat_uid` to query.
```bash
curl -N -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your name?",
    "model": "llama-3.3-70b-versatile",
    "chat_uid": "my-session-1",
    "chatbot_name": "DSIMONAssistant"
  }'
```
The endpoint streams tokens; `curl -N` keeps output unbuffered.

## Local CLI Demo (optional)
You can also try a minimal local run without the API:
```bash
python main.py
```
- Place documents in `./docs` and ensure `./embeddings` exists.
- The script builds embeddings (if missing) and answers a hardcoded query using Groq via LlamaIndex.

## Evaluation with Evidently AI Cloud
Evaluate your RAG system performance using comprehensive metrics and monitoring:

### Setup
1. Install evaluation dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export EVIDENTLY_API_KEY="your-evidently-api-key"
export EVIDENTLY_CLOUD_URL="https://app.evidently.cloud/"
export EVIDENTLY_PROJECT_ID="your-project-id"  # Optional
export RAG_BACKEND_URL="http://localhost:5000"  # If different from default
```

### Run Evaluation
```bash
python run_evaluation.py
```

### What Gets Evaluated
- **Response Generation**: Success rate, response quality
- **Contradiction Detection**: LLM-based factual consistency checking
- **Text Analysis**: Sentiment, length, and other text metrics
- **Performance Metrics**: Response times, error rates

**Note**: Evidently AI's LLM-based contradiction detection requires an OpenAI API key. If you don't have one, the evaluation will still work with basic metrics (sentiment, text length, etc.) and you can implement custom contradiction detection using your Groq setup.

**Alternative**: For contradiction detection without OpenAI, you can modify the evaluator to use Groq directly, but this requires custom implementation as Evidently AI doesn't natively support Groq for LLM-based evaluation.

### Custom Evaluation
```python
from src.evaluator import RAGEvaluator, EvaluationConfig

config = EvaluationConfig(
    backend_url="http://localhost:5000",
    evidently_cloud_url="https://app.evidently.cloud/",
    project_id="your-project-id"
)

evaluator = RAGEvaluator(config)
results = evaluator.evaluate_rag_pipeline(
    questions=["Your question here"],
    reference_answers=["Expected answer"],
    chat_uid="custom-eval-1"
)
```

### Evaluation Logs
- Console: Real-time evaluation progress
- File: `logs/evaluation_logs.log` for detailed analysis

## Streamlit Frontend (optional)
Start an interactive UI to upload files, build embeddings, and chat with conversation history:
```bash
streamlit run streamlit_app.py
```

What you can do in the UI:
- **Sidebar**:
  - **Backend URL**: point to your FastAPI server (default `http://localhost:5000`).
  - **Check backend health**: pings `/health` to verify connectivity.
  - **Model**: choose a Groq model.
  - **Chatbot name**: optional persona injected into the system prompt.
  - **Clear chat history**: clears only the frontend conversation display.
- **1) Session and Documents**:
  - **Chat UID**: auto-generated per session; used to namespace embeddings and memory. Use the same value when chatting.
  - **New Chat UID**: create a fresh server-side memory namespace; use when you want a clean context.
  - **Upload files** (same accepted types as backend) and click **Index files** to generate embeddings for the current `Chat UID`.
- **2) Chat**:
  - Type your message in the chat input at the bottom. Messages and responses appear top-to-bottom and persist during your session.
  - The backend streams tokens; the assistant bubble updates in real time.

Notes:
- Set `RAG_BACKEND_URL` if your FastAPI server is not at `http://localhost:5000`.
- The UI enforces the same accepted file types as the backend.
- Ensure ChromaDB is running and the backend env vars are configured.
- "Clear chat history" resets only the UI; it does not clear the backend memory. Use **New Chat UID** to start a fresh context on the backend.
- If you see a Hugging Face tokenizers warning about parallelism, you can silence it by setting:
  ```bash
  export TOKENIZERS_PARALLELISM=false
  ```

## Logging
- Console logs are colorized and human-friendly.
- File logs are written to `./logs/status_logs.log`.

## Notes and Limitations
- `app.state.chat_memory` is process-local and will not be shared across workers/instances.
- Ensure your ChromaDB instance is reachable from the API.
- Model selection, chunking, and top-k heuristics in `generate_response` are configurable; tune for your data.

## Project Structure
```text
RAG-deepdive/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── logs/
├── chroma/
├── docs/
├── streamlit_app.py
├── run_evaluation.py
└── src/
    ├── config.py
    ├── models.py
    ├── helpers.py
    ├── prompts.py
    ├── exceptions.py
    ├── loghandler.py
    └── evaluator.py
```
