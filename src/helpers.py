"""
Helper utilities for the RAG API and pipeline.

This module provides utilities for:
- File validation and upload to a temporary directory (`FileUtils`).
- Generating and persisting vector embeddings to ChromaDB or Pinecone (`EmbeddingUtils`).
- Initializing ChromaDB or Pinecone connections (`init_chroma`, `init_pinecone`).
- Streaming chat responses using LlamaIndex (`generate_response`).
- Managing in-memory or Redis-backed chat history (`TempAppState`, `init_chat_memory`, `retrieve_chat_memory`).

These utilities support the core functionality of the FastAPI-based RAG system, optimized for
production with retry logic, connection pooling, and comprehensive logging.
"""
import json
import os
import tempfile
import time
import traceback
from typing import List, Optional, Tuple, Any
from pathlib import Path
from retry import retry

# Initialize logger first to ensure availability for imports
API_DIR = Path(__file__).resolve().parent.parent
LOG_FILENAME = str(API_DIR / "logs" / "status_logs.log")
logger = set_logger(
    to_file=True,
    log_file_name=LOG_FILENAME,
    to_console=True,
    custom_formatter=ColorFormatter
)

# Now import dependencies
import chromadb
import redis
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, Request, UploadFile, Form, JSONResponse
from fastapi.responses import PlainTextResponse, StreamingResponse
from llama_index.llms.groq import Groq
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.models import *
from src.prompts import *
from src.config import *
from src.loghandler import *
from src.exceptions import *

# Try importing PineconeVectorStore with fallback
try:
    from llama_index.vector_stores.pinecone import PineconeVectorStore
except ImportError:
    try:
        from llama_index.vector_stores import PineconeVectorStore
        logger.warning("Using fallback import 'llama_index.vector_stores.PineconeVectorStore'.")
    except ImportError:
        logger.error("PineconeVectorStore import failed. Run 'pip install llama-index-vector-stores-pinecone'.")
        raise ImportError("Failed to import PineconeVectorStore.")

DEFAULT_TEMPERATURE = 0.1

# Initialize Redis pool for reuse
redis_pool = None
if USE_REDIS_MEMORY and REDIS_URL:
    try:
        redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis connection pool initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {e}")
        redis_pool = None

def normalize_metadata(metadata: dict) -> dict:
    """
    Normalize metadata values to types supported by ChromaDB (str, int, float, bool, None).

    Converts non-primitive values to JSON strings for compatibility with ChromaDB and Pinecone.

    Args:
        metadata (dict): Metadata dictionary to normalize.

    Returns:
        dict: Normalized metadata dictionary with primitive or JSON-serialized values.
    """
    if not metadata:
        return {}
    normalized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            try:
                normalized[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                normalized[key] = str(value)
    return normalized

class FileUtils:
    """
    Utilities for validating and writing uploaded files to disk.

    Provides methods to check file extensions and handle uploads to a temporary directory.
    """
    ALLOWED_FILES: List[str] = ["txt", "csv", "htm", "html", "pdf", "json", "doc", "docx", "pptx"]

    def is_allowed_file(self, filename: str) -> bool:
        """
        Check if a file has an allowed extension.

        Args:
            filename (str): Name of the file to check.

        Returns:
            bool: True if the file extension is allowed, False otherwise.
        """
        return "." in filename and filename.rsplit(".", 1)[-1].lower() in self.ALLOWED_FILES

    def run_file_checks(self, files: List[UploadFile]) -> JSONResponse:
        """
        Validate a list of uploaded files.

        Checks for missing files, empty filenames, and unsupported extensions.

        Args:
            files (List[UploadFile]): List of uploaded files to validate.

        Returns:
            JSONResponse: Status response (200 for success, 400/415 for errors).
        """
        if not files:
            logger.error("No files provided for upload.")
            return JSONResponse(content={"status": "No file found"}, status_code=400)

        for file in files:
            filename = file.filename
            if not filename:
                logger.error("Empty filename in uploaded files.")
                return JSONResponse(content={"status": "No selected file"}, status_code=400)
            if not self.is_allowed_file(filename):
                message = f"File format {filename.rsplit('.', 1)[-1].lower()} not supported. Allowed: {self.ALLOWED_FILES}"
                logger.warning(message)
                return JSONResponse(content={"status": message}, status_code=415)

        return JSONResponse(content={"status": "success"}, status_code=200)

    async def upload_files(self, files: List[UploadFile], temp_dir: str) -> JSONResponse:
        """
        Write uploaded files to a temporary directory after validation.

        Args:
            files (List[UploadFile]): List of files from the multipart request.
            temp_dir (str): Path to temporary directory for file storage.

        Returns:
            JSONResponse: Status response (200 for success, raises exception on failure).

        Raises:
            UploadError: If file writing fails.
            FileCheckError: If file validation fails.
        """
        file_checks = self.run_file_checks(files)
        if file_checks.status_code != 200:
            raise FileCheckError(file_checks.content["status"])

        try:
            for file in files:
                filepath = os.path.join(temp_dir, file.filename)
                file_content = await file.read()
                with open(filepath, "wb") as buffer:
                    buffer.write(file_content)
            logger.info(f"Successfully uploaded {len(files)} files to {temp_dir}.")
            return JSONResponse(content={"status": "Files uploaded successfully."}, status_code=200)
        except Exception as e:
            message = f"Failed to upload file {file.filename}: {e}"
            logger.error(message)
            raise UploadError(message)

@retry(tries=3, delay=1, backoff=2, exceptions=(Exception,))
def init_pinecone(index_name: str, chat_uid: str) -> PineconeVectorStore:
    """
    Initialize a connection to Pinecone with retry logic.

    Creates or connects to a Pinecone index for cosine similarity searches in the given namespace.

    Args:
        index_name (str): Name of the Pinecone index.
        chat_uid (str): Namespace identifier for embeddings.

    Returns:
        PineconeVectorStore: Configured Pinecone vector store instance.

    Raises:
        ChromaConnectionError: If connection to Pinecone fails after retries.
    """
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not set.")
        raise ChromaConnectionError("PINECONE_API_KEY required for Pinecone connection.")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index {index_name}...")
            pc.create_index(
                name=index_name,
                dimension=384,  # Matches bge-small-en
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index, namespace=chat_uid)
        logger.info(f"Pinecone index {index_name} initialized for namespace {chat_uid}.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone index {index_name}: {e}")
        raise ChromaConnectionError(f"Error connecting to Pinecone: {e}")

@retry(tries=3, delay=1, backoff=2, exceptions=(Exception,))
def init_chroma(collection_name: str, task: str = "retrieve") -> chromadb.api.models.Collection:
    """
    Initialize a ChromaDB connection with retry logic.

    Creates or retrieves a ChromaDB collection for cosine similarity searches.

    Args:
        collection_name (str): Name of the collection.
        task (str, optional): Task type ("retrieve" or "create"). Defaults to "retrieve".

    Returns:
        chromadb.api.models.Collection: ChromaDB collection instance.

    Raises:
        ChromaConnectionError: If connection to ChromaDB fails after retries.
    """
    logger.info(f"Initializing ChromaDB collection {collection_name} for {task}...")
    try:
        chroma_client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=CHROMADB_PORT,
            ssl=CHROMADB_SSL
        )
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection {task}d: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB for {collection_name}: {e}")
        raise ChromaConnectionError(f"Error connecting to ChromaDB: {e}")

class EmbeddingUtils:
    """
    Utilities for generating, storing, and retrieving vector embeddings via ChromaDB or Pinecone.

    Uses HuggingFace embeddings and sentence-level text splitting, with fallbacks to ChromaDB
    if Pinecone is disabled. Optimized for production with caching and retry logic.
    """
    def __init__(self):
        """Initialize embedding utilities with cached HuggingFace model."""
        self.tokenizer = SentenceSplitter()._tokenizer
        self.text_splitter = SentenceSplitter()._split_text
        self.embed_model = DEFAULT_EMBED_MODEL
        self.embed_func = HuggingFaceEmbedding(model_name=self.embed_model)
        logger.info(f"Embedding model initialized: {self.embed_model}")

    async def generate_and_store_embeddings(self, chat_uid: str, documents: List[Document]) -> None:
        """
        Split documents, generate embeddings, and store in ChromaDB or Pinecone.

        Args:
            chat_uid (str): Conversation identifier for namespacing.
            documents (List[Document]): LlamaIndex Document objects to embed.

        Raises:
            EmbeddingError: If embedding generation fails.
            ChromaCollectionError: If storage fails or collection/index is empty.
        """
        logger.info(f"Generating embeddings for {len(documents)} documents, chat_uid: {chat_uid}")
        start_time = time.time()
        try:
            if USE_PINECONE:
                vector_store = init_pinecone(PINECONE_INDEX_NAME, chat_uid)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=storage_context,
                    embed_model=self.embed_func
                )
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                index_size = index.describe_index_stats()["namespaces"].get(chat_uid, {"vector_count": 0})["vector_count"]
                if index_size == 0:
                    raise ChromaCollectionError(f"Pinecone namespace {chat_uid} is empty after upserting.")
                logger.info(f"Stored {index_size} embeddings in Pinecone namespace {chat_uid} in {time.time() - start_time:.2f}s.")
            else:
                collection_name = f"dsimon-{chat_uid}-embeddings"
                chroma_collection = init_chroma(collection_name, task="create")
                doc_split_by_chunk_size = [
                    [
                        {"content": item, "metadata": doc.metadata}
                        for item in self.text_splitter(doc.text, chunk_size=1024)
                    ] if len(self.tokenizer(doc.text)) > 1536 else [{"content": doc.text, "metadata": doc.metadata}]
                    for doc in documents
                ]
                doc_chunks = sum(doc_split_by_chunk_size, [])
                content_list = [doc["content"] for doc in doc_chunks]
                metadata_list = [normalize_metadata(doc["metadata"]) for doc in doc_chunks]
                id_list = [f"embedding-{i+1}" for i in range(len(content_list))]
                embeddings = [self.embed_func.get_text_embedding(item) for item in content_list]
                chroma_collection.upsert(
                    ids=id_list,
                    documents=content_list,
                    metadatas=metadata_list,
                    embeddings=embeddings
                )
                collection_count = chroma_collection.count()
                if collection_count == 0:
                    raise ChromaCollectionError(f"Chroma collection {collection_name} is empty after upserting.")
                logger.info(f"Stored {collection_count} embeddings in Chroma collection {collection_name} in {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to generate/store embeddings for chat_uid {chat_uid}: {e}")
            raise EmbeddingError(f"Error generating embeddings: {e}")

    async def retrieve_embeddings(self, chat_uid: str) -> Tuple[VectorStoreIndex, int]:
        """
        Retrieve a VectorStoreIndex and its size for a given chat session.

        Args:
            chat_uid (str): Conversation identifier for the collection/index.

        Returns:
            Tuple[VectorStoreIndex, int]: Vector store index and collection/index size.

        Raises:
            ChromaCollectionError: If the collection/index is empty or does not exist.
        """
        logger.info(f"Retrieving embeddings for chat_uid: {chat_uid}")
        try:
            if USE_PINECONE:
                vector_store = init_pinecone(PINECONE_INDEX_NAME, chat_uid)
                embeddings = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=self.embed_func)
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                index_size = index.describe_index_stats()["namespaces"].get(chat_uid, {"vector_count": 0})["vector_count"]
                if index_size == 0:
                    raise ChromaCollectionError(f"No embeddings found in Pinecone namespace {chat_uid}.")
                logger.info(f"Retrieved {index_size} embeddings from Pinecone namespace {chat_uid}.")
                return embeddings, index_size
            else:
                collection_name = f"dsimon-{chat_uid}-embeddings"
                chroma_collection = init_chroma(collection_name)
                collection_count = chroma_collection.count()
                if collection_count == 0:
                    raise ChromaCollectionError(f"No embeddings found in ChromaDB for {collection_name}.")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                embeddings = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=self.embed_func)
                logger.info(f"Retrieved {collection_count} embeddings from ChromaDB collection {collection_name}.")
                return embeddings, collection_count
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings for chat_uid {chat_uid}: {e}")
            raise ChromaCollectionError(f"Error retrieving embeddings: {e}")

class TempAppState:
    """
    Container for request-scoped or process-scoped state (prototyping only).

    Attributes:
        chat_memory (ChatMemoryBuffer, optional): In-memory chat history buffer.
        chat_uid (str, optional): Conversation identifier for Redis keys.
    """
    chat_memory: Optional[ChatMemoryBuffer] = None
    chat_uid: Optional[str] = None

def init_chat_memory(choice_k: int) -> ChatMemoryBuffer:
    """
    Initialize a ChatMemoryBuffer for storing conversation history.

    Args:
        choice_k (int): Similarity top-k value to scale token limit.

    Returns:
        ChatMemoryBuffer: Configured chat memory buffer instance.
    """
    token_limit = choice_k * 1024
    logger.info(f"Initializing chat memory with token limit: {token_limit}")
    return ChatMemoryBuffer.from_defaults(token_limit=token_limit)

def retrieve_chat_memory(choice_k: int, app_state: Optional[TempAppState] = None) -> ChatMemoryBuffer:
    """
    Retrieve or create a ChatMemoryBuffer for the current session.

    Uses Redis if `USE_REDIS_MEMORY` is enabled; otherwise falls back to in-memory.

    Args:
        choice_k (int): Similarity top-k value to scale token limit.
        app_state (TempAppState, optional): Application state for in-memory chat history.

    Returns:
        ChatMemoryBuffer: Retrieved or new chat memory buffer.

    Raises:
        ChatEngineError: If Redis connection fails and in-memory fallback is unavailable.
    """
    if USE_REDIS_MEMORY and redis_pool:
        try:
            r = redis.Redis(connection_pool=redis_pool)
            memory_key = f"chat_memory:{app_state.chat_uid if app_state and app_state.chat_uid else 'default'}"
            stored_memory = r.get(memory_key)
            if stored_memory:
                memory_data = json.loads(stored_memory)
                logger.info(f"Retrieved chat memory from Redis for key {memory_key}.")
                return ChatMemoryBuffer.from_defaults(chat_history=memory_data.get("history", []), token_limit=choice_k * 1024)
            memory = init_chat_memory(choice_k)
            r.set(memory_key, json.dumps({"history": memory.chat_history}))
            logger.info(f"Initialized new chat memory in Redis for key {memory_key}.")
            return memory
        except Exception as e:
            logger.warning(f"Redis error for chat memory: {e}. Falling back to in-memory.")
    try:
        if app_state and app_state.chat_memory:
            logger.info("Retrieved in-memory chat history.")
            return app_state.chat_memory
        logger.info("No existing chat memory; creating new.")
        return init_chat_memory(choice_k)
    except Exception as e:
        logger.error(f"Failed to retrieve/create chat memory: {e}")
        raise ChatEngineError(f"Cannot retrieve chat memory: {e}")

async def generate_response(
    query: str,
    chat_uid: str,
    model: str = LLAMA_3_3_70B,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chatbot_name: str = "",
    chat_mode: str = "context",
    verbose: bool = True,
    streaming: bool = True,
    app_state: Optional[TempAppState] = None
) -> StreamingResponse:
    """
    Stream a grounded response from the LlamaIndex chat engine.

    Retrieves embeddings, configures a Groq-based chat engine, and streams responses.
    Persists chat history to Redis if enabled.

    Args:
        query (str): User's query.
        chat_uid (str): Conversation identifier for embeddings.
        model (str, optional): Groq model identifier. Defaults to LLAMA_3_3_70B.
        system_prompt (str, optional): System prompt template. Defaults to DEFAULT_SYSTEM_PROMPT.
        chatbot_name (str, optional): Chatbot name for prompt customization.
        chat_mode (str, optional): LlamaIndex chat mode. Defaults to "context".
        verbose (bool, optional): Enable verbose logging. Defaults to True.
        streaming (bool, optional): Enable streaming responses. Defaults to True.
        app_state (TempAppState, optional): Application state for chat memory.

    Yields:
        str: Response token chunks.

    Raises:
        ChatEngineError: If response generation fails.
    """
    try:
        chatbot_desc = f"Your name is {chatbot_name}. " if chatbot_name else ""
        system_prompt = system_prompt.format(chatbot_desc=chatbot_desc)
        logger.info(f"System prompt: {system_prompt[:100]}...")

        index, index_size = await EmbeddingUtils().retrieve_embeddings(chat_uid)
        Settings.llm = LLMClient().map_task_to_client(task="rag", model=model)

        choice_k = min(30, max(3, index_size // 10 if index_size >= 50 else 5 if index_size >= 20 else 3))
        logger.info(f"Using similarity_top_k: {choice_k} for index size: {index_size}")

        if app_state:
            app_state.chat_uid = chat_uid  # Add chat_uid to state for Redis key
        memory = retrieve_chat_memory(choice_k, app_state)
        if app_state:
            app_state.chat_memory = memory

        chat_engine = index.as_chat_engine(
            chat_mode=chat_mode,
            system_prompt=system_prompt,
            similarity_top_k=choice_k,
            verbose=verbose,
            streaming=streaming,
            memory=memory
        )

        response = chat_engine.stream_chat(query)
        logger.info("Starting response stream...")

        async def stream_tokens():
            try:
                for token in response.response_gen:
                    yield str(token)
                # Persist chat history to Redis if enabled
                if USE_REDIS_MEMORY and redis_pool:
                    try:
                        r = redis.Redis(connection_pool=redis_pool)
                        memory_key = f"chat_memory:{chat_uid}"
                        r.set(memory_key, json.dumps({"history": memory.chat_history}))
                        logger.info(f"Persisted chat history to Redis for key {memory_key}.")
                    except Exception as e:
                        logger.warning(f"Failed to persist chat history to Redis: {e}")
            except Exception as e:
                logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
                raise ChatEngineError(f"Error generating response: {e}")

        return StreamingResponse(stream_tokens(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Response generation failed: {e}\n{traceback.format_exc()}")
        raise ChatEngineError(f"Failed to generate response: {e}")