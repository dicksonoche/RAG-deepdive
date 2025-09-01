"""
Configuration layer for environment-driven settings.

This module loads environment variables from a `.env` file using `python-dotenv` and exposes
them as module-level constants for use across the RAG system. It centralizes configuration
for API keys, database connections, and backend URLs.

Expected environment variables:
- GROQ_API_KEY: API key for accessing Groq models.
- CHROMADB_HOST: Hostname for the ChromaDB HTTP server.
- CHROMADB_PORT: Port for the ChromaDB HTTP server.
- CHROMADB_SSL: If set to a truthy value, enables SSL for ChromaDB connections.
- EVIDENTLY_PROJECT_ID: Project ID for Evidently AI cloud.
- EVIDENTLY_CLOUD_URL: URL for the Evidently AI cloud service.
- RAG_BACKEND_URL: URL for the RAG FastAPI backend.
- EVIDENTLY_API_KEY: API key for Evidently AI cloud.
- PINECONE_API_KEY: API key for Pinecone vector DB (for production).
- PINECONE_INDEX_NAME: Name of the Pinecone index (default: 'dsimon-rag').
- REDIS_URL: URL for Redis (e.g., from Upstash: 'redis://user:pass@host:port').
- USE_PINECONE: Set to 'true' to use Pinecone instead of Chroma (for production).
- USE_REDIS_MEMORY: Set to 'true' to use Redis for chat memory (for production).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Existing vars...
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")
CHROMADB_SSL = bool(os.environ.get("CHROMADB_SSL"))
EVIDENTLY_PROJECT_ID = os.environ.get("EVIDENTLY_PROJECT_ID")
EVIDENTLY_CLOUD_URL = os.environ.get("EVIDENTLY_CLOUD_URL")
RAG_BACKEND_URL = os.environ.get("RAG_BACKEND_URL")
EVIDENTLY_API_KEY = os.environ.get("EVIDENTLY_API_KEY")

# For production
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "dsimon-rag")
REDIS_URL = os.environ.get("REDIS_URL")
USE_PINECONE = os.environ.get("USE_PINECONE", "false").lower() == "true"
USE_REDIS_MEMORY = os.environ.get("USE_REDIS_MEMORY", "false").lower() == "true"