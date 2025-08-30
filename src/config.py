"""Configuration layer for environment-driven settings.

Loads variables from a `.env` file (via `python-dotenv`) and exposes them as module-level constants
consumed by other modules. Expected variables include:
- `GROQ_API_KEY`: API key to access Groq models
- `CHROMADB_HOST`: Hostname for Chroma HTTP server
- `CHROMADB_PORT`: Port for Chroma HTTP server
- `CHROMADB_SSL`: If truthy, connect to Chroma using SSL
"""
import os
from dotenv import load_dotenv
load_dotenv()

# load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")
CHROMADB_SSL = bool(os.environ.get("CHROMADB_SSL")) # retunrs False if there's no CHROMADB_SSL in .env or if CHROMADB_SSL==""
EVIDENTLY_PROJECT_ID = os.environ.get("EVIDENTLY_PROJECT_ID")
EVIDENTLY_CLOUD_URL = os.environ.get("EVIDENTLY_CLOUD_URL")
RAG_BACKEND_URL = os.environ.get("RAG_BACKEND_URL")
EVIDENTLY_API_KEY = os.environ.get("EVIDENTLY_API_KEY")