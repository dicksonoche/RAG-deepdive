"""
Custom exception types for the RAG system.

This module defines specific exception classes for handling errors in file uploads, embedding
generation, file validation, chat engine operations, and ChromaDB interactions. These provide
clearer error handling across the application.
"""
class UploadError(Exception):
    """
    Exception raised when a file upload fails due to IO or storage issues.

    Used in file upload operations to indicate failures during file writing or processing.
    """
    pass

class EmbeddingError(Exception):
    """
    Exception raised when embedding generation fails.

    Used in embedding generation to indicate issues with computing or processing embeddings.
    """
    pass

class FileCheckError(Exception):
    """
    Exception raised when incoming files fail validation checks.

    Used in file validation to indicate issues with file presence, names, or formats.
    """
    pass

class ChatEngineError(Exception):
    """
    Exception raised when the LlamaIndex chat engine encounters a runtime error.

    Used in chat response generation to indicate failures in the chat engine.
    """
    pass

class ChromaConnectionError(Exception):
    """
    Exception raised when the client cannot connect to the ChromaDB server.

    Used in ChromaDB operations to indicate connection failures.
    """
    pass

class ChromaCollectionError(Exception):
    """
    Exception raised when a ChromaDB collection is missing, empty, or invalid.

    Used in ChromaDB operations to indicate issues with collection retrieval or storage.
    """
    pass