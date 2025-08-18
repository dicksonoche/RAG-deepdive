"""Custom exception types for clearer error handling across the app."""
class UploadError(Exception):
    """Raised when a file upload fails due to IO or storage errors."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation fails for any reason."""
    pass

class FileCheckError(Exception):
    """Raised when incoming files fail validation checks."""
    pass

class ChatEngineError(Exception):
    """Raised when the LlamaIndex chat engine encounters a runtime error."""
    pass

class ChromaConnectionError(Exception):
    """Raised when the client cannot connect to the Chroma server."""
    pass

class ChromaCollectionError(Exception):
    """Raised when a Chroma collection is missing, empty, or invalid for the request."""
    pass
