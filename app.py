"""
FastAPI application exposing a simple RAG chat engine.

This module provides a FastAPI-based API for a Retrieval-Augmented Generation (RAG) chat system.
It supports health checks, document indexing with vector embeddings, and streaming chat responses
grounded in indexed documents. The API uses an in-memory chat state for prototyping, which is not
suitable for production-grade multi-process or multi-instance deployments.

Endpoints:
- `GET /health`: Performs a liveness check and returns version information.
- `POST /index`: Uploads files, generates, and stores vector embeddings in ChromaDB using a unique `chat_uid`.
- `POST /chat`: Streams a chat response grounded in previously indexed embeddings for a given `chat_uid`.
"""
from src.helpers import *
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
app_state: TempAppState = app.state
app_state.chat_memory = None  # For prototyping only - not suitable for production

@app.get('/health')
async def health_check():
    """
    Perform a simple liveness and version check for the API.

    Returns:
        JSONResponse: A JSON response containing the application name, version, and a status message.
    """
    return JSONResponse(
        content={
            "application": "DSIMON Chat Engine v1",
            "version": "1.0.0",
            "message": "API endpoint working!"
        }
    )

@app.post("/index")
async def process(
    chat_uid: str = Form(...),
    files: List[UploadFile] = None,
):
    """
    Upload documents and generate vector embeddings for a given conversation.

    This endpoint accepts one or more files, validates them, and generates vector embeddings
    stored in ChromaDB, namespaced by the provided `chat_uid`. Files are temporarily saved to
    a directory for processing.

    Args:
        chat_uid (str): Unique identifier for the chat session, used as a suffix for the Chroma collection name.
        files (List[UploadFile]): List of uploaded files to index. Supported file types are defined in `FileUtils.ALLOWED_FILES`.

    Returns:
        JSONResponse: A JSON response indicating success or failure of the indexing process.

    Raises:
        Exception: If file upload or embedding generation fails, returns a JSONResponse with a 400 or 500 status code.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            await FileUtils().upload_files(files, temp_dir)
        except Exception as e:
            exception = traceback.format_exc()
            message = f"Could not proceed to indexing due to exception:"
            logger.info(f"{message}: {exception}")
            return JSONResponse(
                content={"status": f"{message}: {e}"},
                status_code=500
            )
        
        try:
            documents = SimpleDirectoryReader(temp_dir).load_data()
            await EmbeddingUtils().generate_and_store_embeddings(chat_uid, documents)
            message = "Embeddings generated and stored successfully."
        
            return JSONResponse(
                content={"status": message},
                status_code=200
            )
        except Exception as e:
            exception = traceback.format_exc()
            logger.error(exception)
            return JSONResponse(
                content={
                    "status": f"An error occurred during indexing: {str(e)}. Check the system logs for more information.",
                },
                status_code=400
            )

@app.post("/chat")
async def generate(
    request: Request
):
    """
    Stream a chat response grounded in previously indexed documents.

    This endpoint accepts a JSON payload containing the user's query, model, chat session ID, and
    optional chatbot name. It retrieves embeddings from ChromaDB for the given `chat_uid`, initializes
    a LlamaIndex chat engine with streaming and memory, and streams the response tokens to the client.

    Request JSON body:
        {
            "query": str,
            "model": str,
            "chat_uid": str,
            "chatbot_name": str
        }

    Args:
        request (Request): The incoming HTTP request containing the JSON payload.

    Returns:
        StreamingResponse: A streaming response yielding token chunks from the chat engine.

    Raises:
        Exception: If response generation fails, returns a JSONResponse with a 400 status code.
    """
    query = await request.json()
    logger.info(f"""Chat engine started for conversation {query["chat_uid"]}""")
    logger.info(f"""The user's query is: {query["query"]}""")
    
    try:
        response = generate_response(
            query["query"], query["chat_uid"], query["model"], 
            chatbot_name=query["chatbot_name"], app_state=app.state
        )
        return StreamingResponse(content=response)

    except Exception as e:
        exception = traceback.format_exc()
        logger.error(exception)
        return JSONResponse(
            content={
                "status": f"An error occurred during response generation: {str(e)}. Check the system logs for more information.",
            },
            status_code=400
        )

if __name__=="__main__":
    """
    Entry point for running the FastAPI application using Uvicorn.

    Starts the server on host `0.0.0.0` and port `5000` with auto-reload enabled for development.
    """
    import uvicorn
    logger.info("Starting DSIMON Chat Engine...")
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
    