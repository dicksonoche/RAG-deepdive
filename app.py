"""FastAPI application exposing a simple RAG chat engine.

Endpoints:
- `GET /health`: liveness check
- `POST /index`: upload files and generate/store vector embeddings in ChromaDB (per `chat_uid`)
- `POST /chat`: stream a chat response grounded in the previously indexed embeddings for `chat_uid`

This API is intended for prototyping. The in-process `app.state.chat_memory` is not suitable for
multi-process or multi-instance deployments.
"""
from src.helpers import *

app = FastAPI()

app_state: TempAppState = app.state
app_state.chat_memory = None # for prototyping only - don't use this in production

@app.get('/health')
async def health_check():
    """Simple liveness and version check for the API."""
    return JSONResponse(
        content={
            "application": "AISOC Chat Engine v1",
            "version": "1.0.0",
            "message": "API endpoint working!"
        }
    )

@app.post("/index")
async def process(
    chat_uid: str = Form(...),
    files: List[UploadFile] = None,
    # urls: List[str] = None
): 
    """Upload documents and build vector embeddings for a given conversation.

    Args:
        chat_uid (str): Unique identifier for the chat session; used as the Chroma collection name suffix.
        files (List[UploadFile]): One or more files to index. Supported types are defined in `FileUtils.ALLOWED_FILES`.

    Returns:
        JSONResponse: Status message for success or failure.
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
                    "status": f"An error occured during indexing: {str(e)}.                         Check the system logs for more information.",
                },
                status_code=400
            )

@app.post("/chat")
async def generate(
    request: Request
):
    """Stream a chat response grounded in previously indexed documents.

    Request JSON body:
        {
            "query": str,
            "model": str,
            "chat_uid": str,
            "chatbot_name": str
        }

    Behavior:
    - Loads embeddings for `chat_uid` from Chroma
    - Initializes a LlamaIndex chat engine with streaming and memory
    - Streams token chunks back to the client as they are generated
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
                "status": f"An error occured during response generation: {str(e)}. Check the system logs for more information.",
            },
            status_code=400
        )

if __name__=="__main__":
    import uvicorn
    logger.info("Starting AISOC Chat Engine...")
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
