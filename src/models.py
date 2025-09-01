"""
Model and client configuration helpers for the RAG system.

This module defines constants for supported model names and provides a factory class (`LLMClient`)
to instantiate appropriate clients for base chat or RAG flows using Groq and LlamaIndex.
"""
import groq
from llama_index.llms.groq import Groq
from typing import Literal
from src.config import GROQ_API_KEY

# Model constants via Groq
GPT_OSS_20B = "openai/gpt-oss-20b"                              # 8k token context window
GPT_OSS_120B = "openai/gpt-oss-120b"                            # 8k
LLAMA_3_1_8B = "llama-3.1-8b-instant"                           # 6k
LLAMA_3_3_70B = "llama-3.3-70b-versatile"                       # 12k
LLAMA_4_SCOUT_17B = "meta-llama/llama-4-scout-17b-16e-instruct" # 30k
KIMI_K2 = "moonshotai/kimi-k2-instruct"                         # 10k
QWEN_3_32B = "qwen/qwen3-32b"                                   # 6k
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en"

class LLMClient:
    """
    Factory for obtaining LLM clients based on task type.

    Provides methods to instantiate clients for either direct chat completions ("base" task)
    or RAG pipelines ("rag" task) using Groq and LlamaIndex integrations.

    Attributes:
        task (Literal["base", "rag"]): The type of task for which to instantiate a client. Defaults to "rag".
    """
    task: Literal["base", "rag"] = "rag"

    def get_groq(self):
        """
        Instantiate and return a Groq SDK client.

        Uses the configured `GROQ_API_KEY` to initialize a client for direct chat completions.

        Returns:
            groq.Groq: A configured Groq SDK client instance.
        """
        return groq.Groq(api_key=GROQ_API_KEY)

    def get_groq_from_llama_index(self, model: str):
        """
        Return a LlamaIndex `Groq` LLM wrapper for a specific model.

        Initializes a LlamaIndex-compatible Groq client for use in RAG pipelines.

        Args:
            model (str): The model identifier to use (e.g., "llama-3.3-70b-versatile").

        Returns:
            llama_index.llms.groq.Groq: A configured LlamaIndex Groq LLM instance.
        """
        return Groq(model, GROQ_API_KEY)
    
    def map_task_to_client(self, task: str, model: str):
        """
        Map a task name to the appropriate client instantiation method.

        Selects the correct client (Groq SDK or LlamaIndex Groq) based on the task type
        and returns an instance of the client.

        Args:
            task (str): The task type ("base" or "rag").
            model (str): The model identifier to use for the "rag" task.

        Returns:
            Any: An instance of a model client (Groq SDK or LlamaIndex Groq wrapper).

        Raises:
            KeyError: If an invalid task type is provided.
        """
        task_map = {
            "base": self.get_groq,
            "rag": self.get_groq_from_llama_index
        }
        client = task_map.get(task)
        return client(model)