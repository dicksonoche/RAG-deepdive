"""Default prompts used by the chat engine.

`DEFAULT_SYSTEM_PROMPT` is a format string that accepts `{chatbot_desc}` to prepend an optional
assistant persona or name. It instructs the model to rely on retrieved context from the RAG pipeline.
"""
DEFAULT_SYSTEM_PROMPT = """{chatbot_desc}You are a helpful and honest assistant designed for a RAG-powered application. \
Your goal is to use the provided information below to answer my request. These information has been extracted from \
a set of documents, which could include unstructure PDFs, webpages, databases, etc."""