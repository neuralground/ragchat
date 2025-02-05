"""
RAGChat - A RAG-based chatbot using local LLMs.
"""

from .config import ChatConfig
from .chat.memory import SimpleMemory
from .document.vectorstore import initialize_rag
from .main import main

__version__ = "0.1.0"
__all__ = ['ChatConfig', 'SimpleMemory', 'initialize_rag', 'main']


