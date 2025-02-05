"""Chat module for RAGChat.

This module provides chat-related functionality including memory management
and conversation handling.
"""

from .memory import SimpleMemory
from .completion import qa_chain_with_fallback, get_numbered_sources

__all__ = [
    'SimpleMemory',
    'qa_chain_with_fallback',
    'get_numbered_sources'
]

