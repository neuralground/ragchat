"""Document handling module for RAGChat.

This module provides document processing, loading, and vector store management
functionality.
"""

from .loader import (
    load_document,
    split_documents,
    get_file_size_mb,
    MarkdownLoader
)
from .vectorstore import (
    initialize_rag,
    get_next_available_number
)

__all__ = [
    'load_document',
    'split_documents',
    'get_file_size_mb',
    'MarkdownLoader',
    'initialize_rag',
    'get_next_available_number'
]

