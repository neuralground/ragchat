"""Command-line interface module for RAGChat.

This module provides CLI-related functionality including command processing,
tab completion, and user interaction handling.
"""

from .completer import FileCompleter
from .commands import (
    process_command,
    handle_help_command,
    handle_add_command,
    handle_list_command,
    handle_remove_command,
    handle_reset_command,
    handle_ask_command
)

__all__ = [
    'FileCompleter',
    'process_command',
    'handle_help_command',
    'handle_add_command',
    'handle_list_command',
    'handle_remove_command',
    'handle_reset_command',
    'handle_ask_command'
]

