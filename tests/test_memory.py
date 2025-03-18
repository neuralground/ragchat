#!/usr/bin/env python3
"""
Tests for the memory module.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from ragchat.chat.memory import SimpleMemory


def test_simple_memory_initialization():
    """Test SimpleMemory initialization."""
    memory = SimpleMemory()
    assert memory.chat_memory == []
    assert memory.return_messages is True
    assert memory.max_tokens == 5


def test_simple_memory_save_context():
    """Test saving context to memory."""
    memory = SimpleMemory()
    
    # Save first interaction
    memory.save_context({"question": "Hello"}, {"answer": "Hi there"})
    assert len(memory.chat_memory) == 2
    assert isinstance(memory.chat_memory[0], HumanMessage)
    assert isinstance(memory.chat_memory[1], AIMessage)
    assert memory.chat_memory[0].content == "Hello"
    assert memory.chat_memory[1].content == "Hi there"
    
    # Save second interaction
    memory.save_context({"question": "How are you?"}, {"answer": "I'm good"})
    assert len(memory.chat_memory) == 4
    assert memory.chat_memory[2].content == "How are you?"
    assert memory.chat_memory[3].content == "I'm good"


def test_simple_memory_clear():
    """Test clearing memory."""
    memory = SimpleMemory()
    
    # Save some interactions
    memory.save_context({"question": "Hello"}, {"answer": "Hi there"})
    memory.save_context({"question": "How are you?"}, {"answer": "I'm good"})
    assert len(memory.chat_memory) == 4
    
    # Clear memory
    memory.clear()
    assert len(memory.chat_memory) == 0


def test_simple_memory_load_memory_variables():
    """Test loading memory variables."""
    memory = SimpleMemory()
    
    # Save some interactions
    memory.save_context({"question": "Hello"}, {"answer": "Hi there"})
    memory.save_context({"question": "How are you?"}, {"answer": "I'm good"})
    
    # Load memory variables
    variables = memory.load_memory_variables({})
    assert "chat_history" in variables
    assert len(variables["chat_history"]) == 4
    
    # Check message contents
    messages = variables["chat_history"]
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there"
    assert messages[2].content == "How are you?"
    assert messages[3].content == "I'm good"
    
    # Test get_recent_messages
    recent = memory.get_recent_messages(1)
    assert len(recent) == 2
    assert recent[0].content == "How are you?"
    assert recent[1].content == "I'm good"
