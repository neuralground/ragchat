#!/usr/bin/env python3
"""
Tests for the vectorstore module.
"""
import pytest
from unittest.mock import patch, MagicMock
import os

from ragchat.document.vectorstore import (
    get_embedding_model,
    initialize_rag,
    get_next_available_number,
    add_documents_to_vectorstore,
    reset_collection
)
from ragchat.cli.commands import get_source_metadata


def test_get_vectorstore():
    """Test getting a vectorstore instance."""
    mock_chroma = MagicMock()
    
    with patch("ragchat.document.vectorstore.Chroma", return_value=mock_chroma), \
         patch("ragchat.document.vectorstore.OllamaEmbeddings"), \
         patch("os.makedirs"):
        
        # Test with default parameters
        vectorstore = initialize_rag("ollama:llama3")
        assert vectorstore == mock_chroma
        
        # Test with custom parameters
        vectorstore = initialize_rag(
            embed_model="text-embedding-3-small",
            persist_dir="/custom/dir",
            api_key="test-key"
        )
        assert vectorstore == mock_chroma


def test_add_documents():
    """Test adding documents to the vectorstore."""
    mock_vectorstore = MagicMock()
    mock_docs = [MagicMock(), MagicMock()]
    
    # Test adding documents to vectorstore
    add_documents_to_vectorstore(mock_vectorstore, mock_docs)
    
    # Verify that add_documents was called with the right parameters
    mock_vectorstore.add_documents.assert_called_once_with(mock_docs)
    

def test_get_next_available_number():
    """Test getting the next available source number."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source_number": 1},
            {"source_number": 3},
            {"source_number": 5}
        ]
    }
    
    # Next available should be 2 (first unused number)
    assert get_next_available_number(mock_vectorstore) == 2
    
    # Test with empty collection
    mock_vectorstore.get.return_value = {"metadatas": []}
    assert get_next_available_number(mock_vectorstore) == 1


def test_reset_collection():
    """Test resetting the collection."""
    mock_vectorstore = MagicMock()
    
    with patch("ragchat.document.vectorstore.initialize_rag", return_value=mock_vectorstore):
        # Test resetting the collection
        result = reset_collection(mock_vectorstore, "ollama:llama3")
        
        # Verify that delete_collection was called
        mock_vectorstore.delete_collection.assert_called_once()
        
        # Verify that we got a new vectorstore instance
        assert result == mock_vectorstore


def test_get_source_metadata():
    """Test getting source metadata."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source_number": 1, "source": "doc1.pdf"},
            {"source_number": 2, "source": "doc2.jpg"}
        ]
    }
    
    # Test getting metadata for source_id 1
    metadata = get_source_metadata(mock_vectorstore, 1)
    assert metadata == {"source_number": 1, "source": "doc1.pdf"}
    mock_vectorstore.get.assert_called_with(where={"source_number": 1}, include=["metadatas"])
    
    # Reset mock for next test
    mock_vectorstore.reset_mock()
    
    # Test getting metadata for source_id 2
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source_number": 2, "source": "doc2.jpg"}
        ]
    }
    metadata = get_source_metadata(mock_vectorstore, 2)
    assert metadata == {"source_number": 2, "source": "doc2.jpg"}
    mock_vectorstore.get.assert_called_with(where={"source_number": 2}, include=["metadatas"])
    
    # Test getting metadata for non-existent source_id
    mock_vectorstore.reset_mock()
    mock_vectorstore.get.return_value = {"metadatas": []}
    metadata = get_source_metadata(mock_vectorstore, 999)
    assert metadata is None
