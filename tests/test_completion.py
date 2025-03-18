#!/usr/bin/env python3
"""
Tests for the completion module.
"""
import pytest
from unittest.mock import patch, MagicMock
import base64
import os
import sys

from ragchat.chat.completion import (
    extract_thoughts,
    qa_chain_with_fallback,
    qa_with_images,
    get_numbered_sources
)
from ragchat.config import ChatConfig


def test_extract_thoughts():
    """Test extracting thoughts from text."""
    # Test with string that has thoughts
    text = "This is a response. <think>These are my thoughts.</think> More response."
    result, thoughts = extract_thoughts(text)
    assert result == "This is a response. More response."
    assert thoughts == "These are my thoughts."
    
    # Test with string that has no thoughts
    text = "This is a response with no thoughts."
    result, thoughts = extract_thoughts(text)
    assert result == "This is a response with no thoughts."
    assert thoughts is None
    
    # Test with object that has content attribute
    obj = MagicMock()
    obj.content = "This is content. <think>Content thoughts.</think>"
    result, thoughts = extract_thoughts(obj)
    assert result == "This is content."
    assert thoughts == "Content thoughts."
    
    # Test with dictionary-like object
    obj = {"content": "Dict content. <think>Dict thoughts.</think>"}
    result, thoughts = extract_thoughts(obj)
    assert result == "Dict content."
    assert thoughts == "Dict thoughts."
    
    # Test with non-string, non-dict object
    obj = 12345
    result, thoughts = extract_thoughts(obj)
    assert result == "12345"
    assert thoughts is None


def test_get_numbered_sources():
    """Test getting numbered sources."""
    # Create a mock Chroma vectorstore instead of a list of documents
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source_file": "doc1.pdf", "page": 1, "source_number": 1, "source_id": 1},
            {"source_file": "doc1.pdf", "page": 2, "source_number": 1, "source_id": 1},
            {"source_file": "doc2.jpg", "source_number": 2, "source_id": 2}
        ]
    }
    
    result = get_numbered_sources(mock_vectorstore)
    assert 1 in result
    assert result[1] == "doc1.pdf"
    assert 2 in result
    assert result[2] == "doc2.jpg"


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_qa_with_images(provider):
    """Test question answering with images."""
    # Create a temporary image file
    import tempfile
    from PIL import Image
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
        # Create a simple image
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(temp_path)
    
    try:
        # Mock configuration
        config = ChatConfig(
            model="gpt-4o" if provider == "openai" else "gemma3:27b",
            provider=provider
        )
        
        # Mock image sources
        image_sources = [{"path": temp_path}]
        
        # Mock OpenAI client
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = "OpenAI response"
        
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_openai_response
        
        # Mock Ollama client
        mock_ollama_response = {"message": {"content": "Ollama response"}}
        
        # Test with appropriate provider
        if provider == "openai":
            with patch("ragchat.chat.completion.OpenAI", return_value=mock_openai_client), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = qa_with_images("Test prompt", image_sources, None, config)
                assert result == "OpenAI response"
                mock_openai_client.chat.completions.create.assert_called_once()
        else:
            # For Ollama, create a mock module
            mock_ollama = MagicMock()
            mock_ollama.chat.return_value = mock_ollama_response
            
            # Import the module to patch
            with patch.dict('sys.modules', {'ollama': mock_ollama}):
                # Now we can import it in the function
                result = qa_with_images("Test prompt", image_sources, None, config)
                assert "Ollama response" in str(result)
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_qa_chain_with_fallback():
    """Test question answering with fallback."""
    # Mock configuration
    config = ChatConfig(nofallback=False)
    
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LLM response")
    
    # Mock vectorstore and retriever
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    
    # Mock documents
    mock_docs = [
        MagicMock(page_content="Relevant content", metadata={"source": "doc.pdf", "source_id": 1})
    ]
    
    # Mock memory
    mock_memory = MagicMock()
    mock_memory.load_memory_variables.return_value = {"history": "Chat history"}
    
    # Mock ConversationalRetrievalChain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "LLM response with documents",
        "source_documents": mock_docs
    }
    
    # Test with documents found - pass None as source_id
    with patch("ragchat.chat.completion.get_numbered_sources", return_value={1: "doc.pdf"}), \
         patch("ragchat.chat.completion.ConversationalRetrievalChain.from_llm", return_value=mock_chain):
        
        # Set up mock to return documents
        mock_retriever.invoke.return_value = mock_docs
        
        # Test with no source_id
        result = qa_chain_with_fallback("Test query", mock_vectorstore, mock_llm, mock_memory, None, False)
        assert "LLM response" in str(result["answer"])
        
        # Test with specific source ID
        # For this test, we need to handle the case where a specific source is requested
        # The function will set nofallback=True internally
        with patch("ragchat.chat.completion.extract_thoughts", return_value=("LLM response with documents", None)):
            result = qa_chain_with_fallback("Test query", mock_vectorstore, mock_llm, mock_memory, 1, False)
            assert "LLM response" in str(result["answer"])
        
        # Test with no documents found (fallback)
        mock_retriever.invoke.return_value = []
        result = qa_chain_with_fallback("Test query", mock_vectorstore, mock_llm, mock_memory, None, False)
        assert "LLM response" in str(result["answer"])
        
        # Test with nofallback=True and no documents found
        # We need to adjust our expectations here - it should return a message saying no information is available
        result = qa_chain_with_fallback("Test query", mock_vectorstore, mock_llm, mock_memory, None, True)
        assert "I don't have any information about this" in str(result["answer"])
