#!/usr/bin/env python3
"""
Tests for vision models in RAGChat.
These tests verify that image processing works correctly with different providers.
"""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import base64
from PIL import Image
import numpy as np

from ragchat.config import ChatConfig
from ragchat.chat.completion import qa_with_images
from ragchat.cli.commands import resolve_image_path


@pytest.fixture
def setup_test_images():
    """Create test images for vision model testing."""
    # Create temporary image files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a JPEG image
        jpg_path = os.path.join(temp_dir, "test.jpg")
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(jpg_path)
        
        # Create a PNG image
        png_path = os.path.join(temp_dir, "test.png")
        img.save(png_path)
        
        # Create a WebP image
        webp_path = os.path.join(temp_dir, "test.webp")
        img.save(webp_path, "WEBP")
        
        yield {
            "jpg_path": jpg_path,
            "png_path": png_path,
            "webp_path": webp_path,
            "temp_dir": temp_dir
        }


@pytest.mark.vision
def test_resolve_image_path(setup_test_images):
    """Test resolving image paths."""
    images = setup_test_images
    
    # Test with absolute path
    path, exists = resolve_image_path(images["jpg_path"])
    assert path == images["jpg_path"]
    assert exists is True
    
    # Test with relative path
    rel_path = os.path.relpath(images["jpg_path"])
    path, exists = resolve_image_path(rel_path)
    assert os.path.abspath(path) == os.path.abspath(images["jpg_path"])
    assert exists is True
    
    # Test with non-existent file
    path, exists = resolve_image_path("nonexistent.jpg")
    assert exists is False


@pytest.mark.vision
@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o"),
    ("ollama", "gemma3:27b")
])
def test_qa_with_images(setup_test_images, provider, model):
    """Test question answering with images for different providers."""
    images = setup_test_images
    
    # Skip if API keys are not set in the environment
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Configure test
    config = ChatConfig(
        model=model,
        provider=provider,
        temperature=0.7
    )
    
    # Prepare image sources
    image_sources = [
        {"path": images["jpg_path"]},
        {"path": images["png_path"]},
        {"path": images["webp_path"]}
    ]
    
    # Test with OpenAI
    if provider == "openai":
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = "OpenAI vision model response"
        
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_openai_response
        
        with patch("ragchat.chat.completion.OpenAI", return_value=mock_openai_client), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            
            result = qa_with_images("Describe these images", image_sources, None, config)
            assert result == "OpenAI vision model response"
            
            # Verify that the client was called with the right parameters
            mock_openai_client.chat.completions.create.assert_called_once()
            args, kwargs = mock_openai_client.chat.completions.create.call_args
            
            # Check that all images were included
            assert len(kwargs["messages"][1]["content"]) == 4  # 1 text + 3 images
            assert kwargs["messages"][1]["content"][0]["type"] == "text"
            assert kwargs["messages"][1]["content"][1]["type"] == "image_url"
            assert kwargs["messages"][1]["content"][2]["type"] == "image_url"
            assert kwargs["messages"][1]["content"][3]["type"] == "image_url"
    
    # Test with Ollama
    elif provider == "ollama":
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Ollama vision model response"}}
        
        with patch.dict("sys.modules", {"ollama": mock_ollama}), \
             patch("PIL.Image.open", return_value=MagicMock()):
            
            result = qa_with_images("Describe these images", image_sources, None, config)
            assert result == "Ollama vision model response"
            
            # Verify that ollama.chat was called with the right parameters
            mock_ollama.chat.assert_called_once()
            args, kwargs = mock_ollama.chat.call_args
            
            # Check model and options
            assert kwargs["model"] == model
            assert "temperature" in kwargs["options"]
            
            # Check that messages include system, user and image messages
            messages = kwargs["messages"]
            assert len(messages) == 5  # system + user + 3 images
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert "images" in messages[2]
            assert "images" in messages[3]
            assert "images" in messages[4]


@pytest.mark.vision
def test_webp_conversion(setup_test_images):
    """Test WebP to PNG conversion for compatibility with vision models."""
    images = setup_test_images
    
    # Configure test
    config = ChatConfig(
        model="gpt-4o",
        provider="openai",
        temperature=0.7
    )
    
    # Prepare image sources with WebP
    image_sources = [{"path": images["webp_path"]}]
    
    # Mock OpenAI response
    mock_openai_response = MagicMock()
    mock_openai_response.choices = [MagicMock()]
    mock_openai_response.choices[0].message.content = "WebP conversion test response"
    
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_openai_response
    
    # Test WebP conversion
    with patch("ragchat.chat.completion.OpenAI", return_value=mock_openai_client), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        
        # Use the real PIL Image for this test to verify conversion
        result = qa_with_images("Describe this WebP image", image_sources, None, config)
        assert result == "WebP conversion test response"
        
        # Verify that a PNG file was created from the WebP
        png_path = images["webp_path"].rsplit('.', 1)[0] + '.png'
        assert os.path.exists(png_path)
        
        # Clean up the converted file
        if os.path.exists(png_path):
            os.unlink(png_path)
