#!/usr/bin/env python3
"""
Tests for the configuration module.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ragchat.config import ChatConfig, load_env_file, get_default_base_dir


@patch('pathlib.Path.mkdir')
def test_default_config(mock_mkdir):
    """Test default configuration values."""
    # Mock the default persist_dir
    with patch('ragchat.config.get_default_base_dir', return_value=Path('/mock/home/.ragchat')):
        config = ChatConfig()
        assert config.model == "llama3.3"
        assert config.embed_model == "nomic-embed-text"
        assert config.persist_dir == "/mock/home/.ragchat/chroma_db"
        assert config.nofallback is False
        assert config.debug is False
        assert config.show_thoughts is False
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.api_key is None
        assert config.api_base is None
        assert config.provider is None


@patch('pathlib.Path.mkdir')
def test_config_from_env(mock_mkdir):
    """Test loading configuration from environment variables."""
    with patch.dict(os.environ, {
        'RAGCHAT_MODEL': 'gpt-4o',
        'RAGCHAT_EMBED_MODEL': 'text-embedding-3-small',
        'RAGCHAT_PERSIST_DIR': '/test/dir',
        'RAGCHAT_NOFALLBACK': 'true',
        'RAGCHAT_DEBUG': 'true',
        'RAGCHAT_SHOW_THOUGHTS': 'true',
        'RAGCHAT_TEMPERATURE': '0.5',
        'RAGCHAT_MAX_TOKENS': '1000',
        'RAGCHAT_API_KEY': 'test-key',
        'RAGCHAT_API_BASE': 'https://test.api',
        'RAGCHAT_PROVIDER': 'openai'
    }):
        config = ChatConfig.from_env()
        assert config.model == 'gpt-4o'
        assert config.embed_model == 'text-embedding-3-small'
        assert config.persist_dir == '/test/dir'
        assert config.nofallback is True
        assert config.debug is True
        assert config.show_thoughts is True
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.api_key == 'test-key'
        assert config.api_base == 'https://test.api'
        assert config.provider == 'openai'


@patch('pathlib.Path.mkdir')
def test_get_default_base_dir(mock_mkdir):
    """Test the default base directory function."""
    with patch.dict(os.environ, {}, clear=True), \
         patch('pathlib.Path.home', return_value=Path('/mock/home')):
        base_dir = get_default_base_dir()
        assert base_dir == Path('/mock/home/.ragchat')
    
    with patch.dict(os.environ, {'RAGCHAT_HOME': '/custom/path'}):
        base_dir = get_default_base_dir()
        assert base_dir == Path('/custom/path')


def test_env_file_loading_priority():
    """Test the priority order of .env file loading."""
    # This is a simplified test that just verifies the function runs without errors
    
    # Mock dotenv.load_dotenv to prevent actual file loading
    with patch('dotenv.load_dotenv', return_value=True), \
         patch('pathlib.Path.exists', return_value=True):
        
        # Should run without errors
        load_env_file()
        
        # If we got here, the test passed
        assert True
