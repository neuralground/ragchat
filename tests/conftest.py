#!/usr/bin/env python3
"""
Configuration for pytest.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "vision: mark a test as a vision model test")


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up environment variables for testing."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set up test environment variables
    os.environ["RAGCHAT_DEBUG"] = "true"
    os.environ["RAGCHAT_PERSIST_DIR"] = str(Path(__file__).parent / "test_data")
    
    # Create test data directory if it doesn't exist
    Path(os.environ["RAGCHAT_PERSIST_DIR"]).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
