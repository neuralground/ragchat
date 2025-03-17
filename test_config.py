#!/usr/bin/env python3
"""
Test script for RAGChat configuration loading.
This script helps diagnose how environment variables are being loaded.
"""

import os
from pathlib import Path
from src.ragchat.config import ChatConfig, load_env_file

def check_env_file(path):
    """Check if an environment file exists at the given path."""
    if path.exists():
        return f"✅ Found: {path}"
    else:
        return f"❌ Not found: {path}"

def main():
    """Test configuration loading."""
    print("RAGChat Configuration Test")
    print("=========================")
    
    # Check for .env files
    print("\nChecking for .env files:")
    package_dir = Path(__file__).parent / "src" / "ragchat"
    default_env_path = package_dir / 'default.env'
    print(check_env_file(default_env_path))
    
    user_default_env_path = Path.home() / '.ragchat' / '.env'
    print(check_env_file(user_default_env_path))
    
    home_env_path = Path.home() / '.env'
    print(check_env_file(home_env_path))
    
    if ragchat_home := os.environ.get('RAGCHAT_HOME'):
        ragchat_home_env_path = Path(ragchat_home) / '.env'
        print(check_env_file(ragchat_home_env_path))
    else:
        print("❌ RAGCHAT_HOME environment variable not set")
    
    current_env_path = Path('.env')
    print(check_env_file(current_env_path))
    
    # Load configuration
    print("\nLoading configuration from environment variables...")
    config = ChatConfig.from_env()
    
    # Print configuration
    print("\nConfiguration:")
    print(f"Model: {config.model}")
    print(f"Embed Model: {config.embed_model}")
    print(f"Persist Dir: {config.persist_dir}")
    print(f"No Fallback: {config.nofallback}")
    print(f"Debug: {config.debug}")
    print(f"Show Thoughts: {config.show_thoughts}")
    
    # Print environment variables
    print("\nEnvironment Variables:")
    print(f"RAGCHAT_MODEL = {os.environ.get('RAGCHAT_MODEL', 'not set')}")
    print(f"RAGCHAT_EMBED_MODEL = {os.environ.get('RAGCHAT_EMBED_MODEL', 'not set')}")
    print(f"RAGCHAT_PERSIST_DIR = {os.environ.get('RAGCHAT_PERSIST_DIR', 'not set')}")
    print(f"RAGCHAT_NOFALLBACK = {os.environ.get('RAGCHAT_NOFALLBACK', 'not set')}")
    print(f"RAGCHAT_DEBUG = {os.environ.get('RAGCHAT_DEBUG', 'not set')}")
    print(f"RAGCHAT_SHOW_THOUGHTS = {os.environ.get('RAGCHAT_SHOW_THOUGHTS', 'not set')}")

if __name__ == "__main__":
    main()
