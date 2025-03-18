#!/usr/bin/env python3
"""
Test script for RAGChat configuration loading.
This script helps diagnose how environment variables are being loaded.
"""

import os
from pathlib import Path
from ragchat.config import ChatConfig, load_env_file

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
    home_dir = Path.home()
    ragchat_home = os.environ.get("RAGCHAT_HOME", home_dir / ".ragchat")
    
    env_files = [
        Path(".env"),
        Path(".env.local"),
        home_dir / ".env",
        home_dir / ".env.local",
        Path(ragchat_home) / ".env",
        Path(ragchat_home) / ".env.local"
    ]
    
    for env_file in env_files:
        print(check_env_file(env_file))
    
    # Load configuration
    print("\nLoading configuration:")
    config = ChatConfig()
    
    # Print configuration
    print("\nConfiguration values:")
    print(f"  Model: {config.model}")
    print(f"  Provider: {config.provider}")
    print(f"  Embedding model: {config.embed_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Persist directory: {config.persist_dir}")
    print(f"  Debug mode: {config.debug}")
    print(f"  Show thoughts: {config.show_thoughts}")
    print(f"  No fallback: {config.nofallback}")
    
    # Print environment variables
    print("\nEnvironment variables:")
    env_vars = [var for var in os.environ if var.startswith("RAGCHAT_") or var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]]
    
    for var in sorted(env_vars):
        value = os.environ[var]
        # Mask API keys
        if "API_KEY" in var:
            if value:
                value = value[:4] + "..." + value[-4:]
            else:
                value = "(not set)"
        print(f"  {var}={value}")

if __name__ == "__main__":
    main()
