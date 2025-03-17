# RAGChat

A Retrieval-Augmented Generation (RAG) chatbot using local and cloud LLMs.

## Features

- Document ingestion with support for PDF, HTML, EPUB, Markdown, and text files
- Vector storage with ChromaDB
- Multiple LLM provider support:
  - Local LLMs via Ollama
  - OpenAI (GPT-3.5, GPT-4)
  - Anthropic (Claude 3 models)
  - Google (Gemini models)
- Command-line interface with tab completion
- Conversation memory
- Source tracking and citation

## Installation

```bash
pip install .
```

## Usage

Basic usage:

```bash
python -m ragchat
```

With custom configuration:

```bash
# Using local Ollama model
python -m ragchat --model llama3.3 --embed-model nomic-embed-text --persist-dir ./my_docs

# Using OpenAI GPT-4
python -m ragchat --model gpt-4 --api-key your_openai_api_key

# Using Anthropic Claude
python -m ragchat --model claude-3-opus --api-key your_anthropic_api_key

# Using Google Gemini
python -m ragchat --model gemini-pro --api-key your_google_api_key
```

## Configuration

RAGChat can be configured in multiple ways, with the following priority (highest to lowest):

1. Command line arguments
2. User's `.env` file in the current directory
3. User's `.env` file in `$RAGCHAT_HOME` directory (if set)
4. User's `.env` file in home directory (`~/.env`)
5. User's `.env` file in `~/.ragchat` directory
6. Application's default settings

To create a custom configuration, you can create a `.env` file in any of the locations above with the following variables:

```
# Model settings
RAGCHAT_MODEL=llama3.3
RAGCHAT_EMBED_MODEL=nomic-embed-text
RAGCHAT_TEMPERATURE=0.7
# RAGCHAT_MAX_TOKENS=1024  # Uncomment to set max tokens

# Provider settings
# RAGCHAT_PROVIDER=ollama  # Options: ollama, openai, anthropic, google
# RAGCHAT_API_KEY=         # Your API key (if not using provider-specific env vars)
# RAGCHAT_API_BASE=        # Custom API base URL (if needed)

# Provider-specific API keys (can be set instead of RAGCHAT_API_KEY)
# OPENAI_API_KEY=          # For OpenAI models
# ANTHROPIC_API_KEY=       # For Anthropic Claude models
# GOOGLE_API_KEY=          # For Google Gemini models

# Vector store settings
RAGCHAT_PERSIST_DIR=/path/to/your/vector/store

# Behavior settings
RAGCHAT_NOFALLBACK=false
RAGCHAT_DEBUG=false
RAGCHAT_SHOW_THOUGHTS=false
```

All settings are optional and will fall back to application defaults if not specified. The application comes with sensible defaults, so you don't need to create any `.env` files to get started.

### Configuration Settings Explained

- **RAGCHAT_MODEL**: The LLM model to use for chat responses (default: llama3.3)
- **RAGCHAT_EMBED_MODEL**: The embedding model to use for document indexing (default: nomic-embed-text)
- **RAGCHAT_TEMPERATURE**: Controls the randomness of the model's output (default: 0.7)
- **RAGCHAT_MAX_TOKENS**: Maximum number of tokens in the model's response (default: model-specific)
- **RAGCHAT_PROVIDER**: Explicitly set the LLM provider (default: auto-detected from model name)
- **RAGCHAT_API_KEY**: API key for the LLM provider (default: provider-specific env var)
- **RAGCHAT_API_BASE**: Custom API base URL for the LLM provider (default: provider's standard endpoint)
- **RAGCHAT_PERSIST_DIR**: Custom location for the vector database (default: ~/.ragchat/chroma_db)
- **RAGCHAT_NOFALLBACK**: If true, the model will only answer based on documents in the knowledge base (default: false)
- **RAGCHAT_DEBUG**: If true, enables debug mode with additional logging and shows the "Thinking..." spinner (default: false)
- **RAGCHAT_SHOW_THOUGHTS**: If true, shows the model's chain-of-thought reasoning in the output (default: false)

### Supported Models

#### OpenAI Models
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo
- gpt-4o
- gpt-4o-mini

#### Anthropic Models
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku

#### Google Models
- gemini-pro
- gemini-ultra

#### Ollama Models
- llama3.3 (default)
- mistral
- mixtral
- Any other model available in your Ollama installation

### Embedding Models

- **Ollama**: Any model in your Ollama installation (default: nomic-embed-text)
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **Google**: models/embedding-001, embedding-001

## Commands

- `/help` - Display help message
- `/add <file>` - Add a document to the knowledge base
- `/list` - List loaded documents
- `/remove <file>` - Remove a document
- `/reset` - Clear all documents
- `/clear` - Clear conversation history
- `/ask [N] <question>` - Ask question (optionally from specific source N)
- `/bye` - Exit

## Development

Setup development environment:

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
