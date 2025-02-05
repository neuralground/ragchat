# RAGChat

A Retrieval-Augmented Generation (RAG) chatbot using local LLMs via Ollama.

## Features

- Document ingestion with support for PDF, HTML, EPUB, Markdown, and text files
- Vector storage with ChromaDB
- Local LLM integration via Ollama
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
python -m ragchat --model mistral-small:24b --embed-model nomic-embed-text --persist-dir ./my_docs
```

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
```
