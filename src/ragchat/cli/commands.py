from typing import Dict, List, Tuple, Optional, Any, Union
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm
import logging

from langchain_core.language_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..chat.completion import qa_chain_with_fallback, get_numbered_sources
from ..chat.memory import SimpleMemory
from ..config import ChatConfig, RAGCHAT_HOME_ENV
from ..document.loader import (
    load_document, 
    split_documents,
    get_document_info
)
from ..document.vectorstore import get_next_available_number, reset_collection

def handle_help_command(config: ChatConfig) -> str:
    """Display help information about the chat system."""
    # Determine the provider
    provider = config.provider or "auto-detected"
    
    return f"""
Chat system information:
  Chat model: {config.model}
  Model provider: {provider}
  Embedding model: {config.embed_model}
  Vector store location: {config.persist_dir}
  Fallback to model knowledge: {'disabled' if config.nofallback else 'enabled'}
  Debug mode: {'enabled' if config.debug else 'disabled'}
  Show thoughts: {'enabled' if config.show_thoughts else 'disabled'}

Available commands:
  /help          - Display this help message
  /add <file>    - Add a document to the knowledge base
  /list          - List all documents in the knowledge base
  /remove <file> - Remove a document from the knowledge base
  /reset         - Remove all documents from the knowledge base
  /clear         - Clear conversation history
  /ask [N] <question> - Ask question to specific source N
                  Use [0] to force answer from model knowledge only
  /bye           - Exit the chat

Environment Variables:
  RAGCHAT_HOME - Set custom base directory (default: $HOME/.ragchat)
  RAGCHAT_MODEL - Set the LLM model to use (default: llama3.3)
  RAGCHAT_PROVIDER - Set the model provider (ollama, openai, anthropic, google)
  RAGCHAT_API_KEY - Set the API key for cloud providers
  RAGCHAT_DEBUG - Enable debug mode and show "Thinking..." spinner
  RAGCHAT_SHOW_THOUGHTS - Show model's chain-of-thought reasoning

Tip: Use TAB for command and file completion
"""

def handle_add_command(vectorstore: Chroma, file_path: str) -> str:
    """Add a document to the vector store."""
    try:
        import os
        from ..document.loader import get_document_info
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        # Get document information
        doc_info = get_document_info(file_path)
        file_type = doc_info.get("doc_type", "document")
        content_type = doc_info.get("content_type", "document")
        
        # Get next available source number
        source_number = get_next_available_number(vectorstore)
        
        # Load the document
        documents = load_document(file_path, source_number)
        
        if not documents:
            return f"Error: No content could be extracted from {file_path}"
        
        # Generate a title/description from the document content
        title = ""
        if content_type != "image" and documents:
            # Use the first chunk of content to generate a title
            first_content = documents[0].page_content
            title = generate_document_title(first_content)
            
            # Store the title in the metadata of all documents
            for doc in documents:
                doc.metadata["doc_title"] = title
        
        # Split documents into chunks
        chunks = split_documents(documents)
        
        total_chunks = len(chunks)
        
        # Determine description based on content type
        if content_type == "image":
            desc = f"Processing image"
        elif file_path.lower().endswith('.pdf') and "page_count" in doc_info:
            desc = f"Processing {doc_info['page_count']} pages"
        else:
            desc = f"Processing document"
        
        # Add chunks to vectorstore with progress bar
        with tqdm(total=total_chunks, desc=desc, leave=True) as pbar_chunks:
            for i in range(0, total_chunks, 100):
                batch = chunks[i:i + 100]
                vectorstore.add_documents(batch)
                pbar_chunks.update(len(batch))
        
        # Format success message with document info
        message = f"\n{file_path} added successfully to the knowledge base"
        message += f"\nType: {file_type}"
        message += f"\nSize: {doc_info.get('file_size_str', 'Unknown')}"
        message += f"\nChunks: {total_chunks}"
        if title:
            message += f"\nTitle: {title}"
        
        return message
    except Exception as e:
        return f"Error: {e}"

def handle_list_command(vectorstore: Chroma) -> str:
    """List all documents in the knowledge base with enhanced information."""
    sources = get_numbered_sources(vectorstore)
    if not sources:
        return "No documents in knowledge base"
    
    # Get document metadata for each source
    source_metadata = {}
    results = vectorstore.get(include=['metadatas', 'documents'])
    if results and results['metadatas'] and results['documents']:
        for i, metadata in enumerate(results['metadatas']):
            if 'source_file' in metadata and 'source_number' in metadata:
                source_number = metadata['source_number']
                if source_number not in source_metadata and 'source_file' in metadata:
                    source_metadata[source_number] = {
                        'file_path': metadata['source_file'],
                        'doc_type': metadata.get('doc_type', 'Unknown document type'),
                        'content_type': metadata.get('content_type', 'document'),
                        'file_size': metadata.get('file_size', 0),
                        'chunk_count': 0,
                        'content': [],
                        'doc_title': metadata.get('doc_title', '')
                    }
                
                # Count chunks for this source
                if source_number in source_metadata:
                    source_metadata[source_number]['chunk_count'] += 1
                
                # Collect some content for description generation
                if source_number in source_metadata and len(source_metadata[source_number]['content']) < 5:
                    # Only collect up to 5 chunks to avoid memory issues
                    if i < len(results['documents']):
                        source_metadata[source_number]['content'].append(results['documents'][i])
    
    # Generate descriptions for each source
    result = "\nDocuments in knowledge base:\n"
    for idx, source in sources.items():
        meta = source_metadata.get(idx, {})
        doc_type = meta.get('doc_type', 'Document')
        content_type = meta.get('content_type', 'document')
        chunk_count = meta.get('chunk_count', 0)
        
        # Format file size
        file_size = meta.get('file_size', 0)
        if file_size > 1:
            size_str = f"{file_size:.1f}MB"
        else:
            size_str = f"{file_size * 1024:.0f}KB"
        
        # Get title if available, otherwise generate a description
        title = meta.get('doc_title', '')
        if not title and content_type != 'image':
            description = generate_document_description(meta.get('content', []), content_type)
            title = description
        
        # Format the output
        result += f"[{idx}] {source}\n"
        result += f"    Type: {doc_type}, Size: {size_str}, Chunks: {chunk_count}\n"
        if title:
            result += f"    Title: {title}\n"
        
        # For images, generate a description of the content
        if content_type == 'image':
            description = generate_document_description(meta.get('content', []), content_type)
            if description:
                result += f"    Content: {description}\n"
    
    return result

def generate_document_title(text):
    """Generate a title for a document based on its content."""
    if not text:
        return ""
    
    try:
        # Limit text length to avoid processing too much
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length]
        
        # Look for title patterns
        import re
        
        # Try to find a title in the text
        title_patterns = [
            # Look for "Title: something" pattern
            r'(?:title|subject):\s*([^\n\.]{5,100})',
            # Look for paper title format
            r'^([A-Z][^\.]{10,150}?)(?:\n|\.)',
            # Look for a first line that might be a title
            r'^([^\n\.]{5,100}?)(?:\n|\.)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # Clean up the title
                title = re.sub(r'\s+', ' ', title)
                if len(title) > 10:  # Only use if it's a substantial title
                    return title
        
        # If no title found, use the first sentence if it's reasonable
        first_sentence = text.split('.')[0].strip()
        if 10 <= len(first_sentence) <= 100:
            return first_sentence
        
        # Fall back to first 50 characters
        return text[:50].strip() + "..."
    
    except Exception:
        return ""

def generate_document_description(content_samples, content_type):
    """Generate a brief description of the document based on content samples."""
    if not content_samples:
        return ""
    
    try:
        # Combine content samples
        combined_text = " ".join(content_samples)
        
        # Limit text length to avoid processing too much
        max_length = 1000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        # Simple keyword extraction
        import re
        from collections import Counter
        
        # Remove common punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'to', 'of', 'in', 'for', 'on', 'by', 'with', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'this', 'that', 'these', 'those', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        
        # Get the most common words
        common_words = [word for word, count in word_counts.most_common(5)]
        
        # Generate description based on content type
        if content_type == 'image':
            return f"Image containing text about: {', '.join(common_words)}" if common_words else "Image with limited or no text content"
        else:
            # Try to detect document type based on content
            if any(word in combined_text.lower() for word in ['abstract', 'methodology', 'conclusion', 'references']):
                return f"Academic or research paper about: {', '.join(common_words)}"
            elif any(word in combined_text.lower() for word in ['chapter', 'novel', 'story', 'character']):
                return f"Book or literary work about: {', '.join(common_words)}"
            elif any(word in combined_text.lower() for word in ['code', 'function', 'class', 'method', 'variable']):
                return f"Programming or technical documentation about: {', '.join(common_words)}"
            else:
                return f"Document about: {', '.join(common_words)}" if common_words else "Document with varied content"
    
    except Exception as e:
        # If description generation fails, return a generic description
        return ""

def handle_remove_command(vectorstore: Chroma, identifier: str) -> str:
    """Remove a document from the knowledge base using file path or source number in [#] format."""
    try:
        # Check if identifier uses [#] format
        if identifier.startswith('[') and ']' in identifier:
            try:
                end_bracket = identifier.index(']')
                source_number = int(identifier[1:end_bracket])
                # Find the corresponding file path
                sources = get_numbered_sources(vectorstore)
                if source_number not in sources:
                    return f"No document found with source number: {source_number}"
                file_path = sources[source_number]
            except (ValueError, IndexError):
                return "Invalid source number format. Use: /remove [N] or /remove filename"
        else:
            file_path = identifier
            
        results = vectorstore.get(include=['documents', 'metadatas'])
        if results and results['metadatas']:
            indices_to_remove = [
                i for i, metadata in enumerate(results['metadatas'])
                if metadata.get('source_file') == file_path
            ]
            if indices_to_remove:
                vectorstore._collection.delete(
                    where={"source_file": file_path}
                )
                return f"Removed {len(indices_to_remove)} chunks from {file_path}"
            return f"No documents found from: {file_path}"
    except Exception as e:
        return f"Error removing document: {e}"

def handle_reset_command(
    vectorstore: Chroma, 
    memory: SimpleMemory,
    config: ChatConfig
) -> Tuple[str, bool, Optional[Chroma]]:
    """Reset the knowledge base and conversation memory."""
    try:
        # Reset and reinitialize the collection
        new_vectorstore = reset_collection(vectorstore, config.embed_model, config.persist_dir)
        memory.clear()
        return "Knowledge base reset successfully", True, new_vectorstore
    except Exception as e:
        return f"Error resetting knowledge base: {e}", False, None

def handle_ask_command(
    query: str,
    vectorstore: Chroma,
    llm: BaseChatModel,
    memory: SimpleMemory,
    config: ChatConfig
) -> Dict[str, Any]:
    """Handle the /ask command with optional source specification."""
    source_id = None
    
    # Parse source ID if provided
    if query.startswith('['):
        try:
            end_bracket = query.index(']')
            source_id = int(query[1:end_bracket])
            query = query[end_bracket + 1:].strip()
        except (ValueError, IndexError):
            return {
                "error": "Invalid source ID format. Use: /ask [N] question",
                "source_documents": []
            }
    
    result = qa_chain_with_fallback(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        memory=memory,
        source_id=source_id,
        nofallback=config.nofallback
    )
    
    return format_qa_response(result)

def format_qa_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format the QA response for display."""
    response = {
        "answer": result["answer"],
        "source_documents": [],
        "used_fallback": result.get("used_fallback", False)
    }
    
    # Only include source documents if we're not using fallback
    if not response["used_fallback"] and result.get("source_documents"):
        valid_sources = []
        for doc in result["source_documents"]:
            if (doc.metadata.get("source_number") is not None and 
                doc.metadata.get("source_file") is not None):
                source = f"[{doc.metadata['source_number']}] {doc.metadata['source_file']}"
                # Add page number if available
                if "page_number" in doc.metadata:
                    source += f" (p.{doc.metadata['page_number']})"
                valid_sources.append(source)
        
        if valid_sources:
            response["source_documents"] = sorted(set(valid_sources))
    
    if "thoughts" in result:
        response["thoughts"] = result["thoughts"]
    
    return response

def process_command(
    command: str,
    args: str,
    vectorstore: Chroma,
    llm: BaseChatModel,
    memory: SimpleMemory,
    config: ChatConfig
) -> Dict[str, Any]:
    """Process a command and return the appropriate response."""
    
    # Special handling for reset command to return new vectorstore
    if command == '/reset':
        message, success, new_vectorstore = handle_reset_command(vectorstore, memory, config)
        return {
            'message': message,
            'new_vectorstore': new_vectorstore if success else None
        }
    
    command_handlers = {
        '/help': lambda: {'message': handle_help_command(config)},
        '/add': lambda: {'message': handle_add_command(vectorstore, args)},
        '/list': lambda: {'message': handle_list_command(vectorstore)},
        '/remove': lambda: {'message': handle_remove_command(vectorstore, args)},
        '/clear': lambda: {'message': 'Conversation history cleared', 'clear_memory': True},
        '/ask': lambda: handle_ask_command(args, vectorstore, llm, memory, config),
        '/bye': lambda: {'message': 'Goodbye!', 'exit': True}
    }
    
    handler = command_handlers.get(command)
    if handler:
        return handler()
    return {'message': f"Unknown command: {command}"}
