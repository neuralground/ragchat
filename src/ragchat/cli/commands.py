from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from ..config import ChatConfig, RAGCHAT_HOME_ENV
from ..document.loader import load_document, split_documents
from ..document.vectorstore import get_next_available_number
from ..chat.completion import qa_chain_with_fallback, get_numbered_sources
from ..chat.memory import SimpleMemory

def handle_help_command(config: ChatConfig) -> str:
    """Display help information about the chat system."""
    return f"""
Chat system information:
  Chat model: {config.model}
  Embedding model: {config.embed_model}
  Vector store location: {config.persist_dir}
  Fallback to model knowledge: {'disabled' if config.nofallback else 'enabled'}

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
  {RAGCHAT_HOME_ENV} - Set custom base directory (default: $HOME/.ragchat)

Tip: Use TAB for command and file completion
"""

def handle_add_command(vectorstore: Chroma, file_path: str) -> str:
    """Add a document to the vector store."""
    try:
        source_number = get_next_available_number(vectorstore)
        documents = load_document(file_path, source_number)
        chunks = split_documents(documents)
        
        total_chunks = len(chunks)
        desc = f"Processing {len(documents)} pages" if file_path.lower().endswith('.pdf') else "Processing"
        
        with tqdm(total=total_chunks, desc=desc, leave=True) as pbar_chunks:
            for i in range(0, total_chunks, 100):
                batch = chunks[i:i + 100]
                vectorstore.add_documents(batch)
                pbar_chunks.update(len(batch))
        
        return f"\n{file_path} added successfully to the knowledge base"
    except Exception as e:
        return f"Error: {e}"

def handle_list_command(vectorstore: Chroma) -> str:
    """List all documents in the knowledge base."""
    sources = get_numbered_sources(vectorstore)
    if sources:
        result = "\nDocuments in knowledge base:\n"
        for idx, source in sources.items():
            result += f"[{idx}] {source}\n"
        return result
    return "No documents in knowledge base"

def handle_remove_command(vectorstore: Chroma, file_path: str) -> str:
    """Remove a document from the knowledge base."""
    try:
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

def handle_reset_command(vectorstore: Chroma, memory: SimpleMemory) -> Tuple[str, bool]:
    """Reset the knowledge base and conversation memory."""
    try:
        vectorstore.delete_collection()
        memory.clear()
        return "Knowledge base reset successfully", True
    except Exception as e:
        return f"Error resetting knowledge base: {e}", False

def handle_ask_command(
    query: str,
    vectorstore: Chroma,
    llm: OllamaLLM,
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
    
    if result.get("source_documents"):
        valid_sources = [
            (doc.metadata.get("source_number"), doc.metadata.get("source_file"))
            for doc in result["source_documents"]
            if doc.metadata.get("source_number") is not None
            and doc.metadata.get("source_file") is not None
        ]
        if valid_sources:
            response["source_documents"] = [
                f"[{num}] {file}" for num, file in sorted(set(valid_sources))
            ]
    
    return response

def process_command(
    command: str,
    args: str,
    vectorstore: Chroma,
    llm: OllamaLLM,
    memory: SimpleMemory,
    config: ChatConfig
) -> Dict[str, Any]:
    """Process a command and return the appropriate response."""
    command_handlers = {
        '/help': lambda: {'message': handle_help_command(config)},
        '/add': lambda: {'message': handle_add_command(vectorstore, args)},
        '/list': lambda: {'message': handle_list_command(vectorstore)},
        '/remove': lambda: {'message': handle_remove_command(vectorstore, args)},
        '/clear': lambda: {'message': 'Conversation history cleared', 'clear_memory': True},
        '/reset': lambda: {'message': handle_reset_command(vectorstore, memory)[0]},
        '/ask': lambda: handle_ask_command(args, vectorstore, llm, memory, config),
        '/bye': lambda: {'message': 'Goodbye!', 'exit': True}
    }
    
    handler = command_handlers.get(command)
    if handler:
        return handler()
    return {'message': f"Unknown command: {command}"}

