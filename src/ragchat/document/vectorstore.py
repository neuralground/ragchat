import os
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model(model_name: str, api_key: Optional[str] = None) -> Embeddings:
    """Get an embedding model based on the model name."""
    # Check if it's an Ollama model
    if model_name.startswith("ollama:"):
        # Remove the "ollama:" prefix
        ollama_model = model_name[7:]
        return OllamaEmbeddings(model=ollama_model)
    
    # Check for OpenAI embedding models
    if model_name in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    
    # Check for Google embedding models
    if model_name in ["models/embedding-001", "embedding-001"]:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required for Google embeddings")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    
    # Default to Ollama embeddings
    return OllamaEmbeddings(model=model_name)

def initialize_rag(embed_model: str, persist_dir: Optional[str] = None, api_key: Optional[str] = None) -> Chroma:
    """Initialize the RAG system with the specified embedding model."""
    embeddings = get_embedding_model(embed_model, api_key)
    
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="ragchat_docs"
        )
    
    return Chroma(
        embedding_function=embeddings,
        collection_name="ragchat_docs"
    )

def reset_collection(vectorstore: Chroma, embed_model: str, persist_dir: Optional[str] = None, api_key: Optional[str] = None) -> Chroma:
    """Reset and reinitialize the Chroma collection."""
    try:
        vectorstore.delete_collection()
    except Exception:
        pass  # Collection might not exist
    
    # Create a new collection
    return initialize_rag(embed_model, persist_dir, api_key)

def get_next_available_number(vectorstore: Chroma) -> int:
    """Get the next available source number."""
    used_numbers = set()
    results = vectorstore.get(include=['metadatas'])
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            if 'source_number' in metadata:
                used_numbers.add(metadata['source_number'])
    
    number = 1
    while number in used_numbers:
        number += 1
    return number

def add_documents_to_vectorstore(vectorstore: Chroma, documents: List[Document]) -> None:
    """Add documents to the vectorstore."""
    vectorstore.add_documents(documents)
    if hasattr(vectorstore, "_collection") and hasattr(vectorstore._collection, "persist"):
        vectorstore._collection.persist()
