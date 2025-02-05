import os
from typing import Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def initialize_rag(embed_model: str, persist_dir: Optional[str] = None) -> Chroma:
    embeddings = OllamaEmbeddings(model=embed_model)
    
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

def reset_collection(vectorstore: Chroma, embed_model: str, persist_dir: Optional[str] = None) -> Chroma:
    """Reset and reinitialize the Chroma collection."""
    try:
        vectorstore.delete_collection()
    except Exception:
        pass  # Collection might not exist
    
    # Create a new collection
    return initialize_rag(embed_model, persist_dir)

def get_next_available_number(vectorstore: Chroma) -> int:
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

