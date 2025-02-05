from typing import Dict, List, Optional, Any, Tuple
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_core.memory import BaseMemory
from langchain_chroma import Chroma

def get_numbered_sources(vectorstore: Chroma) -> Dict[int, str]:
    sources = {}
    results = vectorstore.get(include=['metadatas'])
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            if 'source_file' in metadata and 'source_number' in metadata:
                sources[metadata['source_number']] = metadata['source_file']
    return sources

def extract_thoughts(text: str) -> Tuple[str, Optional[str]]:
    """Extract thoughts from text if present."""
    thoughts = None
    
    # Look for thoughts between <think> tags
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    
    if think_start != -1 and think_end != -1:
        thoughts = text[think_start + 7:think_end].strip()
        # Remove the thoughts section from the text
        text = text[:think_start].strip() + " " + text[think_end + 8:].strip()
    
    return text.strip(), thoughts

def qa_chain_with_fallback(
    query: str,
    vectorstore: Chroma,
    llm: OllamaLLM,
    memory: BaseMemory,
    source_id: Optional[int] = None,
    nofallback: bool = False
) -> Dict[str, Any]:
    if source_id == 0:
        response = llm.invoke(query)
        answer, thoughts = extract_thoughts(response)
        return {
            "answer": answer,
            "thoughts": thoughts,
            "source_documents": [],
            "used_fallback": True
        }
        
    sources = get_numbered_sources(vectorstore)
    
    if source_id is not None and source_id != 0:
        if source_id not in sources:
            return {
                "answer": f"Invalid source ID: {source_id}. Use /list to see available sources.",
                "thoughts": None,
                "source_documents": [],
                "used_fallback": False
            }
        target_source = sources[source_id]
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"source_file": target_source}
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    docs = retriever.invoke(query)
    
    if len(docs) > 0:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        result = chain.invoke({"question": query})
        answer, thoughts = extract_thoughts(result["answer"])
        result["answer"] = answer
        result["thoughts"] = thoughts
        result["used_fallback"] = False
        
        if result["source_documents"]:
            source_mapping = {}
            store_results = vectorstore.get(include=['metadatas'])
            if store_results and store_results['metadatas']:
                for metadata in store_results['metadatas']:
                    if 'source_file' in metadata:
                        source_file = metadata['source_file']
                        if 'source_number' in metadata:
                            source_mapping[source_file] = metadata['source_number']
            
            for doc in result["source_documents"]:
                if 'source_file' in doc.metadata and doc.metadata['source_file'] in source_mapping:
                    doc.metadata['source_number'] = source_mapping[doc.metadata['source_file']]
        
        return result
    elif not nofallback:
        response = llm.invoke(query)
        answer, thoughts = extract_thoughts(response)
        return {
            "answer": answer,
            "thoughts": thoughts,
            "source_documents": [],
            "used_fallback": True
        }
    else:
        return {
            "answer": "I don't have any information about this in the loaded documents.",
            "thoughts": None,
            "source_documents": [],
            "used_fallback": False
        }