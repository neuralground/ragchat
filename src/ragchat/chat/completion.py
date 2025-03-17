from typing import Dict, List, Optional, Any, Tuple, Union
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_chroma import Chroma
import sys
import os
import logging
import contextlib
import re

# Configure logging to suppress specific messages
logging.getLogger('chromadb.segment').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout output."""
    # Save the original stdout
    original_stdout = sys.stdout
    # Redirect stdout to /dev/null
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            # Restore stdout
            sys.stdout = original_stdout

def get_numbered_sources(vectorstore: Chroma) -> Dict[int, str]:
    sources = {}
    with suppress_stdout():
        results = vectorstore.get(include=['metadatas'])
    if results and results['metadatas']:
        for metadata in results['metadatas']:
            if 'source_file' in metadata and 'source_number' in metadata:
                sources[metadata['source_number']] = metadata['source_file']
    return sources

def extract_thoughts(text: Union[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract thoughts from text if present."""
    thoughts = None
    
    # Handle AIMessage or other message objects
    if hasattr(text, 'content'):
        text = text.content
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
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
    llm: BaseChatModel,
    memory: BaseMemory,
    source_id: Optional[int] = None,
    nofallback: bool = False
) -> Dict[str, Any]:
    if source_id == 0:
        # Explicitly using model knowledge only
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
        with suppress_stdout():
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "filter": {"source_file": target_source}
                }
            )
        
        # When a specific source is requested, we should not use fallback
        # This ensures we only answer from the specified document
        nofallback = True
    else:
        with suppress_stdout():
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

    with suppress_stdout():
        docs = retriever.invoke(query)
    
    if len(docs) > 0:
        # Custom prompt template that doesn't mention "I don't know" when answering
        qa_prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
If the context contains the information to answer the question, provide a comprehensive answer using only that information.
If the context does not contain the information needed to answer the question, simply respond with: "I don't have information about this in the documents."
Do not say things like "the context doesn't mention" or "the provided information doesn't include" - just answer the question if you can, or say you don't have the information.

Context: {context}

Question: {question}

Answer:""")
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        result = chain.invoke({"question": query})
        answer, thoughts = extract_thoughts(result["answer"])
        
        # Check if the answer indicates no information was found
        no_info_phrases = [
            "I don't have information about this in the documents",
            "The context does not contain",
            "The provided context does not",
            "The information is not provided",
            "Based on the given context, I cannot",
            "The context doesn't mention",
            "I don't know based on the given context",
            "I cannot find information about",
            "There is no information about",
            "No information is provided about"
        ]
        
        no_info_found = any(phrase.lower() in answer.lower() for phrase in no_info_phrases)
        
        if no_info_found and not nofallback:
            # Fall back to model knowledge without mentioning documents
            prompt = f"""Please answer this question based on your knowledge: {query}
            Do not mention anything about documents, context, or sources in your answer.
            Just provide a direct answer as if you were asked this question normally."""
            
            response = llm.invoke(prompt)
            answer, thoughts = extract_thoughts(response)
            
            return {
                "answer": answer,
                "thoughts": thoughts,
                "source_documents": [],
                "used_fallback": True
            }
        else:
            result["answer"] = answer
            result["thoughts"] = thoughts
            result["used_fallback"] = False
            
            if result["source_documents"]:
                source_mapping = {}
                with suppress_stdout():
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
        # Using model knowledge as fallback, but don't mention documents
        prompt = f"""Please answer this question based on your knowledge: {query}
        Do not mention anything about documents, context, or sources in your answer.
        Just provide a direct answer as if you were asked this question normally."""
        
        response = llm.invoke(prompt)
        answer, thoughts = extract_thoughts(response)
        return {
            "answer": answer,
            "thoughts": thoughts,
            "source_documents": [],
            "used_fallback": True
        }
    else:
        # When nofallback is True or a specific source was requested but no info found
        if source_id is not None and source_id != 0:
            return {
                "answer": f"I don't have information about this in document [{source_id}] {sources.get(source_id, '')}.",
                "thoughts": None,
                "source_documents": [],
                "used_fallback": False
            }
        else:
            return {
                "answer": "I don't have any information about this in the loaded documents.",
                "thoughts": None,
                "source_documents": [],
                "used_fallback": False
            }