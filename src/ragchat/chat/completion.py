from typing import Dict, List, Tuple, Optional, Any, Union
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
from contextlib import contextmanager
import re
from ..config import ChatConfig
import base64
import os
from openai import OpenAI

# Configure logging to suppress specific messages
logging.getLogger('chromadb.segment').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)

@contextmanager
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
    seen_files = set()
    
    with suppress_stdout():
        results = vectorstore.get(include=['metadatas'])
    
    if results and results['metadatas']:
        # First pass: collect unique source files with their numbers
        for metadata in results['metadatas']:
            if 'source_file' in metadata and 'source_number' in metadata:
                source_file = metadata['source_file']
                source_number = metadata['source_number']
                
                # Only add each source file once
                if source_file not in seen_files:
                    sources[source_number] = source_file
                    seen_files.add(source_file)
    
    return sources

def extract_thoughts(text: Union[str, Any]) -> Tuple[str, Optional[str]]:
    """Extract thoughts from text if present."""
    try:
        thoughts = None
        
        # If text is already a string, skip all the object handling logic
        if isinstance(text, str):
            # Just look for thoughts between <think> tags
            try:
                think_start = text.find("<think>")
                think_end = text.find("</think>")
                
                if think_start != -1 and think_end != -1:
                    thoughts = text[think_start + 7:think_end].strip()
                    # Remove the thoughts section from the text
                    text = text[:think_start].strip() + " " + text[think_end + 8:].strip()
            except Exception:
                pass
            
            return text.strip(), thoughts
        
        # Handle AIMessage or other message objects with content attribute
        if hasattr(text, 'content') and not isinstance(text, str):
            try:
                text = text.content
            except Exception:
                pass
        
        # Handle dictionary-like objects - only if it's not a string
        if not isinstance(text, str):
            try:
                # Check if it has a get method
                if hasattr(text, 'get') and callable(getattr(text, 'get')):
                    if hasattr(text, 'keys'):
                        # Try common keys
                        for key in ['content', 'text', 'message']:
                            try:
                                # Use get with a default of None to avoid KeyError
                                content = text.get(key)
                                if content is not None:
                                    text = content
                                    break
                            except Exception:
                                continue
            except Exception:
                pass
        
        # Convert to string if it's not already
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = "Error: Unable to convert response to string"
        
        # At this point, text should be a string
        try:
            # Look for thoughts between <think> tags
            think_start = text.find("<think>")
            think_end = text.find("</think>")
            
            if think_start != -1 and think_end != -1:
                thoughts = text[think_start + 7:think_end].strip()
                # Remove the thoughts section from the text
                text = text[:think_start].strip() + " " + text[think_end + 8:].strip()
        except Exception:
            pass
        
        try:
            return text.strip(), thoughts
        except Exception:
            # Return safe values if all else fails
            return str(text) if text else "Error processing response", None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical error in extract_thoughts: {str(e)}")
        return "Error processing response", None

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

def qa_with_images(prompt: str, image_sources: List[Dict], llm: BaseChatModel, config: ChatConfig) -> str:
    """Process a query with both text and images using the OpenAI API."""
    
    # Check if the model supports image processing
    if not config.model.startswith("gpt-4"):
        raise ValueError(f"Model {config.model} does not support image processing. Use GPT-4o or another vision-capable model.")
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can analyze both text and images."},
        {"role": "user", "content": []}
    ]
    
    # Add the text prompt
    messages[1]["content"].append({"type": "text", "text": prompt})
    
    # Add the images
    for img_source in image_sources:
        path = img_source["path"]
        
        # Convert WebP to PNG if needed
        if path.lower().endswith('.webp'):
            try:
                from PIL import Image
                img = Image.open(path)
                png_path = path.rsplit('.', 1)[0] + '.png'
                img.save(png_path, 'PNG')
                path = png_path
            except Exception as e:
                # If conversion fails, try to use the original format
                pass
        
        # Encode the image
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Add the image to the message
        image_url = f"data:image/{'png' if path.lower().endswith('.png') else 'jpeg'};base64,{base64_image}"
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
    
    # Create a client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Call the API
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ],
            max_tokens=config.max_tokens or 2000,
            temperature=config.temperature or 0.7
        )
        
        # Extract the content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content is not None:
                return content
            else:
                return "No response content received from the model."
        else:
            return "No valid response received from the model."
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"