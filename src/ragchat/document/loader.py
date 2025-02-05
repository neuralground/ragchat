import os
import sys
from typing import List
import nltk
import markdown

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
    PyPDFLoader,
    UnstructuredEPubLoader,
    TextLoader
)

from .epub_loader import SimpleEPubLoader

class MarkdownLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            text = markdown.markdown(content)
            text = text.replace('<p>', '').replace('</p>', '\n\n')
            text = text.replace('<h1>', '# ').replace('</h1>', '\n')
            text = text.replace('<h2>', '## ').replace('</h2>', '\n')
            text = text.replace('<h3>', '### ').replace('</h3>', '\n')
            text = text.replace('<code>', '`').replace('</code>', '`')
            text = text.replace('<pre>', '```\n').replace('</pre>', '\n```\n')
            return [Document(page_content=text, metadata={"source": self.file_path})]

def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)

def load_document(file_path: str, source_number: int) -> List[Document]:
    """Load a document file and return a list of Document objects."""
    _, ext = os.path.splitext(file_path)
    
    loaders = {
        '.pdf': PyPDFLoader,
        '.html': UnstructuredHTMLLoader,
        '.epub': SimpleEPubLoader,  # Use our new simple EPUB loader
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.txt': TextLoader
    }
    
    if ext.lower() not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    
    file_size = get_file_size_mb(file_path)
    if file_size > 0.1:
        print(f"Loading {file_size:.1f}MB {ext.upper()} file...")
    
    loader = loaders[ext.lower()](file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source_file"] = file_path
        doc.metadata["source_number"] = source_number
        if "chapter" in doc.metadata:
            doc.metadata["page_number"] = f"ch{doc.metadata['chapter']}"
    
    return documents

def ensure_nltk_data():
    """Ensure NLTK data is available, downloading if necessary."""
    try:
        # Try to find the punkt tokenizer data
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        try:
            print("Downloading required NLTK data...", file=sys.stderr)
            nltk.download('punkt', quiet=True)
            return True
        except Exception as e:
            print(f"Warning: Failed to download NLTK data: {e}", file=sys.stderr)
            return False

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks, with fallback if NLTK is unavailable."""
    text_splitter = None
    
    if ensure_nltk_data():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
    else:
        print("Warning: Using simple text splitting (NLTK data unavailable)", file=sys.stderr)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
    
    try:
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Warning: Document splitting error: {e}", file=sys.stderr)
        # If splitting fails, try with simpler settings
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[" ", ""],
            length_function=len,
            is_separator_regex=False
        ).split_documents(documents)
    