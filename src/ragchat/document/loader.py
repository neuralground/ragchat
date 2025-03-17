import os
import sys
from typing import List, Dict, Any
import nltk
import markdown
import mimetypes
from PIL import Image
import pytesseract

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

class ImageLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        try:
            # Open the image
            image = Image.open(self.file_path)
            
            # Extract text from the image using OCR
            text = pytesseract.image_to_string(image)
            
            # Get image metadata
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # Create metadata
            metadata = {
                "source": self.file_path,
                "image_width": width,
                "image_height": height,
                "image_format": format_type,
                "image_mode": mode,
                "content_type": "image"
            }
            
            # Return document with extracted text and metadata
            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            print(f"Error processing image {self.file_path}: {e}", file=sys.stderr)
            # Return a minimal document if OCR fails
            return [Document(
                page_content=f"Image file: {os.path.basename(self.file_path)}",
                metadata={"source": self.file_path, "content_type": "image", "ocr_failed": True}
            )]

def get_file_size_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)

def get_document_info(file_path: str) -> Dict[str, Any]:
    """Get document information including type, size, and page count."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    file_size_mb = get_file_size_mb(file_path)
    file_size_str = f"{file_size_mb:.1f}MB"
    
    info = {
        "file_path": file_path,
        "file_size": file_size_mb,
        "file_size_str": file_size_str,
        "extension": ext,
        "content_type": "document"
    }
    
    # Determine content type based on extension
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
        info["content_type"] = "image"
        try:
            with Image.open(file_path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["format"] = img.format
                info["doc_type"] = f"{img.format} image ({img.width}x{img.height})"
        except Exception:
            info["doc_type"] = f"Image file"
    
    elif ext == '.pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            info["page_count"] = page_count
            info["doc_type"] = f"PDF document, {page_count} pages"
        except Exception:
            info["doc_type"] = "PDF document"
    
    elif ext in ['.epub']:
        info["doc_type"] = "E-book (EPUB)"
    
    elif ext in ['.md', '.markdown']:
        info["doc_type"] = "Markdown document"
    
    elif ext in ['.html', '.htm']:
        info["doc_type"] = "HTML document"
    
    elif ext in ['.txt']:
        info["doc_type"] = "Text document"
    
    else:
        # Try to determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            info["doc_type"] = mime_type
        else:
            info["doc_type"] = f"{ext[1:].upper()} document"
    
    return info

def load_document(file_path: str, source_number: int) -> List[Document]:
    """Load a document file and return a list of Document objects."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Get document information
    doc_info = get_document_info(file_path)
    
    loaders = {
        '.pdf': PyPDFLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.epub': SimpleEPubLoader,  # Use our simple EPUB loader
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.txt': TextLoader,
        # Image formats
        '.jpg': ImageLoader,
        '.jpeg': ImageLoader,
        '.png': ImageLoader,
        '.gif': ImageLoader,
        '.bmp': ImageLoader,
        '.tiff': ImageLoader,
        '.webp': ImageLoader
    }
    
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    
    file_size = get_file_size_mb(file_path)
    if file_size > 0.1:
        print(f"Loading {file_size:.1f}MB {ext.upper()} file...")
    
    loader = loaders[ext](file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source_file"] = file_path
        doc.metadata["source_number"] = source_number
        doc.metadata["doc_type"] = doc_info["doc_type"]
        doc.metadata["file_size"] = doc_info["file_size"]
        doc.metadata["content_type"] = doc_info["content_type"]
        
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