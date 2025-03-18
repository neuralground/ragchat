#!/usr/bin/env python3
"""
Tests for the document loader module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from ragchat.document.loader import (
    load_document, 
    split_documents,
    get_document_info
)
from ragchat.cli.commands import resolve_image_path


def test_get_document_type():
    """Test document type detection."""
    # Mock os.path.getsize to avoid file not found errors
    with patch("os.path.getsize", return_value=1024):
        # Test different file types
        with patch("pypdf.PdfReader", return_value=MagicMock(pages=[])):
            assert get_document_info("test.pdf")["doc_type"] == "PDF document, 0 pages"
            assert get_document_info("test.PDF")["doc_type"] == "PDF document, 0 pages"
        
        assert get_document_info("test.txt")["doc_type"] == "Text document"
        assert get_document_info("test.md")["doc_type"] == "Markdown document"
        assert get_document_info("test.html")["doc_type"] == "HTML document"
        assert get_document_info("test.htm")["doc_type"] == "HTML document"
        assert get_document_info("test.epub")["doc_type"] == "E-book (EPUB)"
        
        # For image tests, we need to mock Image.open
        with patch("PIL.Image.open") as mock_image:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 100
            mock_img.format = "JPEG"
            mock_image.return_value.__enter__.return_value = mock_img
            
            jpg_info = get_document_info("test.jpg")
            assert "JPEG image" in jpg_info["doc_type"]
            assert "(100x100)" in jpg_info["doc_type"]
            
            jpeg_info = get_document_info("test.jpeg")
            assert "JPEG image" in jpeg_info["doc_type"]
        
        with patch("PIL.Image.open") as mock_image:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 100
            mock_img.format = "PNG"
            mock_image.return_value.__enter__.return_value = mock_img
            
            png_info = get_document_info("test.png")
            assert "PNG image" in png_info["doc_type"]
        
        with patch("PIL.Image.open") as mock_image:
            mock_img = MagicMock()
            mock_img.width = 100
            mock_img.height = 100
            mock_img.format = "WEBP"
            mock_image.return_value.__enter__.return_value = mock_img
            
            webp_info = get_document_info("test.webp")
            assert "WEBP image" in webp_info["doc_type"]


def test_resolve_image_path():
    """Test image path resolution."""
    with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
        # Test with absolute path
        path, exists = resolve_image_path(temp_file.name)
        assert path == temp_file.name
        assert exists is True
        
        # Test with relative path (assuming the test is run from the project root)
        rel_path = os.path.relpath(temp_file.name)
        path, exists = resolve_image_path(rel_path)
        assert os.path.abspath(path) == os.path.abspath(temp_file.name)
        assert exists is True
        
        # Test with non-existent file
        path, exists = resolve_image_path("nonexistent.jpg")
        assert exists is False


@pytest.mark.parametrize("doc_type,file_ext", [
    ("pdf", ".pdf"),
    ("txt", ".txt"),
    ("markdown", ".md"),
    ("html", ".html"),
    ("epub", ".epub"),
    ("image", ".jpg")
])
def test_load_document_calls_correct_loader(doc_type, file_ext):
    """Test that load_document calls the correct loader based on document type."""
    test_path = f"test{file_ext}"
    
    # Create mock loaders
    mock_pdf_loader = MagicMock()
    mock_pdf_loader.return_value.load.return_value = [MagicMock()]
    
    mock_text_loader = MagicMock()
    mock_text_loader.return_value.load.return_value = [MagicMock()]
    
    mock_markdown_loader = MagicMock()
    mock_markdown_loader.return_value.load.return_value = [MagicMock()]
    
    mock_html_loader = MagicMock()
    mock_html_loader.return_value.load.return_value = [MagicMock()]
    
    mock_epub_loader = MagicMock()
    mock_epub_loader.return_value.load.return_value = [MagicMock()]
    
    mock_image_loader = MagicMock()
    mock_image_loader.return_value.load.return_value = [MagicMock()]
    
    # Mock get_document_info
    mock_doc_info = {
        "doc_type": f"{doc_type.upper()} document",
        "file_size": 1.0,
        "content_type": "document"
    }
    
    # Patch all loaders
    with patch("ragchat.document.loader.PyPDFLoader", mock_pdf_loader), \
         patch("ragchat.document.loader.TextLoader", mock_text_loader), \
         patch("ragchat.document.loader.MarkdownLoader", mock_markdown_loader), \
         patch("ragchat.document.loader.UnstructuredHTMLLoader", mock_html_loader), \
         patch("ragchat.document.loader.SimpleEPubLoader", mock_epub_loader), \
         patch("ragchat.document.loader.ImageLoader", mock_image_loader), \
         patch("ragchat.document.loader.get_document_info", return_value=mock_doc_info), \
         patch("ragchat.document.loader.get_file_size_mb", return_value=1.0), \
         patch("os.path.exists", return_value=True):
        
        # Call load_document
        docs = load_document(test_path, 1)
        
        # Verify the correct loader was called
        if doc_type == "pdf":
            mock_pdf_loader.assert_called_once_with(test_path)
        elif doc_type == "txt":
            mock_text_loader.assert_called_once_with(test_path)
        elif doc_type == "markdown":
            mock_markdown_loader.assert_called_once_with(test_path)
        elif doc_type == "html":
            mock_html_loader.assert_called_once_with(test_path)
        elif doc_type == "epub":
            mock_epub_loader.assert_called_once_with(test_path)
        elif doc_type == "image":
            mock_image_loader.assert_called_once_with(test_path)


def test_split_documents():
    """Test document splitting."""
    # Create mock documents
    mock_docs = [
        MagicMock(page_content="This is page 1"),
        MagicMock(page_content="This is page 2"),
        MagicMock(page_content="This is page 3")
    ]
    
    # Patch the text splitter
    mock_splitter = MagicMock()
    mock_splitter.split_documents.return_value = [
        MagicMock(page_content="Split 1"),
        MagicMock(page_content="Split 2"),
        MagicMock(page_content="Split 3"),
        MagicMock(page_content="Split 4")
    ]
    
    with patch("ragchat.document.loader.RecursiveCharacterTextSplitter", return_value=mock_splitter):
        # Call split_documents
        result = split_documents(mock_docs)
        
        # Verify result
        assert len(result) == 4
        mock_splitter.split_documents.assert_called_once_with(mock_docs)


def test_get_document_info():
    """Test document info extraction."""
    # Test with PDF
    with patch("os.path.getsize", return_value=1024), \
         patch("pypdf.PdfReader") as mock_reader:
        # Setup mock reader with pages
        mock_reader.return_value = MagicMock()
        mock_reader.return_value.pages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        info = get_document_info("test.pdf")
        assert "PDF document" in info["doc_type"]
        assert info["file_size_str"] == "0.0MB"
        assert info["extension"] == ".pdf"
    
    # Test with image
    with patch("os.path.getsize", return_value=2048), \
         patch("PIL.Image.open") as mock_image:
        # Setup mock image
        mock_img = MagicMock()
        mock_img.width = 100
        mock_img.height = 100
        mock_img.format = "JPEG"
        mock_image.return_value.__enter__.return_value = mock_img
        
        info = get_document_info("test.jpg")
        assert "JPEG image" in info["doc_type"]
        assert "(100x100)" in info["doc_type"]
        assert info["file_size_str"] == "0.0MB"
        assert info["extension"] == ".jpg"
        assert info["content_type"] == "image"
