import warnings
import epub2txt
from langchain_core.documents import Document

class SimpleEPubLoader:
    """A simple EPUB loader that doesn't depend on NLTK."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> list[Document]:
        """Load EPUB and return documents."""
        try:
            # Suppress the specific FutureWarning from ebooklib
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, 
                                     module='ebooklib.epub')
                # Read the EPUB content
                content = epub2txt.epub2txt(self.file_path)
            
            # Split into chapters if possible (based on newlines)
            chapters = content.split('\n\n\n')
            
            # Create documents, one per substantial chapter
            documents = []
            for i, chapter in enumerate(chapters, 1):
                # Skip empty or very short chapters
                if len(chapter.strip()) < 50:
                    continue
                    
                documents.append(
                    Document(
                        page_content=chapter.strip(),
                        metadata={
                            "source": self.file_path,
                            "chapter": i,
                        }
                    )
                )
            
            if not documents:
                # If no chapters were long enough, create a single document
                documents = [Document(
                    page_content=content.strip(),
                    metadata={
                        "source": self.file_path,
                        "chapter": 1,
                    }
                )]
            
            return documents
            
        except Exception as e:
            raise ValueError(f"Error loading EPUB file: {e}")
