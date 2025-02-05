import os
from pathlib import Path
from dataclasses import dataclass

# Environment variable for base directory override
RAGCHAT_HOME_ENV = "RAGCHAT_HOME"

def get_default_base_dir() -> Path:
    """Get the base directory for ragchat data.
    
    Checks for RAGCHAT_HOME environment variable first,
    falls back to ~/.ragchat if not set.
    """
    if ragchat_home := os.environ.get(RAGCHAT_HOME_ENV):
        base_dir = Path(ragchat_home)
    else:
        base_dir = Path.home() / ".ragchat"
    
    # Ensure directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

@dataclass
class ChatConfig:
    model: str = "mistral-small:24b"
    embed_model: str = "nomic-embed-text"
    persist_dir: str = None
    nofallback: bool = False
    debug: bool = False
    show_thoughts: bool = False  # New option
    
    def __post_init__(self):
        if self.persist_dir is None:
            base_dir = get_default_base_dir()
            self.persist_dir = str(base_dir / "chroma_db")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
