import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Environment variable for base directory override
RAGCHAT_HOME_ENV = "RAGCHAT_HOME"

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env files.
    
    Priority order (from lowest to highest priority):
    1. Application's default .env (lowest priority, always loaded first)
    2. User's ~/.ragchat/.env
    3. User's home directory .env
    4. User's RAGCHAT_HOME/.env (if RAGCHAT_HOME is set)
    5. User's current directory .env (highest priority)
    
    This allows the application to have default settings that can be
    overridden by user-specific .env files if they exist.
    """
    # First load the application's default .env file (lowest priority)
    package_dir = Path(__file__).parent.absolute()
    default_env_path = package_dir / 'default.env'
    if default_env_path.exists():
        load_dotenv(default_env_path, override=False)
    
    # Then try from ~/.ragchat (can override defaults)
    user_default_env_path = Path.home() / '.ragchat' / '.env'
    if user_default_env_path.exists():
        load_dotenv(user_default_env_path, override=True)
    
    # Then try from user's home directory (can override previous)
    home_env_path = Path.home() / '.env'
    if home_env_path.exists():
        load_dotenv(home_env_path, override=True)
    
    # Then try from RAGCHAT_HOME if set (can override previous)
    if ragchat_home := os.environ.get(RAGCHAT_HOME_ENV):
        env_path = Path(ragchat_home) / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
    
    # Finally try from current directory (highest priority)
    current_env_path = Path('.env')
    if current_env_path.exists():
        load_dotenv('.env', override=True)

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
    model: str = "llama3.3"
    embed_model: str = "nomic-embed-text"
    persist_dir: Optional[str] = None
    nofallback: bool = False
    debug: bool = False
    show_thoughts: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    provider: Optional[str] = None
    llm_options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls):
        """Create a ChatConfig instance from environment variables."""
        # Load environment variables
        load_env_file()
        
        # Get values from environment with defaults
        model = os.environ.get('RAGCHAT_MODEL', "llama3.3")
        embed_model = os.environ.get('RAGCHAT_EMBED_MODEL', "nomic-embed-text")
        persist_dir = os.environ.get('RAGCHAT_PERSIST_DIR', None)
        nofallback = os.environ.get('RAGCHAT_NOFALLBACK', '').lower() in ('true', '1', 'yes')
        debug = os.environ.get('RAGCHAT_DEBUG', '').lower() in ('true', '1', 'yes')
        show_thoughts = os.environ.get('RAGCHAT_SHOW_THOUGHTS', '').lower() in ('true', '1', 'yes')
        temperature = float(os.environ.get('RAGCHAT_TEMPERATURE', '0.7'))
        max_tokens = os.environ.get('RAGCHAT_MAX_TOKENS')
        if max_tokens:
            max_tokens = int(max_tokens)
        
        # Provider-specific settings
        provider = os.environ.get('RAGCHAT_PROVIDER')
        api_key = os.environ.get('RAGCHAT_API_KEY')
        api_base = os.environ.get('RAGCHAT_API_BASE')
        
        # Provider-specific API keys (these take precedence if set)
        if not api_key:
            if 'gpt' in model.lower() or provider == 'openai':
                api_key = os.environ.get('OPENAI_API_KEY')
            elif 'claude' in model.lower() or provider == 'anthropic':
                api_key = os.environ.get('ANTHROPIC_API_KEY')
            elif 'gemini' in model.lower() or provider == 'google':
                api_key = os.environ.get('GOOGLE_API_KEY')
        
        # Additional LLM options
        llm_options = {}
        for key, value in os.environ.items():
            if key.startswith('RAGCHAT_LLM_'):
                option_key = key[12:].lower()  # Remove RAGCHAT_LLM_ prefix
                llm_options[option_key] = value
        
        return cls(
            model=model,
            embed_model=embed_model,
            persist_dir=persist_dir,
            nofallback=nofallback,
            debug=debug,
            show_thoughts=show_thoughts,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            provider=provider,
            llm_options=llm_options
        )
    
    def __post_init__(self):
        if self.persist_dir is None:
            base_dir = get_default_base_dir()
            self.persist_dir = str(base_dir / "chroma_db")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
