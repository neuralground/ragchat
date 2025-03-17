"""LLM provider implementations for RAGChat.

This module provides a unified interface for different LLM providers
including OpenAI, Anthropic, Google, and Ollama.
"""

import os
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

def get_llm_from_config(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance from configuration."""
    if config.provider == LLMProvider.OLLAMA:
        return OllamaLLM(
            model=config.model_name,
            temperature=config.temperature,
            **config.additional_kwargs
        )
    
    elif config.provider == LLMProvider.OPENAI:
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set it in the config or as OPENAI_API_KEY environment variable.")
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
            base_url=config.api_base,
            streaming=config.streaming,
            **config.additional_kwargs
        )
    
    elif config.provider == LLMProvider.ANTHROPIC:
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required. Set it in the config or as ANTHROPIC_API_KEY environment variable.")
        
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=api_key,
            anthropic_api_url=config.api_base,
            streaming=config.streaming,
            **config.additional_kwargs
        )
    
    elif config.provider == LLMProvider.GOOGLE:
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set it in the config or as GOOGLE_API_KEY environment variable.")
        
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=api_key,
            **config.additional_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

# Default configurations for popular models
DEFAULT_MODEL_CONFIGS = {
    # OpenAI models
    "gpt-3.5-turbo": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    ),
    "gpt-4": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.7
    ),
    "gpt-4-turbo": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4-turbo-preview",
        temperature=0.7
    ),
    "gpt-4o": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        temperature=0.7
    ),
    "gpt-4o-mini": LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        temperature=0.7
    ),
    
    # Anthropic models
    "claude-3-opus": LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-opus-20240229",
        temperature=0.7
    ),
    "claude-3-sonnet": LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        temperature=0.7
    ),
    "claude-3-haiku": LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.7
    ),
    
    # Google models
    "gemini-pro": LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-pro",
        temperature=0.7
    ),
    "gemini-ultra": LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-ultra",
        temperature=0.7
    ),
    
    # Ollama models (default)
    "llama3.3": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="llama3.3",
        temperature=0.7
    ),
    "mistral": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="mistral",
        temperature=0.7
    ),
    "mixtral": LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name="mixtral",
        temperature=0.7
    )
}

def get_llm(model_name: str, api_key: Optional[str] = None, 
            api_base: Optional[str] = None, temperature: Optional[float] = None,
            max_tokens: Optional[int] = None, provider: Optional[str] = None) -> BaseChatModel:
    """Get an LLM instance based on the model name.
    
    This function provides a simplified interface to create LLM instances
    with sensible defaults based on the model name.
    
    Args:
        model_name: Name of the model to use
        api_key: Optional API key (will use environment variable if not provided)
        api_base: Optional API base URL
        temperature: Optional temperature override
        max_tokens: Optional max tokens override
        provider: Optional provider override (ollama, openai, anthropic, google)
        
    Returns:
        An LLM instance ready to use
    """
    # Check if we have a default config for this model
    if model_name in DEFAULT_MODEL_CONFIGS:
        config = DEFAULT_MODEL_CONFIGS[model_name].model_copy(deep=True)
    else:
        # If provider is specified, use it to determine the config
        if provider:
            if provider.lower() == "openai":
                config = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model_name=model_name,
                    temperature=0.7
                )
            elif provider.lower() == "anthropic":
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model_name=model_name,
                    temperature=0.7
                )
            elif provider.lower() == "google":
                config = LLMConfig(
                    provider=LLMProvider.GOOGLE,
                    model_name=model_name,
                    temperature=0.7
                )
            else:  # Default to Ollama
                config = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model_name=model_name,
                    temperature=0.7
                )
        else:
            # If not in our defaults and no provider specified, assume it's an Ollama model
            config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name=model_name,
                temperature=0.7
            )
    
    # Override with provided parameters
    if api_key:
        config.api_key = api_key
    if api_base:
        config.api_base = api_base
    if temperature is not None:
        config.temperature = temperature
    if max_tokens is not None:
        config.max_tokens = max_tokens
        
    return get_llm_from_config(config)
