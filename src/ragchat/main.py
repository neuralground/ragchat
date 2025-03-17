import argparse
import os
import readline
import sys
import logging
from typing import NoReturn
from pathlib import Path

# Configure logging to suppress specific messages
logging.getLogger('chromadb.segment').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)

from .config import ChatConfig, RAGCHAT_HOME_ENV, load_env_file
from .document.vectorstore import initialize_rag
from .chat.memory import SimpleMemory
from .cli.completer import FileCompleter
from .cli.commands import process_command
from .cli.spinner import ThinkingSpinner
from .chat.llm_providers import get_llm

def parse_args() -> ChatConfig:
    """Parse command line arguments and return ChatConfig."""
    # First load config from .env file
    config = ChatConfig.from_env()
    
    # Then allow command line args to override
    parser = argparse.ArgumentParser(description='Chat with documents using LLMs')
    parser.add_argument(
        '--model',
        default=config.model,
        help=f'LLM model to use for chat (default: {config.model})'
    )
    parser.add_argument(
        '--embed-model',
        default=config.embed_model,
        help=f'Model to use for embeddings (default: {config.embed_model})'
    )
    parser.add_argument(
        '--persist-dir',
        default=config.persist_dir,
        help=f'Directory to persist vector store (default: $HOME/.ragchat/chroma_db or ${RAGCHAT_HOME_ENV}/chroma_db if set)'
    )
    parser.add_argument(
        '--nofallback',
        action='store_true',
        default=config.nofallback,
        help='Disable model answering from its own knowledge when no relevant documents found'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=config.debug,
        help='Enable debug logging'
    )
    parser.add_argument(
        '--show-thoughts',
        action='store_true',
        default=config.show_thoughts,
        help='Show model chain-of-thought when present'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=config.temperature,
        help='Temperature for LLM sampling (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=config.max_tokens,
        help='Maximum tokens for LLM response'
    )
    parser.add_argument(
        '--api-key',
        default=config.api_key,
        help='API key for the LLM provider'
    )
    parser.add_argument(
        '--api-base',
        default=config.api_base,
        help='Base URL for the LLM API'
    )
    parser.add_argument(
        '--provider',
        default=config.provider,
        help='LLM provider (openai, anthropic, google, ollama)'
    )
    args = parser.parse_args()
    
    # Update config with command line args
    config.model = args.model
    config.embed_model = args.embed_model
    config.persist_dir = args.persist_dir
    config.nofallback = args.nofallback
    config.debug = args.debug
    config.show_thoughts = args.show_thoughts
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens
    config.api_key = args.api_key
    config.api_base = args.api_base
    config.provider = args.provider
    
    return config

def setup_readline(vectorstore) -> None:
    """Configure readline with command completion."""
    completer = FileCompleter(vectorstore)
    readline.set_completer(completer.complete)
    # Set completer delims to only include space to handle paths properly
    readline.set_completer_delims(' ')
    if 'libedit' in readline.__doc__:
        # macOS uses libedit instead of GNU readline
        readline.parse_and_bind('bind ^I rl_complete')
    else:
        readline.parse_and_bind('tab: complete')

def initialize_components(config: ChatConfig):
    """Initialize all required components."""
    vectorstore = initialize_rag(
        config.embed_model, 
        config.persist_dir,
        api_key=config.api_key
    )
    
    # Initialize the LLM using the provider system
    llm = get_llm(
        model_name=config.model,
        api_key=config.api_key,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        provider=config.provider
    )
    
    memory = SimpleMemory(max_tokens=5)
    return vectorstore, llm, memory

def print_result(result: dict, config: ChatConfig) -> None:
    """Print command result with proper formatting."""
    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    if "answer" in result:
        if config.show_thoughts and "thoughts" in result and result["thoughts"]:
            print(f"\nThoughts: {result['thoughts']}")
        print(f"\n{result['answer']}")
        
        # Only show sources if there are any and we're not in fallback mode
        if result.get("source_documents") and not result.get("used_fallback", False):
            print("\nSources:", ", ".join(result["source_documents"]))
        print()
    elif "message" in result:
        print(result["message"])

def chat_loop(config: ChatConfig, vectorstore, llm, memory) -> NoReturn:
    spinner = ThinkingSpinner(config)  # Pass config to ThinkingSpinner
    while True:
        try:
            user_input = input(">> ").strip()
            
            if not user_input:
                continue

            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
            else:
                command = ""
                args = user_input

            with spinner:  # Show spinner while processing
                if command:
                    result = process_command(command, args, vectorstore, llm, memory, config)
                else:
                    result = process_command('/ask', args, vectorstore, llm, memory, config)

                if result.get('exit'):
                    break
                if result.get('clear_memory'):
                    memory.clear()
                if result.get('new_vectorstore'):
                    vectorstore = result['new_vectorstore']
                    setup_readline(vectorstore)
            
            print_result(result, config)  # Pass config to print_result

        except KeyboardInterrupt:
            print("\nUse /bye to exit")
            continue
        except Exception as e:
            print(f"Error: {e}")
            if config.debug:
                import traceback
                print(traceback.format_exc())

def main():
    """Application entry point."""
    try:
        # Parse command line arguments
        config = parse_args()
        
        # Initialize components
        vectorstore, llm, memory = initialize_components(config)
        
        # Setup readline completion
        setup_readline(vectorstore)
        
        # Enter main chat loop
        chat_loop(config, vectorstore, llm, memory)
    except Exception as e:
        print(f"Fatal error: {e}")
        if config.debug:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()