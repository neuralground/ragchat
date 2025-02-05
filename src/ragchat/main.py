import argparse
import readline
import sys
from typing import NoReturn

from langchain_ollama import OllamaLLM
from .config import ChatConfig, RAGCHAT_HOME_ENV
from .document.vectorstore import initialize_rag
from .chat.memory import SimpleMemory
from .cli.completer import FileCompleter
from .cli.commands import process_command
from .cli.spinner import ThinkingSpinner

def parse_args() -> ChatConfig:
    """Parse command line arguments and return ChatConfig."""
    parser = argparse.ArgumentParser(description='Chat with documents using Ollama')
    parser.add_argument(
        '--model',
        default='mistral-small:24b',
        help='Ollama model to use for chat (default: mistral-small:24b)'
    )
    parser.add_argument(
        '--embed-model',
        default='nomic-embed-text',
        help='Ollama model to use for embeddings (default: nomic-embed-text)'
    )
    parser.add_argument(
        '--persist-dir',
        help=f'Directory to persist vector store (default: $HOME/.ragchat/chroma_db or ${RAGCHAT_HOME_ENV}/chroma_db if set)'
    )
    parser.add_argument(
        '--nofallback',
        action='store_true',
        help='Disable model answering from its own knowledge when no relevant documents found'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--show-thoughts',
        action='store_true',
        help='Show model chain-of-thought when present'
    )
    args = parser.parse_args()
    return ChatConfig(
        model=args.model,
        embed_model=args.embed_model,
        persist_dir=args.persist_dir,
        nofallback=args.nofallback,
        debug=args.debug,
        show_thoughts=args.show_thoughts
    )

def setup_readline(vectorstore) -> None:
    """Configure readline with command completion."""
    completer = FileCompleter(vectorstore)
    readline.set_completer(completer.complete)
    readline.set_completer_delims('')
    if 'libedit' in readline.__doc__:
        readline.parse_and_bind('bind ^I rl_complete')
    else:
        readline.parse_and_bind('tab: complete')

def initialize_components(config: ChatConfig):
    """Initialize all required components."""
    vectorstore = initialize_rag(config.embed_model, config.persist_dir)
    llm = OllamaLLM(model=config.model)
    memory = SimpleMemory(max_tokens=5)
    return vectorstore, llm, memory

def print_result(result: dict, config: ChatConfig) -> None:
    """Print command result with proper formatting."""
    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    if "answer" in result:
        if config.show_thoughts and "thoughts" in result:
            print(f"\nThoughts: {result['thoughts']}")
        print(f"\n{result['answer']}")
        if result.get("source_documents"):
            print("\nSources:", ", ".join(result["source_documents"]))
        print()
    elif "message" in result:
        print(result["message"])

def chat_loop(config: ChatConfig, vectorstore, llm, memory) -> NoReturn:
    spinner = ThinkingSpinner()
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