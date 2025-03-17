import os
import readline
from typing import Optional, List
from langchain_chroma import Chroma

class FileCompleter:
    def __init__(self, vectorstore: Optional[Chroma] = None):
        self.commands = ['/add', '/bye', '/clear', '/help', '/list', '/remove', '/reset']
        self.vectorstore = vectorstore
        
    def get_loaded_documents(self):
        if not self.vectorstore:
            return []
        try:
            sources = set()
            results = self.vectorstore.get(include=['metadatas'])
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if 'source_file' in metadata:
                        sources.add(metadata['source_file'])
            return sorted(sources)
        except Exception:
            return []
    
    def _complete_path(self, path: str) -> List[str]:
        """Complete a file path with support for directories."""
        if not path:
            path = './'
            
        if os.path.isdir(path):
            # Path is a directory, list its contents
            base_path = path
            path_prefix = ""
        else:
            # Path might be a partial file/directory name
            base_path = os.path.dirname(path) or '.'
            path_prefix = os.path.basename(path)
            
        try:
            # Get all files and directories in the base path
            all_items = os.listdir(base_path)
            
            # Filter items that match the prefix
            matching_items = [item for item in all_items if item.startswith(path_prefix)]
            
            # Construct full paths
            if base_path == '.':
                completed_paths = matching_items
            else:
                # Make sure the base path ends with a slash
                if not base_path.endswith('/'):
                    base_path += '/'
                completed_paths = [f"{base_path}{item}" for item in matching_items]
            
            # Add trailing slash to directories
            for i, item in enumerate(completed_paths):
                full_path = item
                if os.path.isdir(full_path) and not full_path.endswith('/'):
                    completed_paths[i] = f"{full_path}/"
                    
            return completed_paths
        except OSError:
            return []
    
    def complete(self, text: str, state: int):
        try:
            line_buffer = readline.get_line_buffer()
            
            if line_buffer.startswith('/') and ' ' not in line_buffer:
                # Command completion
                responses = [cmd for cmd in self.commands if cmd.startswith(line_buffer)]
                return responses[state] if state < len(responses) else None
            elif line_buffer.startswith('/add '):
                # Extract the path part after "/add "
                prefix = line_buffer.split(' ', 1)[1]
                completed_paths = self._complete_path(prefix)
                
                # We need to return the full line for readline to replace correctly
                if state < len(completed_paths):
                    return completed_paths[state]
                return None
            elif line_buffer.startswith('/remove '):
                # Extract the part after "/remove "
                prefix = line_buffer.split(' ', 1)[1].lstrip()
                
                # Handle [#] format completion
                if prefix.startswith('[') or not prefix:
                    try:
                        sources = {}
                        results = self.vectorstore.get(include=['metadatas'])
                        if results and results['metadatas']:
                            for metadata in results['metadatas']:
                                if 'source_number' in metadata:
                                    sources[metadata['source_number']] = None
                        
                        num_prefix = prefix[1:] if prefix.startswith('[') else ''
                        num_responses = []
                        for num in sorted(sources.keys()):
                            if str(num).startswith(num_prefix):
                                num_responses.append(f"[{num}]")
                        
                        if state < len(num_responses):
                            return '/remove ' + num_responses[state]
                    except Exception:
                        pass
                
                # Handle filename completion
                if not prefix.startswith('['):
                    loaded_docs = self.get_loaded_documents()
                    file_responses = [f for f in loaded_docs if f.startswith(prefix)]
                    
                    if state < len(file_responses):
                        return '/remove ' + file_responses[state]
                
                return None
            else:
                return None
        
        except Exception as e:
            if os.environ.get('RAGCHAT_DEBUG') == 'true':
                print(f"Completion error: {str(e)}")
            return None
