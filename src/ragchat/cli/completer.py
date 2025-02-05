import os
import readline
from typing import Optional
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
    
    def complete(self, text: str, state: int):
        try:
            line_buffer = readline.get_line_buffer()
            
            if line_buffer.startswith('/') and ' ' not in line_buffer:
                responses = [cmd for cmd in self.commands if cmd.startswith(line_buffer)]
            elif line_buffer.startswith('/add '):
                prefix = line_buffer.split(' ', 1)[1].lstrip()
                try:
                    all_files = os.listdir('.')
                    matching_files = [f for f in all_files if f.startswith(prefix)]
                    responses = ['/add ' + f for f in matching_files]
                except OSError:
                    responses = []
            elif line_buffer.startswith('/remove '):
                prefix = line_buffer.split(' ', 1)[1].lstrip()
                loaded_docs = self.get_loaded_documents()
                responses = ['/remove ' + f for f in loaded_docs if f.startswith(prefix)]
            else:
                responses = []
            
            return responses[state] if state < len(responses) else None
            
        except Exception:
            return None

