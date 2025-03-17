import sys
import threading
import time
import itertools
import os
from ..config import ChatConfig

class ThinkingSpinner:
    def __init__(self, config=None):
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.busy = False
        self.spinner_thread = None
        self.config = config
        
        # Determine if debug is enabled by checking both config and environment
        self.debug_enabled = False
        if config is not None and hasattr(config, 'debug'):
            self.debug_enabled = config.debug
        else:
            # Fallback to environment variable
            self.debug_enabled = os.environ.get('RAGCHAT_DEBUG', '').lower() in ('true', '1', 'yes')

    def spinner_task(self):
        while self.busy:
            # Only show spinner if debug is enabled
            if self.debug_enabled:
                sys.stdout.write(f'\r{next(self.spinner_chars)} Thinking... ')
                sys.stdout.flush()
            time.sleep(0.1)
        
        # Clear the line if we were showing the spinner
        if self.debug_enabled:
            sys.stdout.write('\r')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        self.spinner_thread = threading.Thread(target=self.spinner_task)
        self.spinner_thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        if self.spinner_thread:
            self.spinner_thread.join()
