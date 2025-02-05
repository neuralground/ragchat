import sys
import threading
import time
import itertools

class ThinkingSpinner:
    def __init__(self):
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.busy = False
        self.spinner_thread = None

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(f'\r{next(self.spinner_chars)} Thinking... ')
            sys.stdout.flush()
            time.sleep(0.1)
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
