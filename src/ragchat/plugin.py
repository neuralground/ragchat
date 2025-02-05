# src/ragchat/plugin.py
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class NLTKDataBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """Initialize build hook."""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            print("Successfully downloaded NLTK data")
        except Exception as e:
            print(f"Warning: Failed to download NLTK data: {e}", file=sys.stderr)

def get_build_hook():
    """Return the build hook class to use."""
    return NLTKDataBuildHook