#!/usr/bin/env python3
"""
Test script to verify spinner behavior in RAGChat.
This script will run ragchat with different debug settings to verify
that the spinner only appears when debug mode is enabled.
"""

import os
import subprocess
import time
import sys
import tempfile
import re

def run_test(debug_enabled=False):
    """Run ragchat with specified debug setting and verify spinner behavior."""
    print(f"\n{'=' * 50}")
    print(f"Testing with debug_enabled={debug_enabled}")
    print(f"{'=' * 50}")
    
    # Create a temporary file to capture output
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Set environment variables
        env = os.environ.copy()
        if debug_enabled:
            env["RAGCHAT_DEBUG"] = "true"
        else:
            env.pop("RAGCHAT_DEBUG", None)
        
        # Start ragchat process
        process = subprocess.Popen(
            ["python", "-m", "ragchat"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        
        # Wait for startup
        time.sleep(2)
        
        # Send a command to exit
        process.stdin.write("/bye\n")
        process.stdin.flush()
        
        # Capture output
        output = ""
        with open(temp_filename, 'w') as f:
            for line in process.stdout:
                output += line
                f.write(line)
                if "/bye" in line:
                    break
        
        # Wait for process to exit
        process.wait(timeout=5)
        
        # Check for spinner characters
        spinner_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        spinner_found = any(char in output for char in spinner_chars)
        
        # Check for loading messages
        loading_pattern = re.compile(r'Loading|Initializing', re.IGNORECASE)
        loading_found = bool(loading_pattern.search(output))
        
        # Print results
        print(f"Spinner characters found: {spinner_found}")
        print(f"Loading messages found: {loading_found}")
        
        # Verify expectations
        if debug_enabled:
            if not spinner_found:
                print("❌ ERROR: Spinner should be visible in debug mode")
            else:
                print("✅ Spinner correctly shown in debug mode")
        else:
            if spinner_found:
                print("❌ ERROR: Spinner should NOT be visible in normal mode")
            else:
                print("✅ Spinner correctly hidden in normal mode")
        
        # Print a sample of the output
        print("\nOutput sample:")
        lines = output.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            print(f"  {line}")
        
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

def main():
    """Run tests with different debug settings."""
    # Test with debug disabled
    run_test(debug_enabled=False)
    
    # Test with debug enabled
    run_test(debug_enabled=True)
    
    # Print conclusion
    print("\nTests completed. Check the results above to verify spinner behavior.")
    print("The spinner should only be visible when debug mode is enabled.")
    print("This helps ensure that the spinner doesn't interfere with normal operation.")
    print("Especially important for non-interactive usage or when piping output.")

if __name__ == "__main__":
    main()
