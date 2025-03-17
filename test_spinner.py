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
        temp_file_path = temp_file.name
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env["RAGCHAT_DEBUG"] = "true" if debug_enabled else "false"
        
        # Start ragchat process with output redirected to the temp file
        process = subprocess.Popen(
            ["ragchat"],
            stdin=subprocess.PIPE,
            stdout=open(temp_file_path, 'w'),
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        
        # Wait for ragchat to start
        time.sleep(1)
        
        # Send a help command
        process.stdin.write("/help\n")
        process.stdin.flush()
        
        # Wait for response
        time.sleep(2)
        
        # Send exit command
        process.stdin.write("/bye\n")
        process.stdin.flush()
        
        # Wait for process to exit
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process did not exit in time, killing it...")
            process.kill()
            process.wait()
        
        # Read the output from the temp file
        with open(temp_file_path, 'r') as f:
            output = f.read()
        
        # Look specifically for the spinner character followed by "Thinking..."
        # This is a more precise way to detect the spinner
        spinner_pattern = r'[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s+Thinking\.\.\.'
        spinner_shown = bool(re.search(spinner_pattern, output))
        
        if spinner_shown:
            print(f"RESULT: Spinner was shown in output")
            if debug_enabled:
                print("This is EXPECTED when debug=true")
            else:
                print("This is UNEXPECTED when debug=false")
        else:
            print(f"RESULT: No spinner was shown in output")
            if debug_enabled:
                print("This is UNEXPECTED when debug=true")
            else:
                print("This is EXPECTED when debug=false")
        
        # Print a sample of the output for verification
        print(f"\nSample output (first 200 chars):")
        print(output[:200] if output else "(no output)")
        
        return process.returncode, spinner_shown
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def main():
    """Run tests with different debug settings."""
    print("RAGChat Spinner Test")
    print("===================")
    
    # Test with debug disabled
    print("\nTest 1: Debug disabled")
    exit_code1, spinner_shown1 = run_test(debug_enabled=False)
    
    # Test with debug enabled
    print("\nTest 2: Debug enabled")
    exit_code2, spinner_shown2 = run_test(debug_enabled=True)
    
    # Print summary
    print("\nTest Summary")
    print("===========")
    print(f"Test 1 (Debug disabled): {'PASSED' if not spinner_shown1 else 'FAILED'}")
    print(f"Test 2 (Debug enabled): {'PASSED' if spinner_shown2 else 'FAILED'}")
    
    # Return non-zero exit code if any test failed
    if (spinner_shown1 or not spinner_shown2):
        sys.exit(1)

if __name__ == "__main__":
    main()
