#!/usr/bin/env python3
"""
Script to extract all class and function definitions from Python files in src/ directory.
Output format shows the file path as a comment followed by all def and class statements
with the first 5 lines of each method/class indented by a tab.

Usage: python extract_definitions.py
"""

import os
import re
from pathlib import Path

def extract_definitions_from_file(file_path):
    """Extract def and class statements from a Python file along with first 5 lines."""
    definitions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Strip whitespace but preserve indentation for context
            stripped_line = line.strip()
            
            # Look for class or def statements (but not commented out)
            if (stripped_line.startswith('class ') or stripped_line.startswith('def ')) and not stripped_line.startswith('#'):
                # Extract just the signature part (before any colon)
                if ':' in stripped_line:
                    signature = stripped_line.split(':')[0].strip()
                else:
                    signature = stripped_line.strip()
                
                # Capture the first 5 lines after the definition
                body_lines = []
                for i in range(5):
                    next_line_idx = line_num + i
                    if next_line_idx < len(lines):
                        next_line = lines[next_line_idx]
                        # Remove trailing whitespace but preserve leading indentation
                        body_lines.append(next_line.rstrip())
                    else:
                        break
                
                definitions.append((signature, body_lines))
    
    except (UnicodeDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
    
    return definitions

def main():
    """Main function to process all Python files in src/ directory."""
    src_dir = Path("src/ui/")
    
    if not src_dir.exists():
        print("Error: src/ directory not found. Make sure you're running this from the AviaNZ root directory.")
        return
    
    # Find all Python files in src/ recursively
    python_files = []
    for file_path in src_dir.rglob("*.py"):
        python_files.append(file_path)
    
    # Sort files for consistent output
    python_files.sort()
    
    if not python_files:
        print("No Python files found in src/ directory.")
        return
    
    # Open output file for writing
    output_file = "python_definitions.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Process each file
        for file_path in python_files:
            definitions = extract_definitions_from_file(file_path)
            
            if definitions:
                # Convert path to relative path from src/ and remove .py extension for cleaner output
                relative_path = file_path.relative_to("src")
                display_path = str(relative_path).replace('.py', '').replace('/', '.')
                
                f.write(f"# src.{display_path}\n")
                for signature, body_lines in definitions:
                    f.write(f"{signature}\n")
                    # Write the first 5 lines with tab indentation
                    for body_line in body_lines:
                        f.write(f"\t{body_line}\n")
                    f.write("\n")  # Empty line after each definition
                f.write("\n")  # Empty line between files
    
    print(f"Definitions extracted and saved to {output_file}")
    print(f"Processed {len(python_files)} Python files.")

if __name__ == "__main__":
    main()