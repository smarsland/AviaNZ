#!/usr/bin/env python3
"""
Script to analyze .npy files in a folder.
Reports the number of files and the shapes of 2D numpy arrays (spectrograms).
"""

import argparse
import numpy as np
from pathlib import Path


def analyze_npy_files(folder_path):
    """
    Analyze .npy files in the given folder.
    
    Args:
        folder_path: Path to the folder containing .npy files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # Get all .npy files
    npy_files = list(folder.glob("*.npy"))
    
    print(f"Folder: {folder.absolute()}")
    print(f"Total .npy files: {len(npy_files)}")
    print("-" * 60)
    
    if len(npy_files) == 0:
        print("No .npy files found.")
        return
    
    # Analyze each file
    shapes = {}
    errors = []
    
    for npy_file in sorted(npy_files):
        try:
            arr = np.load(npy_file)
            shape = arr.shape
            
            # Store shape counts
            if shape not in shapes:
                shapes[shape] = []
            shapes[shape].append(npy_file.name)
            
            print(f"{npy_file.name}: shape {shape}")
            
        except Exception as e:
            errors.append((npy_file.name, str(e)))
            print(f"{npy_file.name}: ERROR - {e}")
    
    # Summary
    print("-" * 60)
    print("Summary:")
    print(f"  Successfully loaded: {len(npy_files) - len(errors)} files")
    if errors:
        print(f"  Errors: {len(errors)} files")
    
    print("\nShape distribution:")
    for shape, files in sorted(shapes.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {shape}: {len(files)} file(s)")
    
    if errors:
        print("\nFiles with errors:")
        for filename, error in errors:
            print(f"  {filename}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze .npy files in a folder and report their shapes."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing .npy files"
    )
    
    args = parser.parse_args()
    analyze_npy_files(args.folder)


if __name__ == "__main__":
    main()
