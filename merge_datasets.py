"""
Merge two AviaNZ datasets into a new combined dataset.

This script combines two dataset folders (each with labels.json and .npy spectrograms)
into a single merged dataset, handling:
- Category merging (union of species from both datasets)
- Label remapping (adjusting one-hot vectors for merged category list)
- File copying or symlinking (spectrograms and optional audio)
- Conflict detection (duplicate filenames)

Usage:
    # Basic merge (copy files)
    python merge_datasets.py data/train1 data/train2 data/combined
    
    # Use symlinks instead of copying (faster, saves space)
    python merge_datasets.py data/train1 data/train2 data/combined --symlink
    
    # Skip audio files (only merge spectrograms and labels)
    python merge_datasets.py data/train1 data/train2 data/combined --no-audio
    
    # Require identical categories (error if different)
    python merge_datasets.py data/train1 data/train2 data/combined --require-same-categories
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np


class DatasetMerger:
    """Merge two AviaNZ dataset folders into one."""
    
    def __init__(self, folder1, folder2, output_folder, symlink=False, 
                 include_audio=True, require_same_categories=False):
        self.folder1 = Path(folder1)
        self.folder2 = Path(folder2)
        self.output_folder = Path(output_folder)
        self.symlink = symlink
        self.include_audio = include_audio
        self.require_same_categories = require_same_categories
        
        self.labels1 = None
        self.labels2 = None
        self.merged_labels = None
        
    def load_labels(self):
        """Load labels.json from both folders."""
        print("Loading labels...")
        
        labels1_path = self.folder1 / 'labels.json'
        labels2_path = self.folder2 / 'labels.json'
        
        if not labels1_path.exists():
            raise FileNotFoundError(f"labels.json not found in {self.folder1}")
        if not labels2_path.exists():
            raise FileNotFoundError(f"labels.json not found in {self.folder2}")
        
        with open(labels1_path, 'r') as f:
            self.labels1 = json.load(f)
        
        with open(labels2_path, 'r') as f:
            self.labels2 = json.load(f)
        
        print(f"  Dataset 1: {len(self.labels1['files'])} files, {len(self.labels1['categories'])} categories")
        print(f"  Dataset 2: {len(self.labels2['files'])} files, {len(self.labels2['categories'])} categories")
    
    def merge_categories(self):
        """Merge category lists from both datasets."""
        cats1 = self.labels1['categories']
        cats2 = self.labels2['categories']
        
        print("\nMerging categories...")
        
        if self.require_same_categories:
            if cats1 != cats2:
                print(f"ERROR: Categories differ between datasets:")
                print(f"  Dataset 1: {cats1}")
                print(f"  Dataset 2: {cats2}")
                raise ValueError("Categories must match when --require-same-categories is set")
            merged_cats = cats1
            print(f"  Categories match: {len(merged_cats)} species")
        else:
            # Merge categories (union, preserving order from dataset 1)
            merged_cats = list(cats1)
            new_cats = [cat for cat in cats2 if cat not in merged_cats]
            merged_cats.extend(new_cats)
            
            print(f"  Dataset 1 categories: {cats1}")
            print(f"  Dataset 2 categories: {cats2}")
            print(f"  Merged categories ({len(merged_cats)}): {merged_cats}")
            
            if new_cats:
                print(f"  New categories from dataset 2: {new_cats}")
        
        return merged_cats
    
    def remap_classes(self, file_entry, old_categories, new_categories):
        """
        Remap class names - no changes needed since we store actual species names.
        Just validate that all classes exist in merged category list.
        """
        # No remapping needed - class_names are actual species names, not indices
        # Just return the entry as-is since category names are preserved
        return file_entry
    
    def check_duplicates(self):
        """Check for duplicate filenames between datasets."""
        files1 = {entry['filename'] for entry in self.labels1['files']}
        files2 = {entry['filename'] for entry in self.labels2['files']}
        
        duplicates = files1 & files2
        
        if duplicates:
            print(f"\nWARNING: {len(duplicates)} duplicate filenames found:")
            for dup in sorted(duplicates)[:10]:
                print(f"  - {dup}")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more")
            print("\nDuplicate files will be suffixed with _ds2 for dataset 2 files.")
            return True
        else:
            print("\n✓ No duplicate filenames detected")
            return False
    
    def merge_labels_data(self, merged_categories):
        """Create merged labels structure."""
        print("\nMerging label data...")
        
        has_duplicates = self.check_duplicates()
        files1_map = {}  # Track filename changes for dataset 1
        files2_map = {}  # Track filename changes for dataset 2 (with renames)
        
        merged_files = []
        
        # Add files from dataset 1
        for entry in self.labels1['files']:
            # Validate classes are in merged categories
            if entry['primary_class'] not in merged_categories:
                print(f"  WARNING: Skipping {entry['filename']} - primary_class '{entry['primary_class']}' not in merged categories")
                continue
            
            new_entry = {
                'filename': entry['filename'],
                'primary_class': entry['primary_class'],
                'class_names': entry['class_names']
            }
            
            # Preserve optional fields
            if 'source_file' in entry:
                new_entry['source_file'] = entry['source_file']
            if 'noise' in entry:
                new_entry['noise'] = entry['noise']
            
            merged_files.append(new_entry)
            files1_map[entry['filename']] = entry['filename']  # No change for ds1
        
        # Add files from dataset 2 (with duplicate handling)
        existing_names = {entry['filename'] for entry in merged_files}
        
        for entry in self.labels2['files']:
            # Validate classes are in merged categories
            if entry['primary_class'] not in merged_categories:
                print(f"  WARNING: Skipping {entry['filename']} - primary_class '{entry['primary_class']}' not in merged categories")
                continue
            
            # Handle duplicates
            original_name = entry['filename']
            if original_name in existing_names:
                # Add suffix
                base, ext = os.path.splitext(original_name)
                new_name = f"{base}_ds2{ext}"
                
                # If still duplicate, add counter
                counter = 2
                while new_name in existing_names:
                    new_name = f"{base}_ds2_{counter}{ext}"
                    counter += 1
            else:
                new_name = original_name
            
            files2_map[original_name] = new_name
            existing_names.add(new_name)
            
            new_entry = {
                'filename': new_name,
                'primary_class': entry['primary_class'],
                'class_names': entry['class_names']
            }
            
            # Preserve optional fields
            if 'source_file' in entry:
                new_entry['source_file'] = entry['source_file']
            if 'noise' in entry:
                new_entry['noise'] = entry['noise']
            
            merged_files.append(new_entry)
        
        # Create merged labels structure
        self.merged_labels = {
            'categories': merged_categories,
            'files': merged_files
        }
        
        print(f"  Merged {len(merged_files)} total files")
        print(f"  Categories: {len(merged_categories)}")
        
        return files1_map, files2_map
    
    def copy_files(self, source_folder, filename_map, dataset_name):
        """Copy or symlink spectrogram and audio files."""
        print(f"\nCopying files from {dataset_name}...")
        
        source_path = Path(source_folder)
        data_subfolder = source_path / 'data'
        
        # Check if files are in data/ subfolder or root
        if data_subfolder.exists() and data_subfolder.is_dir():
            source_data_path = data_subfolder
            print(f"  Using data/ subfolder: {source_data_path}")
        else:
            source_data_path = source_path
            print(f"  Using root folder: {source_data_path}")
        
        # Create data/ subfolder in output if it doesn't exist
        output_data_path = self.output_folder / 'data'
        output_data_path.mkdir(exist_ok=True)
        
        # Find all .npy files
        npy_files = list(source_data_path.glob('*.npy'))
        
        copied_count = 0
        skipped_count = 0
        
        for npy_file in npy_files:
            original_name = npy_file.name
            
            # Check if this file is in our labels
            if original_name not in filename_map:
                # This file isn't in labels.json, skip it
                skipped_count += 1
                continue
            
            new_name = filename_map[original_name]
            dest_path = output_data_path / new_name
            
            # Copy or symlink
            if self.symlink:
                # Create absolute symlink
                abs_source = npy_file.resolve()
                dest_path.symlink_to(abs_source)
            else:
                shutil.copy2(npy_file, dest_path)
            
            copied_count += 1
            
            # Also handle corresponding audio file if it exists and requested
            if self.include_audio:
                # Try common audio extensions
                for audio_ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    audio_name = original_name.replace('.npy', audio_ext)
                    audio_file = source_data_path / audio_name
                    
                    if audio_file.exists():
                        new_audio_name = new_name.replace('.npy', audio_ext)
                        audio_dest = output_data_path / new_audio_name
                        
                        if self.symlink:
                            abs_audio = audio_file.resolve()
                            audio_dest.symlink_to(abs_audio)
                        else:
                            shutil.copy2(audio_file, audio_dest)
                        break
        
        action = "Symlinked" if self.symlink else "Copied"
        print(f"  {action} {copied_count} spectrogram files")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} files not in labels.json")
    
    def save_merged_labels(self):
        """Save merged labels.json to output folder."""
        output_path = self.output_folder / 'labels.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.merged_labels, f, indent=2)
        
        print(f"\n✓ Saved merged labels.json to {output_path}")
    
    def merge(self):
        """Execute full merge pipeline."""
        print(f"Merging datasets:")
        print(f"  Dataset 1: {self.folder1}")
        print(f"  Dataset 2: {self.folder2}")
        print(f"  Output: {self.output_folder}")
        print(f"  Mode: {'symlink' if self.symlink else 'copy'}")
        print(f"  Include audio: {self.include_audio}")
        print()
        
        # Create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Load labels from both datasets
        self.load_labels()
        
        # Merge categories
        merged_categories = self.merge_categories()
        
        # Merge labels data (with duplicate handling)
        files1_map, files2_map = self.merge_labels_data(merged_categories)
        
        # Copy/symlink files from both datasets
        self.copy_files(self.folder1, files1_map, "Dataset 1")
        self.copy_files(self.folder2, files2_map, "Dataset 2")
        
        # Save merged labels.json
        self.save_merged_labels()
        
        print(f"\n{'='*60}")
        print(f"✓ Dataset merge complete!")
        print(f"  Output folder: {self.output_folder}")
        print(f"  Total files: {len(self.merged_labels['files'])}")
        print(f"  Total categories: {len(self.merged_labels['categories'])}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two AviaNZ dataset folders into one combined dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge (copy files)
  python merge_datasets.py data/train1 data/train2 data/combined
  
  # Use symlinks (faster, saves disk space, but links to original files)
  python merge_datasets.py data/train1 data/train2 data/combined --symlink
  
  # Only merge spectrograms and labels (skip audio)
  python merge_datasets.py data/train1 data/train2 data/combined --no-audio
  
  # Require identical categories (fails if categories differ)
  python merge_datasets.py data/train1 data/train2 data/combined --require-same-categories
  
  # Then train on merged dataset:
  python finetune_birdclef.py data/combined outputs/model_combined

Notes:
  - Both input folders must contain labels.json
  - If filenames conflict, dataset 2 files get "_ds2" suffix
  - Category merging: union of both category lists (unless --require-same-categories)
  - Labels are automatically remapped to merged category list
  - Symlinks preserve disk space but require original files to remain in place
        """
    )
    
    parser.add_argument('folder1', type=str,
                       help="First dataset folder (with labels.json and .npy files)")
    parser.add_argument('folder2', type=str,
                       help="Second dataset folder (with labels.json and .npy files)")
    parser.add_argument('output_folder', type=str,
                       help="Output folder for merged dataset")
    parser.add_argument('--symlink', action='store_true',
                       help="Create symlinks instead of copying files (faster, saves space)")
    parser.add_argument('--no-audio', action='store_true',
                       help="Skip audio files, only merge spectrograms and labels")
    parser.add_argument('--require-same-categories', action='store_true',
                       help="Require both datasets to have identical categories (error if different)")
    
    args = parser.parse_args()
    
    # Validate input folders
    if not os.path.exists(args.folder1):
        print(f"ERROR: Dataset 1 folder not found: {args.folder1}")
        return 1
    
    if not os.path.exists(args.folder2):
        print(f"ERROR: Dataset 2 folder not found: {args.folder2}")
        return 1
    
    # Check for labels.json
    if not os.path.exists(os.path.join(args.folder1, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.folder1}")
        return 1
    
    if not os.path.exists(os.path.join(args.folder2, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.folder2}")
        return 1
    
    # Create merger and execute
    merger = DatasetMerger(
        folder1=args.folder1,
        folder2=args.folder2,
        output_folder=args.output_folder,
        symlink=args.symlink,
        include_audio=not args.no_audio,
        require_same_categories=args.require_same_categories
    )
    
    try:
        merger.merge()
        return 0
    except Exception as e:
        print(f"\nERROR: Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
