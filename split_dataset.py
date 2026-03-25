#!/usr/bin/env python3
"""
Split processed dataset into train/test sets with stratification.

This script takes the output from data_loader.py and creates train/test splits,
maintaining class balance through stratification.

Usage:
    python split_dataset.py <input_folder> <output_base_folder> [options]

Examples:
    # Default 80/20 split
    python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split"
    
    # Custom 70/30 split
    python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split" --test-ratio 0.3
    
    # Different random seed for reproducibility
    python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split" --seed 123
"""

import os
import json
import shutil
import argparse
import random
from collections import defaultdict
import numpy as np


def load_labels(labels_path):
    """Load labels.json file"""
    with open(labels_path, 'r') as f:
        return json.load(f)


def random_split(files, test_ratio, random_state=42):
    """
    Random split ensuring every class appears in BOTH train and test (multilabel-aware).
    
    In multilabel setting, each file can have multiple classes. This function ensures
    that for every class:
    - At least one file containing that class goes to train
    - At least one file containing that class goes to test (if possible)
    
    Algorithm:
    1. Find all classes and which files contain them
    2. For each class, reserve at least one file for train and one for test
    3. Randomly assign remaining files according to test_ratio
    
    Args:
        files: List of file entries with 'class_names' field
        test_ratio: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        train_files, test_files, split_info: Lists of file entries and per-class counts
    """
    random.seed(random_state)
    
    # Build class-to-files mapping
    class_to_files = defaultdict(set)
    for i, f in enumerate(files):
        for cls in f.get('class_names', []):
            class_to_files[cls].add(i)
    
    # Track which files go to train vs test
    assigned = {}  # file_idx -> 'train' or 'test'
    
    # Phase 1: Ensure each class with 2+ files has at least one in train and one in test
    for cls, file_indices in class_to_files.items():
        file_list = list(file_indices)
        
        if len(file_list) == 1:
            # Only one file for this class - put in train
            idx = file_list[0]
            if idx not in assigned:
                assigned[idx] = 'train'
        elif len(file_list) >= 2:
            # Ensure at least one in train and one in test
            random.shuffle(file_list)
            unassigned = [idx for idx in file_list if idx not in assigned]
            
            # Check if we already have representation in both splits
            has_train = any(assigned.get(idx) == 'train' for idx in file_list)
            has_test = any(assigned.get(idx) == 'test' for idx in file_list)
            
            if not has_train and unassigned:
                assigned[unassigned.pop(0)] = 'train'
            if not has_test and unassigned:
                assigned[unassigned.pop(0)] = 'test'
    
    # Phase 2: Randomly assign remaining files according to test_ratio
    for i in range(len(files)):
        if i not in assigned:
            if random.random() < test_ratio:
                assigned[i] = 'test'
            else:
                assigned[i] = 'train'
    
    # Build train and test sets
    train_files = [files[i] for i in range(len(files)) if assigned[i] == 'train']
    test_files = [files[i] for i in range(len(files)) if assigned[i] == 'test']
    
    # Build split info (count by class)
    split_info = {}
    for cls, file_indices in class_to_files.items():
        train_count = sum(1 for idx in file_indices if assigned[idx] == 'train')
        test_count = sum(1 for idx in file_indices if assigned[idx] == 'test')
        split_info[cls] = {
            'train': train_count,
            'test': test_count,
            'total': len(file_indices)
        }
    
    # Final shuffle
    random.shuffle(train_files)
    random.shuffle(test_files)
    
    return train_files, test_files, split_info


def _get_first_class(file_entry):
    """Get first class name for grouping purposes in multilabel setting."""
    if 'class_names' in file_entry and len(file_entry['class_names']) > 0:
        return file_entry['class_names'][0]
    return 'unknown'


def grouped_random_split(files, test_ratio, group_key, random_state=42):
    """
    Split keeping groups intact (e.g., by source recording).
    
    Groups are defined by file_entry[group_key]. All entries with the same
    group id go to either train or test, never both.
    
    Simple random split without stratification - randomly assigns groups to train/test.
    """
    if not group_key:
        raise ValueError("group_key must be provided")

    random.seed(random_state)

    groups = defaultdict(list)
    for entry in files:
        gid = entry.get(group_key)
        if gid is None:
            gid = entry.get('source_file')
        if gid is None:
            gid = entry.get('filename')
        groups[str(gid)].append(entry)

    group_ids = list(groups.keys())
    if len(group_ids) < 2:
        return files, [], {'_all': {'total': len(files), 'train': len(files), 'test': 0}}

    # Simple random shuffle and split
    random.shuffle(group_ids)
    n_test_groups = max(1, int(len(group_ids) * test_ratio))
    if n_test_groups >= len(group_ids):
        n_test_groups = len(group_ids) - 1
    
    test_group_ids = set(group_ids[:n_test_groups])
    train_group_ids = set(group_ids[n_test_groups:])

    # Build file lists
    train_files = []
    test_files = []
    for gid, entries in groups.items():
        if gid in test_group_ids:
            test_files.extend(entries)
        else:
            train_files.extend(entries)

    random.shuffle(train_files)
    random.shuffle(test_files)

    # Compute per-class stats for reporting (counts all classes in multilabel)
    train_by_class = defaultdict(int)
    test_by_class = defaultdict(int)
    for f in train_files:
        for cls in f.get('class_names', []):
            train_by_class[cls] += 1
    for f in test_files:
        for cls in f.get('class_names', []):
            test_by_class[cls] += 1
    
    split_info = {}
    for class_name in set(train_by_class.keys()) | set(test_by_class.keys()):
        split_info[class_name] = {
            'train': train_by_class[class_name],
            'test': test_by_class[class_name],
            'total': train_by_class[class_name] + test_by_class[class_name]
        }
    
    split_info['_meta'] = {
        'group_key': group_key,
        'total_groups': len(group_ids),
        'train_groups': len(train_group_ids),
        'test_groups': len(test_group_ids)
    }
    return train_files, test_files, split_info


def copy_file_and_audio(src_data_folder, dst_data_folder, filename, src_audio_folder=None, dst_audio_folder=None):
    """
    Copy a data file and its corresponding audio file if it exists.
    
    Args:
        src_data_folder: Source data/ folder
        dst_data_folder: Destination data/ folder
        filename: Filename (e.g., 'file_00000001.npy')
        src_audio_folder: Source audio/ folder (optional)
        dst_audio_folder: Destination audio/ folder (optional)
    """
    # Copy main data file
    src_path = os.path.join(src_data_folder, filename)
    dst_path = os.path.join(dst_data_folder, filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: File not found: {src_path}")
    
    # Copy audio file if it exists
    if src_audio_folder and dst_audio_folder and os.path.exists(src_audio_folder):
        # Try to find matching audio file (could be .wav, etc.)
        base_name = os.path.splitext(filename)[0]
        for ext in ['.wav', '.flac', '.mp3']:
            audio_filename = base_name + ext
            src_audio_path = os.path.join(src_audio_folder, audio_filename)
            
            if os.path.exists(src_audio_path):
                dst_audio_path = os.path.join(dst_audio_folder, audio_filename)
                shutil.copy2(src_audio_path, dst_audio_path)
                break


def split_dataset(input_folder, output_base_folder, test_ratio=0.2, random_seed=42, overwrite=False, group_key=None):
    """
    Split a processed dataset into train and test sets.
    
    Args:
        input_folder: Path to processed data folder (output from data_loader.py)
        output_base_folder: Base path for output (will create train/ and test/ subfolders)
        test_ratio: Fraction of data to use for testing (default: 0.2 = 80/20 split)
        random_seed: Random seed for reproducibility
        overwrite: Whether to overwrite existing output folders
    """
    print(f"Splitting dataset from {input_folder}")
    print(f"Test ratio: {test_ratio:.1%} (train: {1-test_ratio:.1%})")
    print(f"Random seed: {random_seed}")
    
    # Load labels
    labels_path = os.path.join(input_folder, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {input_folder}")
    
    metadata = load_labels(labels_path)
    files = metadata['files']
    categories = metadata.get('categories', [])
    
    print(f"\nDataset info:")
    print(f"  Total files: {len(files)}")
    print(f"  Classes: {len(categories)}")
    print(f"  Dataset type: {metadata.get('dataset', 'Unknown')}")
    
    # Check if audio folder exists
    src_audio_folder = os.path.join(input_folder, "audio")
    has_audio = os.path.exists(src_audio_folder) and os.path.isdir(src_audio_folder)
    if has_audio:
        print(f"  Audio folder: present")
    
    # Group files by first class for reporting
    files_by_class = defaultdict(list)
    for file_entry in files:
        files_by_class[_get_first_class(file_entry)].append(file_entry)

    print(f"\nClass distribution:")
    for class_name in sorted(files_by_class.keys()):
        print(f"  {class_name}: {len(files_by_class[class_name])} files")

    # Perform simple random split (no stratification, no grouping)
    print(f"\nPerforming simple random split...")
    train_files, test_files, split_info = random_split(files, test_ratio, random_seed)
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Test: {len(test_files)} files")
    print(f"\nPer-class split:")
    for class_name in sorted(split_info.keys()):
        info = split_info[class_name]
        print(f"  {class_name}: {info['train']} train, {info['test']} test (total: {info['total']})")
    
    # Create output folders
    train_folder = os.path.join(output_base_folder, "train")
    test_folder = os.path.join(output_base_folder, "test")
    
    for folder in [train_folder, test_folder]:
        if os.path.exists(folder):
            if overwrite:
                print(f"\nRemoving existing folder: {folder}")
                shutil.rmtree(folder)
            else:
                raise FileExistsError(f"Output folder {folder} already exists. Use --overwrite to overwrite.")
    
    # Create folder structure
    train_data_folder = os.path.join(train_folder, "data")
    test_data_folder = os.path.join(test_folder, "data")
    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(test_data_folder, exist_ok=True)
    
    # Create audio folders if needed
    train_audio_folder = None
    test_audio_folder = None
    if has_audio:
        train_audio_folder = os.path.join(train_folder, "audio")
        test_audio_folder = os.path.join(test_folder, "audio")
        os.makedirs(train_audio_folder, exist_ok=True)
        os.makedirs(test_audio_folder, exist_ok=True)
    
    src_data_folder = os.path.join(input_folder, "data")
    
    # Copy train files
    print(f"\nCopying train files...")
    for i, file_entry in enumerate(train_files):
        filename = file_entry['filename']
        copy_file_and_audio(src_data_folder, train_data_folder, filename, 
                          src_audio_folder, train_audio_folder)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(train_files):
            print(f"  Copied {i + 1}/{len(train_files)} train files")
    
    # Copy test files
    print(f"\nCopying test files...")
    for i, file_entry in enumerate(test_files):
        filename = file_entry['filename']
        copy_file_and_audio(src_data_folder, test_data_folder, filename,
                          src_audio_folder, test_audio_folder)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(test_files):
            print(f"  Copied {i + 1}/{len(test_files)} test files")
    
    # Save train labels
    train_metadata = metadata.copy()
    train_metadata['files'] = train_files
    train_metadata['split'] = 'train'
    train_metadata['split_info'] = {
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'original_dataset': input_folder
    }
    
    train_labels_path = os.path.join(train_folder, "labels.json")
    with open(train_labels_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    # Save test labels
    test_metadata = metadata.copy()
    test_metadata['files'] = test_files
    test_metadata['split'] = 'test'
    test_metadata['split_info'] = {
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'original_dataset': input_folder
    }
    
    test_labels_path = os.path.join(test_folder, "labels.json")
    with open(test_labels_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Split complete!")
    print(f"Train folder: {train_folder}")
    print(f"Test folder: {test_folder}")
    print(f"{'='*50}")
    
    return train_folder, test_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split processed dataset into train/test sets with stratification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 80/20 split
  python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split"
  
  # Custom 70/30 split
  python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split" --test-ratio 0.3
  
  # Custom 90/10 split with different seed
  python split_dataset.py "Sound Files/DOC_spec" "Sound Files/DOC_split" --test-ratio 0.1 --seed 123
  
  # Overwrite existing split
  python split_dataset.py "Sound Files/GSK_spec" "Sound Files/GSK_split" --overwrite

Output Structure:
  <output_base_folder>/
    train/
      data/               # Training spectrogram files
      audio/              # Training audio files (if --with-audio was used)
      labels.json         # Training labels
    test/
      data/               # Test spectrogram files
      audio/              # Test audio files (if --with-audio was used)
      labels.json         # Test labels
        """
    )
    
    parser.add_argument('input_folder', type=str,
                       help="Path to processed data folder (output from data_loader.py)")
    parser.add_argument('output_base_folder', type=str,
                       help="Base path for output (will create train/ and test/ subfolders)")
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help="Fraction of data to use for testing (default: 0.2 = 80/20 split)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--overwrite', action='store_true',
                       help="Overwrite existing output folders")

    parser.add_argument('--group-key', type=str, default=None,
                       help="Optional metadata key to keep groups intact (prevents leakage). Typical: source_file")
    
    args = parser.parse_args()
    
    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio must be between 0.0 and 1.0")
    
    if not os.path.exists(args.input_folder):
        parser.error(f"Input folder does not exist: {args.input_folder}")
    
    train_folder, test_folder = split_dataset(
        input_folder=args.input_folder,
        output_base_folder=args.output_base_folder,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        overwrite=args.overwrite,
        group_key=args.group_key
    )
    
    print(f"\n✓ Done! Created train and test splits.")
    print(f"\nYou can now use these folders for training:")
    print(f"  Train: {train_folder}")
    print(f"  Test: {test_folder}")
