#!/usr/bin/env python3
"""
Split matched AviaNZ and DOC datasets with proper handling of file-level grouping.

For AviaNZ: Splits at the file level (all segments from same audio file go to 
           train OR test, never both) to prevent data leakage.

For DOC: Matches the species distribution that resulted from the AviaNZ split,
        ensuring both datasets have similar class balance in train/test.

Usage:
    python split_matched_datasets.py <avianz_folder> <doc_folder> <output_base> [options]

Example:
    python split_matched_datasets.py \\
        /path/to/avianz_matched \\
        /path/to/doc_matched \\
        /path/to/matched \\
        --test-ratio 0.25 \\
        --seed 42
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


def get_file_level_groups(files):
    """
    Group file entries by their source audio file.
    
    Returns:
        dict: source_file -> list of file entries
    """
    groups = defaultdict(list)
    for entry in files:
        source_file = entry.get('source_file', entry.get('filename'))
        groups[source_file].append(entry)
    return groups


def split_avianz_by_file(files, test_ratio, random_state=42):
    """
    Split AviaNZ data at the file level to prevent data leakage.
    
    All segments from the same source file go to either train or test, never both.
    Uses simple random assignment of files to train/test.
    
    Returns:
        train_files, test_files, species_distribution
            where species_distribution = {class_name: {'train': count, 'test': count}}
    """
    random.seed(random_state)
    
    # Group by source file
    file_groups = get_file_level_groups(files)
    source_files = list(file_groups.keys())
    
    if len(source_files) < 2:
        print("Warning: Only one source file found, all data going to train")
        return files, [], {}
    
    # Randomly assign files to train or test
    random.shuffle(source_files)
    n_test_files = max(1, int(len(source_files) * test_ratio))
    if n_test_files >= len(source_files):
        n_test_files = len(source_files) - 1
    
    test_source_files = set(source_files[:n_test_files])
    train_source_files = set(source_files[n_test_files:])
    
    # Build train and test sets
    train_files = []
    test_files = []
    
    for source_file, entries in file_groups.items():
        if source_file in test_source_files:
            test_files.extend(entries)
        else:
            train_files.extend(entries)
    
    # Shuffle within train/test
    random.shuffle(train_files)
    random.shuffle(test_files)
    
    # Compute species distribution
    species_dist = compute_species_distribution(train_files, test_files)
    
    print(f"\nAviaNZ file-level split:")
    print(f"  Total source files: {len(source_files)}")
    print(f"  Train files: {len(train_source_files)} ({len(train_files)} segments)")
    print(f"  Test files: {len(test_source_files)} ({len(test_files)} segments)")
    
    return train_files, test_files, species_dist


def compute_species_distribution(train_files, test_files):
    """
    Compute species distribution across train and test sets.
    
    Returns:
        dict: {class_name: {'train': count, 'test': count, 'train_ratio': float}}
    """
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    
    for entry in train_files:
        for class_name in entry.get('class_names', []):
            train_counts[class_name] += 1
    
    for entry in test_files:
        for class_name in entry.get('class_names', []):
            test_counts[class_name] += 1
    
    all_classes = set(train_counts.keys()) | set(test_counts.keys())
    
    distribution = {}
    for class_name in all_classes:
        train_c = train_counts[class_name]
        test_c = test_counts[class_name]
        total = train_c + test_c
        
        distribution[class_name] = {
            'train': train_c,
            'test': test_c,
            'total': total,
            'train_ratio': train_c / total if total > 0 else 0.0
        }
    
    return distribution


def split_doc_by_distribution(files, target_distribution, random_state=42):
    """
    Split DOC data to match the species distribution from AviaNZ split.
    
    For each species, tries to allocate samples to train/test according to
    the train_ratio from target_distribution.
    
    Args:
        files: List of DOC file entries
        target_distribution: Species distribution dict from AviaNZ split
        random_state: Random seed
    
    Returns:
        train_files, test_files, achieved_distribution
    """
    random.seed(random_state)
    
    # Group DOC files by their first class (for splitting purposes only)
    # Note: This is a simplified approach for multi-label data
    files_by_class = defaultdict(list)
    for entry in files:
        class_names = entry.get('class_names', [])
        if class_names:
            # Use first class for grouping/splitting only
            first_class = class_names[0]
            files_by_class[first_class].append(entry)
    
    train_files = []
    test_files = []
    
    # For each class, split according to target distribution
    for class_name, class_files in files_by_class.items():
        if class_name not in target_distribution:
            # Class not in AviaNZ - put all in train
            print(f"  Warning: {class_name} not in AviaNZ data, putting all in train")
            train_files.extend(class_files)
            continue
        
        target_train_ratio = target_distribution[class_name]['train_ratio']
        
        # Shuffle and split
        random.shuffle(class_files)
        n_train = max(1, int(len(class_files) * target_train_ratio))
        
        # Ensure at least one in test if possible
        if n_train >= len(class_files) and len(class_files) > 1:
            n_train = len(class_files) - 1
        
        class_train = class_files[:n_train]
        class_test = class_files[n_train:]
        
        train_files.extend(class_train)
        test_files.extend(class_test)
    
    # Final shuffle
    random.shuffle(train_files)
    random.shuffle(test_files)
    
    # Compute achieved distribution
    achieved_dist = compute_species_distribution(train_files, test_files)
    
    print(f"\nDOC distribution-matched split:")
    print(f"  Train samples: {len(train_files)}")
    print(f"  Test samples: {len(test_files)}")
    
    return train_files, test_files, achieved_dist


def save_split(files, output_folder, split_name, original_metadata, split_info):
    """
    Save split files and metadata.
    
    Args:
        files: List of file entries
        output_folder: Output folder path
        split_name: 'train' or 'test'
        original_metadata: Original labels.json metadata
        split_info: Split information to include
    """
    # Create folders
    data_folder = os.path.join(output_folder, split_name, "data")
    os.makedirs(data_folder, exist_ok=True)
    
    # Copy data files
    src_data_folder = original_metadata.get('data_folder')
    if not src_data_folder:
        # Try to infer from first file
        if files:
            first_file = files[0].get('source_file', '')
            if first_file:
                # Assume data folder is sibling to source files
                src_data_folder = os.path.dirname(os.path.dirname(first_file))
                src_data_folder = os.path.join(src_data_folder, "data")
    
    print(f"\nCopying {len(files)} {split_name} files...")
    for i, entry in enumerate(files):
        filename = entry['filename']
        
        # Try to find source file (may be in different location)
        src_path = None
        if src_data_folder and os.path.exists(src_data_folder):
            src_path = os.path.join(src_data_folder, filename)
        
        if src_path and os.path.exists(src_path):
            dst_path = os.path.join(data_folder, filename)
            shutil.copy2(src_path, dst_path)
        else:
            print(f"  Warning: Could not find {filename}")
        
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  Copied {i + 1}/{len(files)} files")

    # Copy audio files if present (saved by build_matched_datasets.py --with-audio)
    src_audio_folder = None
    if original_metadata.get('data_folder'):
        src_audio_folder = os.path.join(os.path.dirname(original_metadata['data_folder']), 'audio')
    if src_audio_folder and os.path.exists(src_audio_folder):
        audio_folder = os.path.join(output_folder, split_name, 'audio')
        os.makedirs(audio_folder, exist_ok=True)
        print(f"\nCopying audio files to {audio_folder}...")
        copied = 0
        for entry in files:
            wav_name = entry['filename'].replace('.npy', '.wav')
            src = os.path.join(src_audio_folder, wav_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(audio_folder, wav_name))
                copied += 1
        print(f"  Copied {copied}/{len(files)} audio files")
    
    # Save labels
    metadata = original_metadata.copy()
    metadata['files'] = files
    metadata['split'] = split_name
    metadata['split_info'] = split_info
    
    labels_path = os.path.join(output_folder, split_name, "labels.json")
    with open(labels_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved {labels_path}")


def print_distribution_comparison(avianz_dist, doc_dist):
    """Print comparison of species distributions between AviaNZ and DOC."""
    print("\nSpecies distribution comparison:")
    print(f"{'Species':<20} {'AviaNZ Train%':<15} {'DOC Train%':<15} {'Difference':<10}")
    print("-" * 65)
    
    all_species = sorted(set(avianz_dist.keys()) | set(doc_dist.keys()))
    
    for species in all_species:
        avianz_train_pct = avianz_dist.get(species, {}).get('train_ratio', 0.0) * 100
        doc_train_pct = doc_dist.get(species, {}).get('train_ratio', 0.0) * 100
        diff = abs(avianz_train_pct - doc_train_pct)
        
        avianz_str = f"{avianz_train_pct:.1f}%" if species in avianz_dist else "N/A"
        doc_str = f"{doc_train_pct:.1f}%" if species in doc_dist else "N/A"
        diff_str = f"{diff:.1f}%" if (species in avianz_dist and species in doc_dist) else "N/A"
        
        print(f"{species:<20} {avianz_str:<15} {doc_str:<15} {diff_str:<10}")


def save_split_report(output_base, avianz_train, avianz_test, doc_train, doc_test, 
                     avianz_dist, doc_dist, test_ratio, random_seed):
    """
    Save comprehensive split report to JSON for later analysis.
    
    Includes:
    - Overall statistics (train/test counts and ratios)
    - File-level grouping information for AviaNZ
    - Species distribution comparison
    - List of files in each split
    """
    # Get file-level statistics for AviaNZ
    avianz_train_files = set(e.get('source_file') for e in avianz_train)
    avianz_test_files = set(e.get('source_file') for e in avianz_test)
    
    # Calculate actual ratios
    avianz_total = len(avianz_train) + len(avianz_test)
    doc_total = len(doc_train) + len(doc_test)
    
    report = {
        'metadata': {
            'target_test_ratio': test_ratio,
            'random_seed': random_seed,
            'split_method': {
                'avianz': 'file_level_grouped',
                'doc': 'distribution_matched'
            }
        },
        'avianz': {
            'total_samples': avianz_total,
            'train_samples': len(avianz_train),
            'test_samples': len(avianz_test),
            'actual_train_ratio': len(avianz_train) / avianz_total if avianz_total > 0 else 0,
            'actual_test_ratio': len(avianz_test) / avianz_total if avianz_total > 0 else 0,
            'total_source_files': len(avianz_train_files) + len(avianz_test_files),
            'train_source_files': len(avianz_train_files),
            'test_source_files': len(avianz_test_files),
            'species_distribution': avianz_dist,
            'train_files': sorted(list(avianz_train_files)),
            'test_files': sorted(list(avianz_test_files))
        },
        'doc': {
            'total_samples': doc_total,
            'train_samples': len(doc_train),
            'test_samples': len(doc_test),
            'actual_train_ratio': len(doc_train) / doc_total if doc_total > 0 else 0,
            'actual_test_ratio': len(doc_test) / doc_total if doc_total > 0 else 0,
            'species_distribution': doc_dist
        },
        'distribution_comparison': {}
    }
    
    # Add per-species comparison
    all_species = sorted(set(avianz_dist.keys()) | set(doc_dist.keys()))
    for species in all_species:
        avianz_train_pct = avianz_dist.get(species, {}).get('train_ratio', 0.0)
        doc_train_pct = doc_dist.get(species, {}).get('train_ratio', 0.0)
        
        report['distribution_comparison'][species] = {
            'avianz_train_ratio': avianz_train_pct,
            'doc_train_ratio': doc_train_pct,
            'difference': abs(avianz_train_pct - doc_train_pct)
        }
    
    # Save report
    report_path = os.path.join(output_base, 'split_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSplit report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Split matched AviaNZ and DOC datasets with proper file-level grouping",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('avianz_folder', type=str,
                       help="Path to AviaNZ matched dataset")
    parser.add_argument('doc_folder', type=str,
                       help="Path to DOC matched dataset")
    parser.add_argument('output_base', type=str,
                       help="Base output folder (will create avianz_split/ and doc_split/)")
    parser.add_argument('--test-ratio', type=float, default=0.25,
                       help="Target test ratio (default: 0.25)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--overwrite', action='store_true',
                       help="Overwrite existing splits")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.avianz_folder):
        parser.error(f"AviaNZ folder not found: {args.avianz_folder}")
    if not os.path.exists(args.doc_folder):
        parser.error(f"DOC folder not found: {args.doc_folder}")
    if not 0.0 < args.test_ratio < 1.0:
        parser.error("--test-ratio must be between 0.0 and 1.0")
    
    print("="*70)
    print("MATCHED DATASET SPLITTING")
    print("="*70)
    print(f"AviaNZ folder: {args.avianz_folder}")
    print(f"DOC folder: {args.doc_folder}")
    print(f"Output base: {args.output_base}")
    print(f"Test ratio: {args.test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    avianz_labels_path = os.path.join(args.avianz_folder, "labels.json")
    doc_labels_path = os.path.join(args.doc_folder, "labels.json")
    
    avianz_metadata = load_labels(avianz_labels_path)
    doc_metadata = load_labels(doc_labels_path)
    
    avianz_files = avianz_metadata['files']
    doc_files = doc_metadata['files']
    
    # Store data folders in metadata for later
    avianz_metadata['data_folder'] = os.path.join(args.avianz_folder, "data")
    doc_metadata['data_folder'] = os.path.join(args.doc_folder, "data")
    
    print(f"  AviaNZ: {len(avianz_files)} samples")
    print(f"  DOC: {len(doc_files)} samples")
    
    # Split AviaNZ at file level
    print("\n" + "="*70)
    print("STEP 1: Split AviaNZ at file level")
    print("="*70)
    avianz_train, avianz_test, avianz_dist = split_avianz_by_file(
        avianz_files, args.test_ratio, args.seed
    )
    
    # Split DOC to match distribution
    print("\n" + "="*70)
    print("STEP 2: Split DOC to match AviaNZ distribution")
    print("="*70)
    doc_train, doc_test, doc_dist = split_doc_by_distribution(
        doc_files, avianz_dist, args.seed
    )
    
    # Print comparison
    print_distribution_comparison(avianz_dist, doc_dist)
    
    # Save split report
    save_split_report(args.output_base, avianz_train, avianz_test, doc_train, doc_test,
                     avianz_dist, doc_dist, args.test_ratio, args.seed)
    
    # Create output folders
    avianz_output = os.path.join(args.output_base, "avianz_split")
    doc_output = os.path.join(args.output_base, "doc_split")
    
    for folder in [avianz_output, doc_output]:
        if os.path.exists(folder) and not args.overwrite:
            parser.error(f"Output folder exists: {folder}. Use --overwrite to replace.")
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # Save AviaNZ splits
    print("\n" + "="*70)
    print("STEP 3: Save AviaNZ splits")
    print("="*70)
    avianz_split_info = {
        'test_ratio': args.test_ratio,
        'random_seed': args.seed,
        'split_method': 'file_level_grouped',
        'distribution': avianz_dist
    }
    save_split(avianz_train, avianz_output, 'train', avianz_metadata, avianz_split_info)
    save_split(avianz_test, avianz_output, 'test', avianz_metadata, avianz_split_info)
    
    # Save DOC splits
    print("\n" + "="*70)
    print("STEP 4: Save DOC splits")
    print("="*70)
    doc_split_info = {
        'test_ratio': args.test_ratio,
        'random_seed': args.seed,
        'split_method': 'distribution_matched',
        'target_distribution': avianz_dist,
        'achieved_distribution': doc_dist
    }
    save_split(doc_train, doc_output, 'train', doc_metadata, doc_split_info)
    save_split(doc_test, doc_output, 'test', doc_metadata, doc_split_info)
    
    print("\n" + "="*70)
    print("SPLITTING COMPLETE")
    print("="*70)
    print(f"AviaNZ splits: {avianz_output}")
    print(f"DOC splits: {doc_output}")
    print("="*70)


if __name__ == "__main__":
    main()
