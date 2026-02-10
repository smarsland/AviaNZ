#!/usr/bin/env python3
"""
Count species occurrences in processed dataset and display from least to most popular.

Usage:
    python count_species.py <output_folder>
    
Example:
    python count_species.py "Sound Files/GSK_spec"
"""

import os
import sys
import json
import argparse
from collections import Counter


def count_species_from_labels(labels_file):
    """Load labels.json and count species occurrences."""
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    species_counts = Counter()
    
    # Check if this is a processed dataset with 'files' key
    if 'files' in data:
        files = data['files']
        dataset_name = data.get('dataset', 'Unknown')
        
        print(f"Dataset: {dataset_name}")
        print(f"Total files: {len(files)}")
        
        if 'species_counts' in data:
            # Use pre-computed species counts if available
            species_counts = Counter(data['species_counts'])
        else:
            # Count from individual file labels
            for file_info in files:
                if 'class_names' in file_info and file_info['class_names']:
                    for species in file_info['class_names']:
                        species_counts[species] += 1
                elif 'primary_class' in file_info:
                    species_counts[file_info['primary_class']] += 1
    else:
        print("Warning: Unexpected labels.json format")
        return species_counts
    
    return species_counts


def main():
    parser = argparse.ArgumentParser(
        description="Count species occurrences in processed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Count species in AviaNZ processed data
    python count_species.py "Sound Files/GSK_spec"
    
    # Count species in DOC processed data
    python count_species.py "Sound Files/DOC_spec"
        """
    )
    
    parser.add_argument('output_folder', type=str,
                       help="Path to processed output folder containing labels.json")
    parser.add_argument('--reverse', action='store_true',
                       help="Show counts from most to least popular instead")
    
    args = parser.parse_args()
    
    labels_file = os.path.join(args.output_folder, "labels.json")
    
    if not os.path.exists(labels_file):
        print(f"Error: labels.json not found in {args.output_folder}")
        print(f"Expected: {labels_file}")
        return 1
    
    species_counts = count_species_from_labels(labels_file)
    
    if not species_counts:
        print("\nNo species found in dataset!")
        return 1
    
    print(f"\nFound {len(species_counts)} unique species:")
    print("="*60)
    
    # Sort from least to most popular (or reverse if requested)
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=args.reverse)
    
    for species, count in sorted_species:
        print(f"  {species:40s}: {count:5d}")
    
    print("="*60)
    print(f"Total: {sum(species_counts.values())} occurrences across {len(species_counts)} species")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
