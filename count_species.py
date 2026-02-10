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
import pandas as pd
from collections import Counter


def load_bird_name_mapping(csv_path):
    """Load the DOC bird naming map CSV and create mapping to eBird codes"""
    df = pd.read_csv(csv_path)
    
    name_to_ebird = {}
    
    for _, row in df.iterrows():
        ebird_code = row['eBird']
        
        # Skip rows without eBird code
        if pd.isna(ebird_code):
            continue
        
        # Map all name variants to eBird code
        if pd.notna(row['CommonName']):
            name_to_ebird[row['CommonName']] = ebird_code
        if pd.notna(row['ExtraName']):
            name_to_ebird[row['ExtraName']] = ebird_code
        if pd.notna(row['ListDOCBirds']):
            name_to_ebird[row['ListDOCBirds']] = ebird_code
        if pd.notna(row['ScientificName']):
            name_to_ebird[row['ScientificName']] = ebird_code
        
        # Map eBird code to itself
        name_to_ebird[ebird_code] = ebird_code
    
    # Add hardcoded fixes for common variants
    name_to_ebird['Ruru'] = 'morepo2'
    name_to_ebird['Bellbird/Tui'] = 'nezbel1'
    name_to_ebird['Tomtit (Nth Is)'] = 'tomtit1'
    name_to_ebird['Fantail (Nth Is)'] = 'nezfan1'
    name_to_ebird['Fantail (spp)'] = 'nezfan1'
    name_to_ebird['Kaka (Nth Is)'] = 'nezkak1'
    name_to_ebird['Kaka (spp)'] = 'nezkak1'
    name_to_ebird['Tui (spp)'] = 'tui1'
    name_to_ebird['Robin (Nth Is)'] = 'nezrob2'
    name_to_ebird['Pigeon (NZ Kereru Kukupa)'] = 'nezpig2'
    name_to_ebird['Warbler (Grey)'] = 'gryger1'
    name_to_ebird['Magpie (Australian)'] = 'ausmag2'
    name_to_ebird['Myna (Indian)'] = 'commyn'
    name_to_ebird['Gull (Southern Black-backed)'] = 'kelgul'
    name_to_ebird['Plover (Spur-winged)'] = 'maslap1'
    name_to_ebird['Rosella (Eastern)'] = 'easros1'
    name_to_ebird['Cockatoo (Sulphur-crested)'] = 'succoc'
    name_to_ebird['Sparrow (House)'] = 'houspa'
    
    return name_to_ebird


def normalize_to_ebird(name, name_mapping):
    """Normalize any species name to eBird code using the mapping"""
    if name in ['Empty Sample', 'Tree Weta', 'Spy Bird', None, '']:
        return None
    
    if name in name_mapping:
        return name_mapping[name]
    
    # Try case-insensitive match
    name_lower = name.lower()
    for key, value in name_mapping.items():
        if key.lower() == name_lower:
            return value
    
    # Try matching base name (without parentheses)
    if '(' in name:
        base_name = name.split('(')[0].strip()
        if base_name in name_mapping:
            return name_mapping[base_name]
    
    # If no mapping found, return original name
    return name


def count_species_from_labels(labels_file, name_mapping=None):
    """Load labels.json and count species occurrences."""
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    species_counts = Counter()
    unmapped_species = set()
    
    # Per-species stats about single vs multi-label
    species_single_label = Counter()
    species_multi_label = Counter()
    
    # Check if this is a processed dataset with 'files' key
    if 'files' in data:
        files = data['files']
        dataset_name = data.get('dataset', 'Unknown')
        
        print(f"Dataset: {dataset_name}")
        print(f"Total files: {len(files)}")
        
        if 'species_counts' in data and not name_mapping:
            # Use pre-computed species counts if available and no mapping requested
            species_counts = Counter(data['species_counts'])
        else:
            # Count from individual file labels
            for file_info in files:
                species_list = []
                if 'class_names' in file_info and file_info['class_names']:
                    species_list = file_info['class_names']
                elif 'primary_class' in file_info:
                    species_list = [file_info['primary_class']]
                
                is_single_label = (len(species_list) == 1)
                
                # Map species names if mapping provided
                for species in species_list:
                    if name_mapping:
                        mapped_species = normalize_to_ebird(species, name_mapping)
                        if mapped_species:
                            species_counts[mapped_species] += 1
                            if is_single_label:
                                species_single_label[mapped_species] += 1
                            else:
                                species_multi_label[mapped_species] += 1
                        else:
                            unmapped_species.add(species)
                    else:
                        species_counts[species] += 1
                        if is_single_label:
                            species_single_label[species] += 1
                        else:
                            species_multi_label[species] += 1
    else:
        print("Warning: Unexpected labels.json format")
        return species_counts, species_single_label, species_multi_label
    
    if name_mapping and unmapped_species:
        real_unmapped = unmapped_species - {'Empty Sample', 'Tree Weta', 'Spy Bird'}
        if real_unmapped:
            print(f"\nWarning: Could not map {len(real_unmapped)} species to eBird codes:")
            for species in sorted(real_unmapped):
                print(f"  - {species}")
    
    return species_counts, species_single_label, species_multi_label


def main():
    parser = argparse.ArgumentParser(
        description="Count species occurrences in processed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Count species in AviaNZ processed data (original names)
    python count_species.py "Sound Files/GSK_spec"
    
    # Count species mapped to eBird codes
    python count_species.py "Sound Files/GSK_spec" --ebird
    
    # Count species in DOC processed data with custom mapping file
    python count_species.py "Sound Files/DOC_spec" --ebird --mapping "custom_map.csv"
        """
    )
    
    parser.add_argument('output_folder', type=str,
                       help="Path to processed output folder containing labels.json")
    parser.add_argument('--reverse', action='store_true',
                       help="Show counts from most to least popular instead")
    parser.add_argument('--ebird', action='store_true',
                       help="Map all species names to eBird codes")
    parser.add_argument('--mapping', type=str,
                       help="Path to bird name mapping CSV (default: DOC_bird_naming_map.csv)")
    
    args = parser.parse_args()
    
    labels_file = os.path.join(args.output_folder, "labels.json")
    
    if not os.path.exists(labels_file):
        print(f"Error: labels.json not found in {args.output_folder}")
        print(f"Expected: {labels_file}")
        return 1
    
    # Load name mapping if requested
    name_mapping = None
    if args.ebird:
        mapping_file = args.mapping
        if mapping_file is None:
            # Try to find default mapping file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(script_dir, "DOC_bird_naming_map.csv")
        
        if not os.path.exists(mapping_file):
            print(f"Error: Mapping file not found: {mapping_file}")
            return 1
        
        print(f"Loading bird name mapping from {os.path.basename(mapping_file)}...")
        name_mapping = load_bird_name_mapping(mapping_file)
        print(f"Loaded {len(name_mapping)} name mappings\n")
    
    species_counts = count_species_from_labels(labels_file, name_mapping)
    
    if not species_counts:
        print("\nNo species found in dataset!")
        return 1
    
    # Unpack the results
    if isinstance(species_counts, tuple):
        species_counts, species_single_label, species_multi_label = species_counts
    else:
        species_single_label = {}
        species_multi_label = {}
    
    label_type = "eBird codes" if args.ebird else "species"
    print(f"\nFound {len(species_counts)} unique {label_type}:")
    print("="*60)
    
    # Sort from least to most popular (or reverse if requested)
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=args.reverse)
    
    for species, count in sorted_species:
        single = species_single_label.get(species, 0)
        multi = species_multi_label.get(species, 0)
        if single + multi > 0:
            single_pct = 100 * single / (single + multi)
            print(f"  {species:30s}: {count:5d}  ({single:4d} single-label {single_pct:5.1f}%, {multi:4d} multi-label)")
        else:
            print(f"  {species:30s}: {count:5d}")
    
    print("="*60)
    print(f"Total: {sum(species_counts.values())} occurrences across {len(species_counts)} {label_type}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
