"""
Utility to extract taxonomy/species information from BirdClef model checkpoint.
Also helps create species mapping between BirdClef and AviaNZ datasets.
"""

import torch
import json
import pandas as pd
from pathlib import Path


def inspect_birdclef_checkpoint(model_path):
    """
    Inspect a BirdClef model checkpoint to extract species information.
    """
    print(f"Inspecting checkpoint: {model_path}")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            print(f"  - {key}: {type(checkpoint[key])}")
    
    print("\nModel architecture info:")
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Find classifier layer to determine number of classes
    for key in state_dict.keys():
        if 'classifier' in key and 'weight' in key:
            weight_shape = state_dict[key].shape
            print(f"  {key}: {weight_shape}")
            if len(weight_shape) >= 2:
                num_classes = weight_shape[0]
                print(f"  Number of output classes: {num_classes}")
    
    # Try to extract species list
    species_list = None
    if 'taxonomy' in checkpoint:
        species_list = checkpoint['taxonomy']
        print(f"\nFound 'taxonomy' in checkpoint with {len(species_list)} species")
    elif 'species_list' in checkpoint:
        species_list = checkpoint['species_list']
        print(f"\nFound 'species_list' in checkpoint with {len(species_list)} species")
    elif 'class_names' in checkpoint:
        species_list = checkpoint['class_names']
        print(f"\nFound 'class_names' in checkpoint with {len(species_list)} species")
    else:
        print("\nNo species list found in checkpoint!")
        print("You may need to obtain taxonomy.csv from the BirdClef competition")
    
    return species_list, num_classes


def create_birdclef_taxonomy_from_checkpoint(model_path, output_csv='birdclef_taxonomy_extracted.csv'):
    """
    Try to create a taxonomy file from checkpoint if species list is embedded.
    """
    species_list, num_classes = inspect_birdclef_checkpoint(model_path)
    
    if species_list:
        # Create a simple taxonomy dataframe
        df = pd.DataFrame({
            'primary_label': species_list,
            'scientific_name': ['Unknown'] * len(species_list),
            'common_name': ['Unknown'] * len(species_list)
        })
        df.to_csv(output_csv, index=False)
        print(f"\nTaxonomy saved to: {output_csv}")
        return output_csv
    else:
        print("\nCould not extract species list from checkpoint.")
        print("Please download taxonomy.csv from BirdClef 2025 competition:")
        print("https://www.kaggle.com/competitions/birdclef-2025/data")
        return None


def create_species_mapping(birdclef_taxonomy_csv, avianz_mapping_csv='DOC_bird_naming_map.csv', output_json='species_mapping.json'):
    """
    Create mapping between BirdClef species and AviaNZ species.
    Uses eBird codes as the common identifier.
    """
    print(f"\nCreating species mapping...")
    print(f"BirdClef taxonomy: {birdclef_taxonomy_csv}")
    print(f"AviaNZ mapping: {avianz_mapping_csv}")
    
    # Load BirdClef taxonomy
    birdclef_df = pd.read_csv(birdclef_taxonomy_csv)
    birdclef_species = set(birdclef_df['primary_label'].tolist())
    
    # Load AviaNZ species
    avianz_df = pd.read_csv(avianz_mapping_csv)
    avianz_species = set(avianz_df['eBird'].tolist())
    
    # Find common species
    common_species = birdclef_species & avianz_species
    
    print(f"\nBirdClef species: {len(birdclef_species)}")
    print(f"AviaNZ species: {len(avianz_species)}")
    print(f"Common species: {len(common_species)}")
    
    if len(common_species) > 0:
        print(f"\nCommon species examples:")
        for species in sorted(common_species)[:10]:
            # Get common name from AviaNZ
            row = avianz_df[avianz_df['eBird'] == species]
            if not row.empty:
                common_name = row.iloc[0]['CommonName']
                print(f"  {species}: {common_name}")
    
    # Create mapping dictionary
    mapping = {
        'birdclef_species': sorted(birdclef_species),
        'avianz_species': sorted(avianz_species),
        'common_species': sorted(common_species),
        'species_common_names': {}
    }
    
    # Add common names for common species
    for species in common_species:
        row = avianz_df[avianz_df['eBird'] == species]
        if not row.empty:
            mapping['species_common_names'][species] = row.iloc[0]['CommonName']
    
    # Save mapping
    with open(output_json, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nMapping saved to: {output_json}")
    
    # Warn about unmapped species
    avianz_only = avianz_species - birdclef_species
    if len(avianz_only) > 0:
        print(f"\n⚠️  {len(avianz_only)} AviaNZ species not in BirdClef model:")
        for species in sorted(avianz_only)[:5]:
            row = avianz_df[avianz_df['eBird'] == species]
            if not row.empty:
                print(f"  {species}: {row.iloc[0]['CommonName']}")
        if len(avianz_only) > 5:
            print(f"  ... and {len(avianz_only) - 5} more")
    
    return mapping


def main():
    """Main utility function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BirdClef taxonomy and mapping utility')
    parser.add_argument('--model', default='BirdClefModels/model_fold0.pth', help='Path to BirdClef model checkpoint')
    parser.add_argument('--taxonomy', default=None, help='Path to BirdClef taxonomy.csv (from competition)')
    parser.add_argument('--avianz-map', default='DOC_bird_naming_map.csv', help='Path to AviaNZ species mapping')
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect checkpoint, don\'t create mapping')
    
    args = parser.parse_args()
    
    print("BirdClef Taxonomy & Mapping Utility")
    print("=" * 60)
    
    # Step 1: Inspect model checkpoint
    species_list, num_classes = inspect_birdclef_checkpoint(args.model)
    
    if args.inspect_only:
        return
    
    # Step 2: Get or create taxonomy
    taxonomy_csv = args.taxonomy
    if not taxonomy_csv:
        print("\nAttempting to extract taxonomy from checkpoint...")
        taxonomy_csv = create_birdclef_taxonomy_from_checkpoint(args.model)
    
    if not taxonomy_csv:
        print("\n❌ Cannot proceed without taxonomy information")
        print("\nOptions:")
        print("1. Download taxonomy.csv from BirdClef 2025:")
        print("   https://www.kaggle.com/competitions/birdclef-2025/data")
        print("2. If your checkpoint has embedded species list, contact the model provider")
        return
    
    # Step 3: Create species mapping
    if Path(args.avianz_map).exists():
        mapping = create_species_mapping(taxonomy_csv, args.avianz_map)
        
        if len(mapping['common_species']) == 0:
            print("\n⚠️  WARNING: No common species found!")
            print("This means the BirdClef model won't be useful for your dataset.")
            print("Possible issues:")
            print("1. Different eBird code versions")
            print("2. Regional differences (BirdClef global vs AviaNZ NZ-focused)")
            print("3. Taxonomy updates")
    else:
        print(f"\n❌ AviaNZ mapping file not found: {args.avianz_map}")


if __name__ == "__main__":
    main()
