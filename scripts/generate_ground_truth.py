"""
Generate ground truth CSV from labels for evaluation.

This script converts the labels.json from processed data into a CSV format
compatible with the evaluation script, marking which species are present
in each audio segment or chunk.

Supports two modes:
1. Segmented data: Each row represents a pre-extracted segment with labels
2. Long audio chunks: Each row represents a 5-second chunk with row_id tracking
"""

import argparse
import os
import json
import pandas as pd
import numpy as np


def generate_ground_truth_csv(labels_json_path, output_csv_path, min_certainty=50):
    """
    Generate ground truth CSV from labels.json.
    
    Automatically detects dataset type and generates appropriate row IDs:
    - For LongAudioInference: Uses row_id from metadata (e.g., "filename_5")
    - For other datasets: Uses filename as row_id
    
    Args:
        labels_json_path: Path to labels.json file
        output_csv_path: Path to output CSV file
        min_certainty: Minimum certainty threshold for labels (default: 50)
    """
    print(f"Loading labels from {labels_json_path}")
    
    with open(labels_json_path, 'r') as f:
        labels_data = json.load(f)
    
    files = labels_data['files']
    categories = labels_data['categories']
    dataset_type = labels_data.get('dataset', 'Unknown')
    
    print(f"Dataset type: {dataset_type}")
    print(f"Found {len(files)} files")
    print(f"Found {len(categories)} species categories")
    
    file_paths = []
    row_ids = []
    ground_truth_matrix = []
    
    for file_info in files:
        filename = file_info['filename']
        
        gt_vector = np.zeros(len(categories))
        for species in file_info.get('class_names', []):
            if species in categories:
                species_idx = categories.index(species)
                gt_vector[species_idx] = 1
        
        source_file = file_info.get('source_file', filename)
        
        if 'row_id' in file_info:
            row_id = file_info['row_id']
        else:
            row_id = filename
        
        file_paths.append(source_file)
        row_ids.append(row_id)
        ground_truth_matrix.append(gt_vector)
    
    ground_truth_matrix = np.array(ground_truth_matrix)
    
    df = pd.DataFrame(ground_truth_matrix, columns=categories)
    df.insert(0, 'File_Path', file_paths)
    df.insert(1, 'row_id', row_ids)
    
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nSaved ground truth to {output_csv_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total species: {len(categories)}")
    print(f"  Total positive labels: {int(ground_truth_matrix.sum())}")
    
    if len(df) > 0:
        print(f"  Average labels per row: {ground_truth_matrix.sum(axis=1).mean():.2f}")
        
        species_counts = ground_truth_matrix.sum(axis=0)
        print(f"\nTop 10 species by occurrence:")
        for i in np.argsort(species_counts)[::-1][:10]:
            print(f"  {categories[i]}: {int(species_counts[i])} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth CSV from processed data labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ground truth from segmented data (AviaNZ, DOC, etc.)
  python generate_ground_truth.py Sound_Files/AviaNZ_spec/labels.json avia_results/ground_truth.csv
  
  # Generate ground truth from long audio chunks (matches Kaytoo format)
  python generate_ground_truth.py Sound_Files/LongAudio_spec/labels.json long_results/ground_truth.csv
  
  # With custom certainty threshold
  python generate_ground_truth.py labels.json gt.csv --min-certainty 75
        """
    )
    
    parser.add_argument('labels_json', type=str,
                       help="Path to labels.json file from data extraction")
    parser.add_argument('output_csv', type=str,
                       help="Path to output ground truth CSV file")
    parser.add_argument('--min-certainty', type=int, default=50,
                       help="Minimum certainty threshold for labels (default: 50)")
    
    args = parser.parse_args()
    
    generate_ground_truth_csv(args.labels_json, args.output_csv, args.min_certainty)


if __name__ == '__main__':
    main()
