#!/usr/bin/env python3
"""
Add DANN experiment results to all_results.json for visualization.
Usage: python3 add_dann_to_results.py experiments_20260313_064819 /local/scratch/freangi/joe_mo_split/test /local/scratch/freangi/doc_split/test
"""

import json
import sys
import os
from pathlib import Path


def extract_accuracy_from_output(dann_folder, test_folder, csv_pattern):
    """Extract accuracy from DANN prediction CSV."""
    import pandas as pd
    
    # Find the CSV file
    csv_files = list(Path(dann_folder).glob(csv_pattern))
    if not csv_files:
        print(f"    WARNING: No CSV matching {csv_pattern} found")
        return 0.0
    
    csv_path = csv_files[0]
    
    # Load ground truth
    labels_path = Path(test_folder) / 'labels.json'
    if not labels_path.exists():
        print(f"    WARNING: {labels_path} not found")
        return 0.0
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    true_labels = {}
    for item in labels_data['files']:
        true_labels[item['filename']] = item.get('primary_class') or item.get('primary_species')
    
    # Load predictions
    df = pd.read_csv(csv_path)
    class_columns = [col for col in df.columns if col not in ['row_id', 'File_Path']]
    
    correct = 0
    total = 0
    
    for _, row in df.iterrows():
        filename = row['row_id']
        if filename not in true_labels:
            continue
        
        pred_class = class_columns[row[class_columns].values.argmax()]
        true_class = true_labels[filename]
        
        if pred_class == true_class:
            correct += 1
        total += 1
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"    Found accuracy: {accuracy:.2f}%")
    return accuracy


def load_dann_result(dann_folder, name, description, train_dataset, freeze, test_folder1, test_folder2):
    """Load DANN results and convert to standard format."""
    dann_folder = Path(dann_folder)
    
    if not dann_folder.exists():
        print(f"  WARNING: {dann_folder} not found, skipping")
        return None
    
    history_path = dann_folder / 'training_history.json'
    if not history_path.exists():
        print(f"  WARNING: {history_path} not found, skipping")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Extract test accuracies from prediction CSVs
    test1_name = "joe_mo_split"
    test2_name = "doc_split"
    
    print(f"  Extracting test accuracies...")
    test1_acc = extract_accuracy_from_output(dann_folder, test_folder1, 'predictions_joe_mo_split_test.csv')
    test2_acc = extract_accuracy_from_output(dann_folder, test_folder2, 'predictions_doc_split_test.csv')
    
    # Convert DANN history format to standard format
    # DANN uses: train_class_acc, source_val_acc, target_val_acc
    # Standard uses: train_acc, val_acc
    
    # Use target validation accuracy as the "val_acc" since that's what we care about
    val_acc = history.get('target_val_acc', [0] * len(history.get('train_class_acc', [])))
    
    result = {
        'name': name,
        'description': description,
        'train_dataset': train_dataset,
        'freeze_backbone': freeze,
        'final_train_acc': history['train_class_acc'][-1] if history.get('train_class_acc') else 0,
        'final_val_acc': val_acc[-1] if val_acc else None,
        'test1_name': test1_name,
        'test1_acc': test1_acc,
        'test2_name': test2_name,
        'test2_acc': test2_acc,
        'best_val_acc': max(val_acc) if val_acc else None,
        'history': {
            'train_acc': history.get('train_class_acc', []),
            'val_acc': val_acc,
            'train_loss': history.get('train_class_loss', []),
            'val_loss': [],
            'train_macro_f1': [],
            'val_macro_f1': [],
            'train_micro_f1': [],
            'val_micro_f1': [],
            'train_exact_match': [],
            'val_exact_match': [],
            'train_bit_acc': [],
            'val_bit_acc': []
        },
        'output_folder': str(dann_folder)
    }
    
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 add_dann_to_results.py <results_folder> <test_folder1> <test_folder2>")
        print("Example: python3 add_dann_to_results.py experiments_20260313_064819 /local/scratch/freangi/joe_mo_split/test /local/scratch/freangi/doc_split/test")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    test_folder1 = sys.argv[2]
    test_folder2 = sys.argv[3]
    
    if not results_dir.exists():
        print(f"ERROR: Results folder not found: {results_dir}")
        sys.exit(1)
    
    results_file = results_dir / 'all_results.json'
    if not results_file.exists():
        print(f"ERROR: all_results.json not found in {results_dir}")
        sys.exit(1)
    
    print(f"Loading existing results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data['results'])} existing experiments")
    
    # Define DANN experiments to add
    dann_experiments = [
        {
            'folder': results_dir / 'dann_combined_full',
            'name': 'dann_combined_full',
            'description': 'DANN on Combined (full fine-tuning)',
            'train_dataset': 'combined_train',
            'freeze': False
        },
        {
            'folder': results_dir / 'dann_combined_frozen',
            'name': 'dann_combined_frozen',
            'description': 'DANN on Combined (frozen backbone)',
            'train_dataset': 'combined_train',
            'freeze': True
        }
    ]
    
    print("\nLooking for DANN results...")
    added = 0
    
    for dann_exp in dann_experiments:
        print(f"\nProcessing: {dann_exp['name']}")
        result = load_dann_result(
            dann_exp['folder'],
            dann_exp['name'],
            dann_exp['description'],
            dann_exp['train_dataset'],
            dann_exp['freeze'],
            test_folder1,
            test_folder2
        )
        
        if result:
            # Check if already exists
            existing_names = [r['name'] for r in data['results']]
            if result['name'] in existing_names:
                print(f"  Updating existing entry")
                idx = existing_names.index(result['name'])
                data['results'][idx] = result
            else:
                print(f"  Adding new entry")
                data['results'].append(result)
            added += 1
    
    if added > 0:
        # Update experiment count
        data['experiments'] = len(data['results'])
        
        # Save updated results
        backup_file = results_file.with_suffix('.json.backup')
        print(f"\nBacking up original to: {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saving updated results to: {results_file}")
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Successfully added/updated {added} DANN experiments")
        print(f"Total experiments: {len(data['results'])}")
        print(f"\nNow run: python3 regenerate_plots.py {results_dir}")
    else:
        print("\n⚠ No DANN results found to add")
        print("Make sure you've run the DANN experiments first")


if __name__ == '__main__':
    main()
