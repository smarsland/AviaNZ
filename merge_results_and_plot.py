#!/usr/bin/env python3
"""
Quick script to merge all completed experiment results and regenerate plots.
"""

import json
from pathlib import Path
import sys

def extract_accuracy(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        acc = float(value)
        if acc < 1.0:
            acc = acc * 100.0
        return acc
    if isinstance(value, (tuple, list)):
        return extract_accuracy(value[0]) if len(value) > 0 else 0.0
    if isinstance(value, dict):
        return extract_accuracy(value.get('macro_f1', value.get('accuracy', 0.0)))
    return 0.0

def load_experiment_result(exp_dir):
    """Load result from a completed experiment directory."""
    history_path = exp_dir / 'training_history.json'
    
    if not history_path.exists():
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Extract test names from prediction files
        pred_files = list(exp_dir.glob('predictions_*.csv'))
        if len(pred_files) < 2:
            return None
        
        test_names = []
        test_accs = []
        for pred_file in sorted(pred_files):
            # Extract test name from filename: predictions_<test_name>.csv
            test_name = pred_file.stem.replace('predictions_', '')
            test_names.append(test_name)
            
            # Read first line to get accuracy from header (if present)
            with open(pred_file, 'r') as f:
                lines = f.readlines()
                # Try to find accuracy in file or calculate from predictions
                # For now, we'll extract from ground_truth.json if available
                test_accs.append(0.0)  # Placeholder
        
        # Check for ground_truth.json files to get actual test accuracy
        for i, test_name in enumerate(test_names):
            gt_file = exp_dir / f'ground_truth_{test_name}.json'
            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                    if 'accuracy' in gt_data:
                        test_accs[i] = extract_accuracy(gt_data['accuracy'])
                    elif 'exact_match' in gt_data:
                        test_accs[i] = extract_accuracy(gt_data['exact_match'])
        
        # If still no test accuracies, try to extract from CSV
        if all(a == 0.0 for a in test_accs):
            for i, pred_file in enumerate(sorted(pred_files)):
                # Count matches in CSV
                with open(pred_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    total = len(lines)
                    correct = 0
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            pred = parts[1].strip()
                            gt = parts[2].strip()
                            if pred == gt:
                                correct += 1
                    if total > 0:
                        test_accs[i] = (correct / total) * 100.0
        
        train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        final_train_acc = extract_accuracy(history[train_acc_key][-1] if history.get(train_acc_key) else None)
        final_val_acc = extract_accuracy(history[val_acc_key][-1] if history.get(val_acc_key) else None)
        best_val_acc = max([extract_accuracy(v) for v in history.get(val_acc_key, [0.0])]) if history.get(val_acc_key) else 0.0
        
        # Determine description and model type from directory name
        exp_name = exp_dir.name
        if 'joe_mo' in exp_name:
            base_desc = 'Baseline joe_mo'
        elif 'doc' in exp_name:
            base_desc = 'Baseline doc'
        else:
            base_desc = exp_name
        
        desc = f"{base_desc} (BIRDCLEF)"
        if 'pcen' in exp_name.lower():
            desc += " [PCEN]"
        if 'normalized' in exp_name.lower():
            desc += " [normalized]"
        
        result = {
            'name': exp_name,
            'description': desc,
            'model_type': 'birdclef',
            'train_dataset': 'train',
            'freeze_backbone': False,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'test1_name': test_names[0] if len(test_names) > 0 else 'unknown',
            'test1_acc': test_accs[0] if len(test_accs) > 0 else 0.0,
            'test2_name': test_names[1] if len(test_names) > 1 else 'unknown',
            'test2_acc': test_accs[1] if len(test_accs) > 1 else 0.0,
            'best_val_acc': best_val_acc,
            'history': history,
            'output_folder': str(exp_dir.absolute())
        }
        
        return result
        
    except Exception as e:
        print(f"Error loading {exp_dir}: {e}")
        return None

def main():
    output_folder = Path('experiments_matched')
    
    if not output_folder.exists():
        print(f"Error: {output_folder} does not exist")
        sys.exit(1)
    
    print("Loading all experiment results...")
    
    results = []
    for exp_dir in sorted(output_folder.iterdir()):
        if exp_dir.is_dir() and 'baseline' in exp_dir.name:
            print(f"  Loading: {exp_dir.name}")
            result = load_experiment_result(exp_dir)
            if result:
                results.append(result)
                print(f"    Test1: {result['test1_acc']:.2f}%, Test2: {result['test2_acc']:.2f}%")
    
    print(f"\nLoaded {len(results)} experiments")
    
    # Save merged results
    results_dict = {
        'timestamp': '2026-03-26T10:00:00',
        'experiments': len(results),
        'epochs': 10,
        'batch_size': 16,
        'model': 'BirdClefModels/model_fold0.pth',
        'results': results
    }
    
    json_path = output_folder / 'all_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nSaved merged results to: {json_path}")
    
    # Now run the plotting by importing and using the CrossDatasetExperiments class
    print("\nRegenerating plots...")
    
    import run_cross_dataset_experiments
    import numpy as np
    
    # Create a minimal instance just for plotting
    exp = run_cross_dataset_experiments.CrossDatasetExperiments(
        avianz_train='dummy',
        avianz_test='dummy',
        doc_train='dummy',
        doc_test='dummy',
        output_folder=output_folder,
        model_path='BirdClefModels/model_fold0.pth',
        epochs=10,
        batch_size=16
    )
    
    # Load all results
    exp.results = results
    
    # Generate plots
    try:
        exp.generate_summary_table()
        exp.plot_test_accuracy_comparison()
        exp.plot_heatmap()
        exp.plot_validation_vs_test()
        exp.generate_report()
        print("\n✓ Plots regenerated successfully!")
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
