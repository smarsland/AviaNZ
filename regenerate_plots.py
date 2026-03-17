#!/usr/bin/env python3
"""
Regenerate plots from existing experiment results.
Usage: python3 regenerate_plots.py experiments_20260313_064819 [--ignore-dann]
"""

import json
import sys
import argparse
from pathlib import Path

# Import the experiment class
from run_cross_dataset_experiments import CrossDatasetExperiments

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate plots from existing experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('results_folder', help='Path to results folder')
    parser.add_argument('--ignore-dann', action='store_true',
                       help='Exclude DANN experiments from plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_folder)
    
    if not results_dir.exists():
        print(f"ERROR: Results folder not found: {results_dir}")
        sys.exit(1)
    
    results_file = results_dir / 'all_results.json'
    if not results_file.exists():
        print(f"ERROR: all_results.json not found in {results_dir}")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data['results'])} experiments")
    
    # Filter out DANN experiments if requested
    results = data['results']
    if args.ignore_dann:
        results = [r for r in results if not r['name'].startswith('dann')]
        print(f"Filtering DANN experiments: {len(results)} experiments remaining")
    
    # Create experiment object (paths don't need to exist for plotting)
    exp = CrossDatasetExperiments(
        avianz_train='/dummy/path',
        avianz_test='/dummy/path', 
        doc_train='/dummy/path',
        doc_test='/dummy/path',
        output_folder=str(results_dir),
        model_path=data.get('model', 'N/A'),
        epochs=data.get('epochs', 0),
        batch_size=data.get('batch_size', 0)
    )
    
    # Load the filtered results
    exp.results = results
    
    print("\nRegenerating visualizations...")
    print("=" * 60)
    
    try:
        exp.generate_summary_table()
        exp.plot_test_accuracy_comparison()
        exp.plot_heatmap()
        exp.plot_validation_vs_test()
        
        print("\n" + "=" * 60)
        print("✓ All plots successfully regenerated!")
        print("=" * 60)
        print(f"\nFiles in {results_dir}:")
        print("  - summary_table.csv")
        print("  - summary_table.txt")
        print("  - test_accuracy_comparison.png")
        print("  - results_heatmap.png")
        print("  - validation_vs_test.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
