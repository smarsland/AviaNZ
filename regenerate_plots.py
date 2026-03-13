#!/usr/bin/env python3
"""
Regenerate plots from existing experiment results.
Usage: python3 regenerate_plots.py experiments_20260313_064819
"""

import json
import sys
from pathlib import Path

# Import the experiment class
from run_cross_dataset_experiments import CrossDatasetExperiments

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 regenerate_plots.py <results_folder>")
        print("Example: python3 regenerate_plots.py experiments_20260313_064819")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
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
    
    # Create experiment object (paths don't need to exist for plotting)
    exp = CrossDatasetExperiments(
        avianz_train='/dummy/path',
        avianz_test='/dummy/path', 
        doc_train='/dummy/path',
        doc_test='/dummy/path',
        combined_train='/dummy/path',
        output_folder=str(results_dir),
        model_path=data.get('model', 'N/A'),
        epochs=data.get('epochs', 0),
        batch_size=data.get('batch_size', 0)
    )
    
    # Load the results
    exp.results = data['results']
    
    print("\nRegenerating visualizations...")
    print("=" * 60)
    
    try:
        exp.generate_summary_table()
        exp.plot_heatmap()
        exp.plot_test_accuracy_comparison()
        exp.plot_freeze_comparison()
        exp.plot_generalization_gap()
        exp.plot_training_curves()
        
        print("\n" + "=" * 60)
        print("✓ All plots successfully regenerated!")
        print("=" * 60)
        print(f"\nFiles in {results_dir}:")
        print("  - summary_table.csv")
        print("  - summary_table.txt")
        print("  - heatmap.png")
        print("  - test_accuracy_comparison.png")
        print("  - freeze_comparison.png")
        print("  - generalization_gap.png")
        print("  - training_curves.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
