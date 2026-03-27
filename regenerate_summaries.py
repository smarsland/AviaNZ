#!/usr/bin/env python3
"""
Regenerate summary files from existing experiment results.
This is useful after updating the reporting format without rerunning experiments.
"""

import sys
import json
from pathlib import Path
from run_cross_dataset_experiments import CrossDatasetExperiments

def main():
    # Setup for existing experiments_matched folder
    output_base = Path("experiments_matched")
    
    if not output_base.exists():
        print(f"Error: {output_base} not found")
        sys.exit(1)
    
    # Load existing all_results.json
    results_json = output_base / "all_results.json"
    if not results_json.exists():
        print(f"Error: {results_json} not found")
        sys.exit(1)
    
    print(f"Loading results from {results_json}...")
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    # Create experiment runner (just for accessing the summary generation methods)
    # Use dummy paths since we're only regenerating summaries
    runner = CrossDatasetExperiments(
        avianz_train="dummy",
        avianz_test="dummy",
        doc_train="dummy",
        doc_test="dummy",
        output_folder=output_base,
        model_path="BirdClefModels/model_fold0.pth",
        epochs=data['epochs'],
        batch_size=data['batch_size']
    )
    
    # Load the results
    runner.results = data['results']
    
    print(f"Loaded {len(runner.results)} experiment results")
    
    # Regenerate summaries
    print("\nRegenerating summary table...")
    runner.generate_summary_table()
    
    print("\nRegenerating report...")
    runner.generate_report()
    
    print("\n✓ Summaries regenerated successfully!")
    print(f"  - summary_table.csv")
    print(f"  - summary_table.txt")
    print(f"  - report.txt")

if __name__ == '__main__':
    main()
