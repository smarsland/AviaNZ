#!/usr/bin/env python3
"""
Collect all experiment results and generate summary JSON.

This script scans the output folder for completed experiments
and aggregates their results into all_results.json.

Usage:
    python3 collect_results.py <output_folder>
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Collect all experiment results')
    parser.add_argument('output_folder', help='Folder containing experiment subdirectories')
    args = parser.parse_args()
    
    output_folder = Path(args.output_folder)
    
    if not output_folder.exists():
        print(f"ERROR: Output folder not found: {output_folder}")
        return
    
    # Scan for all result.json files in subdirectories
    all_results = []
    result_files = list(output_folder.glob('*/result.json'))
    
    print(f"Found {len(result_files)} experiment results")
    
    for result_file in sorted(result_files):
        try:
            with open(result_file) as f:
                result = json.load(f)
                all_results.append(result)
                print(f"  ✓ {result_file.parent.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {result_file}: {e}")
    
    # Save aggregated results
    summary_file = output_folder / 'all_results.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f" RESULTS COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f" Total experiments collected: {len(all_results)}")
    print(f" Summary saved to: {summary_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
