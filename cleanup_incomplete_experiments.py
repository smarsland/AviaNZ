#!/usr/bin/env python3
"""
Cleanup script for incomplete experiments.

Problem: If experiments halt mid-run, some experiment folders are created
         but don't have a result.json file. This prevents re-running since
         the script thinks those experiments already exist.

Solution: Delete experiment folders without result.json from both the main
          experiments directory AND the shared results directory.

Usage:
    python3 cleanup_incomplete_experiments.py [EXPERIMENTS_DIR] [RESULTS_DIR]

Defaults:
    EXPERIMENTS_DIR: /local/scratch/$USER/experiments_matched
    RESULTS_DIR: ~/results
"""

import os
import sys
import shutil
from pathlib import Path


def main():
    # Default paths (matches run_matched_experiments.sh)
    user = os.environ.get('USER', 'unknown')
    default_experiments_dir = f"/local/scratch/{user}/experiments_matched"
    default_results_dir = str(Path.home() / "results")
    
    # Parse arguments
    experiments_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_experiments_dir)
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(default_results_dir)
    
    # Check if experiments directory exists
    if not experiments_dir.exists():
        print(f"Error: Experiments directory does not exist: {experiments_dir}")
        print()
        print("Usage: python3 cleanup_incomplete_experiments.py [EXPERIMENTS_DIR] [RESULTS_DIR]")
        print("Defaults:")
        print(f"  EXPERIMENTS_DIR: {default_experiments_dir}")
        print(f"  RESULTS_DIR: {default_results_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print(" Cleanup Incomplete Experiments")
    print("=" * 70)
    print(f"  Experiments dir: {experiments_dir}")
    print(f"  Results dir    : {results_dir}")
    print("=" * 70)
    print()
    
    # Remove main all_results.json if it exists (will be regenerated)
    all_results_exp = experiments_dir / "all_results.json"
    if all_results_exp.exists():
        print(f"🗑️  Removing experiments all_results.json: {all_results_exp}")
        all_results_exp.unlink()
    
    if results_dir.exists():
        all_results_res = results_dir / "all_results.json"
        if all_results_res.exists():
            print(f"🗑️  Removing results all_results.json: {all_results_res}")
            all_results_res.unlink()
    
    print()
    
    # Count complete and incomplete experiments
    incomplete = []
    complete = []
    
    # Collect all experiment names from both directories
    exp_names = set()
    
    if experiments_dir.exists():
        exp_names.update(d.name for d in experiments_dir.iterdir() if d.is_dir())
    
    if results_dir.exists():
        exp_names.update(d.name for d in results_dir.iterdir() if d.is_dir())
    
    # Find incomplete experiments (missing result.json in shared results)
    print("Scanning for incomplete experiments...")
    print("(Source of truth: shared results directory)")
    print()
    
    for exp_name in sorted(exp_names):
        exp_dir = experiments_dir / exp_name
        results_exp_dir = results_dir / exp_name if results_dir.exists() else None
        
        # Check if result.json exists in shared results directory (source of truth)
        has_result = results_exp_dir is not None and (results_exp_dir / "result.json").exists()
        
        if not has_result:
            print(f"❌ INCOMPLETE: {exp_name}")
            
            # Remove from experiments directory
            if exp_dir.exists():
                print(f"   → Removing from experiments: {exp_dir}")
                shutil.rmtree(exp_dir)
            
            # Remove from shared results directory
            if results_exp_dir is not None and results_exp_dir.exists():
                print(f"   → Removing from results: {results_exp_dir}")
                shutil.rmtree(results_exp_dir)
            
            incomplete.append(exp_name)
        else:
            print(f"✓ Complete: {exp_name}")
            complete.append(exp_name)
    
    print()
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"  Complete experiments   : {len(complete)} (kept)")
    print(f"  Incomplete experiments : {len(incomplete)} (removed from both locations)")
    print("=" * 70)
    print()
    
    if incomplete:
        print(f"✅ Cleaned up {len(incomplete)} incomplete experiment(s)")
        print("   Removed from:")
        print(f"   - Experiments dir: {experiments_dir}")
        print(f"   - Results dir: {results_dir}")
        print()
        print("   You can now rerun run_matched_experiments.sh")
    else:
        print("✅ No cleanup needed - all experiments are complete!")
    
    print()


if __name__ == '__main__':
    main()
