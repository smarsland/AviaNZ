#!/usr/bin/env python3
"""
Fix PCEN/Box-Cox configs and regenerate predictions on remote machine.
Run this script in /local/scratch/freangi/ on the remote machine.

Usage:
    scp fix_remote_predictions.py freangi@entry.ecs.vuw.ac.nz:~/
    ssh freangi@entry.ecs.vuw.ac.nz
    cd /local/scratch/freangi
    python3 ~/fix_remote_predictions.py
"""

import json
import subprocess
import sys
from pathlib import Path

# Configuration
EXPERIMENTS_BASE = Path('/local/scratch/freangi/experiments_matched')
AVIANZ_TEST = '/local/scratch/freangi/matched/avianz_split/test'
DOC_TEST = '/local/scratch/freangi/matched/doc_split/test'
AVIANZ_CODE = Path.home() / 'AviaNZ'  # Assumes code is in ~/AviaNZ

# Only fix PCEN and Box-Cox (Log experiments are fine since Log is default)
EXPERIMENTS_TO_FIX = {
    'joe_mo_baseline_birdclef_pcen': 'PCEN',
    'doc_baseline_birdclef_pcen': 'PCEN',
    'joe_mo_baseline_birdclef_box-cox': 'Box-Cox',
    'doc_baseline_birdclef_box-cox': 'Box-Cox',
}

def fix_config(config_path, spec_transform):
    """Add spec_transform to config."""
    with open(config_path) as f:
        cfg = json.load(f)
    cfg['spec_transform'] = spec_transform
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  ✓ Fixed config: spec_transform={spec_transform}")

def regenerate_predictions(exp_dir, model_name, config_name):
    """Regenerate test predictions."""
    model_path = exp_dir / model_name
    config_path = exp_dir / config_name
    
    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return False
    
    # Regenerate avianz test
    csv1 = exp_dir / 'predictions_avianz_split_test.csv'
    print(f"  Regenerating avianz_split/test...")
    result = subprocess.run([
        sys.executable,
        str(AVIANZ_CODE / 'predict.py'),
        str(model_path),
        str(config_path),
        AVIANZ_TEST,
        str(csv1)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr[:200]}")
        return False
    
    # Regenerate doc test
    csv2 = exp_dir / 'predictions_doc_split_test.csv'
    print(f"  Regenerating doc_split/test...")
    result = subprocess.run([
        sys.executable,
        str(AVIANZ_CODE / 'predict.py'),
        str(model_path),
        str(config_path),
        DOC_TEST,
        str(csv2)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr[:200]}")
        return False
    
    print(f"  ✓ Predictions regenerated")
    return True

def main():
    print("="*70)
    print("Fixing PCEN/Box-Cox experiments")
    print("="*70)
    print(f"Experiments base: {EXPERIMENTS_BASE}")
    print(f"AviaNZ code: {AVIANZ_CODE}")
    print()
    
    if not EXPERIMENTS_BASE.exists():
        print(f"ERROR: {EXPERIMENTS_BASE} not found")
        print("Make sure you're running this on the remote machine in the right directory")
        return 1
    
    if not AVIANZ_CODE.exists():
        print(f"ERROR: {AVIANZ_CODE} not found")
        print("Make sure AviaNZ code is in ~/AviaNZ")
        return 1
    
    success_count = 0
    for exp_name, spec_transform in EXPERIMENTS_TO_FIX.items():
        print(f"\n{exp_name}:")
        exp_dir = EXPERIMENTS_BASE / exp_name
        
        if not exp_dir.exists():
            print(f"  ⚠️  Experiment folder not found, skipping")
            continue
        
        # Fix both best and final configs
        for config_file in ['birdclef_finetuned_best_config.json', 'birdclef_finetuned_final_config.json']:
            config_path = exp_dir / config_file
            if config_path.exists():
                fix_config(config_path, spec_transform)
        
        # Regenerate predictions using best model
        if regenerate_predictions(exp_dir, 'birdclef_finetuned_best.pt', 'birdclef_finetuned_best_config.json'):
            success_count += 1
    
    print()
    print("="*70)
    print(f"✓ Complete! Fixed {success_count}/{len(EXPERIMENTS_TO_FIX)} experiments")
    print()
    print("Next step: Regenerate summary on your local machine:")
    print("  rsync -avz freangi@entry.ecs.vuw.ac.nz:/local/scratch/freangi/experiments_matched/ experiments_matched/")
    print("  python run_cross_dataset_experiments.py --avianz-train joe_mo_split/train --avianz-test joe_mo_split/test --doc-train doc_split/train --doc-test doc_split/test --output experiments_matched")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
