#!/usr/bin/env python3
"""
Clean cross-dataset experiment pipeline.

Runs four separate experiment suites:
1. Normalization comparison (6 methods × 2 directions = 12 experiments)
2. DANN domain adaptation (2 directions = 2 experiments)
3. Noise intensity sweep (5 levels × 2 directions = 10 experiments)
4. Noise variety sweep (5 levels × 2 directions = 10 experiments)

Total: ~34 experiments

No flags. No skip logic. Just clean experiment loops.
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

import config


def run_experiment(config_dict):
    """
    Run a single training + evaluation experiment.
    
    Returns: dict with results
    """
    name = config_dict['name']
    output_dir = Path(config_dict['output_folder']) / name
    
    # Check if already complete (has result.json)
    result_file = output_dir / 'result.json'
    if result_file.exists():
        print(f"✓ {name} - already complete (loading cached result)")
        with open(result_file) as f:
            return json.load(f)
    
    # Check if training complete but evaluation missing
    model_file = output_dir / 'birdclef_finetuned_best.pt'
    history_file = output_dir / 'training_history.json'
    eval_only = model_file.exists() and history_file.exists()
    
    if eval_only:
        print(f"\n{'='*70}")
        print(f"Running: {name} (EVAL ONLY - model exists)")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
    
    # Build command
    if config_dict['type'] == 'baseline':
        cmd = [
            'python3', 'finetune_birdclef.py',
            config_dict['train'],        # positional: data_folder
            str(output_dir),             # positional: output_folder
            '--pretrained', config_dict['model'],
            '--test-folder', config_dict['test1'],
            '--test-folder2', config_dict['test2'],
            '--epochs', str(config_dict['epochs']),
            '--batch-size', str(config_dict['batch_size']),
            '--spec-transform', config_dict['spec_transform'],
            '--multilabel',
        ]
        
        # Add eval-only flag if model exists
        if eval_only:
            cmd.append('--eval-only')
        
        # Add normalization flags
        if config_dict.get('normalize'):
            cmd.append('--normalize')
        if config_dict.get('normalize_no_median'):
            cmd.append('--normalize-no-median')
        if config_dict.get('median_only'):
            cmd.append('--median-only')
        
        # Add noise args
        if config_dict.get('noise') and config_dict.get('noise') > 0:
            cmd.extend(['--noise', str(config_dict['noise'])])
            if config_dict.get('noise_folder'):
                cmd.extend(['--noise-folder', config_dict['noise_folder']])
            cmd.extend(['--noise-mode', 'both'])
        
        # Add mixup if specified
        if config_dict.get('mixup'):
            cmd.extend(['--mixup', str(config_dict['mixup'])])
    
    elif config_dict['type'] == 'dann':
        cmd = [
            'python3', 'finetune_birdclef.py',
            config_dict['source'],       # positional: data_folder (source domain)
            str(output_dir),             # positional: output_folder
            '--pretrained', config_dict['model'],
            '--test-folder', config_dict['test1'],
            '--test-folder2', config_dict['test2'],
            '--epochs', str(config_dict['epochs']),
            '--batch-size', str(config_dict['batch_size']),
            '--spec-transform', config_dict['spec_transform'],
            '--multilabel',
            '--use-dann',
            '--target-folder', config_dict['target'],
            '--lambda-domain', str(config_dict.get('lambda_domain', 0.3)),
        ]
        
        # Add eval-only flag if model exists
        if eval_only:
            cmd.append('--eval-only')
        
        # DANN always uses Log+normalize (best from normalization study)
        cmd.append('--normalize')
        
        if config_dict.get('mixup'):
            cmd.extend(['--mixup', str(config_dict['mixup'])])
    
    else:
        raise ValueError(f"Unknown experiment type: {config_dict['type']}")
    
    # Run experiment
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {name}")
        print(f"Stopping experiment pipeline due to failure.")
        sys.exit(1)
    
    # Load result
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    else:
        print(f"⚠ WARNING: No result file found for {name}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run all cross-dataset experiments')
    parser.add_argument('--avianz-train', required=True, help='AviaNZ training data folder')
    parser.add_argument('--avianz-test', required=True, help='AviaNZ test data folder')
    parser.add_argument('--doc-train', required=True, help='DOC training data folder')
    parser.add_argument('--doc-test', required=True, help='DOC test data folder')
    parser.add_argument('--output', required=True, help='Output folder for all experiments')
    parser.add_argument('--noise-folder', default=None, help='Noise folder (optional, for noise sweep)')
    parser.add_argument('--model', default='BirdClefModels/model_fold0.pth', help='Pretrained model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--mixup', type=float, default=0.25, help='Mixup alpha')
    
    args = parser.parse_args()
    
    # Validate paths
    for path in [args.avianz_train, args.avianz_test, args.doc_train, args.doc_test]:
        if not os.path.exists(path):
            print(f"ERROR: Path not found: {path}")
            sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # =========================================================================
    # EXPERIMENT SUITE 1: NORMALIZATION COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print(" EXPERIMENT SUITE 1: NORMALIZATION COMPARISON")
    print("="*70)
    print(" Goal: Find which normalization reduces domain shift most")
    print(" Methods: Log, Log+normalize, Log+normalize-no-median,")
    print("          Log+median-only, PCEN, Box-Cox")
    print(" Total: 6 methods × 2 directions = 12 experiments")
    print("="*70 + "\n")
    
    normalization_configs = [
        {'name': 'Log', 'spec_transform': 'Log', 'normalize': False, 'normalize_no_median': False, 'median_only': False},
        {'name': 'Log+normalize', 'spec_transform': 'Log', 'normalize': True, 'normalize_no_median': False, 'median_only': False},
        {'name': 'Log+normalize-no-median', 'spec_transform': 'Log', 'normalize': True, 'normalize_no_median': True, 'median_only': False},
        {'name': 'Log+median-only', 'spec_transform': 'Log', 'normalize': False, 'normalize_no_median': False, 'median_only': True},
        {'name': 'PCEN', 'spec_transform': 'PCEN', 'normalize': False, 'normalize_no_median': False, 'median_only': False},
        {'name': 'Box-Cox', 'spec_transform': 'Box-Cox', 'normalize': False, 'normalize_no_median': False, 'median_only': False},
    ]
    
    for norm_config in normalization_configs:
        # AviaNZ → DOC
        result = run_experiment({
            **norm_config,
            'name': f"avianz_baseline_{norm_config['name']}",
            'type': 'baseline',
            'train': args.avianz_train,
            'test1': args.avianz_test,
            'test2': args.doc_test,
            'model': args.model,
            'output_folder': output_folder,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'mixup': args.mixup,
            'noise': 0.0,
            'noise_folder': args.noise_folder,
        })
        if result:
            all_results.append(result)
        
        # DOC → AviaNZ
        result = run_experiment({
            **norm_config,
            'name': f"doc_baseline_{norm_config['name']}",
            'type': 'baseline',
            'train': args.doc_train,
            'test1': args.doc_test,
            'test2': args.avianz_test,
            'model': args.model,
            'output_folder': output_folder,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'mixup': args.mixup,
            'noise': 0.0,
            'noise_folder': args.noise_folder,
        })
        if result:
            all_results.append(result)
    
    # =========================================================================
    # EXPERIMENT SUITE 2: DOMAIN ADVERSARIAL NEURAL NETWORKS (DANN)
    # =========================================================================
    print("\n" + "="*70)
    print(" EXPERIMENT SUITE 2: DOMAIN ADVERSARIAL TRAINING (DANN)")
    print("="*70)
    print(" Goal: Test if DANN reduces domain shift beyond normalization")
    print(" Using: Log+normalize (best from Suite 1)")
    print(" Total: 2 directions = 2 experiments")
    print("="*70 + "\n")
    
    # AviaNZ → DOC with DANN
    result = run_experiment({
        'name': 'avianz_dann_Log+normalize',
        'type': 'dann',
        'source': args.avianz_train,
        'target': args.doc_train,
        'test1': args.avianz_test,
        'test2': args.doc_test,
        'model': args.model,
        'output_folder': output_folder,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'mixup': args.mixup,
        'spec_transform': 'Log',
        'lambda_domain': 0.3,
    })
    if result:
        all_results.append(result)
    
    # DOC → AviaNZ with DANN
    result = run_experiment({
        'name': 'doc_dann_Log+normalize',
        'type': 'dann',
        'source': args.doc_train,
        'target': args.avianz_train,
        'test1': args.doc_test,
        'test2': args.avianz_test,
        'model': args.model,
        'output_folder': output_folder,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'mixup': args.mixup,
        'spec_transform': 'Log',
        'lambda_domain': 0.3,
    })
    if result:
        all_results.append(result)
    
    # =========================================================================
    # EXPERIMENT SUITE 3: NOISE INTENSITY SWEEP
    # =========================================================================
    if args.noise_folder and os.path.exists(args.noise_folder):
        print("\n" + "="*70)
        print(" EXPERIMENT SUITE 3: NOISE INTENSITY SWEEP")
        print("="*70)
        print(" Goal: Test optimal noise mixing ratio")
        print(" Using: Log+normalize (best from Suite 1)")
        print(" Levels: 0.0, 0.25, 0.5, 0.75, 1.0")
        print(" Total: 5 levels × 2 directions = 10 experiments")
        print("="*70 + "\n")
        
        noise_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for noise_level in noise_levels:
            # AviaNZ → DOC
            result = run_experiment({
                'name': f"avianz_baseline_Log+normalize_intensity{noise_level}",
                'type': 'baseline',
                'train': args.avianz_train,
                'test1': args.avianz_test,
                'test2': args.doc_test,
                'model': args.model,
                'output_folder': output_folder,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'mixup': args.mixup,
                'spec_transform': 'Log',
                'normalize': True,
                'normalize_no_median': False,
                'median_only': False,
                'noise': noise_level,
                'noise_folder': args.noise_folder,
            })
            if result:
                all_results.append(result)
            
            # DOC → AviaNZ
            result = run_experiment({
                'name': f"doc_baseline_Log+normalize_intensity{noise_level}",
                'type': 'baseline',
                'train': args.doc_train,
                'test1': args.doc_test,
                'test2': args.avianz_test,
                'model': args.model,
                'output_folder': output_folder,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'mixup': args.mixup,
                'spec_transform': 'Log',
                'normalize': True,
                'normalize_no_median': False,
                'median_only': False,
                'noise': noise_level,
                'noise_folder': args.noise_folder,
            })
            if result:
                all_results.append(result)
        
        # =====================================================================
        # EXPERIMENT SUITE 4: NOISE VARIETY SWEEP
        # =====================================================================
        print("\n" + "="*70)
        print(" EXPERIMENT SUITE 4: NOISE VARIETY SWEEP")
        print("="*70)
        print(" Goal: Test if more noise variety improves robustness")
        print(" Using: Log+normalize, fixed noise ratio 0.25")
        print(" Levels: 1, 10, 100, 1000, all available noise files")
        print(" Total: ~5 levels × 2 directions = ~10 experiments")
        print("="*70 + "\n")
        
        # Count available noise files
        noise_data_dir = Path(args.noise_folder) / 'data'
        if noise_data_dir.exists():
            total_noise_files = len(list(noise_data_dir.glob('*.npy')))
            print(f"Total available noise files: {total_noise_files}\n")
            
            variety_levels = [1, 10, 100, 1000]
            # Add total if not already in list
            if total_noise_files not in variety_levels and total_noise_files > 0:
                variety_levels.append(total_noise_files)
            
            # Filter out levels larger than available
            variety_levels = [n for n in variety_levels if n <= total_noise_files]
            
            for n_noise in variety_levels:
                # Create subset of noise files
                noise_subset_dir = Path(args.noise_folder).parent / f'noise_subset_{n_noise}'
                
                if not noise_subset_dir.exists():
                    print(f"Creating noise subset: {n_noise} files")
                    noise_subset_dir.mkdir(parents=True, exist_ok=True)
                    (noise_subset_dir / 'data').mkdir(exist_ok=True)
                    
                    # Randomly sample n files
                    import random
                    all_noise_files = list(noise_data_dir.glob('*.npy'))
                    selected_files = random.sample(all_noise_files, min(n_noise, len(all_noise_files)))
                    
                    for f in selected_files:
                        import shutil
                        shutil.copy(f, noise_subset_dir / 'data' / f.name)
                    
                    # Copy labels if exists
                    labels_file = Path(args.noise_folder) / 'labels.json'
                    if labels_file.exists():
                        import shutil
                        shutil.copy(labels_file, noise_subset_dir / 'labels.json')
                
                # AviaNZ → DOC
                result = run_experiment({
                    'name': f"avianz_baseline_Log+normalize_variety{n_noise}",
                    'type': 'baseline',
                    'train': args.avianz_train,
                    'test1': args.avianz_test,
                    'test2': args.doc_test,
                    'model': args.model,
                    'output_folder': output_folder,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'mixup': args.mixup,
                    'spec_transform': 'Log',
                    'normalize': True,
                    'normalize_no_median': False,
                    'median_only': False,
                    'noise': 0.25,  # Fixed ratio
                    'noise_folder': str(noise_subset_dir),
                })
                if result:
                    all_results.append(result)
                
                # DOC → AviaNZ
                result = run_experiment({
                    'name': f"doc_baseline_Log+normalize_variety{n_noise}",
                    'type': 'baseline',
                    'train': args.doc_train,
                    'test1': args.doc_test,
                    'test2': args.avianz_test,
                    'model': args.model,
                    'output_folder': output_folder,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'mixup': args.mixup,
                    'spec_transform': 'Log',
                    'normalize': True,
                    'normalize_no_median': False,
                    'median_only': False,
                    'noise': 0.25,  # Fixed ratio
                    'noise_folder': str(noise_subset_dir),
                })
                if result:
                    all_results.append(result)
        else:
            print(f"⚠ Noise data directory not found: {noise_data_dir}")
    else:
        print("\n⚠ Skipping noise experiments (no noise folder provided)")
    
    # =========================================================================
    # GENERATE SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f" Total experiments run: {len(all_results)}")
    print(f" Results saved to: {output_folder}")
    print("="*70 + "\n")
    
    # Save all results
    results_file = output_folder / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved: {results_file}")


if __name__ == '__main__':
    main()
