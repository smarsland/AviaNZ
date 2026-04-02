#!/usr/bin/env python3
"""
Clean cross-dataset experiment pipeline.
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import signal
import shutil

import config


def copy_result_files(output_dir, results_dir, experiment_name):
    """
    Copy small result files (JSONs, CSVs) to shared results directory.
    Keeps large model files (.pt) in the original output directory.
    
    Args:
        output_dir: Original experiment output directory (contains model files)
        results_dir: Shared results directory for small files
        experiment_name: Name of the experiment (for subdirectory)
    """
    if results_dir is None:
        return  # No shared directory specified, keep all files in output_dir
    
    output_dir = Path(output_dir)
    results_dir = Path(results_dir)
    
    # Create experiment subdirectory in shared results
    exp_results_dir = results_dir / experiment_name
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    # List of small files to copy (exclude large .pt model files)
    small_file_patterns = [
        'result.json',
        'training_history.json',
        '*_config.json',
        'predictions_*.csv',
        'training_curves.png',
    ]
    
    copied_files = []
    for pattern in small_file_patterns:
        for src_file in output_dir.glob(pattern):
            dst_file = exp_results_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_files.append(src_file.name)
    
    if copied_files:
        print(f"  ✓ Copied {len(copied_files)} result files to {exp_results_dir}")
        print(f"    Files: {', '.join(copied_files)}")


def get_available_gpus():
    """Detect available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_list = list(range(torch.cuda.device_count()))
            print(f"DEBUG: Detected {len(gpu_list)} GPUs: {gpu_list}")
            return gpu_list
        else:
            print("DEBUG: torch.cuda.is_available() returned False")
            return []
    except Exception as e:
        print(f"DEBUG: GPU detection failed with error: {e}")
        return []


def run_experiments_parallel(experiments, n_workers, gpu_pool):
    """Run experiments in parallel using a worker pool."""
    if n_workers == 1:
        # Sequential execution
        results = []
        try:
            for exp_config in experiments:
                gpu_id = gpu_pool[0] if gpu_pool else None
                result = run_experiment_with_gpu(exp_config, gpu_id)
                if result:
                    results.append(result)
        except KeyboardInterrupt:
            print("\n⚠️  Ctrl+C detected! Stopping...")
            sys.exit(1)
        return results
    
    # Parallel execution
    results = []
    executor = ProcessPoolExecutor(max_workers=n_workers)
    
    try:
        # Submit all jobs with GPU assignment
        future_to_exp = {}
        for i, exp_config in enumerate(experiments):
            # Round-robin GPU assignment
            gpu_id = gpu_pool[i % len(gpu_pool)] if gpu_pool else None
            future = executor.submit(run_experiment_with_gpu, exp_config, gpu_id)
            future_to_exp[future] = exp_config
        
        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp_config = future_to_exp[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"✓ Completed: {result.get('name', 'unknown')}")
            except Exception as e:
                print(f"❌ Failed: {exp_config.get('name', 'unknown')} - {e}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Ctrl+C detected! Stopping all workers...")
        # Cancel all pending futures
        for future in future_to_exp:
            future.cancel()
        # Force shutdown with wait=False to kill running workers immediately
        executor.shutdown(wait=False, cancel_futures=True)
        print("✓ All workers stopped.")
        sys.exit(1)
    finally:
        executor.shutdown(wait=True)
    
    return results


def run_experiment_with_gpu(config_dict, gpu_id=None):
    """Wrapper to run experiment with specific GPU assignment."""
    if gpu_id is not None:
        config_dict = config_dict.copy()
        config_dict['gpu_id'] = gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    if config_dict.get('is_ast'):
        return run_ast_experiment(config_dict)
    else:
        return run_experiment(config_dict)


def run_ast_experiment(config_dict):
    """
    Run AST training + evaluation experiment using train_models.py.
    
    Returns: dict with results
    """
    name = config_dict['name']
    seed = config_dict.get('seed', 42)
    
    # Always add seed to name for consistency
    name = f"{name}_seed{seed}"
    
    output_dir = Path(config_dict['output_folder']) / name
    
    # Check if already complete or in progress (check shared results directory ONLY)
    if config_dict.get('results_dir'):
        shared_result_dir = Path(config_dict['results_dir']) / name
        
        if shared_result_dir.exists():
            print(f"✓ {name} - skipping (folder exists in shared results)")
            shared_result_file = shared_result_dir / 'result.json'
            if shared_result_file.exists():
                with open(shared_result_file) as f:
                    return json.load(f)
            else:
                # Folder exists but no result.json yet - experiment in progress
                return {
                    'name': name,
                    'experiment_type': 'ast_baseline',
                    'seed': seed,
                    'status': 'in_progress'
                }
    
    # If no shared directory or folder doesn't exist, check if training already complete locally
    result_file = output_dir / 'result.json'
    
    # Create the experiment folder in shared results immediately to claim it
    if config_dict.get('results_dir'):
        shared_result_dir = Path(config_dict['results_dir']) / name
        shared_result_dir.mkdir(parents=True, exist_ok=True)
        print(f"  → Created {shared_result_dir} (claiming experiment)")
    
    # Check if training complete
    model_file = output_dir / 'ast_model_best.pt'
    history_file = output_dir / 'training_history.json'
    
    if model_file.exists() and history_file.exists():
        print(f"\n{'='*70}")
        print(f"Skipping: {name} (MODEL ALREADY EXISTS)")
        print(f"{'='*70}")
        
        # Load complete results from existing files
        result_dict = {
            'name': name,
            'experiment_type': 'ast_baseline',
            'seed': seed,
            'output_folder': str(output_dir),
            'status': 'completed (pre-existing)'
        }
        
        # Load training history
        with open(history_file) as f:
            history = json.load(f)
            if 'val_acc' in history and history['val_acc']:
                val_accs = [v for v in history['val_acc'] if v is not None]
                if val_accs:
                    best_epoch = val_accs.index(max(val_accs))
                    result_dict['best_val_acc'] = max(val_accs) * 100
                    
                    if 'train_acc' in history and history['train_acc']:
                        train_accs = [v for v in history['train_acc'] if v is not None]
                        if len(train_accs) > best_epoch:
                            result_dict['best_train_acc'] = train_accs[best_epoch] * 100
        
        # Determine test set names from actual folder paths
        test1_name = Path(config_dict['test1']).parent.name if config_dict.get('test1') else 'test1'
        test2_name = Path(config_dict['test2']).parent.name if config_dict.get('test2') else 'test2'
        
        result_dict['test1_name'] = test1_name
        result_dict['test2_name'] = test2_name
        
        # Load test results from evaluation reports
        test1_report = output_dir / f'ast_test_{test1_name}_multilabel_report.json'
        if test1_report.exists():
            with open(test1_report) as f:
                report = json.load(f)
                if 'exact_match_accuracy' in report:
                    result_dict['test1_acc'] = report['exact_match_accuracy'] * 100
        
        test2_report = output_dir / f'ast_test_{test2_name}_multilabel_report.json'
        if test2_report.exists():
            with open(test2_report) as f:
                report = json.load(f)
                if 'exact_match_accuracy' in report:
                    result_dict['test2_acc'] = report['exact_match_accuracy'] * 100
        
        # Save/update result file
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Copy result files to shared directory if specified
        if config_dict.get('results_dir'):
            copy_result_files(output_dir, config_dict['results_dir'], name)
        
        return result_dict
    else:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
    
    # Build command for train_models.py
    cmd = [
        'python3', 'train_models.py',
        config_dict['train'],
        str(output_dir),
        '--model', 'ast',
        '--multilabel',
        '--epochs', str(config_dict['epochs']),
        '--batch_size', str(config_dict['batch_size']),
        '--mixup', str(config_dict.get('mixup', 0.25)),
        '--spec-transform', 'Log',
        '--patience', '15',  # Early stopping patience (same as finetune_birdclef.py default)
    ]
    
    # Add test folders if specified
    if config_dict.get('test1'):
        cmd.extend(['--test-folder', config_dict['test1']])
    if config_dict.get('test2'):
        cmd.extend(['--test-folder2', config_dict['test2']])
    
    # Add seed if specified
    if config_dict.get('seed'):
        cmd.extend(['--seed', str(config_dict['seed'])])
    
    # Run experiment
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {name}")
        print(f"Stopping experiment pipeline due to failure.")
        sys.exit(1)
    
    # After training, load results from training history and evaluation reports
    result_dict = {
        'name': name,
        'experiment_type': 'ast_baseline',
        'seed': seed,
        'output_folder': str(output_dir),
        'status': 'completed'
    }
    
    # Load training history for train/val accuracies
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
            if 'val_acc' in history and history['val_acc']:
                val_accs = [v for v in history['val_acc'] if v is not None]
                if val_accs:
                    best_epoch = val_accs.index(max(val_accs))
                    result_dict['best_val_acc'] = max(val_accs) * 100  # Convert to percentage
                    
                    # Get corresponding train accuracy
                    if 'train_acc' in history and history['train_acc']:
                        train_accs = [v for v in history['train_acc'] if v is not None]
                        if len(train_accs) > best_epoch:
                            result_dict['best_train_acc'] = train_accs[best_epoch] * 100  # Convert to percentage
    
    # Load test set evaluations from JSON reports
    # Determine test set names from actual folder paths
    test1_name = Path(config_dict['test1']).parent.name if config_dict.get('test1') else 'test1'
    test2_name = Path(config_dict['test2']).parent.name if config_dict.get('test2') else 'test2'
    
    result_dict['test1_name'] = test1_name
    result_dict['test2_name'] = test2_name
    
    # Try to load test1 results
    test1_report = output_dir / f'ast_test_{test1_name}_multilabel_report.json'
    if test1_report.exists():
        with open(test1_report) as f:
            report = json.load(f)
            if 'exact_match_accuracy' in report:
                result_dict['test1_acc'] = report['exact_match_accuracy'] * 100
    
    # Try to load test2 results
    test2_report = output_dir / f'ast_test_{test2_name}_multilabel_report.json'
    if test2_report.exists():
        with open(test2_report) as f:
            report = json.load(f)
            if 'exact_match_accuracy' in report:
                result_dict['test2_acc'] = report['exact_match_accuracy'] * 100
    
    # Save complete result file
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\n✓ AST experiment complete: {name}")
    if 'test1_acc' in result_dict:
        print(f"  Test1 ({test1_name}): {result_dict['test1_acc']:.1f}%")
    if 'test2_acc' in result_dict:
        print(f"  Test2 ({test2_name}): {result_dict['test2_acc']:.1f}%")
    
    # Copy result files to shared directory if specified
    if config_dict.get('results_dir'):
        copy_result_files(output_dir, config_dict['results_dir'], name)
    
    return result_dict


def run_experiment(config_dict):
    """
    Run a single training + evaluation experiment.
    
    Returns: dict with results
    """
    name = config_dict['name']
    seed = config_dict.get('seed', 42)
    
    # Always add seed to name for consistency across multiple trials
    name = f"{name}_seed{seed}"
    
    output_dir = Path(config_dict['output_folder']) / name
    
    # Check if already complete or in progress (check shared results directory ONLY)
    if config_dict.get('results_dir'):
        shared_result_dir = Path(config_dict['results_dir']) / name
        
        if shared_result_dir.exists():
            print(f"✓ {name} - skipping (folder exists in shared results)")
            shared_result_file = shared_result_dir / 'result.json'
            if shared_result_file.exists():
                with open(shared_result_file) as f:
                    return json.load(f)
            else:
                # Folder exists but no result.json yet - experiment in progress
                return {
                    'name': name,
                    'type': config_dict.get('type', 'baseline'),
                    'seed': seed,
                    'status': 'in_progress'
                }
    
    # If no shared directory or folder doesn't exist, check if training already complete locally
    result_file = output_dir / 'result.json'
    
    # Create the experiment folder in shared results immediately to claim it
    if config_dict.get('results_dir'):
        shared_result_dir = Path(config_dict['results_dir']) / name
        shared_result_dir.mkdir(parents=True, exist_ok=True)
        print(f"  → Created {shared_result_dir} (claiming experiment)")
    
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
        if config_dict.get('median_filter', False):
            cmd.append('--median-filter')
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
        
        # Add seed if specified
        if config_dict.get('seed'):
            cmd.extend(['--seed', str(config_dict['seed'])])
    
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
        
        # Add normalization flags if specified
        if config_dict.get('normalize'):
            cmd.append('--normalize')
            if config_dict.get('median_filter', False):
                cmd.append('--median-filter')
        
        if config_dict.get('mixup'):
            cmd.extend(['--mixup', str(config_dict['mixup'])])
        
        # Add seed if specified
        if config_dict.get('seed'):
            cmd.extend(['--seed', str(config_dict['seed'])])
    
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
    result_data = None
    if result_file.exists():
        with open(result_file) as f:
            result_data = json.load(f)
    else:
        print(f"⚠ WARNING: No result file found for {name}")
    
    # Copy result files to shared directory if specified
    if config_dict.get('results_dir') and result_file.exists():
        copy_result_files(output_dir, config_dict['results_dir'], name)
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description='Run all cross-dataset experiments')
    parser.add_argument('--avianz-train', required=True, help='Waitākere training data folder')
    parser.add_argument('--avianz-test', required=True, help='Waitākere test data folder')
    parser.add_argument('--doc-train', required=True, help='DOC training data folder')
    parser.add_argument('--doc-test', required=True, help='DOC test data folder')
    parser.add_argument('--merged-train', default=None, help='Merged training data folder (DOC + Waitākere) - optional')
    parser.add_argument('--output', required=True, help='Output folder for all experiments (model files)')
    parser.add_argument('--results-dir', default=None, help='Shared results directory for small files (JSONs, CSVs). If not specified, saves to output folder.')
    parser.add_argument('--noise-folder', default=None, help='Noise folder (optional, for noise sweep)')
    parser.add_argument('--model', default='BirdClefModels/model_fold0.pth', help='Pretrained model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--mixup', type=float, default=0.25, help='Mixup alpha')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 590, 573], 
                       help='Random seeds for multiple trials (default: 42 123 456 590 573 for 5 trials)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of experiments to run in parallel (default: 1, use 0 for auto-detect based on GPUs)')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                       help='Specific GPU IDs to use (e.g., --gpu-ids 0 1 2). If not specified, auto-detects all available GPUs.')
    
    args = parser.parse_args()
    
    # Validate paths
    for path in [args.avianz_train, args.avianz_test, args.doc_train, args.doc_test]:
        if not os.path.exists(path):
            print(f"ERROR: Path not found: {path}")
            sys.exit(1)
    
    if args.merged_train and not os.path.exists(args.merged_train):
        print(f"ERROR: Merged train path not found: {args.merged_train}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Setup GPU pool for parallel execution
    print(f"\nDEBUG: args.parallel = {args.parallel}")
    print(f"DEBUG: args.gpu_ids = {args.gpu_ids}")
    
    if args.gpu_ids is not None:
        gpu_pool = args.gpu_ids
        print(f"DEBUG: Using user-specified GPU IDs: {gpu_pool}")
    else:
        gpu_pool = get_available_gpus()
        print(f"DEBUG: Auto-detected GPU pool: {gpu_pool}")
    
    if args.parallel == 0:
        # Auto-detect: use number of GPUs (or 1 if no GPUs)
        n_workers = len(gpu_pool) if gpu_pool else 1
        print(f"DEBUG: Auto-detecting workers from GPU pool: {n_workers}")
    else:
        n_workers = args.parallel
        print(f"DEBUG: Using user-specified worker count: {n_workers}")
    
    print(f"\n{'='*70}")
    print(f" PARALLEL EXECUTION SETUP")
    print(f"{'='*70}")
    print(f" Workers: {n_workers}")
    print(f" GPUs available: {gpu_pool if gpu_pool else 'None (using CPU)'}")
    print(f"{'='*70}\n")
    
    all_results = []
    all_experiments = []
    
    # =========================================================================
    # EXPERIMENT SUITE 1: NORMALIZATION COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print(" EXPERIMENT SUITE 1: NORMALIZATION COMPARISON")
    print("="*70)
    print(" Goal: Find which normalization reduces domain shift most")
    print(" Methods: Log, Log+median+normalize, Log+normalize,")
    print("          Log+median, PCEN, Box-Cox")
    print(f" Seeds: {args.seeds}")
    print(f" Total: 6 methods × 2 directions × {len(args.seeds)} seeds = {6 * 2 * len(args.seeds)} experiments")
    print("="*70 + "\n")
    
    normalization_configs = [
        {'name': 'Log', 'spec_transform': 'Log', 'normalize': False, 'median_filter': False, 'median_only': False},
        {'name': 'Log+median+normalize', 'spec_transform': 'Log', 'normalize': True, 'median_filter': True, 'median_only': False},
        {'name': 'Log+normalize', 'spec_transform': 'Log', 'normalize': True, 'median_filter': False, 'median_only': False},
        {'name': 'Log+median', 'spec_transform': 'Log', 'normalize': False, 'median_filter': True, 'median_only': True},
        {'name': 'PCEN', 'spec_transform': 'PCEN', 'normalize': False, 'median_filter': False, 'median_only': False},
        {'name': 'Box-Cox', 'spec_transform': 'Box-Cox', 'normalize': False, 'median_filter': False, 'median_only': False},
    ]
    
    for seed in args.seeds:
        for norm_config in normalization_configs:
            # Waitākere → DOC
            all_experiments.append({
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
                'seed': seed,
            })
            
            # DOC → Waitākere
            all_experiments.append({
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
                'seed': seed,
            })
    
    # =========================================================================
    # EXPERIMENT SUITE 2: DOMAIN ADVERSARIAL NEURAL NETWORKS (DANN)
    # =========================================================================
    print("\n" + "="*70)
    print(" EXPERIMENT SUITE 2: DOMAIN ADVERSARIAL TRAINING (DANN)")
    print("="*70)
    print(" Goal: Test if DANN reduces domain shift with/without normalization")
    print(" Variants: (1) plain Log, (2) Log+median+normalize")
    print(f" Seeds: {args.seeds}")
    print(f" Total: 2 variants × 2 directions × {len(args.seeds)} seeds = {2 * 2 * len(args.seeds)} experiments")
    print("="*70 + "\n")
    
    dann_configs = [
        {'name': 'Log', 'normalize': False, 'median_filter': False},
        {'name': 'Log+median+normalize', 'normalize': True, 'median_filter': True},
    ]
    
    for seed in args.seeds:
        for dann_config in dann_configs:
            # Waitākere → DOC with DANN
            all_experiments.append({
                'name': f"avianz_dann_{dann_config['name']}",
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
                'normalize': dann_config['normalize'],
                'median_filter': dann_config['median_filter'],
                'lambda_domain': 0.3,
                'seed': seed,
            })
            
            # DOC → Waitākere with DANN
            all_experiments.append({
                'name': f"doc_dann_{dann_config['name']}",
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
                'normalize': dann_config['normalize'],
                'median_filter': dann_config['median_filter'],
                'lambda_domain': 0.3,
                'seed': seed,
            })
    
    # =========================================================================
    # EXPERIMENT SUITE 3: NOISE INTENSITY SWEEP
    # =========================================================================
    if args.noise_folder and os.path.exists(args.noise_folder):
        print("\n" + "="*70)
        print(" EXPERIMENT SUITE 3: NOISE INTENSITY SWEEP")
        print("="*70)
        print(" Goal: Test optimal noise mixing ratio")
        print(" Using: Log baseline (no normalization)")
        print(" Levels: 0.0, 0.2, 0.4, 0.6, 0.8")
        print(f" Seeds: {args.seeds}")
        print(f" Total: 5 levels × 2 directions × {len(args.seeds)} seeds = {5 * 2 * len(args.seeds)} experiments")
        print("="*70 + "\n")
        
        noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        for seed in args.seeds:
            for noise_level in noise_levels:
                # Waitākere → DOC
                all_experiments.append({
                    'name': f"avianz_baseline_Log_intensity{noise_level}",
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
                    'normalize': False,
                    'median_filter': False,
                    'median_only': False,
                    'noise': noise_level,
                    'noise_folder': args.noise_folder,
                    'seed': seed,
                })
                
                # DOC → Waitākere
                all_experiments.append({
                    'name': f"doc_baseline_Log_intensity{noise_level}",
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
                    'normalize': False,
                    'median_filter': False,
                    'median_only': False,
                    'noise': noise_level,
                    'noise_folder': args.noise_folder,
                    'seed': seed,
                })
        
        # =====================================================================
        # EXPERIMENT SUITE 4: NOISE VARIETY SWEEP
        # =====================================================================
        print("\n" + "="*70)
        print(" EXPERIMENT SUITE 4: NOISE VARIETY SWEEP")
        print("="*70)
        print(" Goal: Test if more noise variety improves robustness")
        print(" Using: Log baseline (no normalization), fixed noise ratio 0.2")
        print(" Levels: 1, 10, 100, 1000, all available noise files")
        print(f" Seeds: {args.seeds}")
        print(f" Total: ~5 levels × 2 directions × {len(args.seeds)} seeds = ~{5 * 2 * len(args.seeds)} experiments")
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
            
            # Pre-create all noise subsets (not parallelizable)
            print("Creating noise subsets...")
            for n_noise in variety_levels:
                noise_subset_dir = Path(args.noise_folder).parent / f'noise_subset_{n_noise}'
                
                if not noise_subset_dir.exists():
                    print(f"  Creating noise subset: {n_noise} files")
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
            print("✓ All noise subsets ready\n")
            
            # Now add all experiments to queue
            for seed in args.seeds:
                for n_noise in variety_levels:
                    noise_subset_dir = Path(args.noise_folder).parent / f'noise_subset_{n_noise}'
                    
                    # Waitākere → DOC
                    all_experiments.append({
                        'name': f"avianz_baseline_Log_variety{n_noise}",
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
                        'normalize': False,
                        'median_filter': False,
                        'median_only': False,
                        'noise': 0.2,  # Fixed ratio
                        'noise_folder': str(noise_subset_dir),
                        'seed': seed,
                    })
                    
                    # DOC → Waitākere
                    all_experiments.append({
                        'name': f"doc_baseline_Log_variety{n_noise}",
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
                        'normalize': False,
                        'median_filter': False,
                        'median_only': False,
                        'noise': 0.2,  # Fixed ratio
                        'noise_folder': str(noise_subset_dir),
                        'seed': seed,
                    })
        else:
            print(f"⚠ Noise data directory not found: {noise_data_dir}")
    else:
        print("\n⚠ Skipping noise experiments (no noise folder provided)")
    
    # =========================================================================
    # EXPERIMENT SUITE 5: AST BASELINE (DIFFERENT ARCHITECTURE)
    # =========================================================================
    print("\n" + "="*70)
    print(" EXPERIMENT SUITE 5: AST BASELINE")
    print("="*70)
    print(" Goal: Compare AST architecture to BirdCLEF fine-tuning")
    print(" Using: Audio Spectrogram Transformer (AST) with Log transform")
    print(f" Seeds: {args.seeds}")
    print(f" Total: 2 directions × {len(args.seeds)} seeds = {2 * len(args.seeds)} experiments")
    print("="*70 + "\n")
    
    # AST experiments: loop over all seeds (like other experiments)
    for seed in args.seeds:
        # Waitākere → DOC (AST trained from scratch on Waitākere)
        all_experiments.append({
            'name': 'avianz_ast_baseline',
            'train': args.avianz_train,
            'test1': args.avianz_test,
            'test2': args.doc_test,
            'output_folder': output_folder,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'mixup': args.mixup,
            'seed': seed,
            'is_ast': True,
        })
        
        # DOC → Waitākere (AST trained from scratch on DOC)
        all_experiments.append({
            'name': 'doc_ast_baseline',
            'train': args.doc_train,
            'test1': args.doc_test,
            'test2': args.avianz_test,
            'output_folder': output_folder,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'mixup': args.mixup,
            'seed': seed,
            'is_ast': True,
        })
    
    # =========================================================================
    # EXPERIMENT SUITE 6: MERGED DATASET TRAINING
    # =========================================================================
    if args.merged_train:
        print("\n" + "="*70)
        print(" EXPERIMENT SUITE 6: MERGED DATASET TRAINING")
        print("="*70)
        print(" Goal: Evaluate if training on combined DOC+Waitākere data")
        print("       improves cross-domain performance")
        print(" Using: Log baseline and Log+median+normalize preprocessing")
        print(f" Seeds: {args.seeds}")
        print(f" Total: 2 methods × {len(args.seeds)} seeds = {2 * len(args.seeds)} experiments")
        print("="*70 + "\n")
        
        merged_configs = [
            {'name': 'Log', 'spec_transform': 'Log', 'normalize': False, 'median_filter': False, 'median_only': False},
            {'name': 'Log+median+normalize', 'spec_transform': 'Log', 'normalize': True, 'median_filter': True, 'median_only': False},
        ]
        
        for seed in args.seeds:
            for merge_config in merged_configs:
                # Merged → both test sets
                all_experiments.append({
                    **merge_config,
                    'name': f"merged_baseline_{merge_config['name']}",
                    'type': 'baseline',
                    'train': args.merged_train,
                    'test1': args.doc_test,
                    'test2': args.avianz_test,
                    'model': args.model,
                    'output_folder': output_folder,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'mixup': args.mixup,
                    'noise': 0.0,
                    'noise_folder': args.noise_folder,
                    'seed': seed,
                })
    else:
        print("\n⚠ Skipping merged dataset experiments (no --merged-train provided)")
    
    # =========================================================================
    # RUN ALL EXPERIMENTS IN PARALLEL
    # =========================================================================
    print(f"\n{'='*70}")
    print(f" RUNNING ALL EXPERIMENTS")
    print(f"{'='*70}")
    print(f" Total queued: {len(all_experiments)}")
    print(f" Parallel workers: {n_workers}")
    print(f"{'='*70}\n")
    
    # Add results_dir to all experiment configs
    for exp in all_experiments:
        exp['results_dir'] = args.results_dir
    
    all_results = run_experiments_parallel(all_experiments, n_workers, gpu_pool)
    
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
