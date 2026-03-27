#!/usr/bin/env python3
"""
Simplified cross-dataset experiments for testing generalization.

Runs 2 core experiments (BirdClef CNN only):
1. Train on joe_mo (baseline), test on joe_mo test + doc test
2. Train on doc (baseline), test on doc test + joe_mo test

Generates comparison tables and plots.

Usage:
    python run_cross_dataset_experiments.py \\
        --avianz-train /path/to/joe_mo/train \\
        --avianz-test /path/to/joe_mo/test \\
        --doc-train /path/to/doc/train \\
        --doc-test /path/to/doc/test \\
        --output results/experiments
"""

import argparse
import os
import sys
import json
import csv
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd

import config


class CrossDatasetExperiments:
    """Manages cross-dataset training experiments."""
    
    def __init__(self, avianz_train, avianz_test, doc_train, doc_test,
                 output_folder, model_path, epochs=10, batch_size=32,
                 lambda_domain=0.1, noise_folder=None,
                 noise=None, noise_mode=None, background_prob=None, noise_as_samples=False,
                 mixup=None, force_rerun=False, spec_transform='Log', normalize=False,
                 normalize_no_median=False, experiment_suffix=''):
        self.avianz_train = avianz_train
        self.avianz_test = avianz_test
        self.doc_train = doc_train
        self.doc_test = doc_test
        self.output_folder = Path(output_folder)
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_domain = lambda_domain
        self.noise_folder = noise_folder
        self.force_rerun = force_rerun

        self.noise = noise
        self.noise_mode = noise_mode
        self.background_prob = background_prob
        self.noise_as_samples = noise_as_samples
        self.mixup = mixup
        self.spec_transform = spec_transform
        self.normalize = normalize
        self.normalize_no_median = normalize_no_median
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Define base experiment configurations (INTENTIONALLY MINIMAL)
        base_experiments = [
            # joe_mo experiments
            {
                'name': 'joe_mo_baseline',
                'train': avianz_train,
                'test1': avianz_test,
                'test2': doc_test,
                'type': 'finetune',
                'description': 'Baseline joe_mo'
            },
            # doc experiments
            {
                'name': 'doc_baseline',
                'train': doc_train,
                'test1': doc_test,
                'test2': avianz_test,
                'type': 'finetune',
                'description': 'Baseline doc'
            }
        ]
        
        # Generate experiments (BirdClef only)
        self.experiments = []
        model_type = 'birdclef'
        
        # Add suffix based on spec_transform and normalize
        suffix = ""
        if spec_transform != 'Log' or normalize:
            parts = []
            if spec_transform != 'Log':
                parts.append(spec_transform.lower())
            if normalize:
                parts.append("normalized")
            suffix = "_" + "_".join(parts)
        
        for base_exp in base_experiments:
            exp = base_exp.copy()
            exp['model_type'] = model_type
            exp['freeze'] = False
            exp['name'] = f"{base_exp['name']}_{model_type}{suffix}{experiment_suffix}"
            exp['description'] = f"{base_exp['description']} (BIRDCLEF)"
            if spec_transform != 'Log':
                exp['description'] += f" [{spec_transform}]"
            if normalize:
                exp['description'] += " [normalized]"
            if experiment_suffix:
                exp['description'] += f" {experiment_suffix}"
            self.experiments.append(exp)
        
        self.results = []
        
        print(f"{'='*60}")
        print(f"Cross-Dataset Experiments")
        print(f"{'='*60}")
        print(f"joe_mo train: {avianz_train}")
        print(f"joe_mo test:  {avianz_test}")
        print(f"doc train:    {doc_train}")
        print(f"doc test:     {doc_test}")
        print(f"Freeze strategy: Disabled")
        print(f"Normalization: Disabled")
        print(f"\nExperiment breakdown (2 experiments):")
        print(f"  1) joe_mo baseline (train joe_mo, test joe_mo + doc)")
        print(f"  2) doc baseline (train doc, test doc + joe_mo)")
        print(f"\nTotal: {len(self.experiments)} experiments")
        print(f"{'='*60}")
    
    def extract_accuracy(self, value):
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            # Values are already stored as percentages (0-100), don't convert
            return float(value)
        if isinstance(value, (tuple, list)):
            return self.extract_accuracy(value[0]) if len(value) > 0 else 0.0
        if isinstance(value, dict):
            return self.extract_accuracy(value.get('macro_f1', value.get('accuracy', 0.0)))
        return 0.0
    
    def get_best_epoch_metrics(self, history):
        """Extract train/val metrics from the best validation epoch (not final epoch)."""
        train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        # Find epoch with best validation accuracy
        val_accs = [self.extract_accuracy(v) for v in history.get(val_acc_key, []) if v is not None]
        if not val_accs:
            # No validation data - use final epoch
            train_acc = self.extract_accuracy(history[train_acc_key][-1] if history.get(train_acc_key) else None)
            return train_acc, None, None
        
        best_epoch = val_accs.index(max(val_accs))
        best_val = val_accs[best_epoch]
        best_train = self.extract_accuracy(history[train_acc_key][best_epoch] if history.get(train_acc_key) and len(history[train_acc_key]) > best_epoch else None)
        
        return best_train, best_val, best_val
    
    def is_experiment_complete(self, exp_output):
        """Check if experiment has already been completed successfully."""
        if self.force_rerun:
            return False  # Always return False when force_rerun is enabled
        
        history_path = exp_output / 'training_history.json'
        
        if not history_path.exists():
            return False
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_acc'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
            
            if train_acc_key in history and val_acc_key in history:
                if len(history[train_acc_key]) > 0 and len(history[val_acc_key]) > 0:
                    return True
            
            return False
        except Exception as e:
            print(f"  Warning: Could not read history file: {e}")
            return False
    
    def load_completed_experiment(self, exp):
        """Load results from a previously completed experiment."""
        exp_output = self.output_folder / exp['name']
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Extract meaningful test names (parent directory + folder name)
            test1_path = Path(exp['test1'])
            test2_path = Path(exp['test2'])
            test1_name = f"{test1_path.parent.name}/{test1_path.name}"
            test2_name = f"{test2_path.parent.name}/{test2_path.name}"
            
            if exp['model_type'] == 'ast':
                normalize = exp.get('normalize', False)
                test1_acc = self._evaluate_ast_test_set(exp_output, exp['test1'], test1_name, normalize)
                test2_acc = self._evaluate_ast_test_set(exp_output, exp['test2'], test2_name, normalize)
            else:
                test1_acc = self._extract_test_from_file(exp_output, test1_name, exp['test1'])
                test2_acc = self._extract_test_from_file(exp_output, test2_name, exp['test2'])
            
            # Extract metrics from BEST epoch (same checkpoint used for test evaluation)
            best_train_acc, best_val_acc, best_val = self.get_best_epoch_metrics(history)
            
            if exp['type'] == 'dann':
                train_dataset_name = f"{Path(exp['source']).parent.name} (DANN→{Path(exp['target']).parent.name})"
            else:
                train_dataset_name = Path(exp['train']).name
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'model_type': exp['model_type'],
                'train_dataset': train_dataset_name,
                'freeze_backbone': exp['freeze'],
                'final_train_acc': best_train_acc,
                'final_val_acc': best_val_acc,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': best_val,
                'history': history,
                'output_folder': str(exp_output)
            }
            
            if exp['type'] == 'dann':
                exp_result['lambda_domain'] = exp.get('lambda_domain', self.lambda_domain)
            
            self.results.append(exp_result)
            
            print(f"\n✓ Loaded cached results (best epoch):")
            print(f"  Best train acc: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Best val acc: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error loading completed experiment!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_experiment(self, exp):
        """Run a single training experiment (finetune or DANN), or load if already complete."""
        exp_output = self.output_folder / exp['name']
        
        if self.is_experiment_complete(exp_output):
            print(f"\n{'='*60}")
            print(f"Experiment: {exp['name']}")
            print(f"{'='*60}")
            print(f"✓ Already completed - loading cached results...")
            return self.load_completed_experiment(exp)
        
        return self.run_finetune_experiment(exp)
    
    def run_finetune_experiment(self, exp):
        """Run a single fine-tuning experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp['description']}")
        print(f"Model type: {exp['model_type'].upper()}")
        print(f"Train on: {exp['train']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: disabled")
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            'finetune_birdclef.py',
            exp['train'],
            str(exp_output),
            '--pretrained', self.model_path,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--test-folder', exp['test1'],
            '--test-folder2', exp['test2'],
            '--multilabel',
            '--spec-transform', self.spec_transform
        ]

        if self.normalize:
            cmd.append('--normalize')
        if self.normalize_no_median:
            cmd.append('--normalize-no-median')

        if self.mixup is not None:
            cmd.extend(['--mixup', str(self.mixup)])

        if self.noise is not None:
            cmd.extend(['--noise', str(self.noise)])
            if self.noise_folder:
                cmd.extend(['--noise-folder', str(self.noise_folder)])
            if self.noise_mode is not None:
                cmd.extend(['--noise-mode', str(self.noise_mode)])

        if self.background_prob is not None:
            cmd.extend(['--background-prob', str(self.background_prob)])
        
        print(f"\nRunning: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        # Stream output in real-time (don't capture)
        result = subprocess.run(cmd)
        
        print(f"{'='*60}")
        if result.returncode != 0:
            print(f"\n❌ Finetune experiment failed!")
            print(f"Return code: {result.returncode}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Extract meaningful test names (parent directory + folder name)
            test1_path = Path(exp['test1'])
            test2_path = Path(exp['test2'])
            test1_name = f"{test1_path.parent.name}/{test1_path.name}"
            test2_name = f"{test2_path.parent.name}/{test2_path.name}"
            
            if str(exp['test1']) == str(exp['train']):
                print(f"  ⚠️  WARNING: test1 folder is the same as the train folder: {exp['test1']}")
                print(f"      Test accuracy will be meaninglessly high (measured on training data).")
            if str(exp['test2']) == str(exp['train']):
                print(f"  ⚠️  WARNING: test2 folder is the same as the train folder: {exp['test2']}")
                print(f"      Test accuracy will be meaninglessly high (measured on training data).")
            
            if exp['model_type'] == 'ast':
                print(f"\nRunning test evaluation for AST model...")
                normalize = exp.get('normalize', False)
                test1_acc = self._evaluate_ast_test_set(exp_output, exp['test1'], test1_name, normalize)
                test2_acc = self._evaluate_ast_test_set(exp_output, exp['test2'], test2_name, normalize)
            else:
                # Extract test accuracies from saved test results files
                test1_acc = self._extract_test_from_file(exp_output, test1_name, exp['test1'])
                test2_acc = self._extract_test_from_file(exp_output, test2_name, exp['test2'])
            
            # Extract metrics from BEST epoch (same checkpoint used for test evaluation)
            best_train_acc, best_val_acc, best_val = self.get_best_epoch_metrics(history)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'model_type': exp['model_type'],
                'train_dataset': Path(exp['train']).name,
                'freeze_backbone': exp['freeze'],
                'final_train_acc': best_train_acc,
                'final_val_acc': best_val_acc,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': best_val,
                'history': history,
                'output_folder': str(exp_output)
            }
            
            self.results.append(exp_result)
            
            print(f"\n✓ Experiment complete (best epoch):")
            print(f"  Best train exact match: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Best val exact match: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error processing finetune results!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_dann_experiment(self, exp):
        """Run a DANN domain adaptation experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp['description']}")
        print(f"Model type: {exp['model_type'].upper()}")
        print(f"Source: {exp['source']}")
        print(f"Target: {exp['target']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: {exp['freeze']}")
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
        if exp['model_type'] == 'ast':
            # AST: use train_models.py with DANN support
            # Reduce batch size for DANN since it loads both source+target simultaneously
            ast_dann_batch_size = max(4, self.batch_size // 2)
            cmd = [
                sys.executable,
                'train_models.py',
                exp['source'],
                str(exp_output),
                '--model', 'ast',
                '--epochs', str(self.epochs),
                '--batch_size', str(ast_dann_batch_size),
                '--use-dann',
                '--target-folder', exp['target'],
                '--lambda-domain', str(exp.get('lambda_domain', self.lambda_domain)),
                '--test-folder', exp['test1'],
                '--test-folder2', exp['test2']
            ]
            
            # For AST: freeze first 8 transformer layers to reduce overfitting
            if exp['freeze']:
                cmd.extend(['--freeze-layers', '8'])
            
            normalize = exp.get('normalize', False)
            if normalize:
                cmd.append('--normalize')
        
        else:
            # BirdClef: use finetune_birdclef.py
            cmd = [
                sys.executable,
                'finetune_birdclef.py',
                exp['source'],
                str(exp_output),
                '--pretrained', self.model_path,
                '--epochs', str(self.epochs),
                '--batch-size', str(self.batch_size),
                '--test-folder', exp['test1'],
                '--test-folder2', exp['test2'],
                '--use-dann',
                '--target-folder', exp['target'],
                '--lambda-domain', str(exp.get('lambda_domain', self.lambda_domain)),
                '--multilabel'
            ]
            
            # For BirdClef: freeze first 4 stages (stem, s1, s2, s3) to reduce overfitting
            # Only train s4 + classifier for better generalization
            if exp['freeze']:
                cmd.extend(['--freeze-stages', '4'])
        
        print(f"\nRunning: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        # Stream output in real-time (don't capture)
        result = subprocess.run(cmd)
        
        print(f"{'='*60}")
        if result.returncode != 0:
            print(f"\n❌ DANN experiment failed!")
            print(f"Return code: {result.returncode}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Extract meaningful test names (parent directory + folder name)
            test1_path = Path(exp['test1'])
            test2_path = Path(exp['test2'])
            test1_name = f"{test1_path.parent.name}/{test1_path.name}"
            test2_name = f"{test2_path.parent.name}/{test2_path.name}"
            
            # Extract test accuracies from saved test results files
            test1_acc = self._extract_test_from_file(exp_output, test1_name, exp['test1'])
            test2_acc = self._extract_test_from_file(exp_output, test2_name, exp['test2'])
            
            final_train_acc = self.extract_accuracy(history['train_acc'][-1] if history.get('train_acc') else None)
            final_val_acc = self.extract_accuracy(history['val_acc'][-1] if history.get('val_acc') and history['val_acc'][-1] is not None else None)
            best_val = max([self.extract_accuracy(v) for v in history.get('val_acc', []) if v is not None], default=None)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'model_type': exp['model_type'],
                'train_dataset': f"{Path(exp['source']).parent.name} (DANN→{Path(exp['target']).parent.name})",
                'freeze_backbone': exp['freeze'],
                'lambda_domain': exp.get('lambda_domain', self.lambda_domain),
                'final_train_acc': final_train_acc,
                'final_val_acc': final_val_acc,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': best_val,
                'history': history,
                'output_folder': str(exp_output)
            }
            
            self.results.append(exp_result)
            
            print(f"\n✓ Experiment complete:")
            print(f"  Final train exact match: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Final val exact match: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error processing DANN results!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_cleaner_experiment(self, exp):
        """Run an experiment with spectrogram cleaner (trainable preprocessing)."""
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp['description']}")
        print(f"Model type: {exp['model_type'].upper()}")
        print(f"Train on: {exp['train']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: {exp['freeze']}")
        print(f"Using trainable spectrogram cleaner for domain adaptation")
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
        if exp['model_type'] == 'ast':
            cmd = [
                sys.executable,
                'train_models.py',
                exp['train'],
                str(exp_output),
                '--model', 'ast',
                '--epochs', str(self.epochs),
                '--batch_size', str(self.batch_size),
                '--use-cleaner'
            ]
            
            if self.model_path and not self.model_path.endswith('model_fold0.pth'):
                cmd.extend(['--pretrained', self.model_path])
            
            if exp['freeze']:
                cmd.extend(['--freeze-layers', '8'])
        else:
            cmd = [
                sys.executable,
                'finetune_birdclef.py',
                exp['train'],
                str(exp_output),
                '--pretrained', self.model_path,
                '--epochs', str(self.epochs),
                '--batch-size', str(self.batch_size),
                '--test-folder', exp['test1'],
                '--test-folder2', exp['test2'],
                '--use-cleaner',
                '--multilabel'
            ]
            
            if exp['freeze']:
                cmd.extend(['--freeze-stages', '4'])
        
        print(f"\nRunning: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        result = subprocess.run(cmd)
        
        print(f"{'='*60}")
        if result.returncode != 0:
            print(f"\n❌ Cleaner experiment failed!")
            print(f"Return code: {result.returncode}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Extract meaningful test names (parent directory + folder name)
            test1_path = Path(exp['test1'])
            test2_path = Path(exp['test2'])
            test1_name = f"{test1_path.parent.name}/{test1_path.name}"
            test2_name = f"{test2_path.parent.name}/{test2_path.name}"
            
            if exp['model_type'] == 'ast':
                normalize = exp.get('normalize', False)
                test1_acc = self._evaluate_ast_test_set(exp_output, exp['test1'], test1_name, normalize)
                test2_acc = self._evaluate_ast_test_set(exp_output, exp['test2'], test2_name, normalize)
            else:
                test1_acc = self._extract_test_from_file(exp_output, test1_name, exp['test1'])
                test2_acc = self._extract_test_from_file(exp_output, test2_name, exp['test2'])
            
            train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_acc'
            val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
            
            final_train_acc = self.extract_accuracy(history[train_acc_key][-1] if history.get(train_acc_key) else None)
            final_val_acc = self.extract_accuracy(history[val_acc_key][-1] if history.get(val_acc_key) and history[val_acc_key][-1] is not None else None)
            best_val = max([self.extract_accuracy(v) for v in history.get(val_acc_key, []) if v is not None], default=None)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'model_type': exp['model_type'],
                'train_dataset': Path(exp['train']).name,
                'freeze_backbone': exp['freeze'],
                'final_train_acc': final_train_acc,
                'final_val_acc': final_val_acc,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': best_val,
                'history': history,
                'output_folder': str(exp_output)
            }
            
            self.results.append(exp_result)
            
            print(f"\n✓ Experiment complete:")
            print(f"  Final train exact match: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Final val exact match: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error processing cleaner results!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_ast_test_set(self, model_folder, test_folder, test_name, normalize=False):
        """Evaluate AST model on a test set using predict.py and calculate accuracy."""
        model_path = model_folder / 'ast_model_best.pt'
        config_path = model_folder / 'ast_model_config.json'
        output_csv = model_folder / f'predictions_{test_name}.csv'
        
        if not model_path.exists():
            print(f"  ⚠️  Model not found: {model_path}")
            return 0.0
        
        if not config_path.exists():
            print(f"  ⚠️  Config not found: {config_path}")
            return 0.0
        
        print(f"  Evaluating on {test_name}...")
        cmd = [
            sys.executable,
            'predict.py',
            str(model_path),
            str(config_path),
            test_folder,
            str(output_csv)
        ]
        
        if normalize:
            cmd.append('--normalize')
        
        print(f"  Running prediction...")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"  ❌ Prediction failed for {test_name}")
            return 0.0
        
        if not output_csv.exists():
            print(f"  ❌ Prediction CSV was not created: {output_csv}")
            return 0.0
        
        accuracy = self._compute_accuracy_from_predictions(output_csv, test_folder)
        print(f"  {test_name} Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _compute_accuracy_from_predictions(self, csv_path, test_folder):
        """Compute accuracy by comparing predictions CSV to ground truth labels.json."""
        labels_path = os.path.join(test_folder, 'labels.json')
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in test folder: {test_folder}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
        
        # Load bird name mapping (CommonName -> eBird code)
        name_map = {}
        mapping_path = 'DOC_bird_naming_map.csv'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        common_name = row[0]  # e.g., "Fantail"
                        ebird_code = row[1]   # e.g., "nezfan1"
                        name_map[common_name] = ebird_code
                        # Also map in reverse
                        name_map[ebird_code] = ebird_code
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        categories = labels_data['categories']
        
        # Build ground truth mapping: filename -> primary class name (normalize to eBird code)
        true_labels = {}
        for file_info in labels_data['files']:
            filename = file_info['filename']
            # Use primary_class or primary_species if available
            primary = file_info.get('primary_class') or file_info.get('primary_species')
            if not primary:
                # Fall back to first class in class_names
                class_names = file_info.get('class_names', [])
                primary = class_names[0] if class_names else None
            if primary:
                # Normalize to eBird code and lowercase for case-insensitive comparison
                primary = name_map.get(primary, primary)
                true_labels[filename] = primary.lower()
        
        # Read predictions CSV
        predictions = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            
            # Handle two formats:
            # 1. Probability format: File_Path, row_id, class1, class2, ...
            # 2. Simple format: filename, predicted_class
            if 'predicted_class' in fieldnames:
                # Simple format from model_trainer._save_test_predictions
                for row in reader:
                    filename = row.get('filename', '')
                    pred_class = row.get('predicted_class', '')
                    # Normalize to eBird code and lowercase for case-insensitive comparison
                    pred_class = name_map.get(pred_class, pred_class)
                    predictions[filename] = pred_class.lower()
            else:
                # Probability format from predict.py
                class_columns = [col for col in fieldnames if col not in ['row_id', 'File_Path']]
                
                for row in reader:
                    filename = row.get('row_id', row.get('filename', ''))
                    if not filename or not class_columns:
                        continue
                    
                    # Use argmax to get predicted class (like training code does)
                    class_probs = [float(row[col]) for col in class_columns]
                    pred_idx = class_probs.index(max(class_probs))
                    pred_class = class_columns[pred_idx]
                    # Normalize to eBird code and lowercase for case-insensitive comparison
                    pred_class = name_map.get(pred_class, pred_class)
                    predictions[filename] = pred_class.lower()
        
        # DEBUG: Check first few matches
        debug_count = 0
        for filename, true_class in list(true_labels.items())[:3]:
            pred_class = predictions.get(filename, 'NOT_FOUND')
            print(f"    DEBUG: {filename} | True: {true_class} | Pred: {pred_class}")
            debug_count += 1
        
        # Compare predictions to ground truth
        correct = 0
        total = 0
        mismatches = 0
        for filename, true_class in true_labels.items():
            if filename in predictions:
                if predictions[filename] == true_class:
                    correct += 1
                else:
                    mismatches += 1
                total += 1
        
        print(f"    DEBUG: Total files={len(true_labels)}, Matched={total}, Correct={correct}, Wrong={mismatches}")
        
        if total == 0:
            return 0.0
        
        return (correct / total) * 100.0
    
    def _extract_test_from_file(self, exp_output, test_name, test_folder):
        """Extract test accuracy from saved prediction CSV file."""
        # finetune_birdclef.py saves predictions as predictions_{parent}_{test_name}.csv
        # where parent is the parent directory name and test_name is the folder name.
        # test_name comes as "parent/test", need to convert to "parent_test" for file matching
        
        all_csvs = list(exp_output.glob('*.csv'))
        print(f"  DEBUG: Looking for predictions in {exp_output}")
        print(f"  DEBUG: test_name = '{test_name}'")
        print(f"  DEBUG: All CSV files in directory: {[f.name for f in all_csvs]}")
        print(f"  DEBUG: Using test folder: {test_folder}")
        
        # Convert test_name from "parent/folder" to "parent_folder" for filename matching
        test_name_pattern = test_name.replace('/', '_')
        csv_files = list(exp_output.glob(f'predictions_*{test_name_pattern}*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No prediction CSV found for '{test_name}' in {exp_output}.\n"
                f"Expected pattern: predictions_*{test_name_pattern}*.csv\n"
                f"Available CSVs: {[f.name for f in all_csvs]}"
            )
        
        csv_path = csv_files[0]  # Use first match
        print(f"  DEBUG: Found CSV file: {csv_path.name}")
        
        accuracy = self._compute_accuracy_from_predictions(csv_path, test_folder)
        print(f"  ✓ Extracted {test_name} accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _extract_test_accuracy(self, output, test_name):
        """Extract test accuracy from training output."""
        lines = output.split('\n')
        for line in lines:
            # Match both "joe_mo_split" and "joe_mo_split/test" formats
            if test_name in line and 'Accuracy:' in line:
                try:
                    acc_str = line.split('Accuracy:')[1].strip().rstrip('%')
                    return float(acc_str)
                except:
                    pass
        # Also try matching with /test suffix for DANN output
        for line in lines:
            if f"{test_name}/test" in line and 'Accuracy:' in line:
                try:
                    acc_str = line.split('Accuracy:')[1].strip().rstrip('%')
                    return float(acc_str)
                except:
                    pass
        return 0.0
    
    def run_all_experiments(self):
        """Run all experiments."""
        print(f"\n{'='*60}")
        print(f"Running {len(self.experiments)} experiments...")
        print(f"{'='*60}")
        
        for i, exp in enumerate(self.experiments, 1):
            print(f"\n[{i}/{len(self.experiments)}] Starting experiment: {exp['name']}")
            result = self.run_experiment(exp)
            
            if result is None:
                print(f"\n{'='*60}")
                print(f"❌ EXPERIMENT FAILED: {exp['name']}")
                print(f"{'='*60}")
                print(f"\nSTOPPING NOW - Check error above")
                sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"✓ All {len(self.experiments)} experiments complete!")
        print(f"{'='*60}")
    
    def generate_summary_table(self):
        """Generate summary table of all results."""
        print(f"\nGenerating summary table...")
        
        data = []
        for r in self.results:
            # Handle potentially missing fields from old results
            description = r.get('description', r.get('name', 'Unknown'))
            train_dataset = r.get('train_dataset', 'unknown')
            freeze_backbone = r.get('freeze_backbone', False)
            final_train_acc = r.get('final_train_acc', 0.0)
            final_val_acc = r.get('final_val_acc')
            test1_name = r.get('test1_name', 'test1')
            test1_acc = r.get('test1_acc', 0.0)
            test2_name = r.get('test2_name', 'test2')
            test2_acc = r.get('test2_acc', 0.0)
            
            data.append({
                'Experiment': description,
                'Train Dataset': train_dataset,
                'Freeze': 'Yes' if freeze_backbone else 'No',
                'Train Exact Match': f"{final_train_acc:.2f}",
                'Val Exact Match': f"{final_val_acc:.2f}" if final_val_acc is not None else 'N/A',
                f'Test {test1_name}': f"{test1_acc:.2f}",
                f'Test {test2_name}': f"{test2_acc:.2f}"
            })
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_folder / 'summary_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved to: {csv_path}")
        
        txt_path = self.output_folder / 'summary_table.txt'
        with open(txt_path, 'w') as f:
            f.write(df.to_string(index=False))
        print(f"  Saved to: {txt_path}")
        
        print(f"\n{df.to_string(index=False)}")
        
        return df
    
    def plot_test_accuracy_comparison(self):
        """Plot test exact match accuracy comparison across experiments."""
        print(f"\nGenerating test accuracy comparison plot...")
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        exp_names = []
        joe_mo_scores = []
        doc_scores = []
        
        for r in self.results:
            exp_names.append(r['description'])
            
            # Check for 'avianz' in test name (avianz_split is joe_mo data)
            if 'avianz' in r['test1_name'].lower():
                joe_mo_scores.append(r['test1_acc'])
                doc_scores.append(r['test2_acc'])
            elif 'doc' in r['test1_name'].lower():
                joe_mo_scores.append(r['test2_acc'])
                doc_scores.append(r['test1_acc'])
            else:
                # Fallback
                joe_mo_scores.append(r['test1_acc'])
                doc_scores.append(r['test2_acc'])
        
        x = np.arange(len(exp_names))
        width = 0.38
        
        bars1 = ax.bar(x - width/2, joe_mo_scores, width, label='joe_mo Test', 
                      color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, doc_scores, width, label='doc Test', 
                      color='#A23B72', alpha=0.85, edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Exact Match Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Test Exact Match Accuracy Comparison Across Experiments', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=11, rotation=0, ha='center')
        ax.legend(fontsize=13, loc='upper right', framealpha=0.95, edgecolor='black')
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.set_ylim([0, 110])
        ax.set_axisbelow(True)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'test_accuracy_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_validation_performance(self):
        """Plot validation accuracy comparison across experiments."""
        print(f"\nGenerating validation performance comparison plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        exp_names = []
        val_scores = []
        best_val_scores = []
        
        for r in self.results:
            exp_names.append(r['name'].replace('_', '\n').upper())
            
            # Use final validation accuracy (or 0 if None)
            val_acc = r['final_val_acc'] if r['final_val_acc'] is not None else 0
            best_val = r['best_val_acc'] if r['best_val_acc'] is not None else 0
            
            val_scores.append(val_acc)
            best_val_scores.append(best_val)
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_scores, width, label='Final Val Acc', color='#9b59b6', alpha=0.8)
        bars2 = ax.bar(x + width/2, best_val_scores, width, label='Best Val Acc', color='#3498db', alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Validation Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=9, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show label if not zero
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'validation_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_validation_heatmap(self):
        """Generate heatmap of train dataset vs validation accuracy."""
        print(f"\nGenerating validation performance heatmap...")
        
        # Dynamically determine which training datasets are present
        present_datasets = set()
        for r in self.results:
            if r['name'].startswith('avianz'):
                present_datasets.add('AviaNZ')
            elif r['name'].startswith('doc'):
                present_datasets.add('DOC')
            elif r['name'].startswith('combined'):
                present_datasets.add('Combined')
            elif r['name'].startswith('dann'):
                present_datasets.add('DANN')
        
        # Order datasets consistently
        all_possible = ['AviaNZ', 'DOC', 'Combined', 'DANN']
        train_datasets = [d for d in all_possible if d in present_datasets]
        
        n_datasets = len(train_datasets)
        full_matrix = np.zeros((n_datasets, 1))
        frozen_matrix = np.zeros((n_datasets, 1))
        
        # Create mapping dynamically
        mapping = {}
        if 'AviaNZ' in present_datasets:
            mapping['avianz'] = train_datasets.index('AviaNZ')
        if 'DOC' in present_datasets:
            mapping['doc'] = train_datasets.index('DOC')
        if 'Combined' in present_datasets:
            mapping['combined'] = train_datasets.index('Combined')
        if 'DANN' in present_datasets:
            mapping['dann'] = train_datasets.index('DANN')
        
        for r in self.results:
            # Determine train dataset index
            train_idx = None
            for key, idx in mapping.items():
                if r['name'].startswith(key):
                    train_idx = idx
                    break
            
            if train_idx is None:
                continue
            
            # Get validation score
            val_score = r['final_val_acc'] if r['final_val_acc'] is not None else 0
            
            # Fill matrices
            if r['freeze_backbone']:
                frozen_matrix[train_idx, 0] = val_score
            else:
                full_matrix[train_idx, 0] = val_score
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Full fine-tuning heatmap
        sns.heatmap(full_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, 
                    xticklabels=['Validation'], yticklabels=train_datasets,
                    ax=ax1, linewidths=1, linecolor='gray')
        ax1.set_title('Full Fine-tuning', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('', fontsize=12, labelpad=10)
        ax1.set_ylabel('Training Dataset', fontsize=12, labelpad=10)
        
        # Frozen backbone heatmap
        sns.heatmap(frozen_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, 
                    xticklabels=['Validation'], yticklabels=train_datasets,
                    ax=ax2, linewidths=1, linecolor='gray')
        ax2.set_title('Frozen Backbone', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('', fontsize=12, labelpad=10)
        ax2.set_ylabel('Training Dataset', fontsize=12, labelpad=10)
        
        plt.suptitle('Validation Performance Heatmap', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = self.output_folder / 'validation_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_validation_vs_test(self):
        """Plot validation vs test accuracy."""
        print(f"\nGenerating validation vs test comparison plot...")
        
        if not self.results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Collect all points
        joe_mo_vals = []
        joe_mo_tests = []
        doc_vals = []
        doc_tests = []
        label_positions = []
        
        for r in self.results:
            val_acc = r['final_val_acc'] if r['final_val_acc'] is not None else 0
            
            # Check which test set is which based on test1_name
            if 'joe_mo' in r['test1_name']:
                joe_mo_vals.append(val_acc)
                joe_mo_tests.append(r['test1_acc'])
                doc_vals.append(val_acc)
                doc_tests.append(r['test2_acc'])
            else:
                doc_vals.append(val_acc)
                doc_tests.append(r['test1_acc'])
                joe_mo_vals.append(val_acc)
                joe_mo_tests.append(r['test2_acc'])
        
        # Determine nice axis limits
        all_vals = joe_mo_vals + doc_vals
        all_tests = joe_mo_tests + doc_tests
        
        if all_vals and all_tests:
            min_val = max(0, min(min(all_vals), min(all_tests)) - 10)
            max_val = min(100, max(max(all_vals), max(all_tests)) + 10)
            min_val = int(min_val / 10) * 10
            max_val = int((max_val + 9) / 10) * 10
        else:
            min_val, max_val = 0, 100
        
        # Plot joe_mo points (circles)
        ax.scatter(joe_mo_vals, joe_mo_tests, c='#2E86AB', marker='o', s=250, 
                  alpha=0.8, edgecolors='black', linewidths=2, label='joe_mo Test', zorder=3)
        
        # Plot doc points (squares)
        ax.scatter(doc_vals, doc_tests, c='#A23B72', marker='s', s=250, 
                  alpha=0.8, edgecolors='black', linewidths=2, label='doc Test', zorder=3)
        
        # Collect label data for smart positioning
        for r in self.results:
            val_acc = r['final_val_acc'] if r['final_val_acc'] is not None else 0
            
            if 'joe_mo' in r['test1_name']:
                test_scores = [r['test1_acc'], r['test2_acc']]
            else:
                test_scores = [r['test2_acc'], r['test1_acc']]
            
            ax.plot([val_acc, val_acc], [min(test_scores), max(test_scores)], 
                   'k:', alpha=0.3, linewidth=1.5, zorder=1)
            
            mid_point = (min(test_scores) + max(test_scores)) / 2
            label_positions.append((val_acc, mid_point, r['name'].upper()))
        
        # Sort labels by y-position and adjust for overlaps
        label_positions.sort(key=lambda x: x[1])
        min_spacing = 4
        
        for i in range(len(label_positions)):
            if i > 0:
                prev_y = label_positions[i-1][1]
                curr_y = label_positions[i][1]
                if curr_y - prev_y < min_spacing:
                    label_positions[i] = (label_positions[i][0], prev_y + min_spacing, label_positions[i][2])
        
        # Draw labels with better positioning
        for val_acc, adjusted_y, name in label_positions:
            ax.annotate(name, xy=(val_acc, adjusted_y), xytext=(-15, 0), 
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       ha='right', va='center', 
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='gray', alpha=0.85),
                       zorder=4)
        
        # Add diagonal line (perfect prediction)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2.5, label='Perfect Prediction')
        
        # Configure axes
        ax.set_xlim(min_val - 5, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel('Validation Accuracy (%)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=15, fontweight='bold')
        ax.set_title('Validation vs Test Accuracy', fontsize=18, fontweight='bold', pad=20)
        ax.grid(alpha=0.35, linestyle='--', linewidth=1)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', markersize=13, 
                   label='joe_mo Test', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#A23B72', markersize=13, 
                   label='doc Test', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], color='k', linestyle='--', linewidth=2.5, label='Perfect Prediction')
        ]
        ax.legend(handles=legend_elements, fontsize=13, loc='upper left', framealpha=0.95, 
                 edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        plot_path = self.output_folder / 'validation_vs_test.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_generalization_gap(self):
        """Plot generalization gap (train vs test performance)."""
        print(f"\nGenerating generalization gap plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        exp_names = []
        train_scores = []
        test_scores = []
        gaps = []
        
        for r in self.results:
            exp_names.append(r['name'].replace('_', '\n').upper())
            train_scores.append(r['final_train_acc'])
            
            avg_test = (r['test1_acc'] + r['test2_acc']) / 2
            test_scores.append(avg_test)
            gaps.append(r['final_train_acc'] - avg_test)
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train Acc', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Avg Test Acc', color='#e67e22', alpha=0.8)
        
        for i, gap in enumerate(gaps):
            ax.plot([x[i] - width/2, x[i] + width/2], [train_scores[i], test_scores[i]], 
                   'k--', alpha=0.5, linewidth=1)
            ax.text(x[i], max(train_scores[i], test_scores[i]) + 2, 
                   f'Gap: {gap:.1f}%', ha='center', fontsize=8, color='red')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=9, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'generalization_gap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_training_curves(self):
        """Plot training curves for all experiments."""
        print(f"\nGenerating training curves...")
        
        # Calculate grid size based on number of experiments
        n_exp = len(self.results)
        ncols = 4 if n_exp > 6 else 3
        nrows = (n_exp + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        if n_exp == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, r in enumerate(self.results):
            ax = axes[i]
            history = r['history']
            
            # Handle both train_acc (finetune_birdclef) and train_accuracy (train_models)
            train_key = 'train_accuracy' if 'train_accuracy' in history else 'train_acc'
            val_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
            
            epochs = range(1, len(history[train_key]) + 1)
            
            # Convert to percentage if needed
            train_data = [v * 100 if v < 1 else v for v in history[train_key]]
            ax.plot(epochs, train_data, 'b-', label='Train', linewidth=2)
            
            if history[val_key][0] is not None:
                val_data = [(v * 100 if v < 1 else v) if v is not None else 0 for v in history[val_key]]
                ax.plot(epochs, val_data, 'r-', label='Val', linewidth=2)
            
            ax.axhline(y=r['test1_acc'], color='g', linestyle='--', alpha=0.7, label=f'Test {r["test1_name"]}')
            ax.axhline(y=r['test2_acc'], color='orange', linestyle='--', alpha=0.7, label=f'Test {r["test2_name"]}')
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(r.get('description', r.get('name', 'Unknown')), fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        # Hide any unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_heatmap(self):
        """Generate heatmap of train dataset vs test/validation accuracy."""
        print(f"\nGenerating heatmap...")
        
        if not self.results:
            return
        
        # Create matrix: each experiment is a row, columns are metrics
        n_experiments = len(self.results)
        metric_names = ['Train Acc', 'Val Acc', 'joe_mo Test', 'doc Test']
        matrix = np.zeros((n_experiments, 4))
        exp_names = []
        
        for i, r in enumerate(self.results):
            clean_name = r['name'].replace('_', ' ').title()
            exp_names.append(clean_name)
            
            # Get scores - check for 'avianz' in test name (avianz_split is joe_mo data)
            if 'avianz' in r['test1_name'].lower():
                joe_mo_score = r['test1_acc']
                doc_score = r['test2_acc']
            elif 'doc' in r['test1_name'].lower():
                joe_mo_score = r['test2_acc']
                doc_score = r['test1_acc']
            else:
                # Fallback for unclear naming
                joe_mo_score = r['test1_acc']
                doc_score = r['test2_acc']
            
            # Get train and validation scores
            train_score = r['final_train_acc'] if r['final_train_acc'] is not None else 0
            val_score = r['final_val_acc'] if r['final_val_acc'] is not None else 0
            
            # Fill matrix
            matrix[i, 0] = train_score
            matrix[i, 1] = val_score
            matrix[i, 2] = joe_mo_score
            matrix[i, 3] = doc_score
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)', 'pad': 0.02}, 
                    xticklabels=metric_names, yticklabels=exp_names,
                    ax=ax, linewidths=2, linecolor='white', 
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Experiment Results Overview', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metric', fontsize=14, labelpad=12, fontweight='bold')
        ax.set_ylabel('Experiment', fontsize=14, labelpad=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11, rotation=0)
        ax.tick_params(axis='x', labelsize=12)
        
        plt.tight_layout()
        plot_path = self.output_folder / 'results_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_freeze_comparison(self):
        """Compare frozen vs full fine-tuning."""
        print(f"\nGenerating freeze comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Dynamically determine which training datasets are present (exclude DANN for this plot)
        present_datasets = set()
        for r in self.results:
            if r['name'].startswith('avianz'):
                present_datasets.add('avianz')
            elif r['name'].startswith('doc'):
                present_datasets.add('doc')
            elif r['name'].startswith('combined') and not r['name'].startswith('dann'):
                present_datasets.add('combined')
        
        train_datasets = [d for d in ['avianz', 'doc', 'combined'] if d in present_datasets]
        full_avianz = []
        full_doc = []
        frozen_avianz = []
        frozen_doc = []
        
        for dataset in train_datasets:
            full_res = next((r for r in self.results if f'{dataset}_full' == r['name']), None)
            frozen_res = next((r for r in self.results if f'{dataset}_frozen' == r['name']), None)
            
            if full_res:
                # joe_mo_split is AviaNZ, doc_split is DOC
                if 'joe_mo' in full_res['test1_name']:
                    full_avianz.append(full_res['test1_acc'])
                    full_doc.append(full_res['test2_acc'])
                else:
                    full_avianz.append(full_res['test2_acc'])
                    full_doc.append(full_res['test1_acc'])
            
            if frozen_res:
                # joe_mo_split is AviaNZ, doc_split is DOC
                if 'joe_mo' in frozen_res['test1_name']:
                    frozen_avianz.append(frozen_res['test1_acc'])
                    frozen_doc.append(frozen_res['test2_acc'])
                else:
                    frozen_avianz.append(frozen_res['test2_acc'])
                    frozen_doc.append(frozen_res['test1_acc'])
        
        x = np.arange(len(train_datasets))
        width = 0.35
        
        ax1.bar(x - width/2, full_avianz, width, label='Full Fine-tuning', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, frozen_avianz, width, label='Frozen Backbone', color='lightcoral', alpha=0.8)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('AviaNZ Test Set', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.capitalize() for d in train_datasets])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(x - width/2, full_doc, width, label='Full Fine-tuning', color='steelblue', alpha=0.8)
        ax2.bar(x + width/2, frozen_doc, width, label='Frozen Backbone', color='lightcoral', alpha=0.8)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('DOC Test Set', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.capitalize() for d in train_datasets])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Full Fine-tuning vs Frozen Backbone Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = self.output_folder / 'freeze_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def save_results(self):
        """Save all results to JSON, merging with existing results if present."""
        print(f"\nSaving results...")
        
        json_path = self.output_folder / 'all_results.json'
        
        # Load existing results if file exists
        existing_results = []
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get('results', [])
                print(f"  Found {len(existing_results)} existing results")
            except Exception as e:
                print(f"  Warning: Could not load existing results: {e}")
        
        # Merge results, avoiding duplicates by experiment name
        existing_names = {r['name'] for r in existing_results}
        merged_results = existing_results.copy()
        
        for new_result in self.results:
            if new_result['name'] in existing_names:
                # Replace existing result with new one
                merged_results = [r for r in merged_results if r['name'] != new_result['name']]
                merged_results.append(new_result)
                print(f"  Updated: {new_result['name']}")
            else:
                # Add new result
                merged_results.append(new_result)
                print(f"  Added: {new_result['name']}")
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'experiments': len(merged_results),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'model': self.model_path,
            'results': merged_results
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"  Saved {len(merged_results)} total results to: {json_path}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print(f"\nGenerating comprehensive report...")
        
        report_path = self.output_folder / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CROSS-DATASET TRAINING EXPERIMENTS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Epochs per experiment: {self.epochs}\n")
            f.write(f"Batch size: {self.batch_size}\n\n")
            
            f.write("="*60 + "\n")
            f.write("INDIVIDUAL EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for r in self.results:
                f.write(f"{r['description']}\n")
                f.write(f"{'-'*60}\n")
                f.write(f"  Train dataset: {r['train_dataset']}\n")
                f.write(f"  Freeze backbone: {r['freeze_backbone']}\n")
                f.write(f"  Train accuracy: {r['final_train_acc']:.2f}%\n")
                if r['final_val_acc'] is not None:
                    f.write(f"  Val accuracy: {r['final_val_acc']:.2f}%\n")
                f.write(f"  Test {r['test1_name']}: {r['test1_acc']:.2f}%\n")
                f.write(f"  Test {r['test2_name']}: {r['test2_acc']:.2f}%\n")
                f.write(f"  Avg test accuracy: {(r['test1_acc'] + r['test2_acc'])/2:.2f}%\n")
                f.write(f"  Generalization gap: {r['final_train_acc'] - (r['test1_acc'] + r['test2_acc'])/2:.2f}%\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*60 + "\n\n")
            
            best_overall = max(self.results, key=lambda r: (r['test1_acc'] + r['test2_acc'])/2)
            f.write(f"Best overall performance:\n")
            f.write(f"  {best_overall['description']}\n")
            f.write(f"  Avg test accuracy: {(best_overall['test1_acc'] + best_overall['test2_acc'])/2:.2f}%\n\n")
            
            best_generalization = min(self.results, key=lambda r: r['final_train_acc'] - (r['test1_acc'] + r['test2_acc'])/2)
            f.write(f"Best generalization (smallest gap):\n")
            f.write(f"  {best_generalization['description']}\n")
            f.write(f"  Gap: {best_generalization['final_train_acc'] - (best_generalization['test1_acc'] + best_generalization['test2_acc'])/2:.2f}%\n\n")
            
            full_results = [r for r in self.results if not r['freeze_backbone']]
            frozen_results = [r for r in self.results if r['freeze_backbone']]
            
            if full_results and frozen_results:
                full_avg = np.mean([(r['test1_acc'] + r['test2_acc'])/2 for r in full_results])
                frozen_avg = np.mean([(r['test1_acc'] + r['test2_acc'])/2 for r in frozen_results])
                
                f.write(f"Full fine-tuning vs Frozen backbone:\n")
                f.write(f"  Full avg test accuracy: {full_avg:.2f}%\n")
                f.write(f"  Frozen avg test accuracy: {frozen_avg:.2f}%\n")
                f.write(f"  Difference: {full_avg - frozen_avg:.2f}%\n\n")
            
            f.write("="*60 + "\n")
            f.write("GENERATED FILES\n")
            f.write("="*60 + "\n\n")
            f.write("  - summary_table.csv (tabular results)\n")
            f.write("  - heatmap_full.png (full fine-tuning performance matrix)\n")
            f.write("  - heatmap_frozen.png (frozen backbone performance matrix)\n")
            f.write("  - validation_vs_test_full.png (full fine-tuning: validation as test predictor)\n")
            f.write("  - validation_vs_test_frozen.png (frozen backbone: validation as test predictor)\n")
            f.write("  - all_results.json (detailed JSON)\n")
            f.write("  - report.txt (this file)\n\n")
        
        print(f"  Saved to: {report_path}")
        
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")
    
    def plot_confusion_matrices(self):
        """Plot per-experiment pairwise confusion matrices for multilabel."""
        try:
            import seaborn as sns
            from sklearn.metrics import multilabel_confusion_matrix
        except ImportError:
            print("  Skipping confusion matrices (pip install seaborn scikit-learn)")
            return

        print(f"\n{'='*60}")
        print("Plotting pairwise confusion matrices (multilabel)...")
        print(f"{'='*60}")

        def load_gt_multilabel(test_folder):
            """Load ground truth as multi-hot vectors."""
            labels_path = Path(test_folder) / 'labels.json'
            if not labels_path.exists():
                return None, None
            with open(labels_path) as f:
                data = json.load(f)
            
            all_classes = sorted(data['categories'])
            gt = {}
            for item in data['files']:
                fname = item['filename']
                classes = item.get('class_names', [])
                # Multi-hot vector
                vec = [1 if c in classes else 0 for c in all_classes]
                gt[fname] = vec
            return gt, all_classes

        def load_predictions_multilabel(csv_path, all_classes):
            """Load predictions as multi-hot vectors (threshold at 0.5)."""
            df = pd.read_csv(csv_path)
            preds = {}
            for _, row in df.iterrows():
                fname = row['row_id']
                vec = []
                for c in all_classes:
                    if c in df.columns:
                        vec.append(1 if row[c] >= 0.5 else 0)
                    else:
                        vec.append(0)
                preds[fname] = vec
            return preds

        def build_pairwise_confusion(y_true, y_pred, class_names):
            """
            Build pairwise confusion matrix.
            C[i,j] = how often class i is in ground truth while class j is incorrectly predicted.
            """
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            K = len(class_names)
            confusion = np.zeros((K, K))
            
            for n in range(len(y_true)):
                true_labels = np.where(y_true[n] == 1)[0]
                pred_labels = np.where(y_pred[n] == 1)[0]
                
                for i in true_labels:
                    for j in pred_labels:
                        if i != j:  # Only confusions, not correct predictions
                            confusion[i, j] += 1
            
            # Normalize by row (how often each true class appears)
            row_sums = y_true.sum(axis=0, keepdims=True).T
            confusion_norm = np.where(row_sums > 0, confusion / row_sums, 0.0)
            
            return confusion, confusion_norm

        def save_confusion_matrix(confusion_norm, class_names, title, out_png):
            """Save normalized pairwise confusion heatmap."""
            n = len(class_names)
            cell = max(0.55, min(1.2, 10.0 / n))
            fig, ax = plt.subplots(figsize=(n * cell + 2, n * cell + 1.5))
            
            sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=ax, vmin=0, vmax=1, linewidths=0.3, linecolor='whitesmoke')
            ax.set_xlabel('Incorrectly Predicted', fontsize=11)
            ax.set_ylabel('True Class', fontsize=11)
            ax.set_title(f'{title}\n(pairwise confusion: when row class present, col class wrongly predicted)', 
                        fontsize=11, pad=10)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  Saved: {out_png}")

        gt_doc, classes_doc = load_gt_multilabel(self.doc_test)
        gt_avianz, classes_avianz = load_gt_multilabel(self.avianz_test)
        
        if gt_doc is None:
            print(f"  WARNING: no labels.json in {self.doc_test} — skipping DOC confusion matrices")
        if gt_avianz is None:
            print(f"  WARNING: no labels.json in {self.avianz_test} — skipping AviaNZ confusion matrices")

        for csv_path in sorted(self.output_folder.glob('*/predictions_*.csv')):
            exp_name = csv_path.parent.name
            csv_stem = csv_path.stem
            is_doc = 'doc' in csv_stem.lower()
            
            gt = gt_doc if is_doc else gt_avianz
            classes = classes_doc if is_doc else classes_avianz
            
            if gt is None or classes is None:
                continue

            preds = load_predictions_multilabel(csv_path, classes)
            
            # Align filenames
            common_files = sorted(set(gt.keys()) & set(preds.keys()))
            if not common_files:
                print(f"  ERROR: no matching files in {csv_path.name}")
                continue
            
            y_true = [gt[f] for f in common_files]
            y_pred = [preds[f] for f in common_files]
            
            confusion, confusion_norm = build_pairwise_confusion(y_true, y_pred, classes)
            
            train_on = 'AviaNZ' if 'joe_mo' in exp_name else 'DOC'
            test_on = 'AviaNZ' if not is_doc else 'DOC'
            title = f'Train: {train_on}  →  Test: {test_on} ({exp_name})'
            out_png = csv_path.parent / f'confusion_{csv_stem}.png'
            
            save_confusion_matrix(confusion_norm, classes, title, out_png)

    def run(self):
        """Run complete experiment pipeline."""
        self.run_all_experiments()
        
        if len(self.results) > 0:
            print(f"\n{'='*60}")
            print(f"Generating visualizations and reports...")
            print(f"{'='*60}")
            
            self.save_results()
            
            # Reload ALL results (including previously saved ones) for plotting
            json_path = self.output_folder / 'all_results.json'
            if json_path.exists():
                with open(json_path, 'r') as f:
                    all_data = json.load(f)
                    self.results = all_data['results']
                print(f"\nLoaded {len(self.results)} total experiments for visualization")
            
            self.generate_summary_table()
            self.plot_test_accuracy_comparison()
            self.plot_heatmap()
            self.plot_validation_vs_test()
            self.generate_report()
            self.plot_confusion_matrices()
            
            print(f"\n{'='*60}")
            print(f"✓ ALL COMPLETE!")
            print(f"{'='*60}")
            print(f"\nResults saved to: {self.output_folder}")
            print(f"\nGenerated files:")
            print(f"  - summary_table.csv")
            print(f"  - test_accuracy_comparison.png")
            print(f"  - results_heatmap.png")
            print(f"  - validation_vs_test.png")
            print(f"  - all_results.json")
            print(f"  - report.txt")
            print(f"  - <exp>/confusion_*.png (per-experiment confusion matrices)")
            print(f"\nUse these images in your PDF/paper!")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive cross-dataset training experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--avianz-train', required=True,
                       help='Path to AviaNZ training dataset')
    parser.add_argument('--avianz-test', required=True,
                       help='Path to AviaNZ test dataset')
    parser.add_argument('--doc-train', required=True,
                       help='Path to DOC training dataset')
    parser.add_argument('--doc-test', required=True,
                       help='Path to DOC test dataset')
    parser.add_argument('--output', default='results/cross_dataset_experiments',
                       help='Output folder (default: results/cross_dataset_experiments)')
    parser.add_argument('--model', default='BirdClefModels/model_fold0.pth',
                       help='Path to pretrained BirdCLEF model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per experiment (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lambda-domain', type=float, default=0.3,
                       help='DANN domain loss weight (default: 0.3)')
    parser.add_argument('--noise-folder', default=None,
                       help='Optional noise folder for augmentation mixing (unlabeled background). Used by --noise in train_models.py / finetune_birdclef.py')

    parser.add_argument('--noise', type=float, default=None,
                       help='Expected noise mixing ratio for augmentation (e.g., 0.2 = 20%% expected noise). Only used if provided.')
    parser.add_argument('--noise-mode', type=str, default=None, choices=['full', 'background', 'both'],
                       help='[BirdClef] Noise extraction mode for augmentation mixing. Only used if provided.')
    parser.add_argument('--noise-as-samples', action='store_true',
                       help='[AST] Add noise spectrograms as extra all-zero-label training samples (requires --noise-folder)')
    parser.add_argument('--background-prob', type=float, default=None,
                       help='[BirdClef] Probability of replacing a training sample with its background (labels zeroed). Only used if provided.')
    parser.add_argument('--mixup', type=float, default=None,
                       help='Mixup alpha (0 disables). Only used if provided.')
    parser.add_argument('--spec-transform', type=str, default='Log', choices=['Log', 'PCEN', 'Box-Cox', 'None'],
                       help='Spectrogram transform (default: Log = standard log scaling). PCEN is robust to amplitude variation.')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply background normalization (--normalize flag in finetune_birdclef.py)')
    parser.add_argument('--normalize-no-median', action='store_true',
                       help='Skip median filter in normalization (for ablation studies)')
    parser.add_argument('--experiment-suffix', default='',
                       help='Optional suffix to append to experiment names (e.g., for noise levels)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run of experiments even if results already exist')
    
    args = parser.parse_args()
    
    for path in [args.avianz_train, args.avianz_test, args.doc_train, args.doc_test]:
        if not os.path.exists(path):
            print(f"ERROR: Path not found: {path}")
            return
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return
    
    if args.noise_folder and not os.path.exists(args.noise_folder):
        print(f"WARNING: Noise folder not found: {args.noise_folder}")
        print(f"         DANN will fallback to using source as target (not recommended)")
        args.noise_folder = None
    
    experiments = CrossDatasetExperiments(
        avianz_train=args.avianz_train,
        avianz_test=args.avianz_test,
        doc_train=args.doc_train,
        doc_test=args.doc_test,
        output_folder=args.output,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_domain=args.lambda_domain,
        noise_folder=args.noise_folder,
        force_rerun=args.force,
        noise=args.noise,
        noise_mode=args.noise_mode,
        background_prob=args.background_prob,
        noise_as_samples=args.noise_as_samples,
        mixup=args.mixup,
        spec_transform=args.spec_transform,
        normalize=args.normalize,
        normalize_no_median=args.normalize_no_median,
        experiment_suffix=args.experiment_suffix
    )
    
    experiments.run()


if __name__ == '__main__':
    main()
