#!/usr/bin/env python3
"""
Simplified cross-dataset experiments for domain adaptation testing.

Runs 8 core experiments:
1. joe_mo baseline (no tricks)
2. doc baseline (no tricks)
3. DANN joe_mo→doc
4. DANN doc→joe_mo
5. MMD joe_mo→doc
6. MMD doc→joe_mo
7. joe_mo + normalize
8. doc + normalize

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
                 combined_train, output_folder, model_path, epochs=10, batch_size=32, 
                 lambda_domain=0.1, noise_folder=None, model_type='birdclef'):
        self.avianz_train = avianz_train
        self.avianz_test = avianz_test
        self.doc_train = doc_train
        self.doc_test = doc_test
        self.combined_train = combined_train
        self.output_folder = Path(output_folder)
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_domain = lambda_domain
        self.noise_folder = noise_folder
        self.model_type = model_type
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Group joe_mo experiments together, then doc experiments
        self.experiments = [
            # joe_mo experiments
            {
                'name': 'joe_mo_baseline',
                'train': avianz_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'finetune',
                'normalize': False,
                'description': 'Baseline joe_mo'
            },
            {
                'name': 'joe_mo_normalize',
                'train': avianz_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'finetune',
                'normalize': True,
                'description': 'joe_mo + normalize'
            },
            {
                'name': 'dann_joe_mo_to_doc',
                'source': avianz_train,
                'target': doc_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'dann',
                'lambda_domain': lambda_domain,
                'description': 'DANN joe_mo→doc'
            },
            {
                'name': 'mmd_joe_mo_to_doc',
                'source': avianz_train,
                'target': doc_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'mmd',
                'lambda_domain': lambda_domain,
                'description': 'MMD joe_mo→doc'
            },
            # doc experiments
            {
                'name': 'doc_baseline',
                'train': doc_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': False,
                'type': 'finetune',
                'normalize': False,
                'description': 'Baseline doc'
            },
            {
                'name': 'doc_normalize',
                'train': doc_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': False,
                'type': 'finetune',
                'normalize': True,
                'description': 'doc + normalize'
            },
            {
                'name': 'dann_doc_to_joe_mo',
                'source': doc_train,
                'target': avianz_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': False,
                'type': 'dann',
                'lambda_domain': lambda_domain,
                'description': 'DANN doc→joe_mo'
            },
            {
                'name': 'mmd_doc_to_joe_mo',
                'source': doc_train,
                'target': avianz_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': False,
                'type': 'mmd',
                'lambda_domain': lambda_domain,
                'description': 'MMD doc→joe_mo'
            }
        ]
        
        self.results = []
        
        print(f"{'='*60}")
        print(f"Cross-Dataset Experiments (Simplified)")
        print(f"{'='*60}")
        print(f"joe_mo train: {avianz_train}")
        print(f"joe_mo test:  {avianz_test}")
        print(f"doc train:    {doc_train}")
        print(f"doc test:     {doc_test}")
        print(f"\nExperiment breakdown:")
        print(f"  joe_mo experiments:")
        print(f"    1. joe_mo baseline (no tricks)")
        print(f"    2. joe_mo + normalize")
        print(f"    3. DANN joe_mo→doc")
        print(f"    4. MMD joe_mo→doc")
        print(f"  doc experiments:")
        print(f"    5. doc baseline (no tricks)")
        print(f"    6. doc + normalize")
        print(f"    7. DANN doc→joe_mo")
        print(f"    8. MMD doc→joe_mo")
        print(f"\nTotal: {len(self.experiments)} experiments")
        print(f"{'='*60}")
    
    def extract_accuracy(self, value):
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (tuple, list)):
            return float(value[0]) if len(value) > 0 else 0.0
        if isinstance(value, dict):
            return float(value.get('macro_f1', value.get('accuracy', 0.0)))
        return 0.0
    
    def run_experiment(self, exp):
        """Run a single training experiment (finetune, DANN, or MMD)."""
        if exp['type'] == 'dann':
            return self.run_dann_experiment(exp)
        elif exp['type'] == 'mmd':
            return self.run_mmd_experiment(exp)
        else:
            return self.run_finetune_experiment(exp)
    
    def run_finetune_experiment(self, exp):
        """Run a single fine-tuning experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp['description']}")
        print(f"Train on: {exp['train']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: {exp['freeze']}")
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
        if self.model_type == 'ast':
            cmd = [
                sys.executable,
                'train_models.py',
                exp['train'],
                str(exp_output),
                '--model', 'ast',
                '--epochs', str(self.epochs),
                '--batch_size', str(self.batch_size)
            ]
            
            if self.model_path and not self.model_path.endswith('model_fold0.pth'):
                cmd.extend(['--pretrained', self.model_path])
            
            if exp.get('normalize', False):
                cmd.append('--normalize')
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
                '--test-folder2', exp['test2']
            ]
            
            if exp['freeze']:
                cmd.append('--freeze-backbone')
            
            if exp.get('normalize', False):
                cmd.append('--normalize')
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ Finetune experiment failed!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Use parent folder name for better identification
            test1_name = Path(exp['test1']).parent.name
            test2_name = Path(exp['test2']).parent.name
            
            if self.model_type == 'ast':
                print(f"\nRunning test evaluation for AST model...")
                test1_acc = self._evaluate_ast_test_set(exp_output, exp['test1'], test1_name)
                test2_acc = self._evaluate_ast_test_set(exp_output, exp['test2'], test2_name)
            else:
                test1_acc = self._extract_test_accuracy(result.stdout, test1_name)
                test2_acc = self._extract_test_accuracy(result.stdout, test2_name)
            
            final_train_acc = self.extract_accuracy(history['train_acc'][-1] if history.get('train_acc') else None)
            final_val_acc = self.extract_accuracy(history['val_acc'][-1] if history.get('val_acc') and history['val_acc'][-1] is not None else None)
            best_val = max([self.extract_accuracy(v) for v in history.get('val_acc', []) if v is not None], default=None)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
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
            print(f"  Final train acc: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Final val acc: {exp_result['final_val_acc']:.2f}%")
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
        print(f"Source: {exp['source']}")
        print(f"Target: {exp['target']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: {exp['freeze']}")
        
        if self.model_type == 'ast':
            print(f"\n⚠️  DANN not supported for AST model type. Skipping {exp['name']}.")
            return None
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
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
            '--lambda-domain', str(exp.get('lambda_domain', self.lambda_domain))
        ]
        
        if exp['freeze']:
            cmd.append('--freeze-backbone')
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ DANN experiment failed!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            test1_name = Path(exp['test1']).parent.name
            test2_name = Path(exp['test2']).parent.name
            
            test1_acc = self._extract_test_accuracy(result.stdout, test1_name)
            test2_acc = self._extract_test_accuracy(result.stdout, test2_name)
            
            final_train_acc = self.extract_accuracy(history['train_acc'][-1] if history.get('train_acc') else None)
            final_val_acc = self.extract_accuracy(history['val_acc'][-1] if history.get('val_acc') and history['val_acc'][-1] is not None else None)
            best_val = max([self.extract_accuracy(v) for v in history.get('val_acc', []) if v is not None], default=None)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
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
            print(f"  Final train acc: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Final val acc: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error processing DANN results!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_mmd_experiment(self, exp):
        """Run an MMD domain adaptation experiment."""
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        print(f"Description: {exp['description']}")
        print(f"Source: {exp['source']}")
        print(f"Target: {exp['target']}")
        print(f"Test on: {exp['test1']} and {exp['test2']}")
        print(f"Freeze backbone: {exp['freeze']}")
        print(f"Lambda: {exp.get('lambda_domain', self.lambda_domain)}")
        
        if self.model_type == 'ast':
            print(f"\n⚠️  MMD not supported for AST model type. Skipping {exp['name']}.")
            return None
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
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
            '--use-mmd',
            '--target-folder', exp['target'],
            '--lambda-domain', str(exp.get('lambda_domain', self.lambda_domain))
        ]
        
        if exp['freeze']:
            cmd.append('--freeze-backbone')
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ MMD experiment failed!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"\nStderr:\n{result.stderr}")
            return None
        
        try:
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            test1_name = Path(exp['test1']).parent.name
            test2_name = Path(exp['test2']).parent.name
            
            test1_acc = self._extract_test_accuracy(result.stdout, test1_name)
            test2_acc = self._extract_test_accuracy(result.stdout, test2_name)
            
            final_train_acc = self.extract_accuracy(history['train_acc'][-1] if history.get('train_acc') else None)
            final_val_acc = self.extract_accuracy(history['val_acc'][-1] if history.get('val_acc') and history['val_acc'][-1] is not None else None)
            best_val = max([self.extract_accuracy(v) for v in history.get('val_acc', []) if v is not None], default=None)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'train_dataset': f"{Path(exp['source']).parent.name} (MMD→{Path(exp['target']).parent.name})",
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
            
            print(f"\n✓ MMD experiment complete:")
            print(f"  Final train acc: {exp_result['final_train_acc']:.2f}%")
            if exp_result['final_val_acc'] is not None:
                print(f"  Final val acc: {exp_result['final_val_acc']:.2f}%")
            print(f"  Test {test1_name}: {test1_acc:.2f}%")
            print(f"  Test {test2_name}: {test2_acc:.2f}%")
            
            return exp_result
            
        except Exception as e:
            print(f"\n❌ Error processing MMD results!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_ast_test_set(self, model_folder, test_folder, test_name):
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
        
        if self.model_type == 'ast':
            for exp in self.experiments:
                if exp.get('normalize', False):
                    cmd.append('--normalize')
                    break
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ❌ Prediction failed for {test_name}")
            if result.stderr:
                print(f"  Error: {result.stderr[:500]}")
            return 0.0
        
        accuracy = self._compute_accuracy_from_predictions(output_csv, test_folder)
        print(f"  {test_name} Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def _compute_accuracy_from_predictions(self, csv_path, test_folder):
        """Compute accuracy by comparing predictions CSV to ground truth labels.json."""
        labels_path = os.path.join(test_folder, 'labels.json')
        
        if not os.path.exists(labels_path):
            print(f"  ⚠️  labels.json not found in {test_folder}")
            return 0.0
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        categories = labels_data['categories']
        
        ground_truth = {}
        for file_info in labels_data['files']:
            filename = file_info['filename']
            class_names = file_info.get('class_names', [])
            label_idx = [categories.index(c) for c in class_names if c in categories]
            if label_idx:
                ground_truth[filename] = label_idx[0]
            else:
                ground_truth[filename] = -1
        
        predictions = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                pred_classes = []
                for cat in categories:
                    if cat in row and float(row[cat]) > 0.5:
                        pred_classes.append(categories.index(cat))
                if pred_classes:
                    predictions[filename] = pred_classes[0]
                else:
                    predictions[filename] = -1
        
        correct = 0
        total = 0
        for filename, true_label in ground_truth.items():
            if filename in predictions:
                if predictions[filename] == true_label:
                    correct += 1
                total += 1
        
        if total == 0:
            return 0.0
        
        return (correct / total) * 100.0
    
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
                print(f"Skipping remaining experiments due to failure.")
                break
        
        if len(self.results) == len(self.experiments):
            print(f"\n{'='*60}")
            print(f"✓ All experiments complete!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"⚠ Completed {len(self.results)}/{len(self.experiments)} experiments")
            print(f"{'='*60}")
    
    def generate_summary_table(self):
        """Generate summary table of all results."""
        print(f"\nGenerating summary table...")
        
        data = []
        for r in self.results:
            data.append({
                'Experiment': r['description'],
                'Train Dataset': r['train_dataset'],
                'Freeze': 'Yes' if r['freeze_backbone'] else 'No',
                'Train Acc': f"{r['final_train_acc']:.2f}",
                'Val Acc': f"{r['final_val_acc']:.2f}" if r['final_val_acc'] is not None else 'N/A',
                f'Test {r["test1_name"]}': f"{r['test1_acc']:.2f}",
                f'Test {r["test2_name"]}': f"{r['test2_acc']:.2f}"
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
        """Plot test accuracy comparison across experiments."""
        print(f"\nGenerating test accuracy comparison plot...")
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        exp_names = []
        joe_mo_scores = []
        doc_scores = []
        
        for r in self.results:
            exp_names.append(r['description'])
            
            if 'joe_mo' in r['test1_name']:
                joe_mo_scores.append(r['test1_acc'])
                doc_scores.append(r['test2_acc'])
            else:
                joe_mo_scores.append(r['test2_acc'])
                doc_scores.append(r['test1_acc'])
        
        x = np.arange(len(exp_names))
        width = 0.38
        
        bars1 = ax.bar(x - width/2, joe_mo_scores, width, label='joe_mo Test', 
                      color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, doc_scores, width, label='doc Test', 
                      color='#A23B72', alpha=0.85, edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Test Accuracy Comparison Across Experiments', fontsize=16, fontweight='bold', pad=20)
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
            
            epochs = range(1, len(history['train_acc']) + 1)
            
            ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
            if history['val_acc'][0] is not None:
                val_acc = [v if v is not None else 0 for v in history['val_acc']]
                ax.plot(epochs, val_acc, 'r-', label='Val', linewidth=2)
            
            ax.axhline(y=r['test1_acc'], color='g', linestyle='--', alpha=0.7, label=f'Test {r["test1_name"]}')
            ax.axhline(y=r['test2_acc'], color='orange', linestyle='--', alpha=0.7, label=f'Test {r["test2_name"]}')
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(r['description'], fontsize=10, fontweight='bold')
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
            
            # Get scores
            if 'joe_mo' in r['test1_name']:
                joe_mo_score = r['test1_acc']
                doc_score = r['test2_acc']
            else:
                joe_mo_score = r['test2_acc']
                doc_score = r['test1_acc']
            
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
        """Save all results to JSON."""
        print(f"\nSaving results...")
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'experiments': len(self.results),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'model': self.model_path,
            'results': self.results
        }
        
        json_path = self.output_folder / 'all_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"  Saved to: {json_path}")
    
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
    
    def run(self):
        """Run complete experiment pipeline."""
        self.run_all_experiments()
        
        if len(self.results) > 0:
            print(f"\n{'='*60}")
            print(f"Generating visualizations and reports...")
            print(f"{'='*60}")
            
            self.save_results()
            self.generate_summary_table()
            self.plot_test_accuracy_comparison()
            self.plot_heatmap()
            self.plot_validation_vs_test()
            self.generate_report()
            
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
    parser.add_argument('--combined-train', required=False, default=None,
                       help='Path to combined training dataset (OPTIONAL, not used)')
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
                       help='Noise folder for DANN target domain (unlabeled)')
    parser.add_argument('--model-type', default='birdclef', choices=['birdclef', 'ast'],
                       help='Model type to use: birdclef (fine-tune BirdCLEF CNN) or ast (train AST from scratch or pretrained). Default: birdclef')
    
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
        combined_train=args.combined_train,
        output_folder=args.output,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_domain=args.lambda_domain,
        noise_folder=args.noise_folder,
        model_type=args.model_type
    )
    
    experiments.run()


if __name__ == '__main__':
    main()
