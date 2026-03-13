#!/usr/bin/env python3
"""
Comprehensive cross-dataset training and evaluation experiments.

This script runs multiple training scenarios:
1. Train on dataset1 (avianz), test on both
2. Train on dataset2 (doc), test on both  
3. Train on combined, test on both

Each with:
- Full fine-tuning (all layers)
- Frozen backbone (only last layer)

Generates comparison plots and tables for publication.

Usage:
    python run_cross_dataset_experiments.py \\
        --avianz-train /path/to/avianz_train \\
        --avianz-test /path/to/avianz_test \\
        --doc-train /path/to/doc_train \\
        --doc-test /path/to/doc_test \\
        --combined-train /path/to/combined_train \\
        --output results/experiments
"""

import argparse
import os
import sys
import json
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
                 combined_train, output_folder, model_path, epochs=10, batch_size=32):
        self.avianz_train = avianz_train
        self.avianz_test = avianz_test
        self.doc_train = doc_train
        self.doc_test = doc_test
        self.combined_train = combined_train
        self.output_folder = Path(output_folder)
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.experiments = [
            {
                'name': 'avianz_full',
                'train': avianz_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'finetune',
                'description': 'Train on AviaNZ (full fine-tuning)'
            },
            {
                'name': 'avianz_frozen',
                'train': avianz_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': True,
                'type': 'finetune',
                'description': 'Train on AviaNZ (frozen backbone)'
            },
            {
                'name': 'doc_full',
                'train': doc_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': False,
                'type': 'finetune',
                'description': 'Train on DOC (full fine-tuning)'
            },
            {
                'name': 'doc_frozen',
                'train': doc_train,
                'test1': doc_test,
                'test2': avianz_test,
                'freeze': True,
                'type': 'finetune',
                'description': 'Train on DOC (frozen backbone)'
            },
            {
                'name': 'combined_full',
                'train': combined_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'finetune',
                'description': 'Train on Combined (full fine-tuning)'
            },
            {
                'name': 'combined_frozen',
                'train': combined_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': True,
                'type': 'finetune',
                'description': 'Train on Combined (frozen backbone)'
            },
            {
                'name': 'dann_combined_full',
                'source': avianz_train,
                'target': doc_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': False,
                'type': 'dann',
                'description': 'DANN on Combined (full fine-tuning)'
            },
            {
                'name': 'dann_combined_frozen',
                'source': avianz_train,
                'target': doc_train,
                'test1': avianz_test,
                'test2': doc_test,
                'freeze': True,
                'type': 'dann',
                'description': 'DANN on Combined (frozen backbone)'
            }
        ]
        
        self.results = []
        
        print(f"{'='*60}")
        print(f"Cross-Dataset Training Experiments")
        print(f"{'='*60}")
        print(f"AviaNZ train: {avianz_train}")
        print(f"AviaNZ test: {avianz_test}")
        print(f"DOC train: {doc_train}")
        print(f"DOC test: {doc_test}")
        print(f"Combined train: {combined_train}")
        print(f"Output: {output_folder}")
        print(f"Model: {model_path}")
        print(f"Epochs: {epochs}")
        print(f"\nTotal experiments: {len(self.experiments)}")
    
    def run_experiment(self, exp):
        """Run a single training experiment (finetune or DANN)."""
        if exp['type'] == 'dann':
            return self.run_dann_experiment(exp)
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
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Use parent folder name for better identification
            test1_name = Path(exp['test1']).parent.name
            test2_name = Path(exp['test2']).parent.name
            
            test1_acc = self._extract_test_accuracy(result.stdout, test1_name)
            test2_acc = self._extract_test_accuracy(result.stdout, test2_name)
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'train_dataset': Path(exp['train']).name,
                'freeze_backbone': exp['freeze'],
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1] if history['val_acc'][-1] is not None else None,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': max([v for v in history['val_acc'] if v is not None], default=None),
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
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Experiment failed!")
            print(f"Error: {e}")
            print(f"Output: {e.output}")
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
        
        exp_output = self.output_folder / exp['name']
        exp_output.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            'train_domain_adaptation.py',
            exp['source'],
            exp['target'],
            str(exp_output),
            '--architecture', 'regnety_008',
            '--pretrained', self.model_path,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--lambda-domain', '1.0',
            '--test-folder', exp['test1'],
            '--test-folder2', exp['test2']
        ]
        
        if exp['freeze']:
            cmd.append('--freeze-backbone')
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            history_path = exp_output / 'training_history.json'
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            test1_name = Path(exp['test1']).parent.name
            test2_name = Path(exp['test2']).parent.name
            
            test1_acc = self._extract_test_accuracy(result.stdout, test1_name)
            test2_acc = self._extract_test_accuracy(result.stdout, test2_name)
            
            # Convert DANN history to standard format
            val_acc = history.get('target_val_acc', [0] * len(history.get('train_class_acc', [])))
            
            exp_result = {
                'name': exp['name'],
                'description': exp['description'],
                'train_dataset': 'combined_train',
                'freeze_backbone': exp['freeze'],
                'final_train_acc': history['train_class_acc'][-1] if history.get('train_class_acc') else 0,
                'final_val_acc': val_acc[-1] if val_acc else None,
                'test1_name': test1_name,
                'test1_acc': test1_acc,
                'test2_name': test2_name,
                'test2_acc': test2_acc,
                'best_val_acc': max(val_acc) if val_acc else None,
                'history': {
                    'train_acc': history.get('train_class_acc', []),
                    'val_acc': val_acc,
                    'train_loss': history.get('train_class_loss', []),
                    'val_loss': [],
                    'train_macro_f1': [],
                    'val_macro_f1': [],
                    'train_micro_f1': [],
                    'val_micro_f1': [],
                    'train_exact_match': [],
                    'val_exact_match': [],
                    'train_bit_acc': [],
                    'val_bit_acc': []
                },
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
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Experiment failed!")
            print(f"Error: {e}")
            print(f"Output: {e.output}")
            return None
    
    def _extract_test_accuracy(self, output, test_name):
        """Extract test accuracy from training output."""
        lines = output.split('\n')
        for line in lines:
            if test_name in line and 'Accuracy:' in line:
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
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        exp_names = []
        avianz_scores = []
        doc_scores = []
        colors = []
        
        for r in self.results:
            exp_names.append(r['name'].replace('_', '\n'))
            
            # joe_mo_split is AviaNZ, doc_split is DOC
            if 'joe_mo' in r['test1_name']:
                avianz_scores.append(r['test1_acc'])
                doc_scores.append(r['test2_acc'])
            else:
                avianz_scores.append(r['test2_acc'])
                doc_scores.append(r['test1_acc'])
            
            if r['freeze_backbone']:
                colors.append('lightcoral')
            else:
                colors.append('steelblue')
        
        x = np.arange(len(exp_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, avianz_scores, width, label='AviaNZ Test', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, doc_scores, width, label='DOC Test', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Cross-Dataset Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'test_accuracy_comparison.png'
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
            exp_names.append(r['name'].replace('_', '\n'))
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
        ax.set_xticklabels(exp_names, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'generalization_gap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_training_curves(self):
        """Plot training curves for all experiments."""
        print(f"\nGenerating training curves...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_heatmap(self):
        """Generate 3x2 heatmap of train dataset vs test dataset accuracy."""
        print(f"\nGenerating heatmap...")
        
        # Create matrices for full and frozen
        train_datasets = ['AviaNZ', 'DOC', 'Combined']
        test_datasets = ['AviaNZ Test', 'DOC Test']
        
        full_matrix = np.zeros((3, 2))
        frozen_matrix = np.zeros((3, 2))
        
        mapping = {
            'avianz': 0,
            'doc': 1,
            'combined': 2
        }
        
        for r in self.results:
            # Determine train dataset index
            train_idx = None
            for key, idx in mapping.items():
                if r['name'].startswith(key):
                    train_idx = idx
                    break
            
            if train_idx is None:
                continue
            
            # Get scores (joe_mo is AviaNZ, doc_split is DOC)
            if 'joe_mo' in r['test1_name']:
                avianz_score = r['test1_acc']
                doc_score = r['test2_acc']
            else:
                avianz_score = r['test2_acc']
                doc_score = r['test1_acc']
            
            # Fill matrices
            if r['freeze_backbone']:
                frozen_matrix[train_idx, 0] = avianz_score
                frozen_matrix[train_idx, 1] = doc_score
            else:
                full_matrix[train_idx, 0] = avianz_score
                full_matrix[train_idx, 1] = doc_score
        
        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Full fine-tuning heatmap
        sns.heatmap(full_matrix, annot=True, fmt='.1f', cmap='RdYlGn', 
                    vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, 
                    xticklabels=test_datasets, yticklabels=train_datasets,
                    ax=ax1, linewidths=1, linecolor='gray')
        ax1.set_title('Full Fine-tuning', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Test Dataset', fontsize=12, labelpad=10)
        ax1.set_ylabel('Training Dataset', fontsize=12, labelpad=10)
        
        # Frozen backbone heatmap
        sns.heatmap(frozen_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, 
                    xticklabels=test_datasets, yticklabels=train_datasets,
                    ax=ax2, linewidths=1, linecolor='gray')
        ax2.set_title('Frozen Backbone', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Test Dataset', fontsize=12, labelpad=10)
        ax2.set_ylabel('Training Dataset', fontsize=12, labelpad=10)
        
        plt.suptitle('Cross-Dataset Performance Heatmap', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plot_path = self.output_folder / 'heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def plot_freeze_comparison(self):
        """Compare frozen vs full fine-tuning."""
        print(f"\nGenerating freeze comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        train_datasets = ['avianz', 'doc', 'combined']
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
            f.write("  - heatmap.png (cross-dataset performance matrix)\n")
            f.write("  - test_accuracy_comparison.png (bar chart)\n")
            f.write("  - generalization_gap.png (train vs test)\n")
            f.write("  - training_curves.png (all training curves)\n")
            f.write("  - freeze_comparison.png (frozen vs full)\n")
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
            self.plot_heatmap()
            self.plot_test_accuracy_comparison()
            self.plot_generalization_gap()
            self.plot_training_curves()
            self.plot_freeze_comparison()
            self.generate_report()
            
            print(f"\n{'='*60}")
            print(f"✓ ALL COMPLETE!")
            print(f"{'='*60}")
            print(f"\nResults saved to: {self.output_folder}")
            print(f"\nGenerated files:")
            print(f"  - summary_table.csv")
            print(f"  - heatmap.png")
            print(f"  - test_accuracy_comparison.png")
            print(f"  - generalization_gap.png")
            print(f"  - training_curves.png")
            print(f"  - freeze_comparison.png")
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
    parser.add_argument('--combined-train', required=True,
                       help='Path to combined training dataset')
    parser.add_argument('--output', default='results/cross_dataset_experiments',
                       help='Output folder (default: results/cross_dataset_experiments)')
    parser.add_argument('--model', default='BirdClefModels/model_fold0.pth',
                       help='Path to pretrained BirdCLEF model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per experiment (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    for path in [args.avianz_train, args.avianz_test, args.doc_train, 
                 args.doc_test, args.combined_train]:
        if not os.path.exists(path):
            print(f"ERROR: Path not found: {path}")
            return
    
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return
    
    experiments = CrossDatasetExperiments(
        avianz_train=args.avianz_train,
        avianz_test=args.avianz_test,
        doc_train=args.doc_train,
        doc_test=args.doc_test,
        combined_train=args.combined_train,
        output_folder=args.output,
        model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    experiments.run()


if __name__ == '__main__':
    main()
