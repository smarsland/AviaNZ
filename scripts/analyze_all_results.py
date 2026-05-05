#!/usr/bin/env python3
"""
Show ALL experimental results clearly and sensibly.

Reads from two folder layouts:
1. run_matched_experiments.sh / Kaytoo → {results_dir}/*/result.json
   Name format: {dataset}_{method}_{config}_seed{N}  or  kaytoo_pretrained_seed0
2. run_experiments.sh → {viz_dir}/{model}_on_{dataset}_{transform}/
   Reads *_multilabel_report.json files directly (no result.json written there)

Usage:
    python3 scripts/analyze_all_results.py ~/results
    python3 scripts/analyze_all_results.py ~/results --viz-dir /local/scratch/freangi/visualizations
"""

import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_from_result_json(results_dir):
    """Load experiments written by run_matched_experiments.sh and evaluate_kaytoo.py."""
    results = []
    for result_file in sorted(Path(results_dir).glob('*/result.json')):
        with open(result_file) as f:
            data = json.load(f)

        name = result_file.parent.name
        parts = name.split('_')

        if data.get('type') == 'pretrained':
            model = data.get('model', 'kaytoo')
            category_map = {
                'kaytoo':  'Kaytoo (Pretrained)',
                'birdnet': 'BirdNET (Pretrained)',
            }
            results.append({
                'name': name,
                'train_dataset': 'pretrained',
                'method': 'pretrained',
                'config': model,
                'category': category_map.get(model, f'{model} (Pretrained)'),
                'seed': data.get('seed', 0),
                'test1_name': data.get('test1_name', 'unknown'),
                'test2_name': data.get('test2_name', 'unknown'),
                'test1_acc': data.get('test1_acc', np.nan),
                'test1_acc_labelled': data.get('test1_acc_labelled', np.nan),
                'test2_acc': data.get('test2_acc', np.nan),
                'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
                'status': data.get('status', 'unknown'),
            })
            continue

        train_dataset = parts[0]
        method_type = parts[1]
        config_parts = []
        seed = 0
        for i in range(2, len(parts)):
            if parts[i].startswith('seed'):
                seed = int(parts[i].replace('seed', ''))
                break
            config_parts.append(parts[i])
        config = '_'.join(config_parts)

        if 'ast' in method_type:
            category = 'AST'
            method = 'ast'
        elif method_type == 'dann':
            category = 'DANN'
            method = 'dann'
        elif 'intensity' in config:
            category = 'Noise Intensity'
            method = 'baseline'
        elif 'variety' in config:
            category = 'Noise Variety'
            method = 'baseline'
        else:
            category = 'Normalization'
            method = 'baseline'

        results.append({
            'name': name,
            'train_dataset': train_dataset,
            'method': method,
            'config': config,
            'category': category,
            'seed': seed,
            'test1_name': data.get('test1_name', 'unknown'),
            'test2_name': data.get('test2_name', 'unknown'),
            'test1_acc': data.get('test1_acc', np.nan),
            'test2_acc': data.get('test2_acc', np.nan),
            'status': data.get('status', 'unknown'),
        })
    return results


def _read_reports_from_dir(report_dir, model, row):
    """Read *_multilabel_report.json files from report_dir into row (mutates in place)."""
    for report_file in sorted(report_dir.glob('*_multilabel_report.json')):
        with open(report_file) as f:
            report = json.load(f)
        acc = report.get('exact_match_accuracy', np.nan)
        if acc is not np.nan:
            acc = acc * 100
        acc_lab = report.get('exact_match_accuracy_labelled', np.nan)
        if acc_lab is not np.nan:
            acc_lab = acc_lab * 100
        stem = report_file.stem.replace('_multilabel_report', '')
        if '_model' in stem:
            continue  # validation set, skip
        dataset_name = re.sub(rf'^{model}_test_', '', stem)
        if row['test1_name'] is np.nan or row['test1_name'] == np.nan:
            row['test1_name'] = dataset_name
            row['test1_acc'] = acc
            row['test1_acc_labelled'] = acc_lab
        else:
            row['test2_name'] = dataset_name
            row['test2_acc'] = acc
            row['test2_acc_labelled'] = acc_lab


def load_from_viz_dir(viz_dir):
    """Load experiments written by run_experiments.sh (model_on_dataset_transform layout)."""
    results = []
    standard_pattern = re.compile(r'^(ast|regnet)_on_(avianz|doc|merged)_(.+)$')
    pseudo_pattern = re.compile(r'^(ast|regnet)_pseudo_([a-z]+)_to_([a-z]+)_(.+)_pct(\d+)$')
    for exp_dir in sorted(Path(viz_dir).iterdir()):
        if not exp_dir.is_dir():
            continue

        m = standard_pattern.match(exp_dir.name)
        if m:
            model, train_dataset, transform = m.groups()
            row = {
                'name': exp_dir.name,
                'train_dataset': train_dataset,
                'method': model,
                'config': transform,
                'category': model.upper(),
                'seed': 0,
                'test1_name': np.nan,
                'test2_name': np.nan,
                'test1_acc': np.nan,
                'test1_acc_labelled': np.nan,
                'test2_acc': np.nan,
                'test2_acc_labelled': np.nan,
                'status': 'unknown',
            }
            _read_reports_from_dir(exp_dir, model, row)
            row['status'] = 'completed' if not (np.isnan(row['test1_acc']) and np.isnan(row['test2_acc'])) else 'incomplete'
            results.append(row)
            continue

        m = pseudo_pattern.match(exp_dir.name)
        if m:
            model, source_dataset, target_dataset, transform, pct_int = m.groups()
            # Final results live in the phase3 subfolder
            final_dir = exp_dir / 'phase3_pseudo_target'
            if not final_dir.is_dir():
                continue
            row = {
                'name': exp_dir.name,
                'train_dataset': f'pseudo_{source_dataset}_to_{target_dataset}_pct{pct_int}',
                'method': model,
                'config': transform,
                'category': f'{model.upper()} Pseudo',
                'seed': 0,
                'test1_name': np.nan,
                'test2_name': np.nan,
                'test1_acc': np.nan,
                'test1_acc_labelled': np.nan,
                'test2_acc': np.nan,
                'test2_acc_labelled': np.nan,
                'status': 'unknown',
            }
            _read_reports_from_dir(final_dir, model, row)
            row['status'] = 'completed' if not (np.isnan(row['test1_acc']) and np.isnan(row['test2_acc'])) else 'incomplete'
            results.append(row)

    return results


def load_all_results(results_dir, viz_dir=None):
    results = []
    if Path(results_dir).exists():
        results.extend(load_from_result_json(results_dir))
    if viz_dir and Path(viz_dir).exists():
        results.extend(load_from_viz_dir(viz_dir))
    return pd.DataFrame(results)


def create_overview_table(df, output_dir):
    """Create CSV with ALL results"""
    cols = ['name', 'train_dataset', 'method', 'config', 'category',
            'test1_name', 'test1_acc', 'test1_acc_labelled',
            'test2_name', 'test2_acc', 'test2_acc_labelled', 'status']
    # Only include columns that actually exist (older result.json rows won't have _labelled)
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()
    df_out = df_out.sort_values(['category', 'train_dataset', 'config'])
    df_out.to_csv(output_dir / 'all_results.csv', index=False, float_format='%.2f')
    print(f"✓ all_results.csv ({len(df_out)} experiments)")


def plot_by_category(df, output_dir):
    """Plot each category separately"""
    
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].copy()
        
        if cat_df.empty:
            continue
        
        # Get unique train datasets
        train_datasets = cat_df['train_dataset'].unique()
        
        if category == 'AST':
            # Simple comparison for AST
            fig, axes = plt.subplots(1, len(train_datasets), figsize=(6*len(train_datasets), 5), squeeze=False)
            axes = axes.flatten()
            
            for idx, train_ds in enumerate(train_datasets):
                ax = axes[idx]
                data = cat_df[cat_df['train_dataset'] == train_ds]
                
                if data.empty:
                    ax.axis('off')
                    continue
                
                row = data.iloc[0]
                
                # Plot test1 and test2
                x = [0, 1]
                y = [row['test1_acc'], row['test2_acc']]
                labels = [row['test1_name'], row['test2_name']]
                
                bars = ax.bar(x, y, alpha=0.8, color=['#1f77b4', '#ff7f0e'])
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=0)
                ax.set_ylabel('Accuracy (%)', fontsize=11)
                ax.set_title(f'AST - Train: {train_ds.upper()}', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
                
                # Add values on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{category.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {category.lower().replace(' ', '_')}.png")
        
        elif category in ['Normalization', 'DANN']:
            # Bar chart comparison
            fig, axes = plt.subplots(1, len(train_datasets), figsize=(8*len(train_datasets), 6), squeeze=False)
            axes = axes.flatten()
            
            for idx, train_ds in enumerate(train_datasets):
                ax = axes[idx]
                data = cat_df[cat_df['train_dataset'] == train_ds].copy()
                
                if data.empty:
                    ax.axis('off')
                    continue
                
                # Drop rows with all NaN accuracies
                data = data.dropna(subset=['test1_acc', 'test2_acc'], how='all')
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # Sort by test1_acc
                data = data.sort_values('test1_acc')
                
                y_pos = np.arange(len(data))
                width = 0.35
                
                # Create labels
                labels = [f"{row['method']}/{row['config']}" for _, row in data.iterrows()]
                
                # Plot bars
                ax.barh(y_pos - width/2, data['test1_acc'], width, 
                       label=data.iloc[0]['test1_name'], alpha=0.8)
                ax.barh(y_pos + width/2, data['test2_acc'], width,
                       label=data.iloc[0]['test2_name'], alpha=0.8)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Accuracy (%)', fontsize=11)
                ax.set_title(f'{category} - Train: {train_ds.upper()}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(axis='x', alpha=0.3)
                ax.set_xlim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{category.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {category.lower().replace(' ', '_')}.png")
        
        elif 'Noise' in category:
            # Line plots for noise sweeps
            fig, axes = plt.subplots(1, len(train_datasets), figsize=(7*len(train_datasets), 5), squeeze=False)
            axes = axes.flatten()
            
            for idx, train_ds in enumerate(train_datasets):
                ax = axes[idx]
                data = cat_df[cat_df['train_dataset'] == train_ds].copy()
                
                if data.empty:
                    ax.axis('off')
                    continue
                
                # Extract parameter value
                if 'intensity' in category.lower():
                    data['param'] = data['config'].str.extract(r'intensity([\d.]+)').astype(float)
                    xlabel = 'Noise Intensity'
                else:
                    data['param'] = data['config'].str.extract(r'variety(\d+)').astype(int)
                    xlabel = 'Number of Noise Samples'
                    ax.set_xscale('log')
                
                data = data.sort_values('param')
                
                # Plot lines
                ax.plot(data['param'], data['test1_acc'], marker='o', label=data.iloc[0]['test1_name'], linewidth=2)
                ax.plot(data['param'], data['test2_acc'], marker='s', label=data.iloc[0]['test2_name'], linewidth=2)
                
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel('Accuracy (%)', fontsize=11)
                ax.set_title(f'{category} - Train: {train_ds.upper()}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(alpha=0.3)
                ax.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{category.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {category.lower().replace(' ', '_')}.png")


def create_summary_by_category(df, output_dir):
    """Create category-specific summaries"""
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].copy()
        
        # Add useful derived columns
        cat_df['test1_test2_diff'] = cat_df['test1_acc'] - cat_df['test2_acc']
        cat_df['avg_acc'] = (cat_df['test1_acc'] + cat_df['test2_acc']) / 2
        
        # Sort by average accuracy
        cat_df = cat_df.sort_values('avg_acc', ascending=False)
        
        # Save
        cat_df[['name', 'train_dataset', 'config', 'test1_acc', 'test2_acc', 
               'test1_test2_diff', 'avg_acc']].to_csv(
            output_dir / f'summary_{category.lower().replace(" ", "_")}.csv',
            index=False, float_format='%.2f'
        )
        print(f"✓ summary_{category.lower().replace(' ', '_')}.csv")


def create_report(df, output_dir):
    """Create a markdown report"""
    lines = []
    
    lines.append("# All Experimental Results\n\n")
    lines.append(f"**Total experiments:** {len(df)}\n")
    lines.append(f"**Completed:** {len(df[df['status']=='completed'])}\n")
    lines.append(f"**With valid data:** {len(df.dropna(subset=['test1_acc', 'test2_acc'], how='all'))}\n\n")
    
    lines.append("## Experiments by Category\n\n")
    
    for category in sorted(df['category'].unique()):
        cat_df = df[df['category'] == category]
        lines.append(f"### {category}\n\n")
        lines.append(f"**Count:** {len(cat_df)}\n\n")
        
        # Show best by train dataset
        for train_ds in sorted(cat_df['train_dataset'].unique()):
            ds_df = cat_df[cat_df['train_dataset'] == train_ds].copy()
            ds_df = ds_df.dropna(subset=['test1_acc', 'test2_acc'], how='all')
            
            if ds_df.empty:
                lines.append(f"**Train: {train_ds.upper()}** - No valid data\n\n")
                continue
            
            ds_df['avg'] = (ds_df['test1_acc'] + ds_df['test2_acc']) / 2
            best = ds_df.sort_values('avg', ascending=False).iloc[0]
            
            lines.append(f"**Train: {train_ds.upper()}** (best: `{best['config']}`)\n")
            lines.append(f"- {best['test1_name']}: {best['test1_acc']:.1f}%\n")
            lines.append(f"- {best['test2_name']}: {best['test2_acc']:.1f}%\n")
            lines.append(f"- Average: {best['avg']:.1f}%\n\n")
    
    # Missing data report
    missing_df = df[df['test1_acc'].isna() | df['test2_acc'].isna()]
    if not missing_df.empty:
        lines.append("## Experiments with Missing Data\n\n")
        for _, row in missing_df.iterrows():
            lines.append(f"- `{row['name']}` (status: {row['status']})\n")
        lines.append("\n")
    
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.writelines(lines)
    
    print(f"✓ REPORT.md")


def main():
    parser = argparse.ArgumentParser(description='Analyze all experimental results')
    parser.add_argument('results_dir',
                        help='Output directory (e.g. /local/scratch/freangi/visualizations). '
                             'Scans for both result.json and *_multilabel_report.json automatically.')
    parser.add_argument('--output', '-o', default=None, help='Output directory for analysis files')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print(" ANALYZING ALL RESULTS")
    print("="*70)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}\n")

    sns.set_style("whitegrid")

    print("Loading all results...")
    df = load_all_results(results_dir, viz_dir=results_dir)
    print(f"  {len(df)} experiments loaded")
    print(f"  {len(df.dropna(subset=['test1_acc', 'test2_acc'], how='all'))} with valid accuracies")
    print(f"  Categories: {', '.join(sorted(df['category'].unique()))}\n")
    
    print("Creating outputs...")
    create_overview_table(df, output_dir)
    plot_by_category(df, output_dir)
    create_summary_by_category(df, output_dir)
    create_report(df, output_dir)
    
    print("\n" + "="*70)
    print(" DONE")
    print("="*70)
    print(f" Output: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
