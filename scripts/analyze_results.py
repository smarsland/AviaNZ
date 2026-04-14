#!/usr/bin/env python3
"""
Domain shift experiment analysis matching paper structure.

Experiments structure:
- Train on Dataset X (avianz/doc/merged)
- Test on BOTH X (in-domain) and Y (cross-domain)
- Key metric: domain shift = cross-domain vs in-domain performance

Usage:
    python3 scripts/analyze_results.py results/
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_experiment(name: str):
    """Parse experiment name: {train_dataset}_{method}_{config}_seed{N}"""
    # Special case for AST: avianz_ast_baseline_seed42 (no config)
    ast_pattern = r'^(\w+)_(ast_baseline)_seed(\d+)$'
    ast_match = re.match(ast_pattern, name)
    if ast_match:
        train_on, method, seed = ast_match.groups()
        return {
            'train_on': train_on,
            'method': 'ast',
            'config': 'baseline',  # AST uses baseline config
            'seed': int(seed)
        }
    
    # Regular pattern: dataset_method_config_seed
    pattern = r'^(\w+)_(baseline|dann)_(.+?)_seed(\d+)$'
    match = re.match(pattern, name)
    
    if match:
        train_on, method, config, seed = match.groups()
        return {
            'train_on': train_on,
            'method': method,
            'config': config,
            'seed': int(seed)
        }
    return None


def categorize(config, method):
    """Categorize experiment type"""
    if method == 'ast':
        return 'AST'
    elif method == 'dann':
        return 'DANN'
    elif 'intensity' in config:
        return 'Noise Intensity'
    elif 'variety' in config:
        return 'Noise Variety'
    else:
        # All other baseline experiments (various normalization methods)
        return 'Normalization'


def load_results(results_dir):
    """Load all experiment results"""
    data = []
    
    for result_file in Path(results_dir).glob('*/result.json'):
        try:
            with open(result_file) as f:
                r = json.load(f)
            
            parsed = parse_experiment(result_file.parent.name)
            if not parsed:
                continue
            
            # Determine in-domain vs cross-domain
            train_on = parsed['train_on']
            test1_name = r.get('test1_name', '')
            test2_name = r.get('test2_name', '')
            test1_acc = r.get('test1_acc', np.nan)
            test2_acc = r.get('test2_acc', np.nan)
            
            # Which test set matches training domain?
            if train_on in test1_name or (train_on == 'merged' and 'doc' in test1_name):
                in_domain = test1_acc
                cross_domain = test2_acc
                test_on = test2_name.split('_')[0]  # extract dataset name
            else:
                in_domain = test2_acc
                cross_domain = test1_acc
                test_on = test1_name.split('_')[0]
            
            data.append({
                'train_on': train_on,
                'test_on': test_on,
                'method': parsed['method'],
                'config': parsed['config'],
                'seed': parsed['seed'],
                'in_domain': in_domain,
                'cross_domain': cross_domain,
                'category': categorize(parsed['config'], parsed['method'])
            })
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
    
    return pd.DataFrame(data)


def aggregate_trials(df):
    """Aggregate across seeds"""
    groups = df.groupby(['train_on', 'test_on', 'method', 'config', 'category'])
    
    agg = groups.agg({
        'in_domain': ['mean', 'std', 'count'],
        'cross_domain': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['train_on', 'test_on', 'method', 'config', 'category',
                  'in_domain_mean', 'in_domain_std', 'n_trials',
                  'cross_domain_mean', 'cross_domain_std']
    
    return agg


def calculate_reduction(agg_df):
    """Calculate reduction vs target-domain baseline"""
    # For each row, find baseline trained on test_on with same config
    for idx, row in agg_df.iterrows():
        baseline = agg_df[
            (agg_df['train_on'] == row['test_on']) &
            (agg_df['method'] == 'baseline') &
            (agg_df['config'] == row['config'])
        ]
        
        if not baseline.empty:
            target_baseline = baseline.iloc[0]['in_domain_mean']
            reduction = ((target_baseline - row['cross_domain_mean']) / target_baseline) * 100
            agg_df.at[idx, 'reduction_pct'] = reduction
            agg_df.at[idx, 'target_baseline'] = target_baseline
    
    return agg_df


def plot_normalization_methods(df, output_dir):
    """Plot like Table 1 in paper: normalization comparison"""
    norm_df = df[df['category'] == 'Normalization'].copy()
    
    if norm_df.empty:
        print("No normalization experiments found")
        return
    
    # Get unique transfer directions (excluding merged for clarity)
    transfers = norm_df[norm_df['train_on'] != 'merged'][['train_on', 'test_on']].drop_duplicates()
    
    fig, axes = plt.subplots(1, len(transfers), figsize=(8*len(transfers), 6), squeeze=False)
    axes = axes.flatten()
    
    for idx, (_, transfer) in enumerate(transfers.iterrows()):
        ax = axes[idx]
        train = transfer['train_on']
        test = transfer['test_on']
        
        data = norm_df[(norm_df['train_on'] == train) & (norm_df['test_on'] == test)]
        data = data.sort_values('cross_domain_mean')
        
        x = np.arange(len(data))
        width = 0.35
        
        # In-domain vs cross-domain bars
        ax.barh(x - width/2, data['in_domain_mean'], width, 
                xerr=data['in_domain_std'],
                label=f'In-domain ({train})', alpha=0.8, capsize=3)
        ax.barh(x + width/2, data['cross_domain_mean'], width,
                xerr=data['cross_domain_std'],
                label=f'Cross-domain ({test})', alpha=0.8, capsize=3)
        
        ax.set_yticks(x)
        ax.set_yticklabels(data['config'], fontsize=9)
        ax.set_xlabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'Train on {train.upper()} → Test on {test.upper()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'normalization_methods.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'normalization_methods.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ normalization_methods.png")


def plot_domain_shift_reduction(df, output_dir):
    """Plot reduction percentage for each normalization method"""
    norm_df = df[(df['category'] == 'Normalization') & (df['train_on'] != 'merged')].copy()
    
    if norm_df.empty or 'reduction_pct' not in norm_df.columns:
        print("No reduction data available")
        return
    
    # Create pivot: rows=config, columns=transfer direction
    norm_df['transfer'] = norm_df['train_on'] + '→' + norm_df['test_on']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    transfers = norm_df['transfer'].unique()
    configs = norm_df['config'].unique()
    
    x = np.arange(len(configs))
    width = 0.8 / len(transfers)
    
    for idx, transfer in enumerate(sorted(transfers)):
        data = norm_df[norm_df['transfer'] == transfer].set_index('config')
        values = [data.loc[c, 'reduction_pct'] if c in data.index else 0 for c in configs]
        
        ax.bar(x + idx*width, values, width, label=transfer, alpha=0.8)
    
    ax.set_xlabel('Normalization Method', fontsize=11)
    ax.set_ylabel('Performance Reduction (%)', fontsize=11)
    ax.set_title('Domain Shift: Performance Reduction vs Target Baseline', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width*(len(transfers)-1)/2)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reduction_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'reduction_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ reduction_comparison.png")


def plot_dann_comparison(df, output_dir):
    """Compare DANN vs baseline"""
    # Get DANN and matching baseline experiments
    dann_configs = df[df['method'] == 'dann'][['config', 'train_on', 'test_on']].drop_duplicates()
    
    if dann_configs.empty:
        print("No DANN experiments found")
        return
    
    fig, axes = plt.subplots(len(dann_configs), 1, 
                            figsize=(10, 4*len(dann_configs)), squeeze=False)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(dann_configs.iterrows()):
        ax = axes[idx]
        config = row['config']
        train = row['train_on']
        test = row['test_on']
        
        baseline = df[(df['method'] == 'baseline') & (df['config'] == config) & 
                     (df['train_on'] == train) & (df['test_on'] == test)]
        dann = df[(df['method'] == 'dann') & (df['config'] == config) & 
                 (df['train_on'] == train) & (df['test_on'] == test)]
        
        if baseline.empty or dann.empty:
            ax.axis('off')
            continue
        
        baseline_row = baseline.iloc[0]
        dann_row = dann.iloc[0]
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, 
              [baseline_row['in_domain_mean'], baseline_row['cross_domain_mean']], 
              width, yerr=[baseline_row['in_domain_std'], baseline_row['cross_domain_std']],
              label='Baseline', alpha=0.8, capsize=3)
        ax.bar(x + width/2,
              [dann_row['in_domain_mean'], dann_row['cross_domain_mean']],
              width, yerr=[dann_row['in_domain_std'], dann_row['cross_domain_std']],
              label='DANN', alpha=0.8, capsize=3)
        
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{config} | Train: {train.upper()} → Test: {test.upper()}',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'In-domain\n({train})', f'Cross-domain\n({test})'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dann_vs_baseline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'dann_vs_baseline.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ dann_vs_baseline.png")


def plot_noise_sweeps(df, output_dir):
    """Plot noise intensity and variety sweeps"""
    
    # Noise intensity
    intensity_df = df[df['category'] == 'Noise Intensity'].copy()
    if not intensity_df.empty:
        intensity_df['intensity'] = intensity_df['config'].str.extract(r'intensity([\d.]+)').astype(float)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, train in enumerate(intensity_df['train_on'].unique()):
            if train == 'merged':
                continue
            ax = axes[idx] if idx < 2 else None
            if ax is None:
                break
            
            data = intensity_df[intensity_df['train_on'] == train].sort_values('intensity')
            
            # Get test_on for this training set
            test = data.iloc[0]['test_on']
            
            ax.errorbar(data['intensity'], data['in_domain_mean'], 
                       yerr=data['in_domain_std'],
                       marker='o', label=f'In-domain ({train})', capsize=3, linewidth=2)
            ax.errorbar(data['intensity'], data['cross_domain_mean'],
                       yerr=data['cross_domain_std'],
                       marker='s', label=f'Cross-domain ({test})', capsize=3, linewidth=2)
            
            ax.set_xlabel('Noise Mixing Intensity', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'Train on {train.upper()}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'noise_intensity_sweep.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'noise_intensity_sweep.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ noise_intensity_sweep.png")
    
    # Noise variety
    variety_df = df[df['category'] == 'Noise Variety'].copy()
    if not variety_df.empty:
        variety_df['variety'] = variety_df['config'].str.extract(r'variety(\d+)').astype(int)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, train in enumerate(variety_df['train_on'].unique()):
            if train == 'merged':
                continue
            ax = axes[idx] if idx < 2 else None
            if ax is None:
                break
            
            data = variety_df[variety_df['train_on'] == train].sort_values('variety')
            test = data.iloc[0]['test_on']
            
            ax.errorbar(data['variety'], data['in_domain_mean'],
                       yerr=data['in_domain_std'],
                       marker='o', label=f'In-domain ({train})', capsize=3, linewidth=2)
            ax.errorbar(data['variety'], data['cross_domain_mean'],
                       yerr=data['cross_domain_std'],
                       marker='s', label=f'Cross-domain ({test})', capsize=3, linewidth=2)
            
            ax.set_xscale('log')
            ax.set_xlabel('Number of Noise Samples', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'Train on {train.upper()}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'noise_variety_sweep.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'noise_variety_sweep.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ noise_variety_sweep.png")


def create_summary_tables(df, output_dir):
    """Create CSV summaries matching paper tables"""
    
    # Table 1: Normalization methods
    norm_df = df[df['category'] == 'Normalization'].copy()
    if not norm_df.empty:
        norm_summary = norm_df[['train_on', 'test_on', 'config', 'n_trials',
                                'in_domain_mean', 'in_domain_std',
                                'cross_domain_mean', 'cross_domain_std',
                                'reduction_pct']].copy()
        norm_summary['transfer'] = norm_summary['train_on'] + '→' + norm_summary['test_on']
        norm_summary = norm_summary.sort_values(['transfer', 'cross_domain_mean'], ascending=[True, False])
        norm_summary.to_csv(output_dir / 'table_normalization.csv', index=False, float_format='%.2f')
        print(f"✓ table_normalization.csv")
    
    # Summary by category
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].copy()
        cat_df = cat_df.sort_values('cross_domain_mean', ascending=False)
        cat_df.to_csv(output_dir / f'summary_{category.lower().replace(" ", "_")}.csv',
                     index=False, float_format='%.2f')
        print(f"✓ summary_{category.lower().replace(' ', '_')}.csv")


def generate_report(df, output_dir):
    """Generate markdown report"""
    lines = []
    
    lines.append("# Domain Shift Experiment Results\n\n")
    lines.append(f"**Total experiments:** {len(df)}\n")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    lines.append("## Summary by Transfer Direction\n\n")
    
    for category in ['Normalization', 'DANN', 'Noise Intensity', 'Noise Variety']:
        cat_df = df[df['category'] == category]
        if cat_df.empty:
            continue
        
        lines.append(f"### {category}\n\n")
        
        for train in cat_df['train_on'].unique():
            for test in cat_df[cat_df['train_on'] == train]['test_on'].unique():
                data = cat_df[(cat_df['train_on'] == train) & (cat_df['test_on'] == test)]
                best = data.sort_values('cross_domain_mean', ascending=False).iloc[0]
                
                lines.append(f"**{train.upper()} → {test.upper()}**\n")
                lines.append(f"- Best config: `{best['config']}` ({best['method']})\n")
                lines.append(f"- In-domain: {best['in_domain_mean']:.1f}±{best['in_domain_std']:.1f}%\n")
                lines.append(f"- Cross-domain: {best['cross_domain_mean']:.1f}±{best['cross_domain_std']:.1f}%\n")
                if not np.isnan(best.get('reduction_pct', np.nan)):
                    lines.append(f"- Reduction: {best['reduction_pct']:.1f}%\n")
                lines.append("\n")
    
    # Key findings
    lines.append("## Key Findings\n\n")
    
    # Best overall cross-domain performance
    best_overall = df.sort_values('cross_domain_mean', ascending=False).iloc[0]
    lines.append(f"### Best Cross-Domain Performance\n")
    lines.append(f"- Config: `{best_overall['config']}` ({best_overall['method']})\n")
    lines.append(f"- Transfer: {best_overall['train_on'].upper()} → {best_overall['test_on'].upper()}\n")
    lines.append(f"- Cross-domain accuracy: {best_overall['cross_domain_mean']:.1f}±{best_overall['cross_domain_std']:.1f}%\n")
    lines.append(f"- In-domain accuracy: {best_overall['in_domain_mean']:.1f}±{best_overall['in_domain_std']:.1f}%\n\n")
    
    # Lowest reduction
    if 'reduction_pct' in df.columns:
        best_reduction = df.dropna(subset=['reduction_pct']).sort_values('reduction_pct').iloc[0]
        lines.append(f"### Lowest Performance Reduction\n")
        lines.append(f"- Config: `{best_reduction['config']}` ({best_reduction['method']})\n")
        lines.append(f"- Transfer: {best_reduction['train_on'].upper()} → {best_reduction['test_on'].upper()}\n")
        lines.append(f"- Reduction: {best_reduction['reduction_pct']:.1f}%\n\n")
    
    with open(output_dir / 'RESULTS_SUMMARY.md', 'w') as f:
        f.writelines(lines)
    
    print(f"✓ RESULTS_SUMMARY.md")


def main():
    parser = argparse.ArgumentParser(description='Domain shift experiment analysis')
    parser.add_argument('results_dir', help='Directory with experiment results')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print(" DOMAIN SHIFT ANALYSIS")
    print("="*70)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}\n")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # Load and process data
    print("Loading results...")
    df_raw = load_results(results_dir)
    print(f"  {len(df_raw)} individual experiments\n")
    
    print("Aggregating across trials...")
    df = aggregate_trials(df_raw)
    df = calculate_reduction(df)
    print(f"  {len(df)} unique configurations\n")
    
    # Generate outputs
    print("Creating visualizations...")
    plot_normalization_methods(df, output_dir)
    plot_domain_shift_reduction(df, output_dir)
    plot_dann_comparison(df, output_dir)
    plot_noise_sweeps(df, output_dir)
    
    print("\nCreating summary tables...")
    create_summary_tables(df, output_dir)
    
    print("\nGenerating report...")
    generate_report(df, output_dir)
    
    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70)
    print(f" Output: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
