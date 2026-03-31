#!/usr/bin/env python3
"""
Extract results from cross-dataset experiments for paper.

Processes all_results.json and groups experiments by base name,
computing mean ± std across multiple trials (seeds).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_base_name(exp_name):
    """
    Remove seed suffix from experiment name to get base name.
    
    Examples:
        avianz_baseline_Log_seed123 -> avianz_baseline_Log
        doc_dann_Log+normalize_seed456 -> doc_dann_Log+normalize
        avianz_baseline_Log_intensity0.25_seed42 -> avianz_baseline_Log_intensity0.25
    """
    # Remove _seed### suffix
    base_name = re.sub(r'_seed\d+$', '', exp_name)
    return base_name


def parse_experiment_name(exp_name):
    """
    Parse experiment name to extract key components.
    
    Returns: dict with keys: dataset, method, variant, etc.
    """
    parts = {}
    
    # Extract dataset (avianz or doc)
    if exp_name.startswith('avianz_'):
        parts['source'] = 'avianz'
        parts['target'] = 'doc'
    elif exp_name.startswith('doc_'):
        parts['source'] = 'doc'
        parts['target'] = 'avianz'
    else:
        parts['source'] = 'unknown'
        parts['target'] = 'unknown'
    
    # Extract method (baseline, dann, or AST)
    if '_ast_' in exp_name:
        parts['method'] = 'AST'
        parts['model_architecture'] = 'AST'
    elif '_dann_' in exp_name:
        parts['method'] = 'DANN'
        parts['model_architecture'] = 'CNN'
    else:
        parts['method'] = 'Baseline'
        parts['model_architecture'] = 'CNN'
    
    # Extract spectrogram transform and normalization
    # AST experiments always use Log transform
    if '_ast_' in exp_name:
        parts['transform'] = 'Log'
    elif 'Log+normalize-no-median' in exp_name:
        parts['transform'] = 'Log+normalize-no-median'
    elif 'Log+normalize' in exp_name:
        parts['transform'] = 'Log+normalize'
    elif 'Log+median-only' in exp_name:
        parts['transform'] = 'Log+median-only'
    elif 'PCEN' in exp_name:
        parts['transform'] = 'PCEN'
    elif 'Box-Cox' in exp_name:
        parts['transform'] = 'Box-Cox'
    elif '_Log' in exp_name:
        parts['transform'] = 'Log'
    else:
        parts['transform'] = 'unknown'
    
    # Extract noise intensity if present
    intensity_match = re.search(r'intensity([\d.]+)', exp_name)
    if intensity_match:
        parts['noise_intensity'] = float(intensity_match.group(1))
    else:
        parts['noise_intensity'] = None
    
    # Extract noise variety if present
    variety_match = re.search(r'variety(\d+)', exp_name)
    if variety_match:
        parts['noise_variety'] = int(variety_match.group(1))
    else:
        parts['noise_variety'] = None
    
    return parts


def compute_statistics(results_list):
    """
    Compute mean and std for metrics across multiple trials.
    
    Args:
        results_list: List of result dicts from different seeds
    
    Returns:
        dict with mean and std for each metric
    """
    if not results_list:
        return {}
    
    # Extract metrics
    val_accs = [r['best_val_acc'] * 100 for r in results_list if 'best_val_acc' in r]
    train_accs = [r['best_train_acc'] for r in results_list if 'best_train_acc' in r]
    test1_accs = [r['test1_acc'] for r in results_list if 'test1_acc' in r]
    test2_accs = [r['test2_acc'] for r in results_list if 'test2_acc' in r]
    
    stats = {
        'n_trials': len(results_list),
        'val_acc_mean': np.mean(val_accs) if val_accs else None,
        'val_acc_std': np.std(val_accs, ddof=1) if len(val_accs) > 1 else 0,
        'train_acc_mean': np.mean(train_accs) if train_accs else None,
        'train_acc_std': np.std(train_accs, ddof=1) if len(train_accs) > 1 else 0,
        'test1_acc_mean': np.mean(test1_accs) if test1_accs else None,
        'test1_acc_std': np.std(test1_accs, ddof=1) if len(test1_accs) > 1 else 0,
        'test2_acc_mean': np.mean(test2_accs) if test2_accs else None,
        'test2_acc_std': np.std(test2_accs, ddof=1) if len(test2_accs) > 1 else 0,
    }
    
    # Also get test names from first result
    if results_list:
        stats['test1_name'] = results_list[0].get('test1_name', 'unknown')
        stats['test2_name'] = results_list[0].get('test2_name', 'unknown')
    
    return stats


def calculate_asymmetry_ratios(df, config_pairs):
    """
    Calculate asymmetry ratios for pairs of experiments (A→B vs B→A).
    
    Args:
        df: DataFrame with results
        config_pairs: List of tuples (method, transform, noise_key)
    
    Returns:
        DataFrame with asymmetry calculations
    """
    rows = []
    
    for method, transform, noise_key in config_pairs:
        # Filter for this configuration
        if noise_key is None:
            mask = (df['method'] == method) & (df['transform'] == transform) & \
                   df['noise_intensity'].isna() & df['noise_variety'].isna()
        elif 'intensity' in noise_key:
            intensity = float(noise_key.split('=')[1])
            mask = (df['method'] == method) & (df['transform'] == transform) & \
                   (df['noise_intensity'] == intensity)
        elif 'variety' in noise_key:
            variety = int(noise_key.split('=')[1])
            mask = (df['method'] == method) & (df['transform'] == transform) & \
                   (df['noise_variety'] == variety)
        else:
            continue
        
        subset = df[mask]
        
        # Get both directions
        avianz_to_doc = subset[subset['source_dataset'] == 'avianz']
        doc_to_avianz = subset[subset['source_dataset'] == 'doc']
        
        if len(avianz_to_doc) == 1 and len(doc_to_avianz) == 1:
            a2d = avianz_to_doc.iloc[0]
            d2a = doc_to_avianz.iloc[0]
            
            # Calculate asymmetry ratio
            asymmetry = d2a['reduction_pct'] / a2d['reduction_pct'] if a2d['reduction_pct'] > 0 else np.inf
            
            row = {
                'method': method,
                'transform': transform,
                'noise_config': noise_key or 'None',
                'avianz_to_doc_cross_acc': a2d['cross_domain_acc'],
                'avianz_to_doc_reduction': a2d['reduction_pct'],
                'doc_to_avianz_cross_acc': d2a['cross_domain_acc'],
                'doc_to_avianz_reduction': d2a['reduction_pct'],
                'asymmetry_ratio': asymmetry,
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_noise_experiments(df, output_dir):
    """
    Create visualizations for noise augmentation experiments.
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # =========================================================================
    # PLOT 1: Noise Intensity - Cross-Domain Accuracy
    # =========================================================================
    ax = axes[0, 0]
    intensity_df = df[df['noise_intensity'].notna()].copy()
    
    if len(intensity_df) > 0:
        for source in ['avianz', 'doc']:
            subset = intensity_df[intensity_df['source_dataset'] == source].sort_values('noise_intensity')
            label = 'Waitākere→DOC' if source == 'avianz' else 'DOC→Waitākere'
            ax.errorbar(subset['noise_intensity'], subset['cross_domain_acc'], 
                       yerr=subset['cross_domain_std'], marker='o', label=label, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Noise Intensity', fontsize=12)
        ax.set_ylabel('Cross-Domain Accuracy (%)', fontsize=12)
        ax.set_title('Noise Intensity Effect on Cross-Domain Performance', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Noise Intensity - Reduction %
    # =========================================================================
    ax = axes[0, 1]
    
    if len(intensity_df) > 0:
        for source in ['avianz', 'doc']:
            subset = intensity_df[intensity_df['source_dataset'] == source].sort_values('noise_intensity')
            label = 'Waitākere→DOC' if source == 'avianz' else 'DOC→Waitākere'
            ax.errorbar(subset['noise_intensity'], subset['reduction_pct'], 
                       yerr=subset['reduction_pct_std'], marker='s', label=label, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Noise Intensity', fontsize=12)
        ax.set_ylabel('Performance Reduction (%)', fontsize=12)
        ax.set_title('Noise Intensity Effect on Domain Shift', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 3: Noise Variety - Cross-Domain Accuracy
    # =========================================================================
    ax = axes[1, 0]
    variety_df = df[df['noise_variety'].notna()].copy()
    
    if len(variety_df) > 0:
        for source in ['avianz', 'doc']:
            subset = variety_df[variety_df['source_dataset'] == source].sort_values('noise_variety')
            label = 'Waitākere→DOC' if source == 'avianz' else 'DOC→Waitākere'
            ax.errorbar(subset['noise_variety'], subset['cross_domain_acc'], 
                       yerr=subset['cross_domain_std'], marker='o', label=label, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Noise Samples', fontsize=12)
        ax.set_ylabel('Cross-Domain Accuracy (%)', fontsize=12)
        ax.set_title('Noise Variety Effect on Cross-Domain Performance', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 4: Noise Variety - Reduction %
    # =========================================================================
    ax = axes[1, 1]
    
    if len(variety_df) > 0:
        for source in ['avianz', 'doc']:
            subset = variety_df[variety_df['source_dataset'] == source].sort_values('noise_variety')
            label = 'Waitākere→DOC' if source == 'avianz' else 'DOC→Waitākere'
            ax.errorbar(subset['noise_variety'], subset['reduction_pct'], 
                       yerr=subset['reduction_pct_std'], marker='s', label=label, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Noise Samples', fontsize=12)
        ax.set_ylabel('Performance Reduction (%)', fontsize=12)
        ax.set_title('Noise Variety Effect on Domain Shift', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'noise_augmentation_analysis.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved noise augmentation plots to: {output_file}")
    
    # Also save as PNG for easier viewing
    output_file_png = output_dir / 'noise_augmentation_analysis.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {output_file_png}")
    
    plt.close()


def main():
    # Load results
    results_file = Path('experiments_matched/all_results.json')
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Looking for alternative locations...")
        
        # Try current directory
        if Path('all_results.json').exists():
            results_file = Path('all_results.json')
            print(f"Found: {results_file}")
        else:
            print("No results file found. Exiting.")
            return
    
    print(f"Loading results from: {results_file}")
    with open(results_file) as f:
        all_results = json.load(f)
    
    print(f"Total experiments loaded: {len(all_results)}")
    
    # Group by base name
    grouped = defaultdict(list)
    for result in all_results:
        if not result or 'name' not in result:
            continue
        
        base_name = extract_base_name(result['name'])
        grouped[base_name].append(result)
    
    print(f"Unique experiment configurations: {len(grouped)}")
    
    # Process each group
    rows = []
    for base_name in sorted(grouped.keys()):
        results_list = grouped[base_name]
        
        # Parse experiment name
        exp_info = parse_experiment_name(base_name)
        
        # Compute statistics
        stats = compute_statistics(results_list)
        
        # Combine into row
        row = {
            'experiment': base_name,
            'source_dataset': exp_info['source'],
            'target_dataset': exp_info['target'],
            'method': exp_info['method'],
            'model_architecture': exp_info.get('model_architecture', 'CNN'),
            'transform': exp_info['transform'],
            'noise_intensity': exp_info['noise_intensity'],
            'noise_variety': exp_info['noise_variety'],
            'n_trials': stats['n_trials'],
            'val_acc': stats['val_acc_mean'],
            'val_acc_std': stats['val_acc_std'],
            'train_acc': stats['train_acc_mean'],
            'train_acc_std': stats['train_acc_std'],
            'in_domain_acc': stats['test1_acc_mean'],
            'in_domain_std': stats['test1_acc_std'],
            'cross_domain_acc': stats['test2_acc_mean'],
            'cross_domain_std': stats['test2_acc_std'],
            'test1_name': stats.get('test1_name', ''),
            'test2_name': stats.get('test2_name', ''),
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate reduction percentage
    # NEW: Compare cross-domain performance to in-domain baseline of the TARGET dataset
    # This normalizes by target dataset difficulty, which is more fair
    # For each experiment, find the matching in-domain baseline for its target dataset
    
    df['reduction_pct'] = np.nan
    df['reduction_pct_std'] = np.nan
    df['target_in_domain_acc'] = np.nan
    df['target_in_domain_std'] = np.nan
    
    for idx, row in df.iterrows():
        # Find the in-domain baseline for this experiment's target dataset
        # This is the experiment where source_dataset == current target_dataset
        # and has matching method, transform, and noise parameters
        baseline_mask = (
            (df['source_dataset'] == row['target_dataset']) &
            (df['method'] == row['method']) &
            (df['transform'] == row['transform'])
        )
        
        # Match noise parameters
        if pd.notna(row['noise_intensity']):
            baseline_mask &= (df['noise_intensity'] == row['noise_intensity'])
        else:
            baseline_mask &= df['noise_intensity'].isna()
            
        if pd.notna(row['noise_variety']):
            baseline_mask &= (df['noise_variety'] == row['noise_variety'])
        else:
            baseline_mask &= df['noise_variety'].isna()
        
        baseline = df[baseline_mask]
        
        if len(baseline) == 1:
            target_in_domain = baseline.iloc[0]['in_domain_acc']
            target_in_domain_std = baseline.iloc[0]['in_domain_std']
            
            # Store target in-domain baseline
            df.at[idx, 'target_in_domain_acc'] = target_in_domain
            df.at[idx, 'target_in_domain_std'] = target_in_domain_std
            
            # Calculate reduction: (target_in_domain - cross_domain) / target_in_domain * 100
            df.at[idx, 'reduction_pct'] = (
                (target_in_domain - row['cross_domain_acc']) / target_in_domain * 100
            )
            
            # Error propagation for reduction
            # If Reduction = (A - B) / A, then std_red ≈ sqrt((std_B/A)^2 + ((B*std_A)/A^2)^2) * 100
            with np.errstate(divide='ignore', invalid='ignore'):
                df.at[idx, 'reduction_pct_std'] = np.sqrt(
                    (row['cross_domain_std'] / target_in_domain)**2 + 
                    ((row['cross_domain_acc'] * target_in_domain_std) / target_in_domain**2)**2
                ) * 100
    
    # =========================================================================
    # GENERATE NOISE AUGMENTATION PLOTS (before tables for better flow)
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING NOISE AUGMENTATION VISUALIZATIONS")
    print("="*70)
    
    plot_noise_experiments(df, results_file.parent)
    
    # =========================================================================
    
    # =========================================================================    # Create specific tables for paper
    # =========================================================================
    
    # TABLE 1: Normalization comparison (baseline methods)
    print("\n" + "="*70)
    print("TABLE 1: NORMALIZATION COMPARISON")
    print("="*70)
    print("\nNote: reduction_pct = (target_in_domain - cross_domain) / target_in_domain * 100")
    print("This measures how much worse cross-domain transfer is vs. the target dataset's in-domain baseline.\n")
    
    norm_df = df[
        (df['method'] == 'Baseline') & 
        (df['noise_intensity'].isna()) & 
        (df['noise_variety'].isna())
    ].copy()
    
    if len(norm_df) > 0:
        table1 = norm_df[[
            'source_dataset', 'target_dataset', 'transform',
            'val_acc', 'val_acc_std', 
            'in_domain_acc', 'in_domain_std',
            'target_in_domain_acc', 'target_in_domain_std',
            'cross_domain_acc', 'cross_domain_std', 
            'reduction_pct', 'reduction_pct_std', 'n_trials'
        ]].sort_values(['source_dataset', 'transform'])
        
        print(table1.to_string(index=False))
        
        table1_file = results_file.parent / 'table1_normalization.csv'
        table1.to_csv(table1_file, index=False)
        print(f"\n✓ Saved to: {table1_file}")
    
    # TABLE 2: DANN vs Baseline vs AST
    print("\n" + "="*70)
    print("TABLE 2: MODEL COMPARISON (DANN vs Baseline vs AST)")
    print("="*70)
    
    dann_df = df[
        (df['transform'].isin(['Log', 'Log+normalize'])) &
        (df['noise_intensity'].isna()) & 
        (df['noise_variety'].isna())
    ].copy()
    
    if len(dann_df) > 0:
        table2 = dann_df[[
            'source_dataset', 'target_dataset', 'method', 'model_architecture', 'transform',
            'val_acc', 'val_acc_std', 
            'in_domain_acc', 'in_domain_std',
            'target_in_domain_acc', 'target_in_domain_std',
            'cross_domain_acc', 'cross_domain_std', 
            'reduction_pct', 'reduction_pct_std', 'n_trials'
        ]].sort_values(['source_dataset', 'transform', 'method'])
        
        print(table2.to_string(index=False))
        
        table2_file = results_file.parent / 'table2_dann.csv'
        table2.to_csv(table2_file, index=False)
        print(f"\n✓ Saved to: {table2_file}")
    
    # TABLE 3: Noise intensity sweep
    print("\n" + "="*70)
    print("TABLE 3: NOISE INTENSITY SWEEP")
    print("="*70)
    
    intensity_df = df[df['noise_intensity'].notna()].copy()
    
    if len(intensity_df) > 0:
        table3 = intensity_df[[
            'source_dataset', 'target_dataset', 'noise_intensity',
            'val_acc', 'val_acc_std', 
            'in_domain_acc', 'in_domain_std',
            'target_in_domain_acc', 'target_in_domain_std',
            'cross_domain_acc', 'cross_domain_std', 
            'reduction_pct', 'reduction_pct_std', 'n_trials'
        ]].sort_values(['source_dataset', 'noise_intensity'])
        
        print(table3.to_string(index=False))
        
        table3_file = results_file.parent / 'table3_noise_intensity.csv'
        table3.to_csv(table3_file, index=False)
        print(f"\n✓ Saved to: {table3_file}")
    
    # TABLE 4: Noise variety sweep
    print("\n" + "="*70)
    print("TABLE 4: NOISE VARIETY SWEEP")
    print("="*70)
    
    variety_df = df[df['noise_variety'].notna()].copy()
    
    if len(variety_df) > 0:
        table4 = variety_df[[
            'source_dataset', 'target_dataset', 'noise_variety',
            'val_acc', 'val_acc_std', 
            'in_domain_acc', 'in_domain_std',
            'target_in_domain_acc', 'target_in_domain_std',
            'cross_domain_acc', 'cross_domain_std', 
            'reduction_pct', 'reduction_pct_std', 'n_trials'
        ]].sort_values(['source_dataset', 'noise_variety'])
        
        print(table4.to_string(index=False))
        
        table4_file = results_file.parent / 'table4_noise_variety.csv'
        table4.to_csv(table4_file, index=False)
        print(f"\n✓ Saved to: {table4_file}")
    
    # =========================================================================
    # TABLE 5: Paper-ready summary with reduction% and asymmetry
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 5: PAPER-READY RESULTS (with Reduction% and Asymmetry)")
    print("="*70)
    
    # Define key configurations to compare
    key_configs = [
        ('Baseline', 'Log', None),
        ('Baseline', 'Log+normalize', None),
        ('Baseline', 'Log+normalize-no-median', None),
        ('Baseline', 'PCEN', None),
        ('Baseline', 'Box-Cox', None),
        ('DANN', 'Log+normalize', None),
        ('AST', 'Log', None),
    ]
    
    asymmetry_df = calculate_asymmetry_ratios(df, key_configs)
    
    if len(asymmetry_df) > 0:
        print("\nAsymmetry Analysis:")
        print(asymmetry_df.to_string(index=False))
        
        # Create a paper-ready formatted table
        print("\n\nPAPER TABLE FORMAT:")
        print("-" * 100)
        print(f"{'Method':<25} {'AviaNZ→DOC':<30} {'DOC→AviaNZ':<30} {'Asym':<10}")
        print(f"{'':25} {'Cross-Acc':<15} {'Red%':<15} {'Cross-Acc':<15} {'Red%':<15}")
        print("-" * 100)
        
        for _, row in asymmetry_df.iterrows():
            method_name = f"{row['method']}+{row['transform']}" if row['method'] == 'DANN' else row['transform']
            a2d_acc = f"{row['avianz_to_doc_cross_acc']:.1f}"
            a2d_red = f"{row['avianz_to_doc_reduction']:.1f}"
            d2a_acc = f"{row['doc_to_avianz_cross_acc']:.1f}"
            d2a_red = f"{row['doc_to_avianz_reduction']:.1f}"
            asym = f"{row['asymmetry_ratio']:.2f}×"
            
            print(f"{method_name:<25} {a2d_acc:<15} {a2d_red:<15} {d2a_acc:<15} {d2a_red:<15} {asym:<10}")
        
        print("-" * 100)
        
        # Save to CSV
        table5_file = results_file.parent / 'table5_paper_summary.csv'
        asymmetry_df.to_csv(table5_file, index=False)
        print(f"\n✓ Saved to: {table5_file}")
    
    # =========================================================================
    # In-domain baselines
    # =========================================================================
    print("\n" + "="*70)
    print("IN-DOMAIN BASELINES (for paper)")
    print("="*70)
    
    # Get baseline Log experiments for in-domain accuracy
    baseline_log = df[(df['method'] == 'Baseline') & 
                      (df['transform'] == 'Log') & 
                      df['noise_intensity'].isna() & 
                      df['noise_variety'].isna()]
    
    if len(baseline_log) == 2:
        avianz_baseline = baseline_log[baseline_log['source_dataset'] == 'avianz'].iloc[0]
        doc_baseline = baseline_log[baseline_log['source_dataset'] == 'doc'].iloc[0]
        
        print(f"\nAviaNZ in-domain (Log baseline): {avianz_baseline['in_domain_acc']:.1f}% ± {avianz_baseline['in_domain_std']:.1f}%")
        print(f"DOC in-domain (Log baseline): {doc_baseline['in_domain_acc']:.1f}% ± {doc_baseline['in_domain_std']:.1f}%")
        print(f"\nDifference: {doc_baseline['in_domain_acc'] - avianz_baseline['in_domain_acc']:.1f}pp")
        print(f"(DOC is {doc_baseline['in_domain_acc'] / avianz_baseline['in_domain_acc']:.2f}× easier)")
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal experiment configurations: {len(df)}")
    print(f"Configurations with 3 trials: {len(df[df['n_trials'] == 3])}")
    print(f"Configurations with 1 trial: {len(df[df['n_trials'] == 1])}")
    
    print("\nBest performing configurations by cross-domain accuracy:")
    best_cross_domain = df.nlargest(10, 'cross_domain_acc')[[
        'experiment', 'source_dataset', 'method', 'model_architecture', 'transform', 
        'cross_domain_acc', 'cross_domain_std'
    ]]
    print(best_cross_domain.to_string(index=False))
    
    # =========================================================================
    # AST-specific analysis
    # =========================================================================
    print("\n" + "="*70)
    print("AST vs CNN BASELINE COMPARISON")
    print("="*70)
    
    ast_comparison = df[
        (df['method'].isin(['Baseline', 'AST'])) &
        (df['transform'] == 'Log') &
        df['noise_intensity'].isna() &
        df['noise_variety'].isna()
    ][[
        'source_dataset', 'target_dataset', 'method', 'model_architecture',
        'in_domain_acc', 'in_domain_std',
        'cross_domain_acc', 'cross_domain_std',
        'reduction_pct', 'reduction_pct_std', 'n_trials'
    ]].sort_values(['source_dataset', 'method'])
    
    if len(ast_comparison) > 0:
        print("\nDirect comparison of CNN vs AST architectures (both using Log transform):")
        print(ast_comparison.to_string(index=False))
        
        # Calculate improvement/difference
        print("\n--- AST vs CNN Baseline Differences ---")
        for source in ['avianz', 'doc']:
            ast_row = ast_comparison[(ast_comparison['source_dataset'] == source) & 
                                     (ast_comparison['method'] == 'AST')]
            cnn_row = ast_comparison[(ast_comparison['source_dataset'] == source) & 
                                     (ast_comparison['method'] == 'Baseline')]
            
            if len(ast_row) == 1 and len(cnn_row) == 1:
                ast = ast_row.iloc[0]
                cnn = cnn_row.iloc[0]
                
                in_domain_diff = ast['in_domain_acc'] - cnn['in_domain_acc']
                cross_domain_diff = ast['cross_domain_acc'] - cnn['cross_domain_acc']
                reduction_diff = ast['reduction_pct'] - cnn['reduction_pct']
                
                direction = f"{source}→{ast['target_dataset']}"
                print(f"\n{direction}:")
                print(f"  In-domain: AST {ast['in_domain_acc']:.1f}% vs CNN {cnn['in_domain_acc']:.1f}% (diff: {in_domain_diff:+.1f}pp)")
                print(f"  Cross-domain: AST {ast['cross_domain_acc']:.1f}% vs CNN {cnn['cross_domain_acc']:.1f}% (diff: {cross_domain_diff:+.1f}pp)")
                print(f"  Reduction: AST {ast['reduction_pct']:.1f}% vs CNN {cnn['reduction_pct']:.1f}% (diff: {reduction_diff:+.1f}pp)")
                
                if abs(reduction_diff) > 2:
                    if reduction_diff < 0:
                        print(f"  → AST shows BETTER cross-dataset transfer (less reduction)")
                    else:
                        print(f"  → CNN shows BETTER cross-dataset transfer (less reduction)")
        
        # Save comparison table
        ast_table_file = results_file.parent / 'table_ast_comparison.csv'
        ast_comparison.to_csv(ast_table_file, index=False)
        print(f"\n✓ Saved AST comparison to: {ast_table_file}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
