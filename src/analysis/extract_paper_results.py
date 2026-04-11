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
    
    # Extract dataset (avianz/waitākere, doc, or merged)
    if exp_name.startswith('merged_'):
        parts['source'] = 'merged'
        parts['target'] = 'both'  # Merged trains on both, tests on both
    elif exp_name.startswith('avianz_'):
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
    elif 'Log+median+normalize' in exp_name:
        parts['transform'] = 'Log+median+normalize'
    elif 'Log+normalize' in exp_name:
        parts['transform'] = 'Log+normalize'
    elif 'Log+median' in exp_name:
        parts['transform'] = 'Log+median'
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
    
    # Compute per-class statistics
    # Extract all species from the first result
    if results_list and 'test1_per_class_acc' in results_list[0]:
        species_list = list(results_list[0]['test1_per_class_acc'].keys())
        
        # Compute mean and std for each species
        test1_per_class = {}
        test2_per_class = {}
        
        for species in species_list:
            # Test1 (in-domain) per-class accuracies
            species_test1_accs = [
                r['test1_per_class_acc'][species] 
                for r in results_list 
                if 'test1_per_class_acc' in r and species in r['test1_per_class_acc']
            ]
            if species_test1_accs:
                test1_per_class[species] = {
                    'mean': np.mean(species_test1_accs),
                    'std': np.std(species_test1_accs, ddof=1) if len(species_test1_accs) > 1 else 0
                }
            
            # Test2 (cross-domain) per-class accuracies
            species_test2_accs = [
                r['test2_per_class_acc'][species] 
                for r in results_list 
                if 'test2_per_class_acc' in r and species in r['test2_per_class_acc']
            ]
            if species_test2_accs:
                test2_per_class[species] = {
                    'mean': np.mean(species_test2_accs),
                    'std': np.std(species_test2_accs, ddof=1) if len(species_test2_accs) > 1 else 0
                }
        
        stats['test1_per_class'] = test1_per_class
        stats['test2_per_class'] = test2_per_class
    
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


def plot_noise_experiments_for_paper(df, output_dir):
    """
    Create a simplified figure for the paper showing in-domain test and cross-domain accuracy
    for noise intensity and variety experiments, arranged in a row.
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # =========================================================================
    # PLOT 1: Noise Intensity
    # =========================================================================
    ax = axes[0]
    intensity_df = df[df['noise_intensity'].notna()].copy()
    
    if len(intensity_df) > 0:
        for source in ['avianz', 'doc']:
            subset = intensity_df[intensity_df['source_dataset'] == source].sort_values('noise_intensity')
            label_cross = 'Wait\=akere→DOC' if source == 'avianz' else 'DOC→Wait\=akere'
            label_indomain = 'Wait\=akere test' if source == 'avianz' else 'DOC test'
            
            # Cross-domain (solid line)
            ax.errorbar(subset['noise_intensity'], subset['cross_domain_acc'], 
                       yerr=subset['cross_domain_std'], marker='o', label=label_cross, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
            
            # In-domain test (dashed line)
            ax.errorbar(subset['noise_intensity'], subset['in_domain_acc'], 
                       yerr=subset['in_domain_std'], marker='s', label=label_indomain, 
                       linestyle='--', capsize=5, capthick=2, linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Noise Intensity', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('(a) Noise Intensity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # PLOT 2: Noise Variety
    # =========================================================================
    ax = axes[1]
    variety_df = df[df['noise_variety'].notna()].copy()
    
    if len(variety_df) > 0:
        for source in ['avianz', 'doc']:
            subset = variety_df[variety_df['source_dataset'] == source].sort_values('noise_variety')
            label_cross = 'Wait\=akere→DOC' if source == 'avianz' else 'DOC→Wait\=akere'
            label_indomain = 'Wait\=akere test' if source == 'avianz' else 'DOC test'
            
            # Cross-domain (solid line)
            ax.errorbar(subset['noise_variety'], subset['cross_domain_acc'], 
                       yerr=subset['cross_domain_std'], marker='o', label=label_cross, 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
            
            # In-domain test (dashed line)
            ax.errorbar(subset['noise_variety'], subset['in_domain_acc'], 
                       yerr=subset['in_domain_std'], marker='s', label=label_indomain, 
                       linestyle='--', capsize=5, capthick=2, linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Number of Noise Samples', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('(b) Noise Variety', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'noise_augmentation_paper.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved noise augmentation figure for paper to: {output_file}")
    
    # Also save as PNG for easier viewing
    output_file_png = output_dir / 'noise_augmentation_paper.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {output_file_png}")
    
    plt.close()


def generate_per_species_tables(df, grouped, output_dir):
    """
    Generate per-species accuracy tables for key experiments.
    
    Args:
        df: DataFrame with aggregated results
        grouped: Dict mapping base experiment names to lists of result dicts
        output_dir: Directory to save output files
    """
    print("\n" + "="*70)
    print("PER-SPECIES ACCURACY ANALYSIS")
    print("="*70)
    
    # Define species list (they should all be the same across experiments)
    species_list = [
        'blackbird', 'chaffinch', 'fantail', 'grey warbler', 
        'kaka', 'morepork', 'silvereye', 'tomtit', 'tui/bellbird'
    ]
    
    # =========================================================================
    # TABLE: Per-species accuracy for best normalization method
    # =========================================================================
    print("\n--- Per-Species Results: Log+median+normalize (Best Method) ---\n")
    
    best_method_df = df[
        (df['method'] == 'Baseline') &
        (df['transform'] == 'Log+median+normalize') &
        (df['noise_intensity'].isna()) &
        (df['noise_variety'].isna())
    ]
    
    if len(best_method_df) == 0:
        # Try alternative names
        best_method_df = df[
            (df['method'] == 'Baseline') &
            (df['transform'].str.contains('normalize', case=False, na=False)) &
            (df['noise_intensity'].isna()) &
            (df['noise_variety'].isna())
        ]
    
    if len(best_method_df) > 0:
        per_species_rows = []
        
        # First pass: collect in-domain baselines for each species in each dataset
        species_baselines = {}  # {(dataset, species): {'mean': X, 'std': Y}}
        
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            target = row['target_dataset']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats:
                    # Store in-domain baselines (test1 = same as source)
                    for species in species_list:
                        if species in stats['test1_per_class']:
                            species_baselines[(source, species)] = {
                                'mean': stats['test1_per_class'][species]['mean'],
                                'std': stats['test1_per_class'][species]['std']
                            }
        
        # Second pass: build rows with reduction percentages
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            target = row['target_dataset']
            
            # Get per-class stats from the row (they were stored as JSON-like dicts)
            # We need to re-extract from the original grouped data
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats and 'test2_per_class' in stats:
                    # Create a row for each species
                    for species in species_list:
                        if species in stats['test1_per_class'] and species in stats['test2_per_class']:
                            in_domain = stats['test1_per_class'][species]['mean']
                            cross_domain = stats['test2_per_class'][species]['mean']
                            
                            # Get target dataset's in-domain baseline for this species
                            target_baseline = species_baselines.get((target, species), {}).get('mean', None)
                            
                            # Calculate reduction percentage (like overall analysis)
                            if target_baseline is not None and target_baseline > 0:
                                reduction_pct = (target_baseline - cross_domain) / target_baseline * 100
                            else:
                                reduction_pct = np.nan
                            
                            per_species_rows.append({
                                'source': source,
                                'target': target,
                                'species': species,
                                'in_domain_acc': in_domain,
                                'in_domain_std': stats['test1_per_class'][species]['std'],
                                'cross_domain_acc': cross_domain,
                                'cross_domain_std': stats['test2_per_class'][species]['std'],
                                'target_baseline': target_baseline,
                                'reduction_pct': reduction_pct,
                            })
        
        if per_species_rows:
            per_species_df = pd.DataFrame(per_species_rows)
            
            # Pivot to create a nice table
            print("\nWaitākere → DOC (trained on Waitākere, tested on DOC):")
            waitakere_to_doc = per_species_df[per_species_df['source'] == 'avianz']
            if len(waitakere_to_doc) > 0:
                for _, row in waitakere_to_doc.iterrows():
                    print(f"  {row['species']:20s}: In-domain: {row['in_domain_acc']:.1f}±{row['in_domain_std']:.1f}%  "
                          f"Cross-domain: {row['cross_domain_acc']:.1f}±{row['cross_domain_std']:.1f}%  "
                          f"Target baseline: {row['target_baseline']:.1f}%  "
                          f"Reduction: {row['reduction_pct']:.1f}%")
            
            print("\nDOC → Waitākere (trained on DOC, tested on Waitākere):")
            doc_to_waitakere = per_species_df[per_species_df['source'] == 'doc']
            if len(doc_to_waitakere) > 0:
                for _, row in doc_to_waitakere.iterrows():
                    print(f"  {row['species']:20s}: In-domain: {row['in_domain_acc']:.1f}±{row['in_domain_std']:.1f}%  "
                          f"Cross-domain: {row['cross_domain_acc']:.1f}±{row['cross_domain_std']:.1f}%  "
                          f"Target baseline: {row['target_baseline']:.1f}%  "
                          f"Reduction: {row['reduction_pct']:.1f}%")
            
            # Save CSV
            per_species_file = output_dir / 'table_per_species_best_method.csv'
            per_species_df.to_csv(per_species_file, index=False)
            print(f"\n✓ Saved per-species results to: {per_species_file}")
    
    # =========================================================================
    # TABLE: Compare Log baseline vs Log+normalize per species
    # =========================================================================
    # TABLE: Compare Log baseline vs Log+median+normalize per species
    # =========================================================================
    print("\n" + "="*70)
    print("Per-Species Comparison: Log vs Log+median+normalize")
    print("="*70)
    
    comparison_df = df[
        (df['method'] == 'Baseline') &
        (df['transform'].isin(['Log', 'Log+median+normalize'])) &
        (df['noise_intensity'].isna()) &
        (df['noise_variety'].isna())
    ]
    
    if len(comparison_df) > 0:
        comparison_rows = []
        
        for idx, row in comparison_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            target = row['target_dataset']
            transform = row['transform']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test2_per_class' in stats:  # Focus on cross-domain
                    for species in species_list:
                        if species in stats['test2_per_class']:
                            comparison_rows.append({
                                'source': source,
                                'target': target,
                                'transform': transform,
                                'species': species,
                                'cross_domain_acc': stats['test2_per_class'][species]['mean'],
                                'cross_domain_std': stats['test2_per_class'][species]['std'],
                            })
        
        if comparison_rows:
            comparison_per_species_df = pd.DataFrame(comparison_rows)
            
            # Pivot to show Log vs Log+normalize side by side
            pivot_df = comparison_per_species_df.pivot_table(
                index=['source', 'target', 'species'],
                columns='transform',
                values=['cross_domain_acc', 'cross_domain_std']
            )
            
            print("\nCross-Domain Accuracy by Species and Normalization:")
            print(pivot_df.to_string())
            
            # Save CSV
            comparison_file = output_dir / 'table_per_species_normalization_comparison.csv'
            comparison_per_species_df.to_csv(comparison_file, index=False)
            print(f"\n✓ Saved per-species normalization comparison to: {comparison_file}")
    
    # =========================================================================
    # VISUALIZATION: Per-species domain shift
    # =========================================================================
    print("\n" + "="*70)
    print("Generating Per-Species Domain Shift Visualization")
    print("="*70)
    
    if len(best_method_df) > 0:
        per_species_rows = []
        
        # First pass: collect in-domain baselines
        species_baselines = {}  # {(dataset, species): mean_acc}
        
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats:
                    for species in species_list:
                        if species in stats['test1_per_class']:
                            species_baselines[(source, species)] = stats['test1_per_class'][species]['mean']
        
        # Second pass: build rows with reduction percentages
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            target = row['target_dataset']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats and 'test2_per_class' in stats:
                    for species in species_list:
                        if species in stats['test1_per_class'] and species in stats['test2_per_class']:
                            in_domain = stats['test1_per_class'][species]['mean']
                            cross_domain = stats['test2_per_class'][species]['mean']
                            
                            # Get target dataset's baseline for reduction calculation
                            target_baseline = species_baselines.get((target, species), None)
                            if target_baseline is not None and target_baseline > 0:
                                reduction_pct = (target_baseline - cross_domain) / target_baseline * 100
                            else:
                                reduction_pct = np.nan
                            
                            per_species_rows.append({
                                'source': source,
                                'target': target,
                                'species': species,
                                'in_domain_acc': in_domain,
                                'in_domain_std': stats['test1_per_class'][species]['std'],
                                'cross_domain_acc': cross_domain,
                                'cross_domain_std': stats['test2_per_class'][species]['std'],
                                'reduction_pct': reduction_pct,
                            })
        
        if per_species_rows:
            per_species_df = pd.DataFrame(per_species_rows)
            
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Waitākere → DOC
            ax = axes[0]
            waitakere_data = per_species_df[per_species_df['source'] == 'avianz'].sort_values('reduction_pct', ascending=False)
            
            x = np.arange(len(waitakere_data))
            width = 0.35
            
            ax.bar(x - width/2, waitakere_data['in_domain_acc'], width, 
                   label='In-domain (Waitākere)', alpha=0.8, color='#2E86AB')
            ax.bar(x + width/2, waitakere_data['cross_domain_acc'], width, 
                   label='Cross-domain (DOC)', alpha=0.8, color='#A23B72')
            
            ax.set_xlabel('Species', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title('Waitākere → DOC\n(trained on Waitākere, tested on DOC)', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(waitakere_data['species'], rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, loc='lower left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
            
            # Plot 2: DOC → Waitākere
            ax = axes[1]
            doc_data = per_species_df[per_species_df['source'] == 'doc'].sort_values('reduction_pct', ascending=False)
            
            x = np.arange(len(doc_data))
            
            ax.bar(x - width/2, doc_data['in_domain_acc'], width, 
                   label='In-domain (DOC)', alpha=0.8, color='#2E86AB')
            ax.bar(x + width/2, doc_data['cross_domain_acc'], width, 
                   label='Cross-domain (Waitākere)', alpha=0.8, color='#A23B72')
            
            ax.set_xlabel('Species', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title('DOC → Waitākere\n(trained on DOC, tested on Waitākere)', 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(doc_data['species'], rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9, loc='lower left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
            
            plt.tight_layout()
            
            # Save figure
            output_file = output_dir / 'per_species_domain_shift.pdf'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved per-species visualization to: {output_file}")
            
            output_file_png = output_dir / 'per_species_domain_shift.png'
            plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
            print(f"✓ Saved PNG version to: {output_file_png}")
            
            plt.close()
    
    # =========================================================================
    # GENERATE LATEX TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("LaTeX Table for Paper")
    print("="*70)
    
    if len(best_method_df) > 0:
        per_species_rows = []
        
        # First pass: collect in-domain baselines
        species_baselines = {}  # {(dataset, species): mean_acc}
        
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats:
                    for species in species_list:
                        if species in stats['test1_per_class']:
                            species_baselines[(source, species)] = stats['test1_per_class'][species]['mean']
        
        # Second pass: build rows with reduction percentages
        for idx, row in best_method_df.iterrows():
            exp_name = row['experiment']
            source = row['source_dataset']
            target = row['target_dataset']
            
            base_name = exp_name
            if base_name in grouped:
                results_list = grouped[base_name]
                stats = compute_statistics(results_list)
                
                if 'test1_per_class' in stats and 'test2_per_class' in stats:
                    for species in species_list:
                        if species in stats['test1_per_class'] and species in stats['test2_per_class']:
                            in_domain = stats['test1_per_class'][species]['mean']
                            cross_domain = stats['test2_per_class'][species]['mean']
                            
                            # Get target dataset's baseline for reduction calculation
                            target_baseline = species_baselines.get((target, species), None)
                            if target_baseline is not None and target_baseline > 0:
                                reduction_pct = (target_baseline - cross_domain) / target_baseline * 100
                            else:
                                reduction_pct = np.nan
                            
                            per_species_rows.append({
                                'source': source,
                                'target': target,
                                'species': species,
                                'in_domain_acc': in_domain,
                                'in_domain_std': stats['test1_per_class'][species]['std'],
                                'cross_domain_acc': cross_domain,
                                'cross_domain_std': stats['test2_per_class'][species]['std'],
                                'reduction_pct': reduction_pct,
                            })
        
        if per_species_rows:
            per_species_df = pd.DataFrame(per_species_rows)
            
            # Generate LaTeX table
            latex_lines = []
            latex_lines.append("\\begin{table*}[t]")
            latex_lines.append("\\centering")
            latex_lines.append("\\small")
            latex_lines.append("\\caption{Per-species accuracy with Log+median+normalize preprocessing. "
                             "\\textbf{In-Dom}: In-domain accuracy (trained and tested on same dataset). "
                             "\\textbf{Cross-Dom}: Cross-domain accuracy (trained on source, tested on target). "
                             "\\textbf{Red\\%%}: Reduction percentage relative to target dataset baseline. "
                             "Mean$\\pm$std over 5 seeds.}")
            latex_lines.append("\\label{tab:per_species}")
            latex_lines.append("\\begin{tabular}{@{}lcccccc@{}}")
            latex_lines.append("\\hline")
            latex_lines.append("Species & \\multicolumn{3}{c}{Wait\\=akere→DOC} & \\multicolumn{3}{c}{DOC→Wait\\=akere} \\\\")
            latex_lines.append("\\cline{2-4} \\cline{5-7}")
            latex_lines.append("        & In-Dom & Cross-Dom & Red\\% & In-Dom & Cross-Dom & Red\\% \\\\")
            latex_lines.append("\\hline")
            
            # For each species, get both directions
            for species in species_list:
                waitakere_to_doc = per_species_df[
                    (per_species_df['source'] == 'avianz') & 
                    (per_species_df['species'] == species)
                ]
                doc_to_waitakere = per_species_df[
                    (per_species_df['source'] == 'doc') & 
                    (per_species_df['species'] == species)
                ]
                
                if len(waitakere_to_doc) > 0 and len(doc_to_waitakere) > 0:
                    w2d = waitakere_to_doc.iloc[0]
                    d2w = doc_to_waitakere.iloc[0]
                    
                    # Format species name for LaTeX
                    species_display = species.replace('_', ' ').title()
                    
                    w2d_in = f"{w2d['in_domain_acc']:.1f}$\\pm${w2d['in_domain_std']:.1f}"
                    w2d_cross = f"{w2d['cross_domain_acc']:.1f}$\\pm${w2d['cross_domain_std']:.1f}"
                    w2d_reduction = w2d['reduction_pct']
                    
                    d2w_in = f"{d2w['in_domain_acc']:.1f}$\\pm${d2w['in_domain_std']:.1f}"
                    d2w_cross = f"{d2w['cross_domain_acc']:.1f}$\\pm${d2w['cross_domain_std']:.1f}"
                    d2w_reduction = d2w['reduction_pct']
                    
                    line = f"{species_display:20s} & {w2d_in} & {w2d_cross} & {w2d_reduction:.1f} & {d2w_in} & {d2w_cross} & {d2w_reduction:.1f} \\\\"
                    latex_lines.append(line)
            
            latex_lines.append("\\hline")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\end{table*}")
            
            latex_output = "\n".join(latex_lines)
            print("\n" + latex_output)
            
            # Save LaTeX to file
            latex_file = output_dir / 'table_per_species.tex'
            with open(latex_file, 'w') as f:
                f.write(latex_output)
            print(f"\n✓ Saved LaTeX table to: {latex_file}")


def extract_species_distribution(output_dir):
    """
    Extract and display species distribution from split_report.json.
    
    Args:
        output_dir: Directory containing split_report.json
    """
    split_report_file = output_dir / 'split_report.json'
    
    if not split_report_file.exists():
        print(f"Warning: split_report.json not found at {split_report_file}")
        return
    
    print("\n" + "="*70)
    print("SPECIES DISTRIBUTION ANALYSIS")
    print("="*70)
    
    with open(split_report_file) as f:
        split_data = json.load(f)
    
    # Extract species distribution for both datasets
    avianz_dist = split_data['avianz']['species_distribution']
    doc_dist = split_data['doc']['species_distribution']
    
    # Create a comprehensive table
    species_list = sorted(avianz_dist.keys())
    
    print("\n--- Dataset Composition Summary ---\n")
    print(f"{'Species':<15} {'Waitākere Total':<15} {'Waitākere %':<12} {'DOC Total':<15} {'DOC %':<12}")
    print("-" * 75)
    
    avianz_total = split_data['avianz']['total_samples']
    doc_total = split_data['doc']['total_samples']
    
    rows = []
    for species in species_list:
        avianz_count = avianz_dist[species]['total']
        avianz_pct = (avianz_count / avianz_total) * 100
        
        doc_count = doc_dist[species]['total']
        doc_pct = (doc_count / doc_total) * 100
        
        print(f"{species:<15} {avianz_count:<15} {avianz_pct:>6.1f}%     {doc_count:<15} {doc_pct:>6.1f}%")
        
        rows.append({
            'species': species,
            'avianz_total': avianz_count,
            'avianz_percentage': avianz_pct,
            'doc_total': doc_count,
            'doc_percentage': doc_pct,
        })
    
    print("-" * 75)
    print(f"{'TOTAL':<15} {avianz_total:<15} {100.0:>6.1f}%     {doc_total:<15} {100.0:>6.1f}%")
    
    # Train/test split details
    print("\n--- Train/Test Split Details ---\n")
    print(f"{'Species':<15} {'Waitākere Train':<15} {'Waitākere Test':<15} {'DOC Train':<15} {'DOC Test':<15}")
    print("-" * 80)
    
    for species in species_list:
        avianz_train = avianz_dist[species]['train']
        avianz_test = avianz_dist[species]['test']
        doc_train = doc_dist[species]['train']
        doc_test = doc_dist[species]['test']
        
        print(f"{species:<15} {avianz_train:<15} {avianz_test:<15} {doc_train:<15} {doc_test:<15}")
        
        rows[-len(species_list) + species_list.index(species)].update({
            'avianz_train': avianz_train,
            'avianz_test': avianz_test,
            'doc_train': doc_train,
            'doc_test': doc_test,
        })
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {split_data['avianz']['train_samples']:<15} "
          f"{split_data['avianz']['test_samples']:<15} "
          f"{split_data['doc']['train_samples']:<15} "
          f"{split_data['doc']['test_samples']:<15}")
    
    # Save to CSV
    species_dist_df = pd.DataFrame(rows)
    species_dist_file = output_dir / 'species_distribution.csv'
    species_dist_df.to_csv(species_dist_file, index=False)
    print(f"\n✓ Saved species distribution to: {species_dist_file}")
    
    # Generate LaTeX table for paper
    print("\n--- LaTeX Table for Paper ---\n")
    
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Species distribution across datasets. Both datasets contain identical species with matched distributions.}",
        "\\label{tab:species_distribution}",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\hline",
        "Species & \\multicolumn{2}{c}{Wait\\=akere} & \\multicolumn{2}{c}{DOC} \\\\",
        "\\cline{2-3} \\cline{4-5}",
        "        & N & \\% & N & \\% \\\\",
        "\\hline",
    ]
    
    for row in rows:
        species_display = row['species'].replace('_', ' ').title()
        if species_display == 'Tui/Bellbird':
            species_display = 'T\\=u\\=i/bellbird'
        elif species_display == 'Grey Warbler':
            species_display = 'Grey warbler'
        elif species_display == 'Kaka':
            species_display = 'K\\=ak\\=a'
            
        line = (f"{species_display:<20s} & {row['avianz_total']:>3d} & {row['avianz_percentage']:>4.1f} & "
                f"{row['doc_total']:>3d} & {row['doc_percentage']:>4.1f} \\\\")
        latex_lines.append(line)
    
    latex_lines.append("\\hline")
    latex_lines.append(f"Total & {avianz_total} & 100.0 & {doc_total} & 100.0 \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_output = "\n".join(latex_lines)
    print(latex_output)
    
    # Save LaTeX table
    latex_file = output_dir / 'table_species_distribution.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_output)
    print(f"\n✓ Saved LaTeX table to: {latex_file}")


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
    
    # Extract species distribution first
    extract_species_distribution(results_file.parent)
    
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
    plot_noise_experiments_for_paper(df, results_file.parent)
    
    # =========================================================================
    # GENERATE PER-SPECIES TABLES
    # =========================================================================
    generate_per_species_tables(df, grouped, results_file.parent)
    
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
        (df['transform'].isin(['Log', 'Log+median+normalize'])) &
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
    # TABLE 4.5: MERGED DATASET TRAINING RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 4.5: MERGED DATASET TRAINING (DOC + Waitākere)")
    print("="*70)
    
    merged_df = df[df['source_dataset'] == 'merged'].copy()
    
    if len(merged_df) > 0:
        print("\nMerged dataset experiments train on combined DOC+Waitākere data,")
        print("then test on both DOC and Waitākere test sets separately.\n")
        
        table_merged = merged_df[[
            'experiment', 'transform', 'method',
            'val_acc', 'val_acc_std',
            'in_domain_acc', 'in_domain_std',
            'cross_domain_acc', 'cross_domain_std',
            'n_trials'
        ]].sort_values(['transform'])
        
        print(table_merged.to_string(index=False))
        
        table_merged_file = results_file.parent / 'table_merged_dataset.csv'
        table_merged.to_csv(table_merged_file, index=False)
        print(f"\n✓ Saved to: {table_merged_file}")
        
        # Print summary
        print("\nKey findings:")
        for idx, row in merged_df.iterrows():
            print(f"  {row['transform']}: Cross-domain acc = {row['cross_domain_acc']:.1f}±{row['cross_domain_std']:.1f}%")
    else:
        print("\nNo merged dataset experiments found yet.")
    
    # =========================================================================
    # TABLE 5: Paper-ready summary with reduction% and asymmetry
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 5: PAPER-READY RESULTS (with Reduction% and Asymmetry)")
    print("="*70)
    
    # Define key configurations to compare
    key_configs = [
        ('Baseline', 'Log', None),
        ('Baseline', 'Log+median+normalize', None),
        ('Baseline', 'Log+normalize', None),
        ('Baseline', 'Log+median', None),
        ('Baseline', 'PCEN', None),
        ('Baseline', 'Box-Cox', None),
        ('DANN', 'Log+median+normalize', None),
        ('AST', 'Log', None),
    ]
    
    asymmetry_df = calculate_asymmetry_ratios(df, key_configs)
    
    if len(asymmetry_df) > 0:
        print("\nAsymmetry Analysis:")
        print(asymmetry_df.to_string(index=False))
        
        # Create a paper-ready formatted table
        print("\n\nPAPER TABLE FORMAT:")
        print("-" * 100)
        print(f"{'Method':<25} {'Waitākere→DOC':<30} {'DOC→Waitākere':<30} {'Asym':<10}")
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
