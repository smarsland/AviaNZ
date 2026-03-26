"""
Analyze noise intensity experiment results.
Generate plots and tables showing how noise mixing ratio affects performance.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configuration
RESULTS_DIR = Path("/local/scratch/freangi/experiments_noise_intensity")
OUTPUT_DIR = Path("/home/giotto/Desktop/AviaNZ/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_experiment_results(results_dir):
    """Load all experiment results from directory."""
    results = []
    
    for experiment_dir in results_dir.glob("*_baseline_birdclef_normalized_noise*"):
        # Parse experiment name
        name = experiment_dir.name
        
        # Extract noise count and ratio from suffix
        # Format: doc_baseline_birdclef_normalized_noise10_ratio0.25
        match = re.search(r'noise(\d+)_ratio([\d.]+)', name)
        if not match:
            continue
            
        n_noise = int(match.group(1))
        ratio = float(match.group(2))
        
        # Determine training dataset
        if name.startswith('doc_'):
            train_dataset = 'DOC'
        elif name.startswith('joe_mo_'):
            train_dataset = 'AviaNZ'
        else:
            continue
        
        # Load training history
        history_file = experiment_dir / "training_history.json"
        if not history_file.exists():
            print(f"Warning: No history file for {name}")
            continue
            
        with open(history_file) as f:
            history = json.load(f)
        
        # Get test accuracies (use best epoch or final)
        if "test_accuracies" in history:
            test_accs = history["test_accuracies"]
            
            # Find best epoch based on validation
            val_accs = history.get("val_accuracies", [])
            if val_accs:
                best_epoch = np.argmax(val_accs)
                test_acc_avianz = test_accs.get("avianz_split/test", [0])[best_epoch] if isinstance(test_accs, dict) else 0
                test_acc_doc = test_accs.get("doc_split/test", [0])[best_epoch] if isinstance(test_accs, dict) else 0
            else:
                # Use final epoch
                test_acc_avianz = test_accs.get("avianz_split/test", [0])[-1] if isinstance(test_accs, dict) else 0
                test_acc_doc = test_accs.get("doc_split/test", [0])[-1] if isinstance(test_accs, dict) else 0
        else:
            # Try to read from predictions CSV
            pred_avianz = experiment_dir / "predictions_avianz_split_test.csv"
            pred_doc = experiment_dir / "predictions_doc_split_test.csv"
            
            if pred_avianz.exists():
                df = pd.read_csv(pred_avianz)
                test_acc_avianz = (df['predicted'] == df['true']).mean() * 100
            else:
                test_acc_avianz = 0
                
            if pred_doc.exists():
                df = pd.read_csv(pred_doc)
                test_acc_doc = (df['predicted'] == df['true']).mean() * 100
            else:
                test_acc_doc = 0
        
        # Calculate in-domain and cross-domain accuracies
        if train_dataset == 'AviaNZ':
            in_domain_acc = test_acc_avianz
            cross_domain_acc = test_acc_doc
            test_domain = 'DOC'
        else:  # DOC
            in_domain_acc = test_acc_doc
            cross_domain_acc = test_acc_avianz
            test_domain = 'AviaNZ'
        
        domain_shift = cross_domain_acc - in_domain_acc
        
        results.append({
            'train_dataset': train_dataset,
            'test_domain': test_domain,
            'n_noise': n_noise,
            'noise_ratio': ratio,
            'in_domain_acc': in_domain_acc,
            'cross_domain_acc': cross_domain_acc,
            'domain_shift': domain_shift,
            'avg_acc': (in_domain_acc + cross_domain_acc) / 2
        })
    
    return pd.DataFrame(results)

def plot_noise_intensity_heatmap(df, train_dataset):
    """Plot heatmap of accuracy vs (n_noise, ratio) for a specific training dataset."""
    df_subset = df[df['train_dataset'] == train_dataset].copy()
    
    if len(df_subset) == 0:
        print(f"No data for {train_dataset}")
        return
    
    # Create pivot tables for different metrics
    metrics = {
        'Cross-Domain Accuracy': 'cross_domain_acc',
        'Domain Shift': 'domain_shift',
        'Average Accuracy': 'avg_acc'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (title, metric) in zip(axes, metrics.items()):
        pivot = df_subset.pivot_table(
            values=metric,
            index='n_noise',
            columns='noise_ratio',
            aggfunc='mean'
        )
        
        if metric == 'domain_shift':
            cmap = 'RdYlGn'  # Red (bad) to Green (good) for shift
            vmin, vmax = -35, 0
        else:
            cmap = 'viridis'
            vmin, vmax = None, None
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                   vmin=vmin, vmax=vmax, cbar_kws={'label': '%'})
        ax.set_title(f'{title}\nTrain: {train_dataset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Noise Ratio', fontsize=11)
        ax.set_ylabel('Number of Noise Samples', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'noise_intensity_heatmap_{train_dataset.lower()}.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'noise_intensity_heatmap_{train_dataset.lower()}.pdf', bbox_inches='tight')
    print(f"Saved heatmap for {train_dataset}")
    plt.close()

def plot_noise_intensity_curves(df):
    """Plot curves showing how accuracy changes with ratio for each noise count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, train_dataset in zip(axes, ['AviaNZ', 'DOC']):
        df_subset = df[df['train_dataset'] == train_dataset].copy()
        
        if len(df_subset) == 0:
            continue
        
        # Plot lines for each n_noise
        for n_noise in sorted(df_subset['n_noise'].unique()):
            data = df_subset[df_subset['n_noise'] == n_noise].sort_values('noise_ratio')
            ax.plot(data['noise_ratio'], data['cross_domain_acc'], 
                   marker='o', label=f'{n_noise} samples', linewidth=2)
        
        ax.set_xlabel('Noise Mixing Ratio', fontsize=11)
        ax.set_ylabel('Cross-Domain Accuracy (%)', fontsize=11)
        ax.set_title(f'Train: {train_dataset}', fontsize=12, fontweight='bold')
        ax.legend(title='Noise Count')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([40, 70])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'noise_intensity_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'noise_intensity_curves.pdf', bbox_inches='tight')
    print("Saved intensity curves")
    plt.close()

def create_summary_table(df):
    """Create summary table with best configurations."""
    summary = []
    
    for train_dataset in ['AviaNZ', 'DOC']:
        df_subset = df[df['train_dataset'] == train_dataset]
        
        if len(df_subset) == 0:
            continue
        
        # Find best configuration for cross-domain accuracy
        best_cross = df_subset.loc[df_subset['cross_domain_acc'].idxmax()]
        
        # Find best configuration for minimizing domain shift (closest to 0)
        best_shift = df_subset.loc[df_subset['domain_shift'].abs().idxmin()]
        
        # Find best average accuracy
        best_avg = df_subset.loc[df_subset['avg_acc'].idxmax()]
        
        summary.append({
            'Train Dataset': train_dataset,
            'Metric': 'Best Cross-Domain',
            'N Noise': int(best_cross['n_noise']),
            'Ratio': best_cross['noise_ratio'],
            'In-Domain': f"{best_cross['in_domain_acc']:.1f}%",
            'Cross-Domain': f"{best_cross['cross_domain_acc']:.1f}%",
            'Shift': f"{best_cross['domain_shift']:.1f}pp"
        })
        
        summary.append({
            'Train Dataset': train_dataset,
            'Metric': 'Minimum Shift',
            'N Noise': int(best_shift['n_noise']),
            'Ratio': best_shift['noise_ratio'],
            'In-Domain': f"{best_shift['in_domain_acc']:.1f}%",
            'Cross-Domain': f"{best_shift['cross_domain_acc']:.1f}%",
            'Shift': f"{best_shift['domain_shift']:.1f}pp"
        })
        
        summary.append({
            'Train Dataset': train_dataset,
            'Metric': 'Best Average',
            'N Noise': int(best_avg['n_noise']),
            'Ratio': best_avg['noise_ratio'],
            'In-Domain': f"{best_avg['in_domain_acc']:.1f}%",
            'Cross-Domain': f"{best_avg['cross_domain_acc']:.1f}%",
            'Shift': f"{best_avg['domain_shift']:.1f}pp"
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save to file
    summary_file = RESULTS_DIR / "noise_intensity_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NOISE INTENSITY EXPERIMENTS - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
    
    # Also save as CSV
    summary_df.to_csv(RESULTS_DIR / "noise_intensity_summary.csv", index=False)
    
    print("\nBest Configurations:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Main analysis function."""
    print("Loading experiment results...")
    df = load_experiment_results(RESULTS_DIR)
    
    if len(df) == 0:
        print(f"No results found in {RESULTS_DIR}")
        print("Make sure experiments have completed and results are in the expected format.")
        return
    
    print(f"Loaded {len(df)} experiments")
    print(f"Training datasets: {df['train_dataset'].unique()}")
    print(f"Noise counts: {sorted(df['n_noise'].unique())}")
    print(f"Noise ratios: {sorted(df['noise_ratio'].unique())}")
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_noise_intensity_heatmap(df, 'AviaNZ')
    plot_noise_intensity_heatmap(df, 'DOC')
    plot_noise_intensity_curves(df)
    
    # Create summary table
    print("\nGenerating summary...")
    create_summary_table(df)
    
    # Save full results
    df.to_csv(RESULTS_DIR / "noise_intensity_full_results.csv", index=False)
    print(f"\nFull results saved to {RESULTS_DIR / 'noise_intensity_full_results.csv'}")
    print(f"Figures saved to {OUTPUT_DIR}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
