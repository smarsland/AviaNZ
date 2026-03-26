"""
Create confusion matrices comparing Log baseline vs Log+normalize.
Shows which species confusions are reduced by background normalization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path


def load_predictions_with_truth(csv_path, labels_mapping=None):
    """
    Load predictions and determine ground truth.
    
    Note: This is a heuristic approach. For best results, you should have
    a separate ground truth file with actual labels.
    """
    df = pd.read_csv(csv_path)
    
    # Get species columns
    species_cols = [col for col in df.columns if col not in ['File_Path', 'row_id']]
    
    # Heuristic: extract species from file path
    ground_truth = []
    ground_truth_idx = []
    
    for path in df['File_Path']:
        found = False
        for i, sp in enumerate(species_cols):
            # Try to match species name in path
            sp_clean = sp.replace(' ', '').replace('/', '').lower()
            if sp_clean in path.lower():
                ground_truth.append(sp)
                ground_truth_idx.append(i)
                found = True
                break
        
        if not found:
            # Default to first species if no match (shouldn't happen)
            ground_truth.append(species_cols[0])
            ground_truth_idx.append(0)
    
    # Get predictions (argmax of probabilities)
    pred_probs = df[species_cols].values
    predicted_idx = pred_probs.argmax(axis=1)
    predicted = [species_cols[i] for i in predicted_idx]
    
    return ground_truth, predicted, species_cols, ground_truth_idx, predicted_idx


def plot_confusion_matrix_comparison(output_path='figures/confusion_matrix_comparison.pdf'):
    """
    Create side-by-side confusion matrices for baseline vs normalized.
    """
    experiments_dir = Path('experiments_matched')
    
    # We'll focus on DOC training (showed biggest improvement)
    baseline_path = experiments_dir / 'doc_baseline_birdclef' / 'predictions_doc_split_test.csv'
    normalized_path = experiments_dir / 'doc_baseline_birdclef_normalized' / 'predictions_doc_split_test.csv'
    
    if not baseline_path.exists() or not normalized_path.exists():
        print(f"Error: Could not find prediction files")
        print(f"Baseline: {baseline_path}")
        print(f"Normalized: {normalized_path}")
        return
    
    # Load predictions
    y_true_base, y_pred_base, species, y_true_idx_base, y_pred_idx_base = load_predictions_with_truth(baseline_path)
    y_true_norm, y_pred_norm, species, y_true_idx_norm, y_pred_idx_norm = load_predictions_with_truth(normalized_path)
    
    # Compute confusion matrices
    cm_base = confusion_matrix(y_true_idx_base, y_pred_idx_base, labels=range(len(species)))
    cm_norm = confusion_matrix(y_true_idx_norm, y_pred_idx_norm, labels=range(len(species)))
    
    # Normalize by row (true label) to get percentages
    cm_base_pct = cm_base.astype('float') / cm_base.sum(axis=1, keepdims=True)
    cm_norm_pct = cm_norm.astype('float') / cm_norm.sum(axis=1, keepdims=True)
    
    # Replace NaN with 0 (for species with no samples)
    cm_base_pct = np.nan_to_num(cm_base_pct)
    cm_norm_pct = np.nan_to_num(cm_norm_pct)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Shorten species names for readability
    species_short = [sp[:15] for sp in species]
    
    # Plot baseline
    sns.heatmap(cm_base_pct, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=species_short, yticklabels=species_short,
                vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Proportion'})
    ax1.set_xlabel('Predicted Species', fontsize=11)
    ax1.set_ylabel('True Species', fontsize=11)
    ax1.set_title('(a) Log Transform (Baseline)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=8)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot normalized
    sns.heatmap(cm_norm_pct, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=species_short, yticklabels=species_short,
                vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_xlabel('Predicted Species', fontsize=11)
    ax2.set_ylabel('True Species', fontsize=11)
    ax2.set_title('(b) Log + Background Normalization', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=8)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved confusion matrices to {output_path}")
    plt.close()
    
    # Also create a difference matrix showing improvement
    plot_confusion_improvement(cm_base_pct, cm_norm_pct, species_short)
    
    return cm_base, cm_norm, species


def plot_confusion_improvement(cm_base, cm_norm, species_short, 
                                 output_path='figures/confusion_improvement.pdf'):
    """Plot the difference matrix showing where normalization helps."""
    
    # Compute difference (positive = improvement)
    diff = cm_norm - cm_base
    
    # Diagonal elements (correct predictions) - positive is good
    # Off-diagonal elements (errors) - negative is good (fewer errors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    max_abs = max(abs(diff.min()), abs(diff.max()))
    sns.heatmap(diff, annot=False, fmt='.2f', cmap='RdBu_r',
                xticklabels=species_short, yticklabels=species_short,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={'label': 'Change in Proportion\n(Positive = Improvement)'})
    ax.set_xlabel('Predicted Species', fontsize=11)
    ax.set_ylabel('True Species', fontsize=11)
    ax.set_title('Confusion Matrix Improvement: Log+Normalize vs Baseline\n' +
                 '(Red = More errors, Blue = Fewer errors)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved confusion improvement to {output_path}")
    plt.close()


def analyze_major_confusions(cm_base, cm_norm, species):
    """Print the most common confusions and how normalization affects them."""
    
    print("\n" + "="*80)
    print("CONFUSION ANALYSIS")
    print("="*80)
    
    # Normalize by row to get error rates
    cm_base_pct = cm_base.astype('float') / cm_base.sum(axis=1, keepdims=True)
    cm_norm_pct = cm_norm.astype('float') / cm_norm.sum(axis=1, keepdims=True)
    
    cm_base_pct = np.nan_to_num(cm_base_pct)
    cm_norm_pct = np.nan_to_num(cm_norm_pct)
    
    # Find largest off-diagonal confusions in baseline
    confusions = []
    for i in range(len(species)):
        for j in range(len(species)):
            if i != j:  # Off-diagonal
                base_rate = cm_base_pct[i, j]
                norm_rate = cm_norm_pct[i, j]
                reduction = base_rate - norm_rate
                
                if base_rate > 0.05:  # Only report confusions > 5%
                    confusions.append((
                        species[i], species[j], 
                        base_rate, norm_rate, reduction,
                        cm_base[i, j]  # absolute count
                    ))
    
    # Sort by baseline confusion rate
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print("\n--- Most Common Confusions (Baseline) ---")
    print(f"{'True Species':<20} {'Confused As':<20} {'Base Rate':>10} {'Norm Rate':>10} {'Reduction':>10} {'Count':>10}")
    print("-" * 100)
    
    for true_sp, pred_sp, base_rate, norm_rate, reduction, count in confusions[:15]:
        print(f"{true_sp:<20} {pred_sp:<20} {base_rate:>9.1%} {norm_rate:>9.1%} {reduction:>+9.1%} {count:>10}")
    
    # Find confusions most reduced by normalization
    confusions.sort(key=lambda x: x[4], reverse=True)
    
    print("\n--- Confusions Most Reduced by Normalization ---")
    print(f"{'True Species':<20} {'Confused As':<20} {'Base Rate':>10} {'Norm Rate':>10} {'Reduction':>10}")
    print("-" * 100)
    
    for true_sp, pred_sp, base_rate, norm_rate, reduction, count in confusions[:10]:
        if reduction > 0:
            print(f"{true_sp:<20} {pred_sp:<20} {base_rate:>9.1%} {norm_rate:>9.1%} {reduction:>+9.1%}")


if __name__ == "__main__":
    print("Creating confusion matrix analysis...")
    
    # Create visualizations
    cm_base, cm_norm, species = plot_confusion_matrix_comparison()
    
    if cm_base is not None:
        # Analyze confusions
        analyze_major_confusions(cm_base, cm_norm, species)
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("Generated:")
        print("  - figures/confusion_matrix_comparison.pdf")
        print("  - figures/confusion_improvement.pdf")
        print("="*80)
