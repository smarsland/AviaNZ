"""
Simple evaluation script: take predictions and ground truth, produce confusion matrices.

Usage:
    python evaluate.py \
        --preds predictions.csv \
        --gt ground_truth.csv \
        --output_dir results/
"""

import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_bird_name_mapping(csv_path):
    ebird_to_common = {}
    doc_to_common = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ebird_code = row['eBird']
            common_name = row['CommonName']
            doc_name = row.get('ListDOCBirds', '')
            
            if ebird_code and common_name:
                ebird_to_common[ebird_code] = common_name
            if doc_name and common_name:
                doc_to_common[doc_name] = common_name
    
    return ebird_to_common, doc_to_common


def normalize_species_name(name, ebird_to_common, doc_to_common):
    if name in ebird_to_common:
        return ebird_to_common[name]
    if name in doc_to_common:
        return doc_to_common[name]
    return name


def get_predictions_and_labels(preds_df, gt_df, ebird_to_common, doc_to_common):
    """
    Extract predicted and true labels from dataframes.
    
    For each sample:
    - Predicted label: species with highest probability
    - True labels: all species with value 1.0
    
    Returns lists of (true_label, pred_label) pairs.
    """
    pred_cols = [col for col in preds_df.columns if col not in ['File_Path', 'row_id']]
    gt_cols = [col for col in gt_df.columns if col not in ['File_Path', 'row_id']]
    
    gt_file_to_idx = {}
    for idx, row in gt_df.iterrows():
        base_name = Path(row['File_Path']).name
        gt_file_to_idx[base_name] = idx
    
    pairs = []
    
    for pred_idx, pred_row in preds_df.iterrows():
        pred_file = Path(pred_row['File_Path']).name
        
        if pred_file not in gt_file_to_idx:
            continue
        
        gt_idx = gt_file_to_idx[pred_file]
        gt_row = gt_df.iloc[gt_idx]
        
        pred_species = max(pred_cols, key=lambda col: pred_row.get(col, 0))
        pred_species_norm = normalize_species_name(pred_species, ebird_to_common, doc_to_common)
        
        true_species_list = [col for col in gt_cols if gt_row.get(col, 0) == 1.0]
        
        if not true_species_list:
            continue
        
        for true_species in true_species_list:
            true_species_norm = normalize_species_name(true_species, ebird_to_common, doc_to_common)
            pairs.append((true_species_norm, pred_species_norm))
    
    return pairs


def build_confusion_matrix(pairs):
    """Build confusion matrix from (true, pred) pairs."""
    if not pairs:
        return np.array([[]]), [], []
    
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    
    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=true_labels)
    
    pred_only = [label for label in pred_labels if label not in true_labels]
    
    if pred_only:
        extra_cols = []
        for extra_label in pred_only:
            counts = [0] * len(true_labels)
            for i in range(len(y_true)):
                if y_pred[i] == extra_label:
                    true_idx = true_labels.index(y_true[i])
                    counts[true_idx] += 1
            extra_cols.append(counts)
        
        if extra_cols:
            extra_array = np.array(extra_cols).T
            cm = np.hstack([cm, extra_array])
            all_labels = true_labels + pred_only
        else:
            all_labels = true_labels
    else:
        all_labels = true_labels
    
    return cm, true_labels, all_labels


def save_confusion_matrices(cm, row_labels, col_labels, output_dir, prefix):
    """
    Save confusion matrix in two formats:
    - raw.csv: raw counts
    - norm.csv: normalized by row sums (true label counts)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    row_labels_sorted = sorted(row_labels)
    
    col_labels_in_rows = [c for c in row_labels_sorted if c in col_labels]
    col_labels_extra = sorted([c for c in col_labels if c not in row_labels])
    col_labels_reordered = col_labels_in_rows + col_labels_extra
    
    row_order = [row_labels.index(r) for r in row_labels_sorted]
    col_order = [col_labels.index(c) for c in col_labels_reordered]
    
    cm_reordered = cm[np.ix_(row_order, col_order)]
    
    df_raw = pd.DataFrame(cm_reordered, index=row_labels_sorted, columns=col_labels_reordered)
    raw_path = output_dir / f'{prefix}_confusion_matrix_raw.csv'
    df_raw.to_csv(raw_path)
    
    cm_norm = cm_reordered.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_norm / row_sums
    
    df_norm = pd.DataFrame(cm_norm, index=row_labels_sorted, columns=col_labels_reordered)
    norm_path = output_dir / f'{prefix}_confusion_matrix_norm.csv'
    df_norm.to_csv(norm_path)
    
    print(f"Saved raw confusion matrix: {raw_path}")
    print(f"Saved normalized confusion matrix: {norm_path}")
    
    fig_size_rows = max(10, len(row_labels_sorted) * 0.4)
    fig_size_cols = max(10, len(col_labels_reordered) * 0.4)
    
    plt.figure(figsize=(fig_size_cols, fig_size_rows))
    sns.heatmap(cm_reordered, annot=False, fmt='d', cmap='Blues', 
                xticklabels=col_labels_reordered, yticklabels=row_labels_sorted,
                cbar_kws={'label': 'Count'})
    plt.title(f'{prefix} - Raw Counts', fontsize=16, pad=20)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('Actual Species', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    img_raw_path = output_dir / f'{prefix}_confusion_matrix_raw.png'
    plt.savefig(img_raw_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(fig_size_cols, fig_size_rows))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=col_labels_reordered, yticklabels=row_labels_sorted,
                cbar_kws={'label': 'Proportion'})
    plt.title(f'{prefix} - Normalized by Actual (rows sum to 1)', fontsize=16, pad=20)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('Actual Species', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    img_norm_path = output_dir / f'{prefix}_confusion_matrix_norm.png'
    plt.savefig(img_norm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved raw confusion matrix image: {img_raw_path}")
    print(f"Saved normalized confusion matrix image: {img_norm_path}")
    
    diagonal_sum = np.trace(cm_reordered[:min(len(row_labels_sorted), len(col_labels_in_rows)), :min(len(row_labels_sorted), len(col_labels_in_rows))])
    total_sum = cm_reordered.sum()
    accuracy = diagonal_sum / total_sum if total_sum > 0 else 0
    
    print(f"Accuracy: {diagonal_sum}/{total_sum} = {accuracy:.4f}")
    print(f"Matrix shape: {len(row_labels_sorted)} actual species x {len(col_labels_reordered)} predicted species")


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions against ground truth')
    parser.add_argument('--preds', required=True, help='Path to predictions CSV')
    parser.add_argument('--gt', required=True, help='Path to ground truth CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory for confusion matrices')
    parser.add_argument('--prefix', default='model', help='Prefix for output files (default: model)')
    
    args = parser.parse_args()
    
    mapping_path = Path(__file__).parent / 'DOC_bird_naming_map.csv'
    print(f"Loading mapping: {mapping_path}")
    ebird_to_common, doc_to_common = load_bird_name_mapping(mapping_path)
    print(f"  {len(ebird_to_common)} eBird mappings, {len(doc_to_common)} DOC mappings")
    
    print(f"Loading predictions: {args.preds}")
    preds_df = pd.read_csv(args.preds)
    print(f"  {len(preds_df)} rows, {len(preds_df.columns)} columns")
    
    print(f"Loading ground truth: {args.gt}")
    gt_df = pd.read_csv(args.gt)
    print(f"  {len(gt_df)} rows, {len(gt_df.columns)} columns")
    
    print("\nExtracting predictions and labels...")
    pairs = get_predictions_and_labels(preds_df, gt_df, ebird_to_common, doc_to_common)
    print(f"  {len(pairs)} (true, pred) pairs")
    
    print("\nBuilding confusion matrix...")
    cm, row_labels, col_labels = build_confusion_matrix(pairs)
    
    print("\nSaving confusion matrices...")
    save_confusion_matrices(cm, row_labels, col_labels, args.output_dir, args.prefix)


if __name__ == '__main__':
    main()
