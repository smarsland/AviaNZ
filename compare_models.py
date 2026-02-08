"""
Model Comparison Script
Compares multiple model predictions against ground truth labels
Generates unified metrics CSV and individual confusion matrices for each model
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, multilabel_confusion_matrix,
    hamming_loss, jaccard_score, average_precision_score, roc_auc_score
)
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob


def normalize_row_id(row_id):
    """Normalize row IDs to handle different formats"""
    import re
    
    row_id = row_id.strip()
    row_id = row_id.replace('\uf025', '/')
    row_id = row_id.replace('\\', '/')
    
    while '//' in row_id:
        row_id = row_id.replace('//', '/')
    
    if ':/' in row_id or ':\\' in row_id:
        markers = ['Joe_MoDone/', 'Avianzmoreporkfilter/', 'Rurudone/']
        found_marker = False
        for marker in markers:
            if marker in row_id:
                row_id = row_id.split(marker, 1)[1]
                found_marker = True
                break
        
        if not found_marker:
            row_id = re.sub(r'^[A-Z]:[/\\]', '', row_id)
            if row_id.startswith('Rurudone/'):
                row_id = row_id.split('/', 1)[1] if '/' in row_id else row_id
    
    folder_prefixes_to_remove = ['Avianzmoreporkfilter/', 'Tier1ruru/', 'Tier2ruru/', 'Tier3ruru/', 'Eglintonruru/']
    for prefix in folder_prefixes_to_remove:
        if row_id.startswith(prefix):
            row_id = row_id[len(prefix):]
            break
    
    row_id = row_id.replace('.wav_', '_')
    row_id = row_id.replace('.flac_', '_')
    row_id = row_id.replace('.WAV_', '_')
    row_id = row_id.replace('.FLAC_', '_')
    row_id = re.sub(r'\.(wav|flac|WAV|FLAC)$', '', row_id)
    
    return row_id


def load_ground_truth(labels_path):
    """Load and process ground truth labels from labels.json"""
    print("Loading ground truth labels...")
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = {}
    for file_info in data['files']:
        row_id = normalize_row_id(file_info['row_id'])
        class_names = file_info.get('class_names', [])
        
        if class_names and class_names != ['Empty Sample']:
            ground_truth[row_id] = set(class_names)
        else:
            ground_truth[row_id] = set()
    
    print(f"Loaded {len(ground_truth)} ground truth samples")
    return ground_truth, data['categories']


def split_validation_set(ground_truth, val_size=5000, seed=42):
    """Split a validation set from ground truth for threshold optimization"""
    np.random.seed(seed)
    all_ids = list(ground_truth.keys())
    np.random.shuffle(all_ids)
    val_size = min(val_size, len(all_ids) // 5)
    val_ids = set(all_ids[:val_size])
    return val_ids


def load_predictions(pred_path, threshold=0.5):
    """Load predictions and apply threshold"""
    print(f"Loading predictions from {os.path.basename(pred_path)}...")
    df = pd.read_csv(pred_path)
    
    row_id_col = 'row_id'
    meta_cols = ['File_Path', 'row_id']
    species_cols = [col for col in df.columns if col not in meta_cols]
    
    print(f"  Found {len(species_cols)} species in predictions")
    
    predictions = {}
    for idx, row in df.iterrows():
        row_id = normalize_row_id(row[row_id_col])
        pred_species = set()
        
        for species in species_cols:
            if pd.notna(row[species]) and row[species] >= threshold:
                pred_species.add(species)
        
        predictions[row_id] = pred_species
    
    print(f"  Loaded {len(predictions)} prediction samples")
    return predictions, species_cols


def load_predictions_raw(pred_path):
    """Load raw prediction probabilities without thresholding"""
    df = pd.read_csv(pred_path)
    row_id_col = 'row_id'
    meta_cols = ['File_Path', 'row_id']
    species_cols = [col for col in df.columns if col not in meta_cols]
    predictions_probs = {}
    for idx, row in df.iterrows():
        row_id = normalize_row_id(row[row_id_col])
        probs = {species: row[species] for species in species_cols if pd.notna(row[species])}
        predictions_probs[row_id] = probs
    return predictions_probs, species_cols


def apply_threshold_to_probs(predictions_probs, threshold):
    """Convert probability dictionary to thresholded set predictions"""
    predictions = {}
    for row_id, probs in predictions_probs.items():
        pred_species = {species for species, prob in probs.items() if prob >= threshold}
        predictions[row_id] = pred_species
    return predictions


def load_bird_naming_map(csv_path):
    """Load the DOC bird naming map CSV and create comprehensive mappings"""
    df = pd.read_csv(csv_path)
    
    name_to_common = {}
    
    for _, row in df.iterrows():
        common_name = row['CommonName']
        
        if pd.notna(row['eBird']):
            name_to_common[row['eBird']] = common_name
        if pd.notna(row['ExtraName']):
            name_to_common[row['ExtraName']] = common_name
        if pd.notna(row['ListDOCBirds']):
            name_to_common[row['ListDOCBirds']] = common_name
        if pd.notna(row['ScientificName']):
            name_to_common[row['ScientificName']] = common_name
        
        name_to_common[common_name] = common_name
    
    # Hardcoded fixes for common mismatches
    name_to_common['Ruru'] = 'Morepork'
    name_to_common['Bellbird/Tui'] = 'Bellbird'
    name_to_common['Tomtit (Nth Is)'] = 'Tomtit'
    name_to_common['Fantail (Nth Is)'] = 'Fantail'
    name_to_common['Fantail (spp)'] = 'Fantail'
    name_to_common['Kaka (Nth Is)'] = 'Kaka'
    name_to_common['Kaka (spp)'] = 'Kaka'
    name_to_common['Tui (spp)'] = 'Tui'
    name_to_common['Robin (Nth Is)'] = 'New Zealand Robin'
    name_to_common['Pigeon (NZ Kereru Kukupa)'] = 'Kereru'
    name_to_common['Warbler (Grey)'] = 'Grey Warbler'
    name_to_common['Magpie (Australian)'] = 'Australian Magpie'
    name_to_common['Myna (Indian)'] = 'Common Myna'
    name_to_common['Gull (Southern Black-backed)'] = 'Southern Black-backed Gull'
    name_to_common['Plover (Spur-winged)'] = 'Spur-winged Plover'
    name_to_common['Rosella (Eastern)'] = 'Eastern Rosella'
    name_to_common['Cockatoo (Sulphur-crested)'] = 'Sulphur-crested Cockatoo'
    name_to_common['Sparrow (House)'] = 'House Sparrow'
    
    return name_to_common


def normalize_to_common_name(name, name_mapping):
    """Normalize any species name to CommonName using the mapping"""
    if name in ['Empty Sample', 'Tree Weta', 'Spy Bird', None, '']:
        return None
    
    if name in name_mapping:
        return name_mapping[name]
    
    name_lower = name.lower()
    for key, value in name_mapping.items():
        if key.lower() == name_lower:
            return value
    
    if '(' in name:
        base_name = name.split('(')[0].strip()
        if base_name in name_mapping:
            return name_mapping[base_name]
    
    print(f"WARNING: No mapping found for species '{name}'")
    return None


def align_predictions_with_gt(predictions, ground_truth, name_mapping):
    """Convert all predictions and ground truth to CommonName format"""
    aligned_predictions = {}
    unmapped_species = set()
    mapped_count = 0
    total_pred_species = set()
    
    for row_id, pred_species in predictions.items():
        # Map prediction species to CommonName
        common_species = set()
        for species in pred_species:
            total_pred_species.add(species)
            common = normalize_to_common_name(species, name_mapping)
            if common:
                common_species.add(common)
                mapped_count += 1
            else:
                unmapped_species.add(species)
        aligned_predictions[row_id] = common_species
    
    if unmapped_species:
        # Filter out known non-bird species that are intentionally excluded
        real_unmapped = unmapped_species - {'Spy Bird', 'Tree Weta'}
        if real_unmapped:
            print(f"  ERROR: {len(real_unmapped)} unmapped prediction species (not in DOC_bird_naming_map.csv):")
            for sp in sorted(real_unmapped)[:20]:
                print(f"    - {sp}")
            if len(real_unmapped) > 20:
                print(f"    ... and {len(real_unmapped) - 20} more")
            raise ValueError(f"Cannot evaluate model with {len(real_unmapped)} unmapped species!")
        else:
            print(f"  Info: Ignoring {len(unmapped_species)} non-bird species: {sorted(unmapped_species)}")
    
    return aligned_predictions


def normalize_ground_truth(ground_truth, name_mapping):
    """Convert ground truth to CommonName format"""
    normalized_gt = {}
    unmapped_species = set()
    
    for row_id, gt_species in ground_truth.items():
        common_species = set()
        for species in gt_species:
            common = normalize_to_common_name(species, name_mapping)
            if common:
                common_species.add(common)
            else:
                unmapped_species.add(species)
        normalized_gt[row_id] = common_species
    
    if unmapped_species:
        print(f"  Warning: {len(unmapped_species)} unmapped GT species: {sorted(unmapped_species)}")
    
    return normalized_gt


def create_binary_matrix(predictions, ground_truth, all_species):
    """Create binary matrices for predictions and ground truth"""
    common_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    
    n_samples = len(common_ids)

    all_species_sorted = sorted(all_species)
    n_species = len(all_species_sorted)
    species_to_idx = {species: idx for idx, species in enumerate(all_species_sorted)}
    
    y_true = np.zeros((n_samples, n_species), dtype=int)
    y_pred = np.zeros((n_samples, n_species), dtype=int)
    
    for sample_idx, row_id in enumerate(common_ids):
        gt_species = ground_truth[row_id]
        for species in gt_species:
            if species in species_to_idx:
                y_true[sample_idx, species_to_idx[species]] = 1
        
        pred_species = predictions[row_id]
        for species in pred_species:
            if species in species_to_idx:
                y_pred[sample_idx, species_to_idx[species]] = 1
    
    return y_true, y_pred, all_species_sorted, common_ids


def optimize_threshold(predictions_probs, ground_truth, name_mapping, val_ids, all_species, metric='f1_micro'):
    """Find optimal threshold using validation set"""
    thresholds_to_test = np.linspace(0, 1.0, 21)
    best_threshold = 0.5
    best_score = 0.0
    print(f"  Optimizing threshold on {len(val_ids)} validation samples (metric: {metric})...")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'F1-Micro':<10} {'F1-Macro':<10} {'F1-Weighted':<12}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    for threshold in thresholds_to_test:
        predictions = apply_threshold_to_probs(predictions_probs, threshold)
        predictions_normalized = align_predictions_with_gt(predictions, ground_truth, name_mapping)
        val_ground_truth = {k: v for k, v in ground_truth.items() if k in val_ids}
        val_predictions = {k: v for k, v in predictions_normalized.items() if k in val_ids}
        y_true, y_pred, species_list, common_ids = create_binary_matrix(
            val_predictions, val_ground_truth, all_species
        )
        if len(common_ids) == 0:
            continue
        
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"  {threshold:<12.1f} {acc:<10.4f} {f1_micro:<10.4f} {f1_macro:<10.4f} {f1_weighted:<12.4f}")
        
        if metric == 'accuracy':
            score = acc
        elif metric == 'f1_micro':
            score = f1_micro
        elif metric == 'f1_macro':
            score = f1_macro
        elif metric == 'f1_weighted':
            score = f1_weighted
        else:
            score = f1_micro
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"  -> Optimal threshold: {best_threshold} ({metric}={best_score:.4f})")
    return best_threshold


def calculate_metrics(y_true, y_pred, species_list):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    if y_true.size == 0 or y_pred.size == 0:
        print("Warning: Empty prediction or ground truth data")
        return metrics, []
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    try:
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    except ZeroDivisionError:
        metrics['hamming_loss'] = 0.0
    
    for avg in ['micro', 'macro', 'weighted', 'samples']:
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    per_class_metrics = []
    for idx, species in enumerate(species_list):
        y_true_species = y_true[:, idx]
        y_pred_species = y_pred[:, idx]
        
        if y_true_species.sum() == 0:
            continue
        
        tn, fp, fn, tp = confusion_matrix(y_true_species, y_pred_species, labels=[0, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics.append({
            'species': species,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(tp + fn)
        })
    
    return metrics, per_class_metrics


def sample_level_analysis(predictions, ground_truth, common_ids):
    """Analyze predictions at the sample level"""
    if len(common_ids) == 0:
        return {
            'exact_match': 0,
            'partial_match': 0,
            'true_negatives': 0,
            'exact_match_rate': 0.0,
            'partial_match_rate': 0.0,
            'true_negative_rate': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1': 0.0
        }
    
    sample_metrics = {
        'exact_match': 0,
        'partial_match': 0,
        'true_negatives': 0,
        'avg_precision_per_sample': [],
        'avg_recall_per_sample': [],
        'avg_f1_per_sample': []
    }
    
    for row_id in common_ids:
        gt = ground_truth[row_id]
        pred = predictions[row_id]
        
        if gt == pred:
            sample_metrics['exact_match'] += 1
        
        if len(gt & pred) > 0:
            sample_metrics['partial_match'] += 1
        
        if len(gt) == 0 and len(pred) == 0:
            sample_metrics['true_negatives'] += 1
        
        if len(gt) > 0 or len(pred) > 0:
            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)
            
            precision = tp / len(pred) if len(pred) > 0 else 0
            recall = tp / len(gt) if len(gt) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            sample_metrics['avg_precision_per_sample'].append(precision)
            sample_metrics['avg_recall_per_sample'].append(recall)
            sample_metrics['avg_f1_per_sample'].append(f1)
    
    sample_metrics['exact_match_rate'] = sample_metrics['exact_match'] / len(common_ids)
    sample_metrics['partial_match_rate'] = sample_metrics['partial_match'] / len(common_ids)
    sample_metrics['true_negative_rate'] = sample_metrics['true_negatives'] / len(common_ids)
    
    if sample_metrics['avg_precision_per_sample']:
        sample_metrics['mean_precision'] = np.mean(sample_metrics['avg_precision_per_sample'])
        sample_metrics['mean_recall'] = np.mean(sample_metrics['avg_recall_per_sample'])
        sample_metrics['mean_f1'] = np.mean(sample_metrics['avg_f1_per_sample'])
    
    return sample_metrics


def generate_confusion_matrices(y_true, y_pred, species_list, model_name, output_dir, gt_species, threshold=None):
    """Generate and save confusion matrices for a single model"""
    print(f"\nGenerating confusion matrices for {model_name}...")
    if threshold is not None:
        model_name_with_threshold = f"{model_name}_t{threshold:.2f}"
    else:
        model_name_with_threshold = model_name
    
    confusion_data = []
    
    for idx, species in enumerate(species_list):
        y_true_species = y_true[:, idx]
        y_pred_species = y_pred[:, idx]
        
        cm = confusion_matrix(y_true_species, y_pred_species, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        confusion_data.append({
            'Species': species,
            'True_Negative': int(tn),
            'False_Positive': int(fp),
            'False_Negative': int(fn),
            'True_Positive': int(tp),
            'Total_Actual_Positive': int(tp + fn),
            'Total_Actual_Negative': int(tn + fp),
            'Total_Predicted_Positive': int(tp + fp),
            'Total_Predicted_Negative': int(tn + fn)
        })
    
    df = pd.DataFrame(confusion_data)
    output_path = f"{output_dir}/{model_name_with_threshold.replace(' ', '_')}_confusion_matrices.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")
    
    has_no_birds = 'No Birds' in species_list
    row_species_set = set(gt_species)
    if has_no_birds:
        row_species_set.add('No Birds')
    row_species = sorted(row_species_set)
    
    gt_sorted = sorted(gt_species)
    non_gt_species = sorted(set(species_list) - gt_species - {'No Birds'})
    if has_no_birds:
        col_species = gt_sorted + non_gt_species + ['No Birds']
    else:
        col_species = gt_sorted + non_gt_species
    
    species_confusion = pd.DataFrame(0, index=row_species, columns=col_species)
    
    for sample_idx in range(y_true.shape[0]):
        true_species = [species_list[i] for i in range(len(species_list)) if y_true[sample_idx, i] == 1]
        pred_species = [species_list[i] for i in range(len(species_list)) if y_pred[sample_idx, i] == 1]
        
        for true_sp in true_species:
            if true_sp in row_species:
                for pred_sp in pred_species:
                    if pred_sp in col_species:
                        species_confusion.loc[true_sp, pred_sp] += 1
    
    output_path = f"{output_dir}/{model_name_with_threshold.replace(' ', '_')}_species_confusion.csv"
    species_confusion.to_csv(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")
    
    if len(row_species) > 0:
        fig_height = max(10, len(row_species) * 0.5)
        fig_width = max(12, len(col_species) * 0.5)
        annot_size = max(6, min(10, 200 // max(len(row_species), len(col_species))))
        
        species_confusion_norm = species_confusion.div(species_confusion.sum(axis=1), axis=0).fillna(0)
        
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(species_confusion_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                    annot_kws={'size': annot_size},
                    cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        plt.title(f'{model_name} - Species Confusion Matrix (Normalized)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Species', fontsize=12, fontweight='bold')
        plt.ylabel('True Species (GT only)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        img_path = f"{output_dir}/{model_name_with_threshold.replace(' ', '_')}_confusion_normalized.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(img_path)}")
    
    return confusion_data


def discover_prediction_files(results_folder):
    """Discover all prediction CSV files in the results folder"""
    pred_files = []
    
    for filename in os.listdir(results_folder):
        if filename.endswith('.csv') and filename != 'labels.csv':
            if 'pred' in filename.lower() or 'prediction' in filename.lower():
                pred_files.append(filename)
    
    return sorted(pred_files)


def get_model_name_from_filename(filename):
    """Extract a clean model name from the prediction filename"""
    name = filename.replace('.csv', '')
    name = name.replace('_preds', '').replace('preds_', '').replace('prediction_probabilities', 'kaytoo')
    
    if name.startswith('ast_'):
        name = name.replace('ast_', 'AST_')
    
    return name.replace('_', ' ').title()


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple bird species detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python compare_models.py MoreporkResults\n'
               '  python compare_models.py Joe_MO_Results --threshold 0.3'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing labels.json and prediction CSV files'
    )
    parser.add_argument(
        '--ignore-non-gt-preds',
        action='store_true',
        help='Ignore predictions for species not present in the ground truth label set (restricts evaluation and threshold optimization to GT species only)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for positive prediction (default: 0.5, ignored if --optimize-threshold is used)'
    )
    parser.add_argument(
        '--optimize-threshold',
        action='store_true',
        help='Automatically find optimal threshold per model using validation set'
    )
    parser.add_argument(
        '--val-size',
        type=int,
        default=5000,
        help='Validation set size for threshold optimization (default: 5000)'
    )
    parser.add_argument(
        '--optimization-metric',
        type=str,
        default='f1_micro',
        choices=['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted'],
        help='Metric to optimize when finding threshold (default: f1_micro). Use accuracy for imbalanced datasets.'
    )
    parser.add_argument(
        '--birds-only',
        action='store_true',
        help='Exclude samples with no birds from evaluation (only evaluate bird detection performance)'
    )
    
    args = parser.parse_args()
    
    base_dir = '/home/giotto/Desktop/AviaNZ'
    results_folder = os.path.join(base_dir, args.folder)
    
    labels_path = os.path.join(results_folder, 'labels.json')
    naming_map_path = os.path.join(base_dir, 'DOC_bird_naming_map.csv')
    
    if not os.path.exists(labels_path):
        print(f"Error: labels.json not found in {results_folder}")
        return
    
    output_dir = os.path.join(results_folder, 'comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MULTI-MODEL BIRD SPECIES DETECTION EVALUATION")
    print("="*80)
    print(f"\nResults folder: {args.folder}")
    print(f"Output directory: {output_dir}")
    if args.optimize_threshold:
        print(f"Threshold: AUTO-OPTIMIZE (validation size: {args.val_size}, metric: {args.optimization_metric})")
    else:
        print(f"Threshold: {args.threshold} (fixed)")
    print(f"Ignore non-GT predictions: {args.ignore_non_gt_preds}")
    print(f"Birds only (exclude empty samples): {args.birds_only}")
    
    print("\nLoading bird naming map...")
    name_mapping = load_bird_naming_map(naming_map_path)
    print(f"  Loaded {len(name_mapping)} name mappings")
    
    ground_truth, gt_categories = load_ground_truth(labels_path)
    
    pred_files = discover_prediction_files(results_folder)
    
    if not pred_files:
        print(f"\nError: No prediction CSV files found in {results_folder}")
        print("Looking for files with 'pred' or 'prediction' in the name")
        return
    
    print(f"\nFound {len(pred_files)} prediction files:")
    for pf in pred_files:
        print(f"  - {pf}")
    
    all_models = {}
    val_ids = None  # Will be set after filtering
    
    for pred_file in pred_files:
        model_name = get_model_name_from_filename(pred_file)
        pred_path = os.path.join(results_folder, pred_file)
        
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"{'='*80}")
        
        if args.optimize_threshold:
            predictions_probs, species_cols = load_predictions_raw(pred_path)
            predictions = None
        else:
            predictions, species_cols = load_predictions(pred_path, args.threshold)
            predictions_probs = None
        
        all_species_normalized = set()
        for species in species_cols:
            common = normalize_to_common_name(species, name_mapping)
            if common:
                all_species_normalized.add(common)
        print(f"  Model can predict {len(all_species_normalized)} species after normalization")
        
        if 'ground_truth_normalized' not in locals():
            print("\nNormalizing ground truth...")
            ground_truth_normalized = normalize_ground_truth(ground_truth, name_mapping)
            
            # Filter out empty samples if birds-only mode is enabled
            if args.birds_only:
                original_count = len(ground_truth_normalized)
                ground_truth_normalized = {k: v for k, v in ground_truth_normalized.items() if len(v) > 0}
                filtered_count = original_count - len(ground_truth_normalized)
                print(f"  Filtered out {filtered_count} empty samples ({filtered_count/original_count*100:.1f}%)")
            
            # Split validation set AFTER filtering for birds-only
            if args.optimize_threshold:
                val_ids = split_validation_set(ground_truth_normalized, val_size=args.val_size)
                print(f"\nValidation set: {len(val_ids)} samples ({len(val_ids)/len(ground_truth_normalized)*100:.1f}%)")
            
            all_species_in_gt = set()
            for species_set in ground_truth_normalized.values():
                all_species_in_gt.update(species_set)
            print(f"  Total unique species in ground truth: {len(all_species_in_gt)}")
        
        all_species_for_model = all_species_normalized | all_species_in_gt

        if args.ignore_non_gt_preds:
            all_species_for_model = all_species_in_gt
        
        if args.optimize_threshold:
            optimal_threshold = optimize_threshold(
                predictions_probs, ground_truth_normalized, name_mapping, 
                val_ids, all_species_for_model,
                metric=args.optimization_metric
            )
            predictions = apply_threshold_to_probs(predictions_probs, optimal_threshold)
        else:
            optimal_threshold = args.threshold
        
        model_name_with_threshold = f"{model_name}_t{optimal_threshold:.2f}" if args.optimize_threshold else model_name
        clean_name = model_name_with_threshold.replace(' ', '_')
        confusion_csv = os.path.join(output_dir, f"{clean_name}_confusion_matrices.csv")
        species_csv = os.path.join(output_dir, f"{clean_name}_species_confusion.csv")
        confusion_png = os.path.join(output_dir, f"{clean_name}_confusion_normalized.png")

        already_processed = os.path.exists(confusion_csv) and os.path.exists(species_csv) and os.path.exists(confusion_png)
        if already_processed:
            print(f"  Found existing confusion outputs for threshold {optimal_threshold:.2f} (will reuse plots, still recomputing metrics)")
        
        print(f"  Normalizing species names...")
        predictions_normalized = align_predictions_with_gt(predictions, ground_truth_normalized, name_mapping)

        if args.ignore_non_gt_preds:
            predictions_normalized = {k: (v & all_species_in_gt) for k, v in predictions_normalized.items()}
        
        all_models[model_name] = {
            'predictions': predictions_normalized,
            'all_species': all_species_for_model,
            'threshold': optimal_threshold
        }

    # NOTE: Do not merge metrics across runs.
    # Results depend on labels.json contents and CLI flags; reusing old rows can silently produce stale comparisons.
    existing_metrics = {}
    
    all_metrics = []
    
    for model_name, model_data in all_models.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        predictions_normalized = model_data['predictions']
        all_species_for_model = model_data['all_species']
        optimal_threshold = model_data['threshold']
        
        y_true, y_pred, species_list, common_ids = create_binary_matrix(
            predictions_normalized, ground_truth_normalized, all_species_for_model
        )
        
        overall_metrics, per_class_metrics = calculate_metrics(y_true, y_pred, species_list)
        sample_metrics = sample_level_analysis(predictions_normalized, ground_truth_normalized, common_ids)
        
        print(f"  Evaluated on {len(common_ids)} samples")
        print(f"  Species evaluated: {len(species_list)} (GT: {len(all_species_in_gt)}, Considered: {len(all_species_for_model)})")
        print(f"  Accuracy: {overall_metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {overall_metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (micro): {overall_metrics.get('f1_micro', 0):.4f}")

        # Only regenerate confusion plots if they don't already exist for this threshold.
        model_name_with_threshold = f"{model_name}_t{optimal_threshold:.2f}" if args.optimize_threshold else model_name
        clean_name = model_name_with_threshold.replace(' ', '_')
        confusion_csv = os.path.join(output_dir, f"{clean_name}_confusion_matrices.csv")
        species_csv = os.path.join(output_dir, f"{clean_name}_species_confusion.csv")
        confusion_png = os.path.join(output_dir, f"{clean_name}_confusion_normalized.png")
        if os.path.exists(confusion_csv) and os.path.exists(species_csv) and os.path.exists(confusion_png):
            print(f"  Confusion outputs already exist for threshold {optimal_threshold:.2f}, skipping plot generation")
        else:
            generate_confusion_matrices(
                y_true, y_pred, species_list, model_name, output_dir,
                all_species_in_gt, threshold=optimal_threshold
            )
        
        metrics_row = {
            'model': model_name,
            'threshold': optimal_threshold,
            'num_samples': len(common_ids),
            'num_species_evaluated': len(species_list),
            **overall_metrics,
            'exact_match_rate': sample_metrics['exact_match_rate'],
            'partial_match_rate': sample_metrics['partial_match_rate']
        }
        all_metrics.append(metrics_row)
        
        model_data['overall_metrics'] = overall_metrics
        model_data['per_class_metrics'] = per_class_metrics
        model_data['sample_metrics'] = sample_metrics
    
    print(f"\n{'='*80}")
    print("SAVING COMPARISON RESULTS")
    print(f"{'='*80}")
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_output = os.path.join(output_dir, 'all_models_metrics.csv')
    metrics_df.to_csv(metrics_output, index=False)
    print(f"\nSaved unified metrics: {os.path.basename(metrics_output)}")
    
    print("\nSaved confusion matrices for each model:")
    for model_name in all_models.keys():
        clean_name = model_name.replace(' ', '_')
        print(f"  {clean_name}_confusion_matrices.csv")
        print(f"  {clean_name}_species_confusion.csv")
        print(f"  {clean_name}_confusion_normalized.png")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
