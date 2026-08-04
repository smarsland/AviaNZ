#!/usr/bin/env python3
"""
Analyse all experimental results, evaluating exactly 7 conditions based on
the prediction CSV files that actually exist in each experiment directory.

Conditions:
1. DOC matched with validation thresholds
2. DOC matched with DOC matched thresholds
3. DOC matched with AviaNZ matched thresholds
4. AviaNZ matched with validation thresholds
5. AviaNZ matched with DOC matched thresholds
6. AviaNZ matched with AviaNZ matched thresholds
7. Validation with validation thresholds

For each condition: macro F1, micro F1, exact accuracy, labelled-only exact accuracy.
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support as _prf
from sklearn.metrics import jaccard_score as _jaccard


# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

def _apply_pred_remap(df: pd.DataFrame, pred_remap: dict) -> pd.DataFrame:
    """Remap prediction columns: merge multiple sources into one target by max."""
    from collections import defaultdict
    pred_cols = [c for c in df.columns if not c.startswith('true_')]
    true_cols = [c for c in df.columns if c.startswith('true_')]

    groups = defaultdict(list)
    for col in pred_cols:
        groups[pred_remap.get(col, col)].append(col)

    new_pred = {}
    for target, sources in groups.items():
        if len(sources) == 1 and sources[0] == target:
            new_pred[target] = df[sources[0]]
        else:
            new_pred[target] = df[sources].max(axis=1)

    result = pd.DataFrame(new_pred, index=df.index)
    for tc in true_cols:
        result[tc] = df[tc]
    return result


def metrics_from_csv(csv_path: Path, apply_thresholds=None, pred_remap=None) -> dict:
    """
    Compute metrics from a single predictions CSV.
    Returns:
      half_*      : fixed threshold 0.5
      oracle_*    : per-class thresholds tuned on this split
      cross_*     : when apply_thresholds is given (dict mapping class->threshold)
    All metrics are computed only over classes present in the ground truth.
    """
    nan_result = {
        'half_precision': np.nan, 'half_recall': np.nan, 'half_f1': np.nan,
        'half_micro_f1': np.nan, 'half_jaccard': np.nan,
        'half_acc': np.nan, 'half_acc_labelled': np.nan,
        'oracle_precision': np.nan, 'oracle_recall': np.nan, 'oracle_f1': np.nan,
        'oracle_micro_f1': np.nan, 'oracle_jaccard': np.nan,
        'oracle_acc': np.nan, 'oracle_acc_labelled': np.nan,
        'oracle_thresholds': None,           # numpy array (for internal use)
        'oracle_threshold_dict': {},         # dict class->threshold (for cross‑application)
        'oracle_class_names': [],
        'cross_f1': np.nan, 'cross_micro_f1': np.nan,
        'cross_acc': np.nan, 'cross_acc_labelled': np.nan,
    }

    df = pd.read_csv(csv_path, index_col='filename')
    if pred_remap:
        df = _apply_pred_remap(df, pred_remap)

    class_cols = [c for c in df.columns if not c.startswith('true_')]
    true_cols = [c for c in df.columns if c.startswith('true_')]
    if not true_cols:
        return nan_result

    true_class_names = [c[len('true_'):] for c in true_cols]
    pred_name_to_idx = {c: i for i, c in enumerate(class_cols)}

    # Align predictions to true classes (missing -> 0)
    probs = np.zeros((len(df), len(true_cols)), dtype=np.float32)
    for j, tc in enumerate(true_class_names):
        if tc in pred_name_to_idx:
            probs[:, j] = df[class_cols[pred_name_to_idx[tc]]].astype(np.float32).values

    trues = df[true_cols].values.astype(np.int32)

    # Only classes present in ground truth
    present = trues.sum(axis=0) > 0
    present_idx = np.where(present)[0]
    if len(present_idx) == 0:
        return nan_result

    probs_p = probs[:, present_idx]
    trues_p = trues[:, present_idx]
    present_class_names = [true_class_names[i] for i in present_idx]
    labelled = trues_p.sum(axis=1) > 0

    def macro_metrics(preds):
        """Precision, recall, macro F1, micro F1, macro Jaccard."""
        if preds.shape[1] == 0:
            return (np.nan,)*5
        p, r, f1, _ = _prf(trues_p, preds, average='macro', zero_division=0)
        mf1 = _prf(trues_p, preds, average='micro', zero_division=0)[2]
        jac = float(_jaccard(trues_p, preds, average='macro', zero_division=0))
        return float(p), float(r), float(f1), float(mf1), jac

    # ----- threshold 0.5 -----
    preds_half = (probs_p >= 0.5).astype(int)
    p_h, r_h, f1_h, mf1_h, jac_h = macro_metrics(preds_half)
    correct_h = np.all(preds_half == trues_p, axis=1)
    acc_h = float(correct_h.mean() * 100)
    acc_lab_h = float(correct_h[labelled].mean() * 100) if labelled.any() else np.nan

    # ----- oracle thresholds (per class, tuned on this split) -----
    candidates = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    n_present = len(present_idx)
    thresh_arr = np.full(n_present, 1.0, dtype=np.float32)
    for c in range(n_present):
        tc = trues_p[:, c]
        pc = probs_p[:, c]
        if tc.sum() == 0:
            continue
        preds_all = (pc[np.newaxis, :] >= candidates[:, np.newaxis]).astype(np.int32)
        pos_mask = preds_all.sum(axis=1) > 0
        if not pos_mask.any():
            continue
        tp = (preds_all * tc[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fp = (preds_all * (1 - tc)[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fn = ((1 - preds_all) * tc[np.newaxis, :]).sum(axis=1).astype(np.float32)
        denom = 2 * tp + fp + fn
        f1s = np.where(denom > 0, 2 * tp / denom, 0.0)
        f1s[~pos_mask] = -1.0
        thresh_arr[c] = candidates[np.argmax(f1s)]

    preds_oracle = (probs_p >= thresh_arr[np.newaxis, :]).astype(int)
    p_o, r_o, f1_o, mf1_o, jac_o = macro_metrics(preds_oracle)
    correct_o = np.all(preds_oracle == trues_p, axis=1)
    acc_o = float(correct_o.mean() * 100)
    acc_lab_o = float(correct_o[labelled].mean() * 100) if labelled.any() else np.nan

    # Build dict mapping class->threshold for cross‑application
    thresh_dict = {cls: float(thresh_arr[i]) for i, cls in enumerate(present_class_names)}

    result = {
        'half_precision': p_h, 'half_recall': r_h, 'half_f1': f1_h,
        'half_micro_f1': mf1_h, 'half_jaccard': jac_h,
        'half_acc': acc_h, 'half_acc_labelled': acc_lab_h,
        'oracle_precision': p_o, 'oracle_recall': r_o, 'oracle_f1': f1_o,
        'oracle_micro_f1': mf1_o, 'oracle_jaccard': jac_o,
        'oracle_acc': acc_o, 'oracle_acc_labelled': acc_lab_o,
        'oracle_thresholds': thresh_arr,
        'oracle_threshold_dict': thresh_dict,
        'oracle_class_names': present_class_names,
        'cross_f1': np.nan, 'cross_micro_f1': np.nan,
        'cross_acc': np.nan, 'cross_acc_labelled': np.nan,
    }

    # ----- cross thresholds (if provided) -----
    if apply_thresholds is not None:
        # apply_thresholds should be a dict mapping class name -> threshold
        if isinstance(apply_thresholds, dict):
            thresh_cross = np.full(n_present, 0.5, dtype=np.float32)
            for i, cls in enumerate(present_class_names):
                thresh_cross[i] = apply_thresholds.get(cls, 0.5)
            preds_cross = (probs_p >= thresh_cross[np.newaxis, :]).astype(int)
            _, _, f1_c, mf1_c, _ = macro_metrics(preds_cross)
            correct_c = np.all(preds_cross == trues_p, axis=1)
            acc_c = float(correct_c.mean() * 100)
            acc_lab_c = float(correct_c[labelled].mean() * 100) if labelled.any() else np.nan
            result['cross_f1'] = f1_c
            result['cross_micro_f1'] = mf1_c
            result['cross_acc'] = acc_c
            result['cross_acc_labelled'] = acc_lab_c
        else:
            # If an array is passed, try to use it only if lengths match (fallback)
            thresh_arr = np.asarray(apply_thresholds, dtype=np.float32)
            if len(thresh_arr) == n_present:
                preds_cross = (probs_p >= thresh_arr[np.newaxis, :]).astype(int)
                _, _, f1_c, mf1_c, _ = macro_metrics(preds_cross)
                correct_c = np.all(preds_cross == trues_p, axis=1)
                acc_c = float(correct_c.mean() * 100)
                acc_lab_c = float(correct_c[labelled].mean() * 100) if labelled.any() else np.nan
                result['cross_f1'] = f1_c
                result['cross_micro_f1'] = mf1_c
                result['cross_acc'] = acc_c
                result['cross_acc_labelled'] = acc_lab_c

    return result


def _read_adaptive_from_dir(exp_dir: Path, row: dict, pred_remap: dict = None):
    """
    Populate row with metrics from predictions CSVs.
    Uses only files that actually exist.
    """
    val_csv = exp_dir / 'predictions_val.csv'
    doc_csv = exp_dir / 'predictions_doc_split.csv'
    avianz_csv = exp_dir / 'predictions_avianz_split.csv'

    # Name fields: set from filenames if present
    if val_csv.exists():
        row['val_name'] = 'validation'
    if doc_csv.exists():
        row['doc_name'] = 'doc_split'
    if avianz_csv.exists():
        row['avianz_name'] = 'avianz_split'

    # Store thresholds per dataset as DICTs (class->threshold)
    thresholds = {}

    # ---- Validation split ----
    if val_csv.exists():
        m = metrics_from_csv(val_csv, pred_remap=pred_remap)
        thresholds['validation'] = m['oracle_threshold_dict']   # store dict
        # fixed-0.5 metrics
        row['val_macro_f1'] = m['half_f1']
        row['val_micro_f1'] = m['half_micro_f1']
        row['val_acc'] = m['half_acc']
        row['val_acc_labelled'] = m['half_acc_labelled']
        # condition 7: val with val thresholds
        if m['oracle_threshold_dict']:
            m2 = metrics_from_csv(val_csv, apply_thresholds=m['oracle_threshold_dict'],
                                  pred_remap=pred_remap)
            row['val_validation_threshold_macro_f1'] = m2['cross_f1']
            row['val_validation_threshold_micro_f1'] = m2['cross_micro_f1']
            row['val_validation_threshold_acc'] = m2['cross_acc']
            row['val_validation_threshold_acc_labelled'] = m2['cross_acc_labelled']

    # ---- DOC split ----
    if doc_csv.exists():
        m = metrics_from_csv(doc_csv, pred_remap=pred_remap)
        thresholds['doc'] = m['oracle_threshold_dict']
        row['doc_macro_f1'] = m['half_f1']
        row['doc_micro_f1'] = m['half_micro_f1']
        row['doc_acc'] = m['half_acc']
        row['doc_acc_labelled'] = m['half_acc_labelled']
        # condition 2: doc with doc thresholds
        if m['oracle_threshold_dict']:
            m2 = metrics_from_csv(doc_csv, apply_thresholds=m['oracle_threshold_dict'],
                                  pred_remap=pred_remap)
            row['doc_matched_doc_threshold_macro_f1'] = m2['cross_f1']
            row['doc_matched_doc_threshold_micro_f1'] = m2['cross_micro_f1']
            row['doc_matched_doc_threshold_acc'] = m2['cross_acc']
            row['doc_matched_doc_threshold_acc_labelled'] = m2['cross_acc_labelled']

    # ---- AviaNZ split ----
    if avianz_csv.exists():
        m = metrics_from_csv(avianz_csv, pred_remap=pred_remap)
        thresholds['avianz'] = m['oracle_threshold_dict']
        row['avianz_macro_f1'] = m['half_f1']
        row['avianz_micro_f1'] = m['half_micro_f1']
        row['avianz_acc'] = m['half_acc']
        row['avianz_acc_labelled'] = m['half_acc_labelled']
        # condition 6: avianz with avianz thresholds
        if m['oracle_threshold_dict']:
            m2 = metrics_from_csv(avianz_csv, apply_thresholds=m['oracle_threshold_dict'],
                                  pred_remap=pred_remap)
            row['avianz_matched_avianz_threshold_macro_f1'] = m2['cross_f1']
            row['avianz_matched_avianz_threshold_micro_f1'] = m2['cross_micro_f1']
            row['avianz_matched_avianz_threshold_acc'] = m2['cross_acc']
            row['avianz_matched_avianz_threshold_acc_labelled'] = m2['cross_acc_labelled']

    # ---- Cross conditions ----
    # Condition 1: DOC with validation thresholds
    if doc_csv.exists() and thresholds.get('validation'):
        m = metrics_from_csv(doc_csv, apply_thresholds=thresholds['validation'],
                             pred_remap=pred_remap)
        row['doc_matched_validation_threshold_macro_f1'] = m['cross_f1']
        row['doc_matched_validation_threshold_micro_f1'] = m['cross_micro_f1']
        row['doc_matched_validation_threshold_acc'] = m['cross_acc']
        row['doc_matched_validation_threshold_acc_labelled'] = m['cross_acc_labelled']

    # Condition 3: DOC with AviaNZ thresholds
    if doc_csv.exists() and thresholds.get('avianz'):
        m = metrics_from_csv(doc_csv, apply_thresholds=thresholds['avianz'],
                             pred_remap=pred_remap)
        row['doc_matched_avianz_threshold_macro_f1'] = m['cross_f1']
        row['doc_matched_avianz_threshold_micro_f1'] = m['cross_micro_f1']
        row['doc_matched_avianz_threshold_acc'] = m['cross_acc']
        row['doc_matched_avianz_threshold_acc_labelled'] = m['cross_acc_labelled']

    # Condition 4: AviaNZ with validation thresholds
    if avianz_csv.exists() and thresholds.get('validation'):
        m = metrics_from_csv(avianz_csv, apply_thresholds=thresholds['validation'],
                             pred_remap=pred_remap)
        row['avianz_matched_validation_threshold_macro_f1'] = m['cross_f1']
        row['avianz_matched_validation_threshold_micro_f1'] = m['cross_micro_f1']
        row['avianz_matched_validation_threshold_acc'] = m['cross_acc']
        row['avianz_matched_validation_threshold_acc_labelled'] = m['cross_acc_labelled']

    # Condition 5: AviaNZ with DOC thresholds
    if avianz_csv.exists() and thresholds.get('doc'):
        m = metrics_from_csv(avianz_csv, apply_thresholds=thresholds['doc'],
                             pred_remap=pred_remap)
        row['avianz_matched_doc_threshold_macro_f1'] = m['cross_f1']
        row['avianz_matched_doc_threshold_micro_f1'] = m['cross_micro_f1']
        row['avianz_matched_doc_threshold_acc'] = m['cross_acc']
        row['avianz_matched_doc_threshold_acc_labelled'] = m['cross_acc_labelled']


# -----------------------------------------------------------------------------
#  Loading experiments (unchanged from previous working version)
# -----------------------------------------------------------------------------

def _extract_report_metrics(report):
    """Pull scalar metrics from a multilabel_report.json dict."""
    def pct(v):
        return v * 100 if not (np.isnan(v) or v != v) else np.nan
    macro = report.get('macro avg', {})
    return {
        'acc': pct(report.get('exact_match_accuracy', np.nan)),
        'acc_lab': pct(report.get('exact_match_accuracy_labelled', np.nan)),
        'acc_bg': pct(report.get('exact_match_accuracy_background', np.nan)),
        'macro_p': macro.get('precision', np.nan),
        'macro_r': macro.get('recall', np.nan),
        'macro_f1': macro.get('f1-score', np.nan),
        'jaccard': report.get('jaccard_score', np.nan),
    }


def _read_reports_from_dir(report_dir, model, row):
    """Read multilabel_report.json files and set row fields."""
    for report_file in sorted(report_dir.glob('*_multilabel_report.json')):
        with open(report_file) as f:
            report = json.load(f)
        stem = report_file.stem.replace('_multilabel_report', '')
        if '_model' in stem:
            continue  # validation set summary, skip

        m = _extract_report_metrics(report)
        # Determine which split from the stem
        if 'val' in stem or 'validation' in stem:
            row['val_name'] = re.sub(rf'^{model}_test_', '', stem)
            row['val_acc'] = m['acc']
            row['val_acc_labelled'] = m['acc_lab']
            row['val_acc_background'] = m['acc_bg']
            row['val_macro_precision'] = m['macro_p']
            row['val_macro_recall'] = m['macro_r']
            row['val_macro_f1'] = m['macro_f1']
            row['val_jaccard'] = m['jaccard']
        elif 'doc' in stem:
            row['doc_name'] = re.sub(rf'^{model}_test_', '', stem)
            row['doc_acc'] = m['acc']
            row['doc_acc_labelled'] = m['acc_lab']
            row['doc_acc_background'] = m['acc_bg']
            row['doc_macro_precision'] = m['macro_p']
            row['doc_macro_recall'] = m['macro_r']
            row['doc_macro_f1'] = m['macro_f1']
            row['doc_jaccard'] = m['jaccard']
        elif 'avianz' in stem:
            row['avianz_name'] = re.sub(rf'^{model}_test_', '', stem)
            row['avianz_acc'] = m['acc']
            row['avianz_acc_labelled'] = m['acc_lab']
            row['avianz_acc_background'] = m['acc_bg']
            row['avianz_macro_precision'] = m['macro_p']
            row['avianz_macro_recall'] = m['macro_r']
            row['avianz_macro_f1'] = m['macro_f1']
            row['avianz_jaccard'] = m['jaccard']


def _empty_row(name, train_dataset, model, config):
    return {
        'name': name,
        'train_dataset': train_dataset,
        'method': model,
        'config': config,
        'category': model.upper(),
        'seed': 0,
        # Fixed threshold 0.5 metrics
        'val_macro_f1': np.nan, 'val_micro_f1': np.nan,
        'val_acc': np.nan, 'val_acc_labelled': np.nan,
        'doc_macro_f1': np.nan, 'doc_micro_f1': np.nan,
        'doc_acc': np.nan, 'doc_acc_labelled': np.nan,
        'avianz_macro_f1': np.nan, 'avianz_micro_f1': np.nan,
        'avianz_acc': np.nan, 'avianz_acc_labelled': np.nan,
        # Names
        'val_name': np.nan, 'doc_name': np.nan, 'avianz_name': np.nan,
        # Extra from reports (optional)
        'val_acc_background': np.nan, 'val_macro_precision': np.nan,
        'val_macro_recall': np.nan, 'val_jaccard': np.nan,
        'doc_acc_background': np.nan, 'doc_macro_precision': np.nan,
        'doc_macro_recall': np.nan, 'doc_jaccard': np.nan,
        'avianz_acc_background': np.nan, 'avianz_macro_precision': np.nan,
        'avianz_macro_recall': np.nan, 'avianz_jaccard': np.nan,
        # 7 conditions
        'doc_matched_validation_threshold_macro_f1': np.nan,
        'doc_matched_validation_threshold_micro_f1': np.nan,
        'doc_matched_validation_threshold_acc': np.nan,
        'doc_matched_validation_threshold_acc_labelled': np.nan,
        'doc_matched_doc_threshold_macro_f1': np.nan,
        'doc_matched_doc_threshold_micro_f1': np.nan,
        'doc_matched_doc_threshold_acc': np.nan,
        'doc_matched_doc_threshold_acc_labelled': np.nan,
        'doc_matched_avianz_threshold_macro_f1': np.nan,
        'doc_matched_avianz_threshold_micro_f1': np.nan,
        'doc_matched_avianz_threshold_acc': np.nan,
        'doc_matched_avianz_threshold_acc_labelled': np.nan,
        'avianz_matched_validation_threshold_macro_f1': np.nan,
        'avianz_matched_validation_threshold_micro_f1': np.nan,
        'avianz_matched_validation_threshold_acc': np.nan,
        'avianz_matched_validation_threshold_acc_labelled': np.nan,
        'avianz_matched_doc_threshold_macro_f1': np.nan,
        'avianz_matched_doc_threshold_micro_f1': np.nan,
        'avianz_matched_doc_threshold_acc': np.nan,
        'avianz_matched_doc_threshold_acc_labelled': np.nan,
        'avianz_matched_avianz_threshold_macro_f1': np.nan,
        'avianz_matched_avianz_threshold_micro_f1': np.nan,
        'avianz_matched_avianz_threshold_acc': np.nan,
        'avianz_matched_avianz_threshold_acc_labelled': np.nan,
        'val_validation_threshold_macro_f1': np.nan,
        'val_validation_threshold_micro_f1': np.nan,
        'val_validation_threshold_acc': np.nan,
        'val_validation_threshold_acc_labelled': np.nan,
        'status': 'unknown',
    }


def load_from_exp_dirs(results_dir, pred_remap=None):
    """Scan experiment directories and load metrics."""
    results = []
    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'analysis':
            continue
        # Pseudo-label runs may have phase3 subdir
        report_dir = exp_dir / 'phase3_pseudo_target'
        if not report_dir.is_dir():
            report_dir = exp_dir

        # Check for predictions CSVs or report JSONs
        has_data = (any(report_dir.glob('predictions_*.csv')) or
                    any(report_dir.glob('*_multilabel_report.json')))
        if not has_data:
            continue

        name = exp_dir.name
        model = 'ast' if name.startswith('ast') else 'regnet' if name.startswith('regnet') else 'unknown'
        seed_m = re.search(r'_seed(\d+)$', name)
        seed = int(seed_m.group(1)) if seed_m else 0

        row = _empty_row(name, name, model, name)
        row['seed'] = seed

        # Read report JSONs if present
        _read_reports_from_dir(report_dir, model, row)
        # Read prediction CSVs and compute all conditions
        _read_adaptive_from_dir(exp_dir, row, pred_remap=pred_remap)

        # Determine status
        has_any = not (np.isnan(row['val_acc']) and np.isnan(row['doc_acc']) and np.isnan(row['avianz_acc']))
        row['status'] = 'completed' if has_any else 'incomplete'
        results.append(row)

    return results


def load_all_results(results_dir, pred_remap=None):
    results = []
    seen = set()
    for row in load_from_exp_dirs(results_dir, pred_remap):
        if row['name'] not in seen:
            results.append(row)
            seen.add(row['name'])
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
#  Output functions (condensed)
# -----------------------------------------------------------------------------

def create_overview_table(df, output_dir):
    cols = [
        'name', 'train_dataset', 'method', 'config', 'category',
        'val_macro_f1', 'val_micro_f1', 'val_acc', 'val_acc_labelled',
        'doc_macro_f1', 'doc_micro_f1', 'doc_acc', 'doc_acc_labelled',
        'avianz_macro_f1', 'avianz_micro_f1', 'avianz_acc', 'avianz_acc_labelled',
        'doc_matched_validation_threshold_macro_f1', 'doc_matched_validation_threshold_micro_f1',
        'doc_matched_validation_threshold_acc', 'doc_matched_validation_threshold_acc_labelled',
        'doc_matched_doc_threshold_macro_f1', 'doc_matched_doc_threshold_micro_f1',
        'doc_matched_doc_threshold_acc', 'doc_matched_doc_threshold_acc_labelled',
        'doc_matched_avianz_threshold_macro_f1', 'doc_matched_avianz_threshold_micro_f1',
        'doc_matched_avianz_threshold_acc', 'doc_matched_avianz_threshold_acc_labelled',
        'avianz_matched_validation_threshold_macro_f1', 'avianz_matched_validation_threshold_micro_f1',
        'avianz_matched_validation_threshold_acc', 'avianz_matched_validation_threshold_acc_labelled',
        'avianz_matched_doc_threshold_macro_f1', 'avianz_matched_doc_threshold_micro_f1',
        'avianz_matched_doc_threshold_acc', 'avianz_matched_doc_threshold_acc_labelled',
        'avianz_matched_avianz_threshold_macro_f1', 'avianz_matched_avianz_threshold_micro_f1',
        'avianz_matched_avianz_threshold_acc', 'avianz_matched_avianz_threshold_acc_labelled',
        'val_validation_threshold_macro_f1', 'val_validation_threshold_micro_f1',
        'val_validation_threshold_acc', 'val_validation_threshold_acc_labelled',
        'val_name', 'doc_name', 'avianz_name',
        'val_acc_background', 'val_macro_precision', 'val_macro_recall', 'val_jaccard',
        'doc_acc_background', 'doc_macro_precision', 'doc_macro_recall', 'doc_jaccard',
        'avianz_acc_background', 'avianz_macro_precision', 'avianz_macro_recall', 'avianz_jaccard',
        'status',
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy().sort_values(['category', 'train_dataset', 'config'])
    df_out.to_csv(output_dir / 'all_results.csv', index=False, float_format='%.4f')
    print(f"✓ all_results.csv ({len(df_out)} experiments)")


def create_per_class_table(results_dir, output_dir):
    """Create per-class metrics from report JSONs."""
    rows = []
    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == 'analysis':
            continue
        if not any(exp_dir.glob('*_multilabel_report.json')):
            continue
        model = 'ast' if exp_dir.name.startswith('ast') else 'regnet' if exp_dir.name.startswith('regnet') else 'unknown'
        for report_file in exp_dir.glob('*_multilabel_report.json'):
            stem = report_file.stem.replace('_multilabel_report', '')
            if '_model' in stem:
                continue
            test_split = re.sub(rf'^{model}_test_', '', stem)
            with open(report_file) as f:
                report = json.load(f)
            skip = {'macro avg','micro avg','exact_match_accuracy',
                    'exact_match_accuracy_labelled','exact_match_accuracy_background',
                    'num_samples','num_labelled_samples','num_background_samples',
                    'hamming_loss','jaccard_score'}
            for cls, m in report.items():
                if cls in skip or not isinstance(m, dict):
                    continue
                rows.append({
                    'experiment': exp_dir.name,
                    'test_split': test_split,
                    'class': cls,
                    'tp': m.get('tp', np.nan),
                    'fp': m.get('fp', np.nan),
                    'tn': m.get('tn', np.nan),
                    'fn': m.get('fn', np.nan),
                    'precision': m.get('precision', np.nan),
                    'recall': m.get('recall', np.nan),
                    'f1': m.get('f1-score', np.nan),
                    'support': m.get('support', np.nan),
                })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'per_class_metrics.csv', index=False, float_format='%.4f')
        print(f"✓ per_class_metrics.csv ({len(rows)} rows)")


def plot_by_category(df, output_dir):
    """Simplified plots per category."""
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].copy()
        if cat_df.empty:
            continue
        train_datasets = cat_df['train_dataset'].unique()
        fig, axes = plt.subplots(1, len(train_datasets), figsize=(6*len(train_datasets), 5), squeeze=False)
        axes = axes.flatten()
        for idx, train_ds in enumerate(train_datasets):
            ax = axes[idx]
            data = cat_df[cat_df['train_dataset'] == train_ds]
            if data.empty:
                ax.axis('off'); continue
            row = data.iloc[0]
            x = [0,1,2]
            y = [row['val_acc'], row['doc_acc'], row['avianz_acc']]
            labels = [row['val_name'], row['doc_name'], row['avianz_name']]
            bars = ax.bar(x, y, alpha=0.8, color=['#1f77b4','#ff7f0e','#2ca02c'])
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel('Accuracy (%)'); ax.set_title(f'{category} - {train_ds.upper()}')
            ax.set_ylim(0,100); ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x()+bar.get_width()/2., h, f'{h:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(output_dir / f'{category.lower().replace(" ","_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {category.lower().replace(' ','_')}.png")


def create_summary_by_category(df, output_dir):
    for category in df['category'].unique():
        cat_df = df[df['category'] == category].copy()
        cat_df['avg_acc'] = (cat_df['val_acc'] + cat_df['doc_acc'] + cat_df['avianz_acc']) / 3
        cat_df = cat_df.sort_values('avg_acc', ascending=False)
        cat_df[['name','train_dataset','config','val_acc','doc_acc','avianz_acc','avg_acc']].to_csv(
            output_dir / f'summary_{category.lower().replace(" ","_")}.csv', index=False, float_format='%.2f')
        print(f"✓ summary_{category.lower().replace(' ','_')}.csv")


def create_report(df, output_dir):
    lines = ["# All Experimental Results\n\n"]
    lines.append(f"**Total experiments:** {len(df)}\n")
    lines.append(f"**Completed:** {len(df[df['status']=='completed'])}\n")
    has = df.dropna(subset=['val_acc','doc_acc','avianz_acc'], how='all')
    lines.append(f"**With valid data:** {len(has)}\n\n")
    lines.append("## Experiments by Category\n\n")
    for category in sorted(df['category'].unique()):
        cat_df = df[df['category'] == category]
        lines.append(f"### {category}\n\n**Count:** {len(cat_df)}\n\n")
        for train_ds in sorted(cat_df['train_dataset'].unique()):
            ds = cat_df[cat_df['train_dataset']==train_ds].dropna(subset=['val_acc','doc_acc','avianz_acc'], how='all')
            if ds.empty:
                lines.append(f"**Train: {train_ds.upper()}** - No valid data\n\n")
                continue
            ds['avg'] = (ds['val_acc'] + ds['doc_acc'] + ds['avianz_acc']) / 3
            best = ds.sort_values('avg', ascending=False).iloc[0]
            lines.append(f"**Train: {train_ds.upper()}** (best: `{best['config']}`)\n")
            lines.append(f"- Validation: {best['val_acc']:.1f}%\n")
            lines.append(f"- DOC Matched: {best['doc_acc']:.1f}%\n")
            lines.append(f"- AviaNZ Matched: {best['avianz_acc']:.1f}%\n")
            lines.append(f"- Average: {best['avg']:.1f}%\n\n")
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.writelines(lines)
    print("✓ REPORT.md")


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

_STANDARD_REMAP = 'new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird'

def main():
    parser = argparse.ArgumentParser(description='Analyse experimental results – 7 conditions based on existing CSVs.')
    parser.add_argument('results_dir', help='Directory containing experiment subfolders.')
    parser.add_argument('--output', '-o', default=None, help='Output directory for analysis files.')
    parser.add_argument('--pred-remap', default=None,
                        help='Comma-separated old:new remaps, e.g. "tui:tui/bellbird"')
    parser.add_argument('--merge', action='store_true',
                        help=f'Apply standard NZ remap ({_STANDARD_REMAP})')
    args = parser.parse_args()

    pred_remap = None
    remap_str = args.pred_remap or (_STANDARD_REMAP if args.merge else None)
    if remap_str:
        pred_remap = {}
        for pair in remap_str.split(','):
            old, new = pair.split(':')
            pred_remap[old.strip()] = new.strip()
        print(f'  Pred remap: {pred_remap}')

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print(" ANALYZING ALL RESULTS (7 conditions, based on existing CSVs)")
    print("="*70)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}\n")

    sns.set_style("whitegrid")

    print("Loading all results...")
    df = load_all_results(results_dir, pred_remap=pred_remap)
    print(f"  {len(df)} experiments loaded")
    if df.empty:
        print("  No experiments found.")
        return

    has_val = df['val_acc'].notna().sum()
    has_doc = df['doc_acc'].notna().sum()
    has_avianz = df['avianz_acc'].notna().sum()
    print(f"  {has_val} with validation accuracy")
    print(f"  {has_doc} with DOC matched accuracy")
    print(f"  {has_avianz} with AviaNZ matched accuracy")
    print(f"  Categories: {', '.join(sorted(df['category'].unique()))}\n")

    print("Creating outputs...")
    create_overview_table(df, output_dir)
    create_per_class_table(results_dir, output_dir)
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