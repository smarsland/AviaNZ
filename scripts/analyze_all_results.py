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
            row = {
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
                'test1_acc_background': data.get('test1_acc_background', np.nan),
                'test1_macro_precision': np.nan,
                'test1_macro_recall': np.nan,
                'test1_macro_f1': np.nan,
                'test1_jaccard': np.nan,
                'test1_adaptive_f1': np.nan, 'test1_adaptive_acc': np.nan, 'test1_adaptive_acc_labelled': np.nan,
                'test1_cross_f1': np.nan, 'test1_cross_acc': np.nan, 'test1_cross_acc_labelled': np.nan,
                'test2_acc': data.get('test2_acc', np.nan),
                'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
                'test2_acc_background': data.get('test2_acc_background', np.nan),
                'test2_macro_precision': np.nan,
                'test2_macro_recall': np.nan,
                'test2_macro_f1': np.nan,
                'test2_jaccard': np.nan,
                'test2_adaptive_f1': np.nan, 'test2_adaptive_acc': np.nan, 'test2_adaptive_acc_labelled': np.nan,
                'test2_cross_f1': np.nan, 'test2_cross_acc': np.nan, 'test2_cross_acc_labelled': np.nan,
                'status': data.get('status', 'unknown'),
            }
            _read_adaptive_from_dir(result_file.parent, row)
            results.append(row)
            continue

        if data.get('type') == 'finetuned':
            model = data.get('model', 'kaytoo_finetuned')
            row = {
                'name': name,
                'train_dataset': 'finetuned',
                'method': 'finetuned',
                'config': model,
                'category': 'Kaytoo (Finetuned)',
                'seed': data.get('seed', 0),
                'test1_name': data.get('test1_name', 'unknown'),
                'test2_name': data.get('test2_name', 'unknown'),
                'test1_acc': data.get('test1_acc', np.nan),
                'test1_acc_labelled': data.get('test1_acc_labelled', np.nan),
                'test1_acc_background': data.get('test1_acc_background', np.nan),
                'test1_macro_precision': np.nan,
                'test1_macro_recall': np.nan,
                'test1_macro_f1': np.nan,
                'test1_jaccard': np.nan,
                'test1_adaptive_f1': np.nan, 'test1_adaptive_acc': np.nan, 'test1_adaptive_acc_labelled': np.nan,
                'test1_cross_f1': np.nan, 'test1_cross_acc': np.nan, 'test1_cross_acc_labelled': np.nan,
                'test2_acc': data.get('test2_acc', np.nan),
                'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
                'test2_acc_background': data.get('test2_acc_background', np.nan),
                'test2_macro_precision': np.nan,
                'test2_macro_recall': np.nan,
                'test2_macro_f1': np.nan,
                'test2_jaccard': np.nan,
                'test2_adaptive_f1': np.nan, 'test2_adaptive_acc': np.nan, 'test2_adaptive_acc_labelled': np.nan,
                'test2_cross_f1': np.nan, 'test2_cross_acc': np.nan, 'test2_cross_acc_labelled': np.nan,
                'status': data.get('status', 'unknown'),
            }
            _read_adaptive_from_dir(result_file.parent, row)
            results.append(row)
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

        row = {
            'name': name,
            'train_dataset': train_dataset,
            'method': method,
            'config': config,
            'category': category,
            'seed': seed,
            'test1_name': data.get('test1_name', 'unknown'),
            'test2_name': data.get('test2_name', 'unknown'),
            'test1_acc': data.get('test1_acc', np.nan),
            'test1_acc_labelled': data.get('test1_acc_labelled', np.nan),
            'test1_acc_background': data.get('test1_acc_background', np.nan),
            'test1_macro_precision': np.nan,
            'test1_macro_recall': np.nan,
            'test1_macro_f1': np.nan,
            'test1_jaccard': np.nan,
            'test1_adaptive_f1': np.nan, 'test1_adaptive_acc': np.nan, 'test1_adaptive_acc_labelled': np.nan,
            'test1_cross_f1': np.nan, 'test1_cross_acc': np.nan, 'test1_cross_acc_labelled': np.nan,
            'test2_acc': data.get('test2_acc', np.nan),
            'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
            'test2_acc_background': data.get('test2_acc_background', np.nan),
            'test2_macro_precision': np.nan,
            'test2_macro_recall': np.nan,
            'test2_macro_f1': np.nan,
            'test2_jaccard': np.nan,
            'test2_adaptive_f1': np.nan, 'test2_adaptive_acc': np.nan, 'test2_adaptive_acc_labelled': np.nan,
            'test2_cross_f1': np.nan, 'test2_cross_acc': np.nan, 'test2_cross_acc_labelled': np.nan,
            'status': data.get('status', 'unknown'),
        }
        _read_adaptive_from_dir(result_file.parent, row)
        results.append(row)
    return results


def _extract_report_metrics(report):
    """Pull scalar summary metrics out of one *_multilabel_report.json dict."""
    def pct(v):
        return v * 100 if not (v is np.nan or v != v) else np.nan

    macro = report.get('macro avg', {})
    return {
        'acc':        pct(report.get('exact_match_accuracy', np.nan)),
        'acc_lab':    pct(report.get('exact_match_accuracy_labelled', np.nan)),
        'acc_bg':     pct(report.get('exact_match_accuracy_background', np.nan)),
        'macro_p':    macro.get('precision', np.nan),
        'macro_r':    macro.get('recall', np.nan),
        'macro_f1':   macro.get('f1-score', np.nan),
        'jaccard':    report.get('jaccard_score', np.nan),
    }


def _metrics_from_csv(csv_path: Path, apply_thresholds: np.ndarray = None) -> dict:
    """Compute metrics at threshold=0.5, self-tuned oracle thresholds, and optionally
    cross-tuned thresholds supplied via *apply_thresholds*.

    Reads the CSV exactly once regardless of how many threshold sets are evaluated.

    Returns a dict with keys:
        half_precision, half_recall, half_f1, half_jaccard,
        half_acc, half_acc_labelled,
        oracle_precision, oracle_recall, oracle_f1, oracle_jaccard,
        oracle_acc, oracle_acc_labelled,
        oracle_thresholds,           # 1-D array of per-class thresholds
        cross_f1, cross_acc, cross_acc_labelled  (only when apply_thresholds given)
    All NaN when no true_ columns are present.
    """
    from sklearn.metrics import f1_score as _f1
    from sklearn.metrics import precision_recall_fscore_support as _prf
    from sklearn.metrics import jaccard_score as _jaccard

    nan_result = {k: np.nan for k in (
        'half_precision', 'half_recall', 'half_f1', 'half_jaccard',
        'half_acc', 'half_acc_labelled',
        'oracle_precision', 'oracle_recall', 'oracle_f1', 'oracle_jaccard',
        'oracle_acc', 'oracle_acc_labelled',
        'cross_f1', 'cross_acc', 'cross_acc_labelled',
    )}
    nan_result['oracle_thresholds'] = None

    df = pd.read_csv(csv_path, index_col='filename')
    class_cols = [c for c in df.columns if not c.startswith('true_')]
    true_cols  = [c for c in df.columns if c.startswith('true_')]
    if not true_cols:
        return nan_result

    probs = df[class_cols].values.astype(np.float32)
    trues = df[true_cols].values.astype(np.int32)
    labelled = trues.sum(axis=1) > 0

    # Only average over classes that actually appear in the ground truth
    present = trues.sum(axis=0) > 0
    present_idx = np.where(present)[0]

    def _acc(preds):
        correct = np.all(preds == trues, axis=1)
        a = correct.mean() * 100
        al = correct[labelled].mean() * 100 if labelled.any() else np.nan
        return a, al

    def _macro(preds):
        if len(present_idx) == 0:
            return np.nan, np.nan, np.nan, np.nan
        t = trues[:, present_idx]
        p = preds[:, present_idx]
        prec, rec, f1, _ = _prf(t, p, average='macro', zero_division=0)
        jac = float(_jaccard(t, p, average='macro', zero_division=0))
        return float(prec), float(rec), float(f1), jac

    # --- threshold 0.5, restricted to test-set classes ---
    # For models trained on a superset vocabulary (e.g. 120 classes), kbird
    # normalisation across all classes suppresses individual class probs well
    # below 0.5, even for confident predictions.  Restricting to the classes
    # that actually appear in the test set (present_idx) before thresholding
    # matches how evaluate_kaytoo.py evaluates Kaytoo (which only scores over
    # valid_cols = test-set eBird codes).  Without this restriction, fixed-
    # threshold F1 is near-zero for large-vocabulary models while adaptive F1
    # is reasonable — a misleading asymmetry.
    if len(present_idx) > 0:
        probs_restricted = probs[:, present_idx]
        trues_restricted = trues[:, present_idx]
        labelled_restricted = trues_restricted.sum(axis=1) > 0
        preds_restricted = (probs_restricted >= 0.5).astype(int)
        _p, _r, _f, _ = _prf(trues_restricted, preds_restricted,
                              average='macro', zero_division=0)
        prec_half, rec_half, f1_half = float(_p), float(_r), float(_f)
        jac_half = float(_jaccard(trues_restricted, preds_restricted,
                                  average='macro', zero_division=0))
        correct_restricted = np.all(preds_restricted == trues_restricted, axis=1)
        acc_half = float(correct_restricted.mean() * 100)
        acc_lab_half = float(
            correct_restricted[labelled_restricted].mean() * 100
            if labelled_restricted.any() else np.nan
        )
    else:
        prec_half = rec_half = f1_half = jac_half = np.nan
        acc_half = acc_lab_half = np.nan

    # --- oracle per-class thresholds (tuned on this same split) ---
    # Only tune thresholds for classes that actually appear in the ground truth
    # (present_idx).  Non-present classes get threshold=1.0 so they never fire.
    # Without this, a class with all-zero ground truth gets a near-zero threshold
    # (f1=0 at any firing threshold beats f1=-1 at silence), causing it to fire
    # on every sample including background → exact-match accuracy craters to 0%.
    candidates = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    thresholds = np.full(probs.shape[1], 1.0, dtype=np.float32)  # default: never fire
    for c in present_idx:
        tc = trues[:, c]
        pc = probs[:, c]
        # preds_all: shape (201, n_samples)
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
        thresholds[c] = candidates[np.argmax(f1s)]
    preds_oracle = (probs >= thresholds[np.newaxis, :]).astype(int)
    # Accuracy uses only the present_idx columns (same restriction as F1) so
    # non-present classes that happen to have probs < 1.0 don't penalise it.
    preds_oracle_restricted = preds_oracle[:, present_idx]
    trues_oracle_restricted = trues[:, present_idx]
    labelled_oracle = trues_oracle_restricted.sum(axis=1) > 0
    correct_oracle = np.all(preds_oracle_restricted == trues_oracle_restricted, axis=1)
    acc_oracle     = float(correct_oracle.mean() * 100)
    acc_lab_oracle = float(correct_oracle[labelled_oracle].mean() * 100
                           if labelled_oracle.any() else np.nan)
    prec_oracle, rec_oracle, f1_oracle, jac_oracle = _macro(preds_oracle)

    result = {
        'half_precision':      prec_half,
        'half_recall':         rec_half,
        'half_f1':             f1_half,
        'half_jaccard':        jac_half,
        'half_acc':            float(acc_half),
        'half_acc_labelled':   float(acc_lab_half),
        'oracle_precision':    prec_oracle,
        'oracle_recall':       rec_oracle,
        'oracle_f1':           f1_oracle,
        'oracle_jaccard':      jac_oracle,
        'oracle_acc':          float(acc_oracle),
        'oracle_acc_labelled': float(acc_lab_oracle),
        'oracle_thresholds':   thresholds,
        'cross_f1':            np.nan,
        'cross_acc':           np.nan,
        'cross_acc_labelled':  np.nan,
    }

    # --- cross thresholds: tuned on the other split, applied here ---
    # probs/trues already loaded above — no re-read needed
    if apply_thresholds is not None and len(apply_thresholds) == probs.shape[1]:
        preds_cross = (probs >= apply_thresholds[np.newaxis, :]).astype(int)
        _, _, f1_cross, _ = _macro(preds_cross)
        # Restrict accuracy to present_idx columns only (same reason as oracle)
        preds_cross_r = preds_cross[:, present_idx]
        trues_cross_r = trues[:, present_idx]
        labelled_cross = trues_cross_r.sum(axis=1) > 0
        correct_cross  = np.all(preds_cross_r == trues_cross_r, axis=1)
        acc_cross      = float(correct_cross.mean() * 100)
        acc_lab_cross  = float(correct_cross[labelled_cross].mean() * 100
                               if labelled_cross.any() else np.nan)
        result['cross_f1']           = f1_cross
        result['cross_acc']          = acc_cross
        result['cross_acc_labelled'] = acc_lab_cross

    return result


def _read_adaptive_from_dir(exp_dir: Path, row: dict):
    """Populate adaptive_*, cross_*, and (when missing) half_* metrics in *row*
    from predictions CSVs.

    - adaptive (self-tuned): thresholds tuned on the same split they are evaluated on.
    - cross-tuned: thresholds tuned on the *other* split, then applied to this split.

    Precision, recall, f1 and jaccard are all computed only over the classes
    that actually appear in the test set ground truth.
    """
    csvs = sorted(exp_dir.glob('predictions_*.csv'))
    if not csvs:
        return
    prefixes = ['test1', 'test2']

    # Single pass per CSV: compute self-tuned oracle thresholds.
    metrics_list = [_metrics_from_csv(csv_path) for csv_path in csvs]

    # Second pass: apply each split's oracle thresholds to the *other* split.
    # _metrics_from_csv reads the CSV again here, but the oracle threshold search
    # is skipped (apply_thresholds provided), so it is much cheaper than the first pass.
    if len(metrics_list) == 2:
        for i in range(2):
            other_thresholds = metrics_list[1 - i]['oracle_thresholds']
            if other_thresholds is not None:
                cross = _metrics_from_csv(csvs[i], apply_thresholds=other_thresholds)
                metrics_list[i]['cross_f1']           = cross['cross_f1']
                metrics_list[i]['cross_acc']          = cross['cross_acc']
                metrics_list[i]['cross_acc_labelled'] = cross['cross_acc_labelled']

    for prefix, m in zip(prefixes, metrics_list):
        # always write oracle (self-tuned adaptive-threshold) metrics
        row[f'{prefix}_adaptive_f1']             = m['oracle_f1']
        row[f'{prefix}_adaptive_acc']            = m['oracle_acc']
        row[f'{prefix}_adaptive_acc_labelled']   = m['oracle_acc_labelled']
        # cross-tuned metrics (other split's thresholds applied here)
        row[f'{prefix}_cross_f1']                = m['cross_f1']
        row[f'{prefix}_cross_acc']               = m['cross_acc']
        row[f'{prefix}_cross_acc_labelled']      = m['cross_acc_labelled']
        # fill half-threshold metrics when not already set by a report JSON
        if np.isnan(row.get(f'{prefix}_macro_precision', np.nan)): row[f'{prefix}_macro_precision'] = m['half_precision']
        if np.isnan(row.get(f'{prefix}_macro_recall',    np.nan)): row[f'{prefix}_macro_recall']    = m['half_recall']
        if np.isnan(row.get(f'{prefix}_macro_f1',        np.nan)): row[f'{prefix}_macro_f1']        = m['half_f1']
        if np.isnan(row.get(f'{prefix}_jaccard',         np.nan)): row[f'{prefix}_jaccard']         = m['half_jaccard']
        if np.isnan(row.get(f'{prefix}_acc',             np.nan)): row[f'{prefix}_acc']             = m['half_acc']
        if np.isnan(row.get(f'{prefix}_acc_labelled',    np.nan)): row[f'{prefix}_acc_labelled']    = m['half_acc_labelled']


def _read_reports_from_dir(report_dir, model, row):
    """Read *_multilabel_report.json files from report_dir into row (mutates in place)."""
    for report_file in sorted(report_dir.glob('*_multilabel_report.json')):
        with open(report_file) as f:
            report = json.load(f)
        stem = report_file.stem.replace('_multilabel_report', '')
        if '_model' in stem:
            continue  # validation set, skip
        dataset_name = re.sub(rf'^{model}_test_', '', stem)
        m = _extract_report_metrics(report)
        if row['test1_name'] is np.nan or row['test1_name'] == np.nan:
            row['test1_name'] = dataset_name
            row['test1_acc'] = m['acc']
            row['test1_acc_labelled'] = m['acc_lab']
            row['test1_acc_background'] = m['acc_bg']
            row['test1_macro_precision'] = m['macro_p']
            row['test1_macro_recall'] = m['macro_r']
            row['test1_macro_f1'] = m['macro_f1']
            row['test1_jaccard'] = m['jaccard']
        else:
            row['test2_name'] = dataset_name
            row['test2_acc'] = m['acc']
            row['test2_acc_labelled'] = m['acc_lab']
            row['test2_acc_background'] = m['acc_bg']
            row['test2_macro_precision'] = m['macro_p']
            row['test2_macro_recall'] = m['macro_r']
            row['test2_macro_f1'] = m['macro_f1']
            row['test2_jaccard'] = m['jaccard']


def _empty_row(name, train_dataset, model, transform):
    return {
        'name': name,
        'train_dataset': train_dataset,
        'method': model,
        'config': transform,
        'category': model.upper(),
        'seed': 0,
        'test1_name': np.nan, 'test2_name': np.nan,
        'test1_acc': np.nan, 'test1_acc_labelled': np.nan, 'test1_acc_background': np.nan,
        'test1_macro_precision': np.nan, 'test1_macro_recall': np.nan,
        'test1_macro_f1': np.nan, 'test1_jaccard': np.nan,
        'test1_adaptive_f1': np.nan, 'test1_adaptive_acc': np.nan, 'test1_adaptive_acc_labelled': np.nan,
        'test1_cross_f1': np.nan, 'test1_cross_acc': np.nan, 'test1_cross_acc_labelled': np.nan,
        'test2_acc': np.nan, 'test2_acc_labelled': np.nan, 'test2_acc_background': np.nan,
        'test2_macro_precision': np.nan, 'test2_macro_recall': np.nan,
        'test2_macro_f1': np.nan, 'test2_jaccard': np.nan,
        'test2_adaptive_f1': np.nan, 'test2_adaptive_acc': np.nan, 'test2_adaptive_acc_labelled': np.nan,
        'test2_cross_f1': np.nan, 'test2_cross_acc': np.nan, 'test2_cross_acc_labelled': np.nan,
        'status': 'unknown',
    }


def load_from_viz_dir(viz_dir):
    """Load experiments written by run_experiments.sh (model_on_dataset_transform layout)."""
    results = []
    standard_pattern = re.compile(r'^(ast|regnet)_on_(avianz|doc|merged|large_doc|large_avianz|combined)_(.+)$')
    all_species_pattern = re.compile(r'^(ast|regnet)_all_species_(.+?)_seed(\d+)$')
    pseudo_pattern = re.compile(r'^(ast|regnet)_pseudo_([a-z]+)_to_([a-z]+)_(.+)_pct(\d+)$')
    for exp_dir in sorted(Path(viz_dir).iterdir()):
        if not exp_dir.is_dir():
            continue

        # Ensemble directories: ensemble_all, ensemble_avianz, ensemble_doc, etc.
        if exp_dir.name.startswith('ensemble'):
            row = _empty_row(exp_dir.name, 'all', 'ensemble', exp_dir.name)
            row['category'] = 'ENSEMBLE'
            _read_reports_from_dir(exp_dir, 'ensemble', row)
            row['status'] = 'completed' if not (np.isnan(row['test1_acc']) and np.isnan(row['test2_acc'])) else 'incomplete'
            results.append(row)
            continue

        m = standard_pattern.match(exp_dir.name)
        if m:
            model, train_dataset, transform = m.groups()
            row = _empty_row(exp_dir.name, train_dataset, model, transform)
            _read_reports_from_dir(exp_dir, model, row)
            _read_adaptive_from_dir(exp_dir, row)
            row['status'] = 'completed' if not (np.isnan(row['test1_acc']) and np.isnan(row['test2_acc'])) else 'incomplete'
            results.append(row)
            continue

        m = all_species_pattern.match(exp_dir.name)
        if m:
            model, transform, seed = m.groups()
            row = _empty_row(exp_dir.name, 'all_species', model, transform)
            row['seed'] = int(seed)
            _read_reports_from_dir(exp_dir, model, row)
            _read_adaptive_from_dir(exp_dir, row)
            row['status'] = 'completed' if not (np.isnan(row['test1_acc']) and np.isnan(row['test2_acc'])) else 'incomplete'
            results.append(row)
            continue

        m = pseudo_pattern.match(exp_dir.name)
        if m:
            model, source_dataset, target_dataset, transform, pct_int = m.groups()
            final_dir = exp_dir / 'phase3_pseudo_target'
            if not final_dir.is_dir():
                continue
            row = _empty_row(exp_dir.name, f'pseudo_{source_dataset}_to_{target_dataset}_pct{pct_int}', model, transform)
            row['category'] = f'{model.upper()} Pseudo'
            _read_reports_from_dir(final_dir, model, row)
            _read_adaptive_from_dir(exp_dir, row)
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
    cols = [
        'name', 'train_dataset', 'method', 'config', 'category',
        # F1 scores immediately after identifiers
        'test1_macro_f1', 'test1_adaptive_f1', 'test1_cross_f1',
        'test2_macro_f1', 'test2_adaptive_f1', 'test2_cross_f1',
        # then the rest
        'test1_name', 'test1_acc', 'test1_acc_labelled', 'test1_acc_background',
        'test1_macro_precision', 'test1_macro_recall', 'test1_jaccard',
        'test1_adaptive_acc', 'test1_adaptive_acc_labelled',
        'test1_cross_acc', 'test1_cross_acc_labelled',
        'test2_name', 'test2_acc', 'test2_acc_labelled', 'test2_acc_background',
        'test2_macro_precision', 'test2_macro_recall', 'test2_jaccard',
        'test2_adaptive_acc', 'test2_adaptive_acc_labelled',
        'test2_cross_acc', 'test2_cross_acc_labelled',
        'status',
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()
    df_out = df_out.sort_values(['category', 'train_dataset', 'config'])
    df_out.to_csv(output_dir / 'all_results.csv', index=False, float_format='%.4f')
    print(f"✓ all_results.csv ({len(df_out)} experiments)")


def create_per_class_table(viz_dir, output_dir):
    """Create per_class_metrics.csv: one row per (experiment, test split, class)."""
    standard_pattern = re.compile(r'^(ast|regnet)_on_(avianz|doc|merged|large_doc|large_avianz|combined)_(.+)$')
    rows = []
    for exp_dir in sorted(Path(viz_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        m = standard_pattern.match(exp_dir.name)
        if not m:
            continue
        model, train_dataset, _ = m.groups()
        for report_file in sorted(exp_dir.glob('*_multilabel_report.json')):
            stem = report_file.stem.replace('_multilabel_report', '')
            if '_model' in stem:
                continue
            test_split = re.sub(rf'^{model}_test_', '', stem)
            with open(report_file) as f:
                report = json.load(f)
            skip_keys = {'macro avg', 'micro avg', 'exact_match_accuracy',
                         'exact_match_accuracy_labelled', 'exact_match_accuracy_background',
                         'num_samples', 'num_labelled_samples', 'num_background_samples',
                         'hamming_loss', 'jaccard_score'}
            for class_name, metrics in report.items():
                if class_name in skip_keys or not isinstance(metrics, dict):
                    continue
                rows.append({
                    'experiment': exp_dir.name,
                    'train_dataset': train_dataset,
                    'test_split': test_split,
                    'class': class_name,
                    'tp': metrics.get('tp', np.nan),
                    'fp': metrics.get('fp', np.nan),
                    'tn': metrics.get('tn', np.nan),
                    'fn': metrics.get('fn', np.nan),
                    'precision': metrics.get('precision', np.nan),
                    'recall': metrics.get('recall', np.nan),
                    'f1': metrics.get('f1-score', np.nan),
                    'support': metrics.get('support', np.nan),
                })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'per_class_metrics.csv', index=False, float_format='%.4f')
        print(f"✓ per_class_metrics.csv ({len(rows)} rows)")
    else:
        print("  (no per-class data found — re-run experiments to populate tp/fp/tn/fn)")


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


def print_model_comparison(df):
    """
    Print a clear terminal comparison of the best REGNET model vs Kaytoo and
    BirdNET pretrained baselines.
    """
    def _fmt(val):
        if val is np.nan or (isinstance(val, float) and np.isnan(val)):
            return "  N/A"
        return f"{val:5.1f}%"

    def _fmtf(val):
        if val is np.nan or (isinstance(val, float) and np.isnan(val)):
            return "   N/A"
        return f"{val:.3f}"

    def _collect(row):
        return {
            't1':     _fmt(row.get('test1_acc',                   np.nan)),
            't1lab':  _fmt(row.get('test1_acc_labelled',           np.nan)),
            't1f1':   _fmtf(row.get('test1_macro_f1',             np.nan)),
            # self-tuned (oracle) for test1
            't1af1':  _fmtf(row.get('test1_adaptive_f1',          np.nan)),
            't1a':    _fmt(row.get('test1_adaptive_acc',           np.nan)),
            't1alab': _fmt(row.get('test1_adaptive_acc_labelled',  np.nan)),
            # cross-tuned for test1 (thresholds tuned on test2, applied to test1)
            't1xf1':  _fmtf(row.get('test1_cross_f1',             np.nan)),
            't1x':    _fmt(row.get('test1_cross_acc',              np.nan)),
            't1xlab': _fmt(row.get('test1_cross_acc_labelled',     np.nan)),
            't2':     _fmt(row.get('test2_acc',                   np.nan)),
            't2lab':  _fmt(row.get('test2_acc_labelled',           np.nan)),
            't2f1':   _fmtf(row.get('test2_macro_f1',             np.nan)),
            # self-tuned (oracle) for test2
            't2af1':  _fmtf(row.get('test2_adaptive_f1',          np.nan)),
            't2a':    _fmt(row.get('test2_adaptive_acc',           np.nan)),
            't2alab': _fmt(row.get('test2_adaptive_acc_labelled',  np.nan)),
            # cross-tuned for test2 (thresholds tuned on test1, applied to test2)
            't2xf1':  _fmtf(row.get('test2_cross_f1',             np.nan)),
            't2x':    _fmt(row.get('test2_cross_acc',              np.nan)),
            't2xlab': _fmt(row.get('test2_cross_acc_labelled',     np.nan)),
            't1n':    str(row.get('test1_name', '?')),
            't2n':    str(row.get('test2_name', '?')),
        }

    def _build_rows(sort_col1, sort_col2):
        """Return rows list ranked by avg of sort_col1+sort_col2 for REGNET groups."""
        rows = []

        # ---- Top-3 REGNET (matched) -----------------------------------------
        base_df = df[
            (df['category'] == 'REGNET') &
            (~df['train_dataset'].isin(['large_doc', 'large_avianz']))
        ].copy()
        base_df = base_df.dropna(subset=[sort_col1, sort_col2])
        if not base_df.empty:
            base_df['_avg'] = (base_df[sort_col1] + base_df[sort_col2]) / 2
            for rank, (_, r) in enumerate(base_df.sort_values('_avg', ascending=False).head(3).iterrows(), 1):
                rows.append((f'REGNET #{rank} (matched)', r['name'], _collect(r)))
        else:
            rows.append(('REGNET (matched)', '(no results)', {}))

        # ---- Top-3 REGNET (large DOC) ---------------------------------------
        large_df = df[
            (df['category'] == 'REGNET') &
            (df['train_dataset'] == 'large_doc')
        ].copy()
        large_df = large_df.dropna(subset=[sort_col1, sort_col2])
        if not large_df.empty:
            large_df['_avg'] = (large_df[sort_col1] + large_df[sort_col2]) / 2
            for rank, (_, r) in enumerate(large_df.sort_values('_avg', ascending=False).head(3).iterrows(), 1):
                rows.append((f'REGNET #{rank} (large DOC)', r['name'], _collect(r)))

        # ---- Kaytoo pretrained ----------------------------------------------
        k_df = df[df['category'] == 'Kaytoo (Pretrained)']
        if not k_df.empty:
            rows.append(('Kaytoo (Pretrained)', '', _collect(k_df.iloc[0])))
        else:
            rows.append(('Kaytoo (Pretrained)', '(no results)', {}))

        # ---- BirdNET pretrained ---------------------------------------------
        b_df = df[df['category'] == 'BirdNET (Pretrained)']
        if not b_df.empty:
            rows.append(('BirdNET (Pretrained)', '', _collect(b_df.iloc[0])))
        else:
            rows.append(('BirdNET (Pretrained)', '(no results)', {}))

        return rows

    # Build per-table row lists ranked by the metric that table displays
    rows_half   = _build_rows('test1_macro_f1',    'test2_macro_f1')
    rows_t1tune = _build_rows('test1_adaptive_f1', 'test2_cross_f1')
    rows_t2tune = _build_rows('test1_cross_f1',    'test2_adaptive_f1')

    # ---- Print --------------------------------------------------------------
    # Determine test-set names from the first populated row across any table
    t1n = t2n = '?'
    for rows in (rows_half, rows_t1tune, rows_t2tune):
        for _, _, m in rows:
            if m:
                t1n = m['t1n']
                t2n = m['t2n']
                break
        if t1n != '?':
            break

    W = 88

    def _print_table(title, rows, acc_key1, acl_key1, f1_key1, acc_key2, acl_key2, f1_key2, f1_label, note):
        print("\n" + "=" * W)
        print(f" {title}")
        print("=" * W)
        hdr = (f"  {'':26s}  {t1n:>12s}               {t2n:>12s}\n"
               f"  {'':26s}  {'acc':>7s}  {'(lab)':>7s}  {f1_label:>6s}"
               f"     {'acc':>7s}  {'(lab)':>7s}  {f1_label:>6s}")
        print(hdr)
        print("-" * W)
        for label, cfg_name, m in rows:
            if not m:
                print(f"  {label}")
                continue
            print(
                f"  {label:<26s}"
                f"  {m[acc_key1]:>7s}  ({m[acl_key1]:>5s})  {m[f1_key1]:>6s}"
                f"     {m[acc_key2]:>7s}  ({m[acl_key2]:>5s})  {m[f1_key2]:>6s}"
            )
            if cfg_name:
                print(f"    {cfg_name}")
        print()
        print(f"  acc = exact-match accuracy (all samples)  lab = labelled samples only")
        print(f"  {note}")
        print("=" * W)

    _print_table(
        title="RESULTS — threshold 0.5  (ranked by avg macro-F1 @ 0.5)",
        rows=rows_half,
        acc_key1='t1', acl_key1='t1lab', f1_key1='t1f1',
        acc_key2='t2', acl_key2='t2lab', f1_key2='t2f1',
        f1_label='F1',
        note="F1  = macro-F1 at fixed threshold 0.5",
    )
    _print_table(
        title=f"RESULTS — {t1n} thresholds applied to both  (ranked by avg F1†)",
        rows=rows_t1tune,
        acc_key1='t1a', acl_key1='t1alab', f1_key1='t1af1',
        acc_key2='t2x', acl_key2='t2xlab', f1_key2='t2xf1',
        f1_label='F1†',
        note=f"F1† = macro-F1 with per-class thresholds tuned on {t1n}, applied to both",
    )
    _print_table(
        title=f"RESULTS — {t2n} thresholds applied to both  (ranked by avg F1‡)",
        rows=rows_t2tune,
        acc_key1='t1x', acl_key1='t1xlab', f1_key1='t1xf1',
        acc_key2='t2a', acl_key2='t2alab', f1_key2='t2af1',
        f1_label='F1‡',
        note=f"F1‡ = macro-F1 with per-class thresholds tuned on {t2n}, applied to both",
    )
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze all experimental results')
    parser.add_argument('results_dir',
                        help='Output directory (e.g. /local/scratch/freangi/matched_tests). '
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
    if df.empty:
        print("  No experiments found — nothing to analyze.")
        return
    valid_acc_cols = [c for c in ['test1_acc', 'test2_acc'] if c in df.columns]
    print(f"  {len(df.dropna(subset=valid_acc_cols, how='all')) if valid_acc_cols else 0} with valid accuracies")
    print(f"  Categories: {', '.join(sorted(df['category'].unique()))}\n")
    
    print("Creating outputs...")
    create_overview_table(df, output_dir)
    create_per_class_table(results_dir, output_dir)
    plot_by_category(df, output_dir)
    create_summary_by_category(df, output_dir)
    create_report(df, output_dir)

    print_model_comparison(df)
    
    print("\n" + "="*70)
    print(" DONE")
    print("="*70)
    print(f" Output: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
