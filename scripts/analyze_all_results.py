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
                'test1_adaptive_f1': np.nan,
                'test2_acc': data.get('test2_acc', np.nan),
                'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
                'test2_acc_background': data.get('test2_acc_background', np.nan),
                'test2_macro_precision': np.nan,
                'test2_macro_recall': np.nan,
                'test2_macro_f1': np.nan,
                'test2_jaccard': np.nan,
                'test2_adaptive_f1': np.nan,
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
            'test1_adaptive_f1': np.nan,
            'test2_acc': data.get('test2_acc', np.nan),
            'test2_acc_labelled': data.get('test2_acc_labelled', np.nan),
            'test2_acc_background': data.get('test2_acc_background', np.nan),
            'test2_macro_precision': np.nan,
            'test2_macro_recall': np.nan,
            'test2_macro_f1': np.nan,
            'test2_jaccard': np.nan,
            'test2_adaptive_f1': np.nan,
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


def _adaptive_f1_from_predictions(csv_path: Path, tune_csv: Path | None = None) -> float:
    """Compute macro-F1 using per-class thresholds tuned to maximise it.

    Thresholds are found on *tune_csv* (defaults to *csv_path* itself — oracle
    upper bound).  Returns nan when no true_ columns are present.
    """
    from sklearn.metrics import f1_score as _f1

    def _load(p):
        df = pd.read_csv(p, index_col='filename')
        class_cols = [c for c in df.columns if not c.startswith('true_')]
        true_cols  = [c for c in df.columns if c.startswith('true_')]
        if not true_cols:
            return None, None, class_cols
        return df[class_cols].values.astype(np.float32), df[true_cols].values.astype(np.int32), class_cols

    probs, trues, classes = _load(csv_path)
    if trues is None:
        return np.nan

    t_probs, t_trues, _ = _load(tune_csv) if tune_csv and tune_csv != csv_path else (probs, trues, classes)
    if t_trues is None:
        return np.nan

    # Per-class threshold sweep
    candidates = np.linspace(0.0, 1.0, 201)
    thresholds = np.full(len(classes), 0.5, dtype=np.float32)
    for c in range(len(classes)):
        best = -1.0
        for t in candidates:
            preds = (t_probs[:, c] >= t).astype(int)
            if preds.sum() == 0:
                continue
            f = _f1(t_trues[:, c], preds, zero_division=0)
            if f > best:
                best = f
                thresholds[c] = t

    preds = (probs >= thresholds[np.newaxis, :]).astype(int)
    _, _, f1, _ = __import__('sklearn.metrics', fromlist=['precision_recall_fscore_support']).precision_recall_fscore_support(
        trues, preds, average='macro', zero_division=0)
    return float(f1)


def _read_adaptive_from_dir(exp_dir: Path, row: dict):
    """Populate test1_adaptive_f1 / test2_adaptive_f1 in *row* from predictions CSVs.

    Thresholds are tuned on the same split (oracle) — good enough for a
    diagnostic upper bound.  Returns silently when no predictions CSVs exist
    or when they predate the true_ column addition.
    """
    csvs = sorted(exp_dir.glob('predictions_*.csv'))
    if not csvs:
        return
    slots = ['test1_adaptive_f1', 'test2_adaptive_f1']
    for slot, csv_path in zip(slots, csvs):
        row[slot] = _adaptive_f1_from_predictions(csv_path)


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
        'test1_adaptive_f1': np.nan,
        'test2_acc': np.nan, 'test2_acc_labelled': np.nan, 'test2_acc_background': np.nan,
        'test2_macro_precision': np.nan, 'test2_macro_recall': np.nan,
        'test2_macro_f1': np.nan, 'test2_jaccard': np.nan,
        'test2_adaptive_f1': np.nan,
        'status': 'unknown',
    }


def load_from_viz_dir(viz_dir):
    """Load experiments written by run_experiments.sh (model_on_dataset_transform layout)."""
    results = []
    standard_pattern = re.compile(r'^(ast|regnet)_on_(avianz|doc|merged|large_doc|large_avianz)_(.+)$')
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
        'test1_name', 'test1_acc', 'test1_acc_labelled', 'test1_acc_background',
        'test1_macro_precision', 'test1_macro_recall', 'test1_macro_f1', 'test1_jaccard',
        'test1_adaptive_f1',
        'test2_name', 'test2_acc', 'test2_acc_labelled', 'test2_acc_background',
        'test2_macro_precision', 'test2_macro_recall', 'test2_macro_f1', 'test2_jaccard',
        'test2_adaptive_f1',
        'status',
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].copy()
    df_out = df_out.sort_values(['category', 'train_dataset', 'config'])
    df_out.to_csv(output_dir / 'all_results.csv', index=False, float_format='%.4f')
    print(f"✓ all_results.csv ({len(df_out)} experiments)")


def create_per_class_table(viz_dir, output_dir):
    """Create per_class_metrics.csv: one row per (experiment, test split, class)."""
    standard_pattern = re.compile(r'^(ast|regnet)_on_(avianz|doc|merged|large_doc|large_avianz)_(.+)$')
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


def print_model_comparison(df, tuned_lookup=None):
    """
    Print a clear terminal comparison of the best REGNET model vs Kaytoo and
    BirdNET pretrained baselines.

    If tuned_lookup is provided (from tune_thresholds_for_experiments), shows
    a second row for each model with per-species tuned threshold results.
    Kaytoo/BirdNET cannot be tuned post-hoc (no raw scores saved).
    """
    def _fmt(val):
        if val is np.nan or (isinstance(val, float) and np.isnan(val)):
            return "  N/A"
        return f"{val:5.1f}%"

    def _collect(row):
        return {
            't1':    _fmt(row.get('test1_acc',          np.nan)),
            't1lab': _fmt(row.get('test1_acc_labelled', np.nan)),
            't2':    _fmt(row.get('test2_acc',          np.nan)),
            't2lab': _fmt(row.get('test2_acc_labelled', np.nan)),
            't1n':   str(row.get('test1_name', '?')),
            't2n':   str(row.get('test2_name', '?')),
        }

    rows = []
    exp_names = {}  # label -> exp_name, for tuned lookup

    # ---- Best REGNET (matched / sweep datasets) ----------------------------
    regnet_df = df[
        (df['category'] == 'REGNET') &
        (~df['train_dataset'].isin(['large_doc', 'large_avianz']))
    ].copy()
    regnet_df = regnet_df.dropna(subset=['test1_acc', 'test2_acc'])
    if not regnet_df.empty:
        regnet_df['_avg'] = (regnet_df['test1_acc'] + regnet_df['test2_acc']) / 2
        best = regnet_df.sort_values('_avg', ascending=False).iloc[0]
        rows.append(('Best REGNET (matched)', best['name'], _collect(best)))
        exp_names['Best REGNET (matched)'] = best['name']
    else:
        rows.append(('Best REGNET (matched)', '(no results)', {}))

    # ---- Best REGNET on large DOC dataset -----------------------------------
    large_df = df[
        (df['category'] == 'REGNET') &
        (df['train_dataset'] == 'large_doc')
    ].copy()
    large_df = large_df.dropna(subset=['test1_acc', 'test2_acc'])
    if not large_df.empty:
        large_df['_avg'] = (large_df['test1_acc'] + large_df['test2_acc']) / 2
        best_large = large_df.sort_values('_avg', ascending=False).iloc[0]
        rows.append(('Best REGNET (large DOC)', best_large['name'], _collect(best_large)))
        exp_names['Best REGNET (large DOC)'] = best_large['name']
    # skip silently if not yet run

    # ---- Kaytoo pretrained --------------------------------------------------
    k_df = df[df['category'] == 'Kaytoo (Pretrained)']
    if not k_df.empty:
        rows.append(('Kaytoo (Pretrained)', '', _collect(k_df.iloc[0])))
        exp_names['Kaytoo (Pretrained)'] = k_df.iloc[0]['name']
    else:
        rows.append(('Kaytoo (Pretrained)', '(no results)', {}))

    # ---- BirdNET pretrained -------------------------------------------------
    b_df = df[df['category'] == 'BirdNET (Pretrained)']
    if not b_df.empty:
        rows.append(('BirdNET (Pretrained)', '', _collect(b_df.iloc[0])))
        exp_names['BirdNET (Pretrained)'] = b_df.iloc[0]['name']
    else:
        rows.append(('BirdNET (Pretrained)', '(no results)', {}))

    # ---- Print --------------------------------------------------------------
    # Determine test-set names from first row that has them
    t1n = t2n = '?'
    for _, _, m in rows:
        if m:
            t1n = m['t1n']
            t2n = m['t2n']
            break

    W = 84
    hdr2 = f"  {t1n:>10s} acc  (labelled)   {t2n:>10s} acc  (labelled)"

    print("\n" + "=" * W)
    print(" BEST MODEL vs BASELINES")
    print("=" * W)
    print(hdr2)
    print("-" * W)
    for label, cfg_name, m in rows:
        if not m:
            print(f"  {label}")
            continue
        print(
            f"  {label:<26s}"
            f"  {m['t1']:>7s}  ({m['t1lab']:>7s})"
            f"     {m['t2']:>7s}  ({m['t2lab']:>7s})"
        )
        if cfg_name:
            print(f"    {cfg_name}")

        # Show tuned row if available
        if tuned_lookup is not None:
            exp = exp_names.get(label)
            if exp:
                # Cross-split: tune on t2 → eval on t1, tune on t1 → eval on t2
                r1 = tuned_lookup.get((exp, t1n))
                r2 = tuned_lookup.get((exp, t2n))
                if r1 or r2:
                    def _tf(r, key):
                        return f"{r[key]:5.1f}%" if r else "  N/A "
                    t1_acc = _tf(r1, 'acc') if r1 else "  N/A "
                    t1_lab = _tf(r1, 'acc_lab') if r1 else "  N/A "
                    t2_acc = _tf(r2, 'acc') if r2 else "  N/A "
                    t2_lab = _tf(r2, 'acc_lab') if r2 else "  N/A "
                    print(
                        f"  {'  (tuned thresholds)':<26s}"
                        f"  {t1_acc:>7s}  ({t1_lab:>7s})"
                        f"     {t2_acc:>7s}  ({t2_lab:>7s})"
                    )

    print()
    print("  acc     = exact-match accuracy on all samples (incl. background)")
    print("  labelled = exact-match accuracy on bird-call samples only")
    if tuned_lookup is not None:
        print("  tuned   = per-class F1-optimal threshold (tuned on opposite split, applied independently per species)")
    print("=" * W + "\n")


def _load_predictions_with_gt(exp_dir, split, data_base=None):
    """Load predictions_{split}.csv and ground truth from labels.json.

    Returns (probs, y_true, class_names) as numpy arrays, or None if unavailable.

    Supported CSV formats:
      - Training experiments: columns = filename, class1, class2, ...
      - rerun_predictions.py (_v2): columns = row_id, class1, ..., y_class1, ...

    labels.json is found by:
      1. If data_base is given: data_base/{split}/test/labels.json
      2. Otherwise: derived from the filename paths inside the CSV
         (e.g. /some/path/data/file_XXX.npy → /some/path/labels.json)
    """
    csv_path = exp_dir / f"predictions_{split}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    # Detect which column holds the file identifier
    if 'filename' in df.columns:
        id_col = 'filename'
    elif 'row_id' in df.columns:
        id_col = 'row_id'
    else:
        return None

    # Class columns: exclude the id column and any ground-truth columns (y_{class})
    class_names = [
        c for c in df.columns
        if c != id_col and not c.startswith('y_')
    ]
    if not class_names:
        return None

    # Locate labels.json
    if data_base is not None:
        labels_path = Path(data_base) / split / "test" / "labels.json"
    else:
        first_file = Path(df[id_col].iloc[0])
        labels_path = first_file.parent.parent / "labels.json"

    if not labels_path.exists():
        return None

    with open(labels_path) as f:
        labels_data = json.load(f)

    # Map basename → multi-hot label vector
    label_map = {}
    for file_info in labels_data.get('files', []):
        basename = Path(file_info['filename']).name
        vec = [0] * len(class_names)
        for cls in file_info.get('class_names', []):
            if cls in class_names:
                vec[class_names.index(cls)] = 1
        label_map[basename] = vec

    probs, y_true = [], []
    for _, row in df.iterrows():
        basename = Path(row[id_col]).name
        label = label_map.get(basename)
        if label is None:
            continue
        probs.append(row[class_names].values.astype(np.float32))
        y_true.append(label)

    if not probs:
        return None
    return np.array(probs), np.array(y_true, dtype=np.float32), class_names


def _find_best_thresholds(probs, y_true, class_names, n_steps=100):
    """Grid-search the F1-optimal threshold independently for each class.

    For each class the threshold that maximises binary F1 on that class is
    found via a grid search over [0.02, 0.98].  Classes with no positive
    ground-truth samples default to 0.5.

    Returns a dict {class: threshold} with a potentially different value for
    every class.
    """
    from sklearn.metrics import f1_score as sk_f1
    thresholds = np.linspace(0.02, 0.98, n_steps)
    y_int = y_true.astype(int)
    result = {}
    for i, cls in enumerate(class_names):
        gt_col   = y_int[:, i]
        prob_col = probs[:, i]
        if gt_col.sum() == 0:
            result[cls] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            y_pred = (prob_col >= t).astype(int)
            f1 = sk_f1(gt_col, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        result[cls] = best_t
    return result


def _eval_with_thresholds(probs, y_true, class_names, thresholds):
    """Evaluate exact-match accuracy and macro-F1 using per-species thresholds."""
    from sklearn.metrics import f1_score as sk_f1
    thresh_vec = np.array([thresholds[c] for c in class_names], dtype=np.float32)
    y_pred = (probs >= thresh_vec).astype(int)
    exact  = np.all(y_true.astype(int) == y_pred, axis=1)
    acc    = float(np.mean(exact)) * 100
    labelled = y_true.sum(axis=1) > 0
    acc_lab  = float(np.mean(exact[labelled])) * 100 if labelled.any() else float('nan')
    macro_f1 = float(sk_f1(y_true.astype(int), y_pred, average='macro', zero_division=0))
    return acc, acc_lab, macro_f1


def tune_thresholds_for_experiments(results_dir, output_dir, data_base=None):
    """
    For every experiment directory that has predictions_{split}.csv files,
    tune per-species thresholds on one split and evaluate on the other
    (cross-split, so the reported numbers are honest).

    Saves tuned_thresholds_results.csv and prints a summary table.
    """
    SPLITS = [("avianz_split", "doc_split"), ("doc_split", "avianz_split")]

    standard_pattern = re.compile(
        r'^(ast|regnet|cnn)_on_(avianz|doc|merged|large_doc|large_avianz)_(.+)$'
    )
    pretrained_pattern = re.compile(
        r'^(kaytoo|birdnet)_pretrained_.*$'
    )

    rows = []
    for exp_dir in sorted(Path(results_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        if not standard_pattern.match(exp_dir.name) and not pretrained_pattern.match(exp_dir.name):
            continue

        # Need predictions for both splits to do cross-split tuning
        data = {}
        for split in ("avianz_split", "doc_split"):
            result = _load_predictions_with_gt(exp_dir, split, data_base=data_base)
            if result is not None:
                data[split] = result  # (probs, y_true, class_names)

        if len(data) < 2:
            continue  # not enough data for cross-split

        for tune_split, eval_split in SPLITS:
            tune_probs, tune_gt, class_names = data[tune_split]
            eval_probs, eval_gt, _           = data[eval_split]

            thresholds = _find_best_thresholds(tune_probs, tune_gt, class_names)

            # Baseline (fixed 0.5)
            fixed_t = {c: 0.5 for c in class_names}
            acc_fixed,   acc_lab_fixed,   f1_fixed   = _eval_with_thresholds(eval_probs, eval_gt, class_names, fixed_t)
            acc_tuned,   acc_lab_tuned,   f1_tuned   = _eval_with_thresholds(eval_probs, eval_gt, class_names, thresholds)

            rows.append({
                'experiment':      exp_dir.name,
                'tune_on':         tune_split,
                'eval_on':         eval_split,
                'acc_fixed':       round(acc_fixed,   2),
                'acc_lab_fixed':   round(acc_lab_fixed, 2),
                'macro_f1_fixed':  round(f1_fixed,    4),
                'acc_tuned':       round(acc_tuned,   2),
                'acc_lab_tuned':   round(acc_lab_tuned, 2),
                'macro_f1_tuned':  round(f1_tuned,    4),
                'acc_delta':       round(acc_tuned   - acc_fixed,   2),
                'f1_delta':        round(f1_tuned    - f1_fixed,    4),
                'thresholds':      thresholds,
            })

    if not rows:
        print("  No experiments with predictions_{split}.csv found — nothing to tune.")
        return {}

    # Save CSV (without the thresholds dict column)
    df_out = pd.DataFrame([{k: v for k, v in r.items() if k != 'thresholds'} for r in rows])
    out_csv = output_dir / 'tuned_thresholds_results.csv'
    df_out.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"✓ tuned_thresholds_results.csv ({len(df_out)} rows)")

    # Print summary table
    W = 100
    print("\n" + "=" * W)
    print(" PER-SPECIES THRESHOLD TUNING  (tune on one split → evaluate on the other)")
    print("=" * W)
    print(f"  {'Experiment':<45s}  {'Eval on':<14s}  {'Acc(fixed)':<12} {'Acc(tuned)':<12} {'ΔAcc':>6}  {'F1(fixed)':<11} {'F1(tuned)':<11} {'ΔF1':>7}")
    print("-" * W)

    for r in sorted(rows, key=lambda x: x['experiment']):
        print(
            f"  {r['experiment']:<45s}  {r['eval_on']:<14s}"
            f"  {r['acc_fixed']:>8.1f}%   {r['acc_tuned']:>8.1f}%  {r['acc_delta']:>+6.1f}%"
            f"  {r['macro_f1_fixed']:>8.4f}   {r['macro_f1_tuned']:>8.4f}  {r['f1_delta']:>+7.4f}"
        )

    # Average gain
    avg_acc_delta = np.nanmean([r['acc_delta'] for r in rows])
    avg_f1_delta  = np.nanmean([r['f1_delta']  for r in rows])
    print("-" * W)
    print(f"  {'Average gain':<45s}  {'':14s}  {'':>12} {'':>12} {avg_acc_delta:>+6.1f}%  {'':>11} {'':>11} {avg_f1_delta:>+7.4f}")
    print("=" * W + "\n")

    # Per-class threshold summary across all experiments
    all_thresh_vals = []
    for r in rows:
        all_thresh_vals.extend(r['thresholds'].values())
    if all_thresh_vals:
        print(f"  Per-class tuned thresholds: median={np.median(all_thresh_vals):.3f}  "
              f"(range {min(all_thresh_vals):.3f}–{max(all_thresh_vals):.3f})")
    print()
    # Build lookup: (exp_name, eval_split) -> {'acc', 'acc_lab', 'macro_f1'}
    tuned_lookup = {}
    for r in rows:
        tuned_lookup[(r['experiment'], r['eval_on'])] = {
            'acc':      r['acc_tuned'],
            'acc_lab':  r['acc_lab_tuned'],
            'f1':       r['macro_f1_tuned'],
        }
    return tuned_lookup

def main():
    parser = argparse.ArgumentParser(description='Analyze all experimental results')
    parser.add_argument('results_dir',
                        help='Output directory (e.g. /local/scratch/freangi/matched_tests). '
                             'Scans for both result.json and *_multilabel_report.json automatically.')
    parser.add_argument('--output', '-o', default=None, help='Output directory for analysis files')
    parser.add_argument('--tune-thresholds', action='store_true',
                        help='Tune per-species thresholds on one test split and evaluate on the other '
                             '(cross-split). Reads existing predictions_{split}.csv files — no '
                             'model re-run needed.')
    parser.add_argument('--data-base', default=None, metavar='DIR',
                        help='Root directory containing {avianz_split,doc_split}/test/labels.json. '
                             'Required when labels.json paths in the prediction CSVs point to a '
                             'remote server (e.g. /local/scratch/freangi/matched). '
                             'Sync just those two files: '
                             'rsync server:/local/scratch/freangi/matched/avianz_split/test/labels.json DATA_BASE/avianz_split/test/ '
                             'and the same for doc_split.')
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

    # Optionally tune thresholds first, then incorporate into the comparison table
    tuned_lookup = None
    if args.tune_thresholds:
        print("\nTuning per-species thresholds (cross-split)...")
        tuned_lookup = tune_thresholds_for_experiments(results_dir, output_dir, data_base=args.data_base)

    # Print best-model comparison to terminal (with tuned rows if available)
    print_model_comparison(df, tuned_lookup=tuned_lookup)
    
    print("\n" + "="*70)
    print(" DONE")
    print("="*70)
    print(f" Output: {output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
