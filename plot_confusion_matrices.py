"""
Plot test confusion matrices from experiments_matched prediction CSVs.

Ground truth comes from labels.json in the test folders (same as
finetune_birdclef._compute_accuracy_from_csv).  Pass the two test folders
so the script can find their labels.json.

Usage:
    python plot_confusion_matrices.py \
        --exp-dir experiments_matched \
        --doc-test   /local/scratch/freangi/doc_split/test \
        --avianz-test /local/scratch/freangi/joe_mo_split/test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_ground_truth(test_folder):
    """Load {filename -> primary_class} from test_folder/labels.json."""
    labels_path = Path(test_folder) / 'labels.json'
    if not labels_path.exists():
        raise FileNotFoundError(f'labels.json not found in {test_folder}')
    with open(labels_path) as f:
        data = json.load(f)
    gt = {}
    for item in data['files']:
        fname = item['filename']
        label = item.get('primary_class') or item.get('primary_species')
        if not label:
            cl = item.get('class_names', [])
            label = cl[0] if cl else None
        if fname and label:
            gt[fname] = label
    print(f'  Loaded {len(gt)} ground-truth labels from {labels_path}')
    return gt


def plot_and_save(y_true, y_pred, title, out_png):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalise
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    n = len(labels)
    cell = max(0.55, min(1.2, 10.0 / n))
    fig, ax = plt.subplots(figsize=(n * cell + 2, n * cell + 1.5))

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, vmin=0, vmax=1,
        linewidths=0.3, linecolor='whitesmoke',
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Also save raw-count CSV
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        str(out_png).replace('.png', '_counts.csv')
    )

    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    print(f'  Saved: {out_png}  (n={len(y_true)}, acc={acc:.1f}%, {n} classes)')


def process_csv(csv_path, gt, exp_name, csv_stem):
    df = pd.read_csv(csv_path)
    class_cols = [c for c in df.columns if c not in ('File_Path', 'row_id')]

    y_true, y_pred = [], []
    missing = 0
    for _, row in df.iterrows():
        fname = row['row_id']
        if fname not in gt:
            missing += 1
            continue
        pred = class_cols[row[class_cols].values.argmax()]
        y_true.append(gt[fname])
        y_pred.append(pred)

    if missing:
        print(f'  {missing} rows not in labels.json (skipped)')
    if not y_true:
        print(f'  ERROR: no matching rows — wrong labels.json for this CSV?')
        return

    train_on = 'AviaNZ' if 'joe_mo' in exp_name else 'DOC'
    test_on  = 'AviaNZ' if 'avianz' in csv_stem else 'DOC'
    title = f'Train: {train_on}  →  Test: {test_on}\n({exp_name})'

    out_png = csv_path.parent / f'confusion_{csv_stem}.png'
    plot_and_save(y_true, y_pred, title, out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', default='experiments_matched')
    parser.add_argument('--doc-test', required=True,
                        help='Folder containing labels.json for the DOC test set')
    parser.add_argument('--avianz-test', required=True,
                        help='Folder containing labels.json for the AviaNZ test set')
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        sys.exit(f'ERROR: {exp_dir} does not exist')

    print('Loading ground-truth labels...')
    gt_doc    = load_ground_truth(args.doc_test)
    gt_avianz = load_ground_truth(args.avianz_test)

    csv_files = sorted(exp_dir.glob('*/predictions_*.csv'))
    if not csv_files:
        sys.exit('No prediction CSVs found under ' + str(exp_dir))

    print(f'\nProcessing {len(csv_files)} CSVs...')
    for csv_path in csv_files:
        exp_name = csv_path.parent.name
        csv_stem = csv_path.stem
        print(f'\n{exp_name} / {csv_path.name}')

        is_doc = 'doc' in csv_stem.lower()
        gt = gt_doc if is_doc else gt_avianz
        process_csv(csv_path, gt, exp_name, csv_stem)

    print('\nDone.')


if __name__ == '__main__':
    main()
