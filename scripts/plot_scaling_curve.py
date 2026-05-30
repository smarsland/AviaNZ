"""
Plot the data-scaling curve.  Saves THREE separate images:

  scaling_curve_f1.png          — macro-F1 (@ 0.5 and tuned threshold)
  scaling_curve_acc.png         — exact accuracy, all samples (@ 0.5 and tuned)
  scaling_curve_acc_labelled.png— exact accuracy, labelled only (@ 0.5 and tuned)

Each image is 2 rows × 2 cols:
  row 0: threshold 0.5   | AviaNZ  |  DOC
  row 1: tuned threshold | AviaNZ  |  DOC

Usage:
    python scripts/plot_scaling_curve.py [scaling_tests/ [matched_tests/]]

If matched_tests/ is not given as a second argument, the script looks for it
automatically next to scaling_tests/.  From matched_tests it adds:
  • Kaytoo (finetuned)          — if kaytoo_finetuned results are present
  • RegNet baseline (full data) — regnet_on_doc_baseline
  • Best RegNet (full data)     — top-1 by avg macro-F1
"""
import sys
import re
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _extract_baseline_vals(row):
    """Extract the 12 metric values from a CSV row, returning None on error."""
    try:
        return {
            'avianz_f1':               float(row['test1_macro_f1']),
            'doc_f1':                  float(row['test2_macro_f1']),
            'avianz_adaptive_f1':      float(row['test1_adaptive_f1']),
            'doc_adaptive_f1':         float(row['test2_adaptive_f1']),
            'avianz_acc':              float(row['test1_acc']) / 100.0,
            'doc_acc':                 float(row['test2_acc']) / 100.0,
            'avianz_adaptive_acc':     float(row['test1_adaptive_acc']) / 100.0,
            'doc_adaptive_acc':        float(row['test2_adaptive_acc']) / 100.0,
            'avianz_acc_lab':          float(row['test1_acc_labelled']) / 100.0,
            'doc_acc_lab':             float(row['test2_acc_labelled']) / 100.0,
            'avianz_adaptive_acc_lab': float(row['test1_adaptive_acc_labelled']) / 100.0,
            'doc_adaptive_acc_lab':    float(row['test2_adaptive_acc_labelled']) / 100.0,
        }
    except (ValueError, KeyError):
        return None


def load_matched_baselines(matched_dir):
    """Load representative baselines from matched_tests/analysis/all_results.csv.

    Returns a dict suitable for merging into the baselines dict used by
    make_figure().  Returns an empty dict if the CSV is missing or invalid.
    Adds:
      - Kaytoo (finetuned)          if any kaytoo_finetuned* row exists
      - RegNet baseline (full data) for regnet_on_doc_baseline
      - Best RegNet (full data)     top-1 REGNET row by avg(F1_test1, F1_test2)
    """
    csv_path = os.path.join(matched_dir, 'analysis', 'all_results.csv')
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    result = {}

    # Kaytoo finetuned
    for row in rows:
        if 'kaytoo' in row['name'].lower() and 'finetuned' in row['name'].lower():
            vals = _extract_baseline_vals(row)
            if vals:
                vals.update({'color': '#d35400', 'ls': '-.'})
                result['Kaytoo (finetuned)'] = vals
                break

    regnet_rows = [r for r in rows if r.get('category', '') == 'REGNET']

    # RegNet baseline (full data)
    for row in regnet_rows:
        if row['name'] == 'regnet_on_doc_baseline':
            vals = _extract_baseline_vals(row)
            if vals:
                vals.update({'color': '#95a5a6', 'ls': '--'})
                result['RegNet baseline (corrected DOC labels)'] = vals
            break

    # Best RegNet by avg macro-F1 across both test sets
    def _avg_f1(row):
        try:
            return (float(row['test1_macro_f1']) + float(row['test2_macro_f1'])) / 2
        except (ValueError, KeyError):
            return -1.0

    best = max(regnet_rows, key=_avg_f1, default=None)
    if best:
        vals = _extract_baseline_vals(best)
        if vals:
            short_name = best['name'].replace('regnet_on_doc_', '')
            vals.update({'color': '#c0392b', 'ls': '--'})
            result[f'Best RegNet (corrected DOC labels): {short_name}'] = vals

    return result


def make_figure(points, baselines, panel_specs, col_map, suptitle, seeds, seed_colors):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(suptitle, fontsize=13)

    all_ns = sorted(set(p[0] for s in seeds for p in points[s]))

    for row_i, col_i, key, title, ylabel in panel_specs:
        ax = axes[row_i][col_i]
        idx = col_map[key]

        for si, seed in enumerate(seeds):
            pts = points[seed]
            ns   = [p[0] for p in pts]
            vals = [p[idx] for p in pts]
            label = f'RegNet (seed {seed})' if len(seeds) > 1 else 'RegNet (full DOC)'
            ax.plot(ns, vals, 'o-', color=seed_colors[si % len(seed_colors)],
                    linewidth=2, markersize=5, label=label, zorder=3)

        for bname, bvals in baselines.items():
            bval = bvals.get(key)
            if bval is None:
                continue
            ax.axhline(bval, color=bvals['color'], linestyle=bvals['ls'],
                       linewidth=1.5, label=bname, zorder=2)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Training samples per class (N)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(all_ns)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'scaling_tests'
    csv_path = os.path.join(results_dir, 'analysis', 'all_results.csv')
    out_dir  = os.path.join(results_dir, 'analysis')

    # Resolve matched_tests dir (explicit arg or auto-detect sibling)
    if len(sys.argv) > 2:
        matched_dir = sys.argv[2]
    else:
        candidate = os.path.join(os.path.dirname(os.path.abspath(results_dir)),
                                 'matched_tests')
        matched_dir = candidate if os.path.isdir(candidate) else None

    if not os.path.exists(csv_path):
        print(f'ERROR: {csv_path} not found')
        sys.exit(1)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract scaling rows
    pattern = re.compile(r'regnet_on_doc_scaling_.*?N(\d+)_seed(\d+)')
    # tuple indices:
    #  0=N, 1=avianz_f1, 2=doc_f1, 3=avianz_adaptive_f1, 4=doc_adaptive_f1,
    #  5=avianz_acc, 6=doc_acc, 7=avianz_adaptive_acc, 8=doc_adaptive_acc,
    #  9=avianz_acc_lab, 10=doc_acc_lab, 11=avianz_adaptive_acc_lab, 12=doc_adaptive_acc_lab
    points = {}
    for row in rows:
        m = pattern.search(row['name'])
        if not m:
            continue
        n    = int(m.group(1))
        seed = int(m.group(2))
        try:
            entry = (
                n,
                float(row['test1_macro_f1']),
                float(row['test2_macro_f1']),
                float(row['test1_adaptive_f1']),
                float(row['test2_adaptive_f1']),
                float(row['test1_acc']) / 100.0,
                float(row['test2_acc']) / 100.0,
                float(row['test1_adaptive_acc']) / 100.0,
                float(row['test2_adaptive_acc']) / 100.0,
                float(row['test1_acc_labelled']) / 100.0,
                float(row['test2_acc_labelled']) / 100.0,
                float(row['test1_adaptive_acc_labelled']) / 100.0,
                float(row['test2_adaptive_acc_labelled']) / 100.0,
            )
        except (ValueError, KeyError):
            continue
        points.setdefault(seed, []).append(entry)

    if not points:
        print('No scaling rows found in CSV.')
        sys.exit(1)

    for seed in points:
        points[seed].sort(key=lambda x: x[0])

    # Baselines
    baselines = {}
    for row in rows:
        name = row['name']
        if 'kaytoo' in name.lower():
            key = 'Kaytoo (pretrained)'
            color, ls = '#e67e22', '--'
        elif 'birdnet' in name.lower():
            key = 'BirdNET (pretrained)'
            color, ls = '#8e44ad', ':'
        else:
            continue
        try:
            baselines[key] = {
                'avianz_f1':              float(row['test1_macro_f1']),
                'doc_f1':                 float(row['test2_macro_f1']),
                'avianz_adaptive_f1':     float(row['test1_adaptive_f1']),
                'doc_adaptive_f1':        float(row['test2_adaptive_f1']),
                'avianz_acc':             float(row['test1_acc']) / 100.0,
                'doc_acc':                float(row['test2_acc']) / 100.0,
                'avianz_adaptive_acc':    float(row['test1_adaptive_acc']) / 100.0,
                'doc_adaptive_acc':       float(row['test2_adaptive_acc']) / 100.0,
                'avianz_acc_lab':         float(row['test1_acc_labelled']) / 100.0,
                'doc_acc_lab':            float(row['test2_acc_labelled']) / 100.0,
                'avianz_adaptive_acc_lab':float(row['test1_adaptive_acc_labelled']) / 100.0,
                'doc_adaptive_acc_lab':   float(row['test2_adaptive_acc_labelled']) / 100.0,
                'color': color, 'ls': ls,
            }
        except (ValueError, KeyError):
            continue

    # Merge in matched-test baselines (kaytoo finetuned + representative RegNets)
    if matched_dir:
        matched_baselines = load_matched_baselines(matched_dir)
        if matched_baselines:
            print(f'Loaded {len(matched_baselines)} baseline(s) from {matched_dir}:')
            for k in matched_baselines:
                print(f'  • {k}')
        baselines.update(matched_baselines)
    else:
        print('No matched_tests dir found; skipping matched baselines.')

    seeds = sorted(points.keys())
    seed_colors = ['#2980b9', '#27ae60', '#16a085', '#8e44ad']
    SUPTITLE = ('RegNetY-008 — Data Scaling on Noisy DOC Labels (N samples/class)\n'
                'Evaluated on Human-Reviewed Matched Test Sets')

    col_map = {
        'avianz_f1':               1,
        'doc_f1':                  2,
        'avianz_adaptive_f1':      3,
        'doc_adaptive_f1':         4,
        'avianz_acc':              5,
        'doc_acc':                 6,
        'avianz_adaptive_acc':     7,
        'doc_adaptive_acc':        8,
        'avianz_acc_lab':          9,
        'doc_acc_lab':             10,
        'avianz_adaptive_acc_lab': 11,
        'doc_adaptive_acc_lab':    12,
    }

    # ---- Image 1: macro-F1 ----
    f1_panels = [
        (0, 0, 'avianz_f1',           'AviaNZ test set (reviewed) — macro-F1 @ thr 0.5',    'macro-F1 @ 0.5'),
        (0, 1, 'doc_f1',              'DOC test set (reviewed) — macro-F1 @ thr 0.5',       'macro-F1 @ 0.5'),
        (1, 0, 'avianz_adaptive_f1',  'AviaNZ test set (reviewed) — macro-F1 (tuned thr)',  'macro-F1 (tuned)'),
        (1, 1, 'doc_adaptive_f1',     'DOC test set (reviewed) — macro-F1 (tuned thr)',     'macro-F1 (tuned)'),
    ]
    fig1 = make_figure(points, baselines, f1_panels, col_map, SUPTITLE + '\nmacro-F1', seeds, seed_colors)
    out1 = os.path.join(out_dir, 'scaling_curve_f1.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved: {out1}')

    # ---- Image 2: exact accuracy (all samples) ----
    acc_panels = [
        (0, 0, 'avianz_acc',           'AviaNZ test set (reviewed) — accuracy @ thr 0.5',    'accuracy @ 0.5'),
        (0, 1, 'doc_acc',              'DOC test set (reviewed) — accuracy @ thr 0.5',       'accuracy @ 0.5'),
        (1, 0, 'avianz_adaptive_acc',  'AviaNZ test set (reviewed) — accuracy (tuned thr)',  'accuracy (tuned)'),
        (1, 1, 'doc_adaptive_acc',     'DOC test set (reviewed) — accuracy (tuned thr)',     'accuracy (tuned)'),
    ]
    fig2 = make_figure(points, baselines, acc_panels, col_map, SUPTITLE + '\nExact Accuracy — all samples', seeds, seed_colors)
    out2 = os.path.join(out_dir, 'scaling_curve_acc.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')

    # ---- Image 3: exact accuracy (labelled only) ----
    acc_lab_panels = [
        (0, 0, 'avianz_acc_lab',           'AviaNZ test set (reviewed) — accuracy @ thr 0.5 (labelled)',    'accuracy @ 0.5 (labelled)'),
        (0, 1, 'doc_acc_lab',              'DOC test set (reviewed) — accuracy @ thr 0.5 (labelled)',       'accuracy @ 0.5 (labelled)'),
        (1, 0, 'avianz_adaptive_acc_lab',  'AviaNZ test set (reviewed) — accuracy (tuned thr, labelled)',  'accuracy (tuned, labelled)'),
        (1, 1, 'doc_adaptive_acc_lab',     'DOC test set (reviewed) — accuracy (tuned thr, labelled)',     'accuracy (tuned, labelled)'),
    ]
    fig3 = make_figure(points, baselines, acc_lab_panels, col_map, SUPTITLE + '\nExact Accuracy — labelled samples only', seeds, seed_colors)
    out3 = os.path.join(out_dir, 'scaling_curve_acc_labelled.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {out3}')


if __name__ == '__main__':
    main()
