"""
Plot the data-scaling curve: AviaNZ and DOC macro-F1 vs N samples/class.

Reads scaling_tests/analysis/all_results.csv and saves a figure to
scaling_tests/analysis/scaling_curve.png.

Usage:
    python scripts/plot_scaling_curve.py [scaling_tests/]
"""
import sys
import re
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'scaling_tests'
    csv_path = os.path.join(results_dir, 'analysis', 'all_results.csv')
    out_path = os.path.join(results_dir, 'analysis', 'scaling_curve.png')

    if not os.path.exists(csv_path):
        print(f'ERROR: {csv_path} not found')
        sys.exit(1)

    # Parse rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract scaling rows: name must match regnet_on_doc_scaling_N{N}_seed*
    pattern = re.compile(r'regnet_on_doc_scaling_N(\d+)_seed(\d+)')
    points = {}  # seed -> list of (N, avianz_f1, doc_f1, avianz_adaptive, doc_adaptive)
    for row in rows:
        m = pattern.search(row['name'])
        if not m:
            continue
        n = int(m.group(1))
        seed = int(m.group(2))
        try:
            avianz_f1      = float(row['test1_macro_f1'])
            doc_f1         = float(row['test2_macro_f1'])
            avianz_adaptive = float(row['test1_adaptive_f1'])
            doc_adaptive    = float(row['test2_adaptive_f1'])
        except (ValueError, KeyError):
            continue
        points.setdefault(seed, []).append((n, avianz_f1, doc_f1, avianz_adaptive, doc_adaptive))

    if not points:
        print('No scaling rows found in CSV.')
        sys.exit(1)

    # Get baselines (kaytoo and birdnet)
    baselines = {}
    for row in rows:
        name = row['name']
        if 'kaytoo' in name.lower():
            baselines['Kaytoo (pretrained)'] = {
                'avianz': float(row['test1_macro_f1']),
                'doc':    float(row['test2_macro_f1']),
                'avianz_adaptive': float(row['test1_adaptive_f1']),
                'doc_adaptive':    float(row['test2_adaptive_f1']),
                'color': '#e67e22',
                'ls': '--',
            }
        elif 'birdnet' in name.lower():
            baselines['BirdNET (pretrained)'] = {
                'avianz': float(row['test1_macro_f1']),
                'doc':    float(row['test2_macro_f1']),
                'avianz_adaptive': float(row['test1_adaptive_f1']),
                'doc_adaptive':    float(row['test2_adaptive_f1']),
                'color': '#8e44ad',
                'ls': ':',
            }

    # Sort each seed's points by N
    for seed in points:
        points[seed].sort(key=lambda x: x[0])

    # -----------------------------------------------------------------------
    # Figure: 2 rows × 2 cols
    #   [0,0] AviaNZ F1 @ 0.5    [0,1] DOC F1 @ 0.5
    #   [1,0] AviaNZ adaptive F1 [1,1] DOC adaptive F1
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RegNetY-008 (BirdClef pretrained) — Data Scaling on Noisy DOC Labels\n'
                 'Evaluated on Human-Reviewed Matched Test Sets', fontsize=13)

    seeds = sorted(points.keys())
    seed_colors = ['#2980b9', '#27ae60', '#c0392b', '#16a085']

    panel_specs = [
        (0, 0, 'avianz',          'AviaNZ test — macro-F1 @ 0.5',           'F1 @ threshold 0.5'),
        (0, 1, 'doc',             'DOC test — macro-F1 @ 0.5',              'F1 @ threshold 0.5'),
        (1, 0, 'avianz_adaptive', 'AviaNZ test — macro-F1 (tuned threshold)', 'F1 (per-class threshold)'),
        (1, 1, 'doc_adaptive',    'DOC test — macro-F1 (tuned threshold)',    'F1 (per-class threshold)'),
    ]

    col_map = {
        'avianz':          1,
        'doc':             2,
        'avianz_adaptive': 3,
        'doc_adaptive':    4,
    }

    for row_i, col_i, key, title, ylabel in panel_specs:
        ax = axes[row_i][col_i]
        idx = col_map[key]

        for si, seed in enumerate(seeds):
            pts = points[seed]
            ns   = [p[0] for p in pts]
            vals = [p[idx] for p in pts]
            label = f'RegNet (seed {seed})' if len(seeds) > 1 else 'RegNet (noisy DOC)'
            ax.plot(ns, vals, 'o-', color=seed_colors[si % len(seed_colors)],
                    linewidth=2, markersize=5, label=label, zorder=3)

        # Baseline horizontals
        for bname, bvals in baselines.items():
            bval = bvals[key] if key in bvals else None
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
        all_ns = [p[0] for s in seeds for p in points[s]]
        ax.set_xticks(sorted(set(all_ns)))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
