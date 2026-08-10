#!/usr/bin/env python3
"""
Single script to produce a 2x3 comparison figure for the 4 NZ bird models.

Six conditions (2 rows x 3 columns):
  Row 1 - large test sets, oracle thresholds tuned on same data:
    (1) Combined-DOC test,          thresholds from combined-DOC
    (2) Combined-AviaNZ test,       thresholds from combined-AviaNZ
    (3) Combined (DOC+AviaNZ) test, thresholds from combined (DOC+AviaNZ)
  Row 2 - matched test sets, thresholds transferred from large sets:
    (4) Matched-DOC test,           thresholds from combined-DOC
    (5) Matched-AviaNZ test,        thresholds from combined-AviaNZ
    (6) Matched combined test,       thresholds from combined (DOC+AviaNZ)

Each model is evaluated on ALL its own classes - no label normalisation.
Thresholds are computed per-class from the source split and matched to the test
split by class name.  Each model's CSVs use a consistent label scheme, so
within-model threshold transfer is exact.

Two figures are always produced:
  <out>           - standard mode: thresholds computed on all source classes.
  <out>_test_classes - restricted mode: source data is first filtered to only
                       classes present in the test ground truth before thresholds
                       are computed, then applied to the test set.

Note: macro F1 is not directly comparable across models with very different
class counts (BirdNET 9 vs Kaytoo ~85 vs RegNet 120).

Usage:
    python3 scripts/plot_results.py
    python3 scripts/plot_results.py --model-tests /path/to/model_tests
    python3 scripts/plot_results.py --out figure.png --metric macro_f1
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODELS = [
    # (experiment_dir_name,              display_label,               csv_layout)
    ("birdnet_pretrained_seed0",         "BirdNET\n(pretrained)",    "subdir"),
    ("kaytoo_pretrained_seed0",          "Kaytoo\n(pretrained)",     "subdir"),
    ("regnet_on_doc_bgsub",              "RegNet+BgSub\n(DOC)",      "flat"),
    ("regnet_combined_bgsubtract_seed0", "RegNet+BgSub\n(combined)", "flat"),
]

BAR_COLOR = {
    "birdnet_pretrained_seed0":         "#C44E52",
    "kaytoo_pretrained_seed0":          "#E07A3A",
    "regnet_on_doc_bgsub":              "#2B7BB9",
    "regnet_combined_bgsubtract_seed0": "#27AE60",
}
BAR_EDGE = {
    "birdnet_pretrained_seed0":         "#7B241C",
    "kaytoo_pretrained_seed0":          "#7D3010",
    "regnet_on_doc_bgsub":              "#1A5276",
    "regnet_combined_bgsubtract_seed0": "#1E8449",
}


# ---------------------------------------------------------------------------
# CSV path resolution
# ---------------------------------------------------------------------------

def find_csv(model_dir, layout, split):
    """Return the predictions CSV path, or None if the file does not exist."""
    if layout == "subdir":
        subdir_map = {
            "combined_doc":    ("combined_dataset__doc_split",    "predictions_doc_split.csv"),
            "combined_avianz": ("combined_dataset__avianz_split", "predictions_avianz_split.csv"),
            "matched_doc":     ("matched__doc_split",             "predictions_doc_split.csv"),
            "matched_avianz":  ("matched__avianz_split",          "predictions_avianz_split.csv"),
        }
        sub, fname = subdir_map[split]
        p = model_dir / sub / fname
    else:
        flat_map = {
            "combined_doc":    "predictions_combined_dataset__doc_split.csv",
            "combined_avianz": "predictions_combined_dataset__avianz_split.csv",
            "matched_doc":     "predictions_matched__doc_split.csv",
            "matched_avianz":  "predictions_matched__avianz_split.csv",
        }
        p = model_dir / flat_map[split]
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a predictions CSV.  Returns None if missing or all-NaN rows."""
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, index_col="filename")
    pred_cols = [c for c in df.columns if not c.startswith("true_")]
    df = df[df[pred_cols].notna().any(axis=1)]
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds(df, n_cands=101):
    """
    Per-class threshold optimisation maximising per-class F1 on df.

    Only classes that have at least one positive in the ground truth are
    included.  Classes with no positives are omitted - no fallback.

    Returns dict: class_name -> float threshold in [0, 1].
    """
    pred_cols = [c for c in df.columns if not c.startswith("true_")]
    candidates = np.linspace(0.0, 1.0, n_cands, dtype=np.float32)
    thresholds = {}

    for cls in pred_cols:
        tc = "true_" + cls
        if tc not in df.columns:
            continue
        valid = df[[cls, tc]].dropna()
        if valid.empty:
            continue
        y_prob = valid[cls].values.astype(np.float32)
        y_true = valid[tc].values.astype(np.int32)
        if y_true.sum() == 0:
            continue  # no positives in source - cannot tune, omit

        preds = (y_prob[np.newaxis, :] >= candidates[:, np.newaxis]).astype(np.int32)
        tp    = ( preds *  y_true[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fp    = ( preds * (1 - y_true)[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fn    = ((1 - preds) * y_true[np.newaxis, :]).sum(axis=1).astype(np.float32)
        denom = 2 * tp + fp + fn
        f1s   = np.where(denom > 0, 2 * tp / denom, 0.0)
        thresholds[cls] = float(candidates[np.argmax(f1s)])

    return thresholds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(test_df, thresholds, test_classes_only=False):
    """
    Apply thresholds to test_df and return macro F1, micro F1, exact accuracy.

    Default mode (test_classes_only=False):
      Evaluated on ALL classes that have a tuned threshold AND a prediction
      column in test_df.  Classes absent from the test ground truth (no true_
      column or all-zero true labels) contribute F1=0 to the macro average,
      penalising models that predict irrelevant species on the test set.

    test_classes_only=True:
      Evaluated only on classes that have positives in the test ground truth.
      Classes without a tuned threshold fall back to threshold 0.5.
    """
    nan_result = {"macro_f1": np.nan, "micro_f1": np.nan, "exact_acc": np.nan,
                  "n_classes": 0}

    pred_cols = [c for c in test_df.columns if not c.startswith("true_")]
    test_df = test_df[test_df[pred_cols].notna().any(axis=1)]
    if test_df.empty:
        return nan_result

    if test_classes_only:
        # Only classes with actual positives in the test ground truth.
        present = [
            cls for cls in pred_cols
            if "true_" + cls in test_df.columns
            and int(test_df["true_" + cls].fillna(0).sum()) > 0
        ]
        thresh = {cls: thresholds.get(cls, 0.5) for cls in present}
    else:
        # All classes the model predicts for which a threshold was tuned.
        # Missing true_ columns → all-zero ground truth → F1=0 for that class.
        present = [cls for cls in pred_cols if cls in thresholds]
        thresh  = {cls: thresholds[cls] for cls in present}

    if not present:
        return nan_result

    # Build ground truth: use true_ column when available, zeros otherwise.
    y_true = np.column_stack([
        test_df["true_" + cls].fillna(0).values.astype(np.int32)
        if "true_" + cls in test_df.columns
        else np.zeros(len(test_df), dtype=np.int32)
        for cls in present
    ])
    y_pred = np.column_stack([
        (test_df[cls].fillna(0).values >= thresh[cls]).astype(np.int32)
        for cls in present
    ])

    per_f1 = []
    for j in range(len(present)):
        tp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 1)).sum())
        fp = int(((y_pred[:, j] == 1) & (y_true[:, j] == 0)).sum())
        fn = int(((y_pred[:, j] == 0) & (y_true[:, j] == 1)).sum())
        denom = 2 * tp + fp + fn
        per_f1.append(2 * tp / denom if denom > 0 else 0.0)

    macro_f1 = float(np.mean(per_f1))

    tp_all  = int(((y_pred == 1) & (y_true == 1)).sum())
    fp_all  = int(((y_pred == 1) & (y_true == 0)).sum())
    fn_all  = int(((y_pred == 0) & (y_true == 1)).sum())
    denom_m = 2 * tp_all + fp_all + fn_all
    micro_f1 = float(2 * tp_all / denom_m) if denom_m > 0 else 0.0

    exact_acc = float(np.all(y_pred == y_true, axis=1).mean() * 100)

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "exact_acc": exact_acc,
            "n_classes": len(present)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_test_classes(test_df):
    """Return the set of class names that have positives in the test ground truth."""
    pred_cols = [c for c in test_df.columns if not c.startswith("true_")]
    return {
        cls for cls in pred_cols
        if "true_" + cls in test_df.columns
        and int(test_df["true_" + cls].fillna(0).sum()) > 0
    }


def restrict_to_classes(df, classes):
    """Keep only prediction and ground-truth columns for the given class names."""
    keep = [c for c in df.columns
            if c in classes or (c.startswith("true_") and c[5:] in classes)]
    return df[keep] if keep else df.iloc[:, 0:0]


def run_conditions(model_data, metric, conditions, test_classes_only=False):
    """Compute *metric* for every (condition, model) combination.

    When test_classes_only=True the threshold-source DataFrame is restricted
    to only the classes with positives in the test ground truth before
    thresholds are computed.
    """
    results = [{} for _ in conditions]
    for ci, (_title, thresh_splits, test_split) in enumerate(conditions):
        for exp_name, _label, _layout in MODELS:
            splits = model_data[exp_name]

            src_frames = [splits[s] for s in thresh_splits if splits[s] is not None]
            if not src_frames:
                results[ci][exp_name] = np.nan
                continue
            src_df = pd.concat(src_frames)

            if isinstance(test_split, list):
                test_frames = [splits[s] for s in test_split if splits[s] is not None]
            else:
                test_frames = [splits[test_split]] if splits[test_split] is not None else []
            if not test_frames:
                results[ci][exp_name] = np.nan
                continue
            test_df = pd.concat(test_frames)

            if test_classes_only:
                test_cls = get_test_classes(test_df)
                src_df   = restrict_to_classes(src_df, test_cls)

            thresholds = compute_thresholds(src_df)
            metrics    = evaluate(test_df, thresholds,
                                  test_classes_only=test_classes_only)
            results[ci][exp_name] = metrics[metric]
            n = metrics["n_classes"]
            v = metrics[metric]
            vs = f"{v:.3f}" if not np.isnan(v) else "nan"
            print(f"  cond {ci+1}  {exp_name:42s}  {metric}={vs}  n_classes={n}")
    return results


def draw_figure(results, metric, conditions, out_path, note=""):
    """Draw the 2x3 bar-chart figure and save it to *out_path*."""
    exp_names = [n for n, _, _ in MODELS]
    labels    = [lbl for _, lbl, _ in MODELS]
    x     = np.arange(len(MODELS))
    bar_w = 0.6
    ymax  = 115 if metric == "exact_acc" else 1.15

    metric_label = {
        "macro_f1":  "Macro F1",
        "micro_f1":  "Micro F1",
        "exact_acc": "Exact Accuracy (%)",
    }[metric]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for ci, (title, _, _) in enumerate(conditions):
        ax = axes[ci]
        vals = [results[ci].get(n, np.nan) for n in exp_names]

        for i, (v, exp_name) in enumerate(zip(vals, exp_names)):
            if np.isnan(v):
                ax.bar(i, 0, bar_w, color="#DDDDDD", edgecolor="#AAAAAA",
                       linewidth=0.8, alpha=0.6)
                ax.text(i, ymax * 0.02, "N/A", ha="center", va="bottom",
                        fontsize=8, color="#888888")
            else:
                ax.bar(i, v, bar_w,
                       color=BAR_COLOR[exp_name], edgecolor=BAR_EDGE[exp_name],
                       linewidth=0.8, zorder=3)
                ax.text(i, v + ymax * 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8, zorder=5)

        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(metric_label if ci % 3 == 0 else "", fontsize=9)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ci == 0:
            ax.annotate("LARGE TEST SETS\n(oracle thresholds)",
                        xy=(-0.22, 0.5), xycoords="axes fraction",
                        fontsize=8, fontweight="bold", color="#555555",
                        ha="center", va="center", rotation=90)
        if ci == 3:
            ax.annotate("MATCHED TEST SETS\n(transferred thresholds)",
                        xy=(-0.22, 0.5), xycoords="axes fraction",
                        fontsize=8, fontweight="bold", color="#555555",
                        ha="center", va="center", rotation=90)

    note_str = f"  |  {note}" if note else ""
    fig.suptitle(
        f"Model comparison \u2014 {metric_label}{note_str}\n"
        "Each model evaluated on all its own classes  |  "
        "per-class thresholds tuned on source split",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0.06, 0.02, 1.0, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tests", default=None,
                        help="Path to model_tests/ directory (auto-detected if omitted).")
    parser.add_argument("--out", default="results_figure.png",
                        help="Output figure filename (default: results_figure.png).")
    parser.add_argument("--metric", default="macro_f1",
                        choices=["macro_f1", "micro_f1", "exact_acc"],
                        help="Metric to plot (default: macro_f1).")
    args = parser.parse_args()

    if args.model_tests:
        model_tests = Path(args.model_tests)
    else:
        here = Path(__file__).resolve().parent
        for candidate in [here.parent / "model_tests", here / "model_tests"]:
            if candidate.is_dir():
                model_tests = candidate
                break
        else:
            print("ERROR: cannot find model_tests/. Use --model-tests.", file=sys.stderr)
            sys.exit(1)

    print(f"model_tests : {model_tests}")
    print(f"metric      : {args.metric}\n")

    # ------------------------------------------------------------------ load
    ALL_SPLITS = ("combined_doc", "combined_avianz", "matched_doc", "matched_avianz")
    model_data = {}

    for exp_name, _label, layout in MODELS:
        model_dir = model_tests / exp_name
        if not model_dir.is_dir():
            print(f"WARNING: {model_dir} not found, skipping.")
            model_data[exp_name] = {s: None for s in ALL_SPLITS}
            continue
        splits = {}
        for split in ALL_SPLITS:
            csv = find_csv(model_dir, layout, split)
            df  = load_csv(csv)
            splits[split] = df
            status = f"{len(df)} rows" if df is not None else "missing/empty"
            print(f"  {exp_name:42s}  {split:20s}  {status}")
        model_data[exp_name] = splits

    print()

    # ---------------------------------------------------------------- compute
    CONDITIONS = [
        (
            "Large-DOC test\n(large-DOC thresholds)",
            ["combined_doc"],
            "combined_doc",
        ),
        (
            "Large-AviaNZ test\n(large-AviaNZ thresholds)",
            ["combined_avianz"],
            "combined_avianz",
        ),
        (
            "Large combined test\n(combined thresholds)",
            ["combined_doc", "combined_avianz"],
            ["combined_doc", "combined_avianz"],
        ),
        (
            "Matched-DOC test\n(large-DOC thresholds)",
            ["combined_doc"],
            "matched_doc",
        ),
        (
            "Matched-AviaNZ test\n(large-AviaNZ thresholds)",
            ["combined_avianz"],
            "matched_avianz",
        ),
        (
            "Matched combined test\n(combined thresholds)",
            ["combined_doc", "combined_avianz"],
            ["matched_doc", "matched_avianz"],
        ),
    ]

    metric_label = {
        "macro_f1":  "Macro F1",
        "micro_f1":  "Micro F1",
        "exact_acc": "Exact Accuracy (%)",
    }[args.metric]

    for test_classes_only in [False, True]:
        mode_desc = (
            "thresholds restricted to test-set classes"
            if test_classes_only else
            "standard (all source classes)"
        )
        print(f"\n=== Mode: {mode_desc} ===")

        results = run_conditions(model_data, args.metric, CONDITIONS,
                                 test_classes_only=test_classes_only)
        print()

        # -------------------------------------------------------- print table
        col_w = 22
        header = f"{'Condition':<45}" + "".join(
            f"{lbl.replace(chr(10), ' '):>{col_w}}" for _, lbl, _ in MODELS
        )
        sep = "=" * len(header)
        print(metric_label)
        print(sep)
        print(header)
        print("-" * len(header))
        for ci, (title, _, _) in enumerate(CONDITIONS):
            row_label = title.replace("\n", " / ")[:44]
            vals = "".join(
                f"{results[ci][n]:>{col_w}.3f}"
                if not np.isnan(results[ci].get(n, np.nan))
                else f"{'--':>{col_w}}"
                for n, _, _ in MODELS
            )
            print(f"{row_label:<45}{vals}")
        print(sep)
        print()

        # -------------------------------------------------------------- figure
        base = Path(args.out)
        if test_classes_only:
            out_path = base.parent / (base.stem + "_test_classes" + base.suffix)
        else:
            out_path = base
        draw_figure(results, args.metric, CONDITIONS, out_path,
                    note="thresholds restricted to test-set classes"
                    if test_classes_only else "")


if __name__ == "__main__":
    main()
