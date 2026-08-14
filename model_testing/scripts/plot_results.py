#!/usr/bin/env python3
"""
Single script to produce 3x3 comparison figures for the 4 NZ bird models.

Nine conditions (3 rows x 3 columns):
  Row 1 - large test sets, oracle thresholds tuned on same data:
    (1) Combined-DOC test,          thresholds from combined-DOC
    (2) Combined-AviaNZ test,       thresholds from combined-AviaNZ
    (3) Combined (DOC+AviaNZ) test, thresholds from combined (DOC+AviaNZ)
  Row 2 - matched test sets, thresholds transferred from large sets:
    (4) Matched-DOC test,           thresholds from combined-DOC
    (5) Matched-AviaNZ test,        thresholds from combined-AviaNZ
    (6) Matched combined test,      thresholds from combined (DOC+AviaNZ)
  Row 3 - matched test sets, oracle thresholds tuned on matched sets:
    (7) Matched-DOC test,           thresholds from Matched-DOC
    (8) Matched-AviaNZ test,        thresholds from Matched-AviaNZ
    (9) Matched combined test,      thresholds from Matched combined

All models are evaluated against a universal class set: the union of every
GT-positive class name across all models and all splits.  Macro F1 always
uses that set as its denominator.  If a model cannot predict a class, its
contribution is F1=0.  If a class is absent from the current test split,
true labels are 0, so any prediction above threshold is an FP (F1=0 for
that class naturally).

Usage:
    python3 scripts/plot_results.py
    python3 scripts/plot_results.py --model-tests /path/to/model_tests
    python3 scripts/plot_results.py --out figure.png
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
# Kaytoo label normalisation  (eBird codes → dataset common names)
# ---------------------------------------------------------------------------

def _load_ebird_to_common(data_dir):
    """Load DOC naming map; returns {eBird_code: lowercase_common_name}."""
    csv_path = Path(data_dir) / "DOC_bird_naming_map.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return {
        str(row["eBird"]).strip(): str(row["CommonName"]).strip().lower()
        for _, row in df.iterrows()
        if pd.notna(row.get("eBird")) and pd.notna(row.get("CommonName"))
    }

# Post-mapping fixes: DOC map names that differ from dataset labels
_KAYTOO_POST_REMAP = {
    "new zealand kaka": "kaka",
    "red-billed gull":  "red billed gull",
}


def _normalize_kaytoo_df(df, ebird_to_common):
    """Rename Kaytoo eBird-code columns to dataset common names."""
    def _map(cls):
        name = ebird_to_common.get(cls, cls)
        return _KAYTOO_POST_REMAP.get(name, name)

    rename = {}
    for col in df.columns:
        cls = col[5:] if col.startswith("true_") else col
        mapped = _map(cls)
        if mapped != cls:
            rename[col] = ("true_" + mapped) if col.startswith("true_") else mapped
    if rename:
        df = df.rename(columns=rename)

    # Merge separate tui + bellbird columns → tui/bellbird
    for cls in ("tui", "bellbird"):
        if cls in df.columns:
            if "tui/bellbird" in df.columns:
                df["tui/bellbird"] = df[["tui/bellbird", cls]].max(axis=1)
            else:
                df = df.rename(columns={cls: "tui/bellbird"})
            if cls in df.columns:
                df = df.drop(columns=[cls])
        tc = "true_" + cls
        if tc in df.columns:
            if "true_tui/bellbird" in df.columns:
                df["true_tui/bellbird"] = df[["true_tui/bellbird", tc]].max(axis=1)
            else:
                df = df.rename(columns={tc: "true_tui/bellbird"})
            if tc in df.columns:
                df = df.drop(columns=[tc])

    return df


# ---------------------------------------------------------------------------
# Canonical ground-truth set
# ---------------------------------------------------------------------------

def get_canonical_gt(model_data, test_splits):
    """Return the union of GT-positive class names from all models for the given splits."""
    if isinstance(test_splits, str):
        test_splits = [test_splits]
    canonical = set()
    for exp_name, _, _ in MODELS:
        for split in test_splits:
            df = model_data[exp_name].get(split)
            if df is None:
                continue
            for tc in df.columns:
                if tc.startswith("true_") and int(df[tc].fillna(0).sum()) > 0:
                    canonical.add(tc[5:])
    return canonical


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODELS = [
    # (experiment_dir_name,              display_label,               csv_layout)
    ("birdnet_pretrained_seed0",         "BirdNET\n(pretrained)",    "subdir"),
    ("kaytoo_pretrained_seed0",          "Kaytoo\n(pretrained)",     "subdir"),
    ("regnet_on_doc_bgsub",              "RegNet+BgSub\n(DOC)",      "flat"),
    ("regnet_combined_bgsubtract_seed0", "RegNet+BgSub\n(combined)", "flat"),
    ("regnet_on_doc_bgsub_reverb",       "RegNet+BgSub+Reverb\n(DOC)","flat"),
]

# Distinct colors for each model - all clearly different
BAR_COLOR = {
    "birdnet_pretrained_seed0":         "#E74C3C",  # Red
    "kaytoo_pretrained_seed0":          "#F39C12",  # Orange
    "regnet_on_doc_bgsub":              "#2E86C1",  # Medium blue
    "regnet_combined_bgsubtract_seed0": "#1A5276",  # Dark blue
    "regnet_on_doc_bgsub_reverb":       "#85C1E9",  # Light blue
}
BAR_EDGE = {
    "birdnet_pretrained_seed0":         "#C0392B",
    "kaytoo_pretrained_seed0":          "#E67E22",
    "regnet_on_doc_bgsub":              "#1A5276",
    "regnet_combined_bgsubtract_seed0": "#0E2F44",
    "regnet_on_doc_bgsub_reverb":       "#5B8CB8",
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
        if y_prob.max() <= 0.0:
            continue  # model never predicts this class - skip threshold tuning

        preds = (y_prob[np.newaxis, :] >= candidates[:, np.newaxis]).astype(np.int32)
        tp    = ( preds *  y_true[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fp    = ( preds * (1 - y_true)[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fn    = ((1 - preds) * y_true[np.newaxis, :]).sum(axis=1).astype(np.float32)
        denom = 2 * tp + fp + fn
        f1s   = np.where(denom > 0, 2 * tp / denom, 0.0)
        thresholds[cls] = float(candidates[np.argmax(f1s)])

    return thresholds


def compute_thresholds_equal_datasets(doc_df, avianz_df, n_cands=101):
    """
    Per-class threshold optimisation giving exactly equal weight to DOC and
    AviaNZ *among the datasets that have GT positives for that class*.

    Classes present only in one dataset get that dataset's oracle threshold.
    Classes in both get the threshold maximising the average of the two F1s.
    """
    doc_pred_cols  = set(c for c in doc_df.columns   if not c.startswith("true_"))
    avianz_pred_cols = set(c for c in avianz_df.columns if not c.startswith("true_"))
    pred_cols = sorted(doc_pred_cols | avianz_pred_cols)

    candidates = np.linspace(0.0, 1.0, n_cands, dtype=np.float32)
    thresholds = {}

    for cls in pred_cols:
        tc = "true_" + cls

        # Build per-dataset frames; treat missing true_ column as all-zero GT.
        def _get(df):
            if cls not in df.columns:
                return pd.DataFrame(columns=[cls, tc])
            if tc not in df.columns:
                sub = df[[cls]].copy(); sub[tc] = 0
                return sub.dropna()
            return df[[cls, tc]].dropna()

        doc   = _get(doc_df)
        avianz = _get(avianz_df)

        doc_gt   = int(doc[tc].sum())   if not doc.empty   else 0
        avianz_gt = int(avianz[tc].sum()) if not avianz.empty else 0

        if doc_gt == 0 and avianz_gt == 0:
            continue

        doc_max   = float(doc[cls].max())   if not doc.empty   else 0.0
        avianz_max = float(avianz[cls].max()) if not avianz.empty else 0.0
        if doc_max <= 0.0 and avianz_max <= 0.0:
            continue

        # Weight only datasets that actually have GT positives for this class.
        n_with_gt = int(doc_gt > 0) + int(avianz_gt > 0)

        best_f1 = -1.0
        best_threshold = 0.0

        for threshold in candidates:
            def f1_for(df):
                if df.empty:
                    return 0.0
                y_prob = df[cls].values.astype(np.float32)
                y_true = df[tc].values.astype(np.int32)
                y_pred = (y_prob >= threshold).astype(np.int32)
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                denom = 2 * tp + fp + fn
                return 2 * tp / denom if denom > 0 else 0.0

            combined_f1 = (
                (f1_for(doc)   if doc_gt   > 0 else 0.0) +
                (f1_for(avianz) if avianz_gt > 0 else 0.0)
            ) / n_with_gt

            if combined_f1 > best_f1:
                best_f1 = combined_f1
                best_threshold = float(threshold)

        thresholds[cls] = best_threshold

    return thresholds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(test_df, thresholds, canonical_gt=None):
    """
    Apply thresholds to test_df and return macro F1, micro F1, exact accuracy.

    canonical_gt: set of class names that defines the evaluation universe for
    this test split (same across all models).  Only classes in canonical_gt
    that have a tuned threshold are in 'present'; all others contribute F1=0.
    When None, falls back to the test_df's own true_ columns.
    """
    nan_result = {"macro_f1": np.nan, "micro_f1": np.nan, "exact_acc": np.nan,
                  "n_classes": 0}

    pred_cols = [c for c in test_df.columns if not c.startswith("true_")]
    test_df = test_df[test_df[pred_cols].notna().any(axis=1)]
    if test_df.empty:
        return nan_result

    if canonical_gt is not None:
        # Restrict to canonical GT classes: consistent denominator across models.
        present = [cls for cls in pred_cols if cls in thresholds and cls in canonical_gt]
        thresh  = {cls: thresholds[cls] for cls in present}
        n_uncovered = sum(1 for cls in canonical_gt if cls not in set(present))
    else:
        present = [cls for cls in pred_cols if cls in thresholds]
        thresh  = {cls: thresholds[cls] for cls in present}
        n_uncovered = sum(
            1 for tc in test_df.columns
            if tc.startswith("true_")
            and int(test_df[tc].fillna(0).sum()) > 0
            and tc[5:] not in present
        )

    if not present and n_uncovered == 0:
        return nan_result

    if not present:
        # Model has no predictions at all for this split; all GT-positive classes
        # contribute F1=0.  Skip the matrix operations.
        return {"macro_f1": 0.0, "micro_f1": 0.0, "exact_acc": np.nan,
                "n_classes": n_uncovered}

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

    macro_f1 = float(np.mean(per_f1 + [0.0] * n_uncovered)) if (per_f1 or n_uncovered) else 0.0

    tp_all  = int(((y_pred == 1) & (y_true == 1)).sum())
    fp_all  = int(((y_pred == 1) & (y_true == 0)).sum())
    fn_all  = int(((y_pred == 0) & (y_true == 1)).sum())
    denom_m = 2 * tp_all + fp_all + fn_all
    micro_f1 = float(2 * tp_all / denom_m) if denom_m > 0 else 0.0

    exact_acc = float(np.all(y_pred == y_true, axis=1).mean() * 100)

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "exact_acc": exact_acc,
            "n_classes": len(present) + n_uncovered}


# ---------------------------------------------------------------------------
# Run conditions
# ---------------------------------------------------------------------------

def run_conditions(model_data, conditions, all_classes):
    """
    Compute all three metrics for every (condition, model) combination.

    Returns:
        results[condition_index][model_name] = {
            "macro_f1": ...,
            "micro_f1": ...,
            "exact_acc": ...,
        }
    """
    results = [{} for _ in conditions]
    print(f"  Universal class set: {len(all_classes)} classes")

    canonical_gts = {
        s: get_canonical_gt(model_data, [s])
        for s in ("combined_doc", "combined_avianz", "matched_doc", "matched_avianz")
    }

    for s, cgt in canonical_gts.items():
        print(f"  canonical GT [{s:25s}]: {len(cgt)} classes")
    print()

    for ci, (_title, thresh_splits, test_split) in enumerate(conditions):
        for exp_name, _label, _layout in MODELS:
            splits = model_data[exp_name]

            src_frames = [
                splits[s] for s in thresh_splits
                if splits[s] is not None
            ]

            if not src_frames:
                results[ci][exp_name] = {
                    "macro_f1": np.nan,
                    "micro_f1": np.nan,
                    "exact_acc": np.nan,
                }
                continue

            test_splits_list = (
                [test_split]
                if isinstance(test_split, str)
                else test_split
            )

            test_frames = [
                splits[s] for s in test_splits_list
                if splits[s] is not None
            ]

            if not test_frames:
                results[ci][exp_name] = {
                    "macro_f1": np.nan,
                    "micro_f1": np.nan,
                    "exact_acc": np.nan,
                }
                continue

            src_df = pd.concat(src_frames)
            test_df = pd.concat(test_frames)

            if isinstance(test_split, str):
                cgt = canonical_gts.get(test_split)
            else:
                cgt = set().union(
                    *(canonical_gts.get(s, set()) for s in test_split)
                )

            thresholds = compute_thresholds(src_df)
            metrics = evaluate(
                test_df,
                thresholds,
                canonical_gt=cgt
            )

            results[ci][exp_name] = metrics

            print(
                f"  cond {ci+1}  "
                f"{exp_name:42s}  "
                f"macro={metrics['macro_f1']:.3f}  "
                f"micro={metrics['micro_f1']:.3f}  "
                f"exact={metrics['exact_acc']:.3f}  "
                f"n_classes={metrics['n_classes']}"
            )

    return results


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_figure(results, metric, conditions, out_path):
    """Draw the 3x3 bar-chart figure and save it to *out_path*."""
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

    # Create figure with more space at bottom for legend
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    # Adjust spacing - tighter bottom to reduce whitespace
    plt.subplots_adjust(top=0.93, bottom=0.10, left=0.12, right=0.97, hspace=0.12, wspace=0.12)

    for ci, (title, _, _) in enumerate(conditions):
        ax = axes[ci]
        vals = [
            results[ci].get(n, {}).get(metric, np.nan)
            for n in exp_names
        ]

        for i, (v, exp_name) in enumerate(zip(vals, exp_names)):
            if np.isnan(v):
                ax.bar(i, 0, bar_w, color="#DDDDDD", edgecolor="#AAAAAA",
                       linewidth=0.8, alpha=0.6)
                ax.text(i, ymax * 0.02, "N/A", ha="center", va="bottom",
                        fontsize=8, color="#888888")
            else:
                ax.bar(i, v, bar_w,
                       color=BAR_COLOR[exp_name], edgecolor=BAR_EDGE[exp_name],
                       linewidth=1.5, zorder=3)
                ax.text(i, v + ymax * 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8, zorder=5)

        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_ylabel("")  # Remove y-axis labels
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove individual titles for rows 2 and 3
        if ci >= 3:
            ax.set_title("")

        # Row labels on the left side
        if ci % 3 == 0:
            row_labels = [
                "Large test sets",
                "Matched test sets\n(transferred thresholds)",
                "Matched test sets\n(oracle thresholds)"
            ]
            # Position row label closer to the plot
            ax.annotate(row_labels[ci // 3], 
                       xy=(-0.2, 0.5), xycoords="axes fraction",
                       fontsize=9, fontweight="bold", color="black",
                       ha="center", va="center", rotation=90)

    # Create legend with single row, placed below subplots
    from matplotlib.patches import Patch
    legend_elements = []
    for i, n in enumerate(exp_names):
        legend_elements.append(
            Patch(facecolor=BAR_COLOR[n], edgecolor=BAR_EDGE[n], 
                  label=MODELS[i][1].replace('\n', ' '))
        )
    
    # Place legend close to the subplots with minimal gap
    legend = fig.legend(handles=legend_elements, 
                       loc='lower center',
                       bbox_to_anchor=(0.5, 0.04),
                       fontsize=10,
                       frameon=True, 
                       fancybox=True, 
                       shadow=True,
                       ncol=5,  # Single row with 5 columns
                       handlelength=2.5,
                       handletextpad=0.8,
                       borderaxespad=0.5)  # Reduced padding

    # Add title at the very top
    fig.suptitle(
        f"{metric_label}",
        fontsize=14, fontweight="bold", y=0.99, x=0.5
    )
    
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
    print()

    # Load DOC naming map so Kaytoo eBird codes are normalised to common names
    here = Path(__file__).resolve().parent
    ebird_to_common = _load_ebird_to_common(here.parent / "data")
    if ebird_to_common:
        print(f"Loaded eBird→common mapping ({len(ebird_to_common)} entries)\n")
    else:
        print("WARNING: DOC_bird_naming_map.csv not found; Kaytoo labels unchanged\n")

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
            if df is not None and exp_name == "kaytoo_pretrained_seed0":
                df = _normalize_kaytoo_df(df, ebird_to_common)
            splits[split] = df
            status = f"{len(df)} rows" if df is not None else "missing/empty"
            print(f"  {exp_name:42s}  {split:20s}  {status}")
        model_data[exp_name] = splits

    print()

    # ---------------------------------------------------------------- compute
    # CONDITIONS: 3 rows x 3 columns
    # Row 1: Large test sets, oracle from same data
    # Row 2: Matched test sets, thresholds transferred from large sets
    # Row 3: Matched test sets, oracle from matched data
    CONDITIONS = [
        # Row 1
        ("DOC",      ["combined_doc"],                         "combined_doc"),
        ("AviaNZ",   ["combined_avianz"],                      "combined_avianz"),
        ("Combined", ["combined_doc", "combined_avianz"],      ["combined_doc", "combined_avianz"]),
        # Row 2
        ("DOC",      ["combined_doc"],                         "matched_doc"),
        ("AviaNZ",   ["combined_avianz"],                      "matched_avianz"),
        ("Combined", ["combined_doc", "combined_avianz"],      ["matched_doc", "matched_avianz"]),
        # Row 3
        ("DOC",      ["matched_doc"],                          "matched_doc"),
        ("AviaNZ",   ["matched_avianz"],                       "matched_avianz"),
        ("Combined", ["matched_doc", "matched_avianz"],        ["matched_doc", "matched_avianz"]),
    ]

    # Build the universal class set once from all loaded data.
    all_classes = set()
    for splits in model_data.values():
        for df in splits.values():
            if df is None:
                continue
            for tc in df.columns:
                if tc.startswith("true_") and int(df[tc].fillna(0).sum()) > 0:
                    all_classes.add(tc[5:])
    print(f"Universal class set: {len(all_classes)} classes\n")

    print("\n=== Computing all metrics ===\n")

    results = run_conditions(
        model_data,
        CONDITIONS,
        all_classes
    )

    # -------------------------------------------------------- print tables

    metrics_to_plot = [
        ("macro_f1", "Macro F1"),
        ("micro_f1", "Micro F1"),
        ("exact_acc", "Exact Accuracy (%)"),
    ]

    col_w = 22

    for metric, metric_label in metrics_to_plot:
        print(f"\n{metric_label}")
        print("=" * 150)

        header = f"{'Condition':<45}" + "".join(
            f"{lbl.replace(chr(10), ' '):>{col_w}}"
            for _, lbl, _ in MODELS
        )

        print(header)
        print("-" * len(header))

        for ci, (title, _, _) in enumerate(CONDITIONS):
            if ci == 3 or ci == 6:
                print("-" * len(header))

            row_label = title.replace("\n", " ")[:44]

            vals = "".join(
                (
                    f"{results[ci][n][metric]:>{col_w}.3f}"
                    if not np.isnan(results[ci][n][metric])
                    else f"{'--':>{col_w}}"
                )
                for n, _, _ in MODELS
            )

            print(f"{row_label:<45}{vals}")

        print("=" * 150)

        # ---------------------------------------------------------- figure

        out_path = Path(args.out)

        # If user supplied results_figure.png, make:
        # results_figure_macro_f1.png
        # results_figure_micro_f1.png
        # results_figure_exact_acc.png

        stem = out_path.stem
        suffix = out_path.suffix

        metric_out = out_path.with_name(
            f"{stem}_{metric}{suffix}"
        )

        draw_figure(
            results,
            metric,
            CONDITIONS,
            metric_out
        )


if __name__ == "__main__":
    main()