#!/usr/bin/env python3
"""
Produce publication-ready summary figures comparing key experiments.

Models:
  1. BirdNET (pretrained)            – external reference baseline
  2. Kaytoo (pretrained)             – reference model trained on all DOC (noisy labels)
  3. RegNet +BgSub (DOC only)        – trained on DOC noisy labels only
  4. RegNet +BgSub (Combined)        – trained on DOC noisy + AviaNZ

Usage:
    python3 scripts/make_summary_figure.py [--workspace .] [--out-dir summary_figure]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Model catalogue - matches your experiments
# ─────────────────────────────────────────────
MODELS = [
    # (experiment_name, short_label, kind)
    # kind: "birdnet" | "kaytoo" | "ours"
    (
        "birdnet_pretrained_seed0",
        "BirdNET\n(pretrained)",
        "birdnet",
    ),
    (
        "kaytoo_pretrained_seed0",
        "Kaytoo\n(pretrained)",
        "kaytoo",
    ),
    (
        "regnet_on_doc_bgsub",
        "RegNet +BgSub\n(DOC noisy)",
        "ours",
    ),
    (
        "regnet_combined_bgsubtract_seed0",
        "RegNet +BgSub\n(trained on combined\nAviaNZ+DOC data)",
        "ours",
    ),
]

# ─────────────────────────────────────────────
#  Colour scheme
# ─────────────────────────────────────────────
KIND_COLOR = {
    "ours":    "#2B7BB9",   # steel-blue
    "kaytoo":  "#E07A3A",   # orange
    "birdnet": "#C44E52",   # red
}
KIND_EDGE = {
    "ours":    "#1A5276",
    "kaytoo":  "#7D3010",
    "birdnet": "#7B241C",
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def resolve_analysis_csv(workspace: Path) -> Path:
    """Find the consolidated analysis CSV, generating it from the experiment root if needed."""
    candidates = [
        workspace / "model_testing" / "model_tests" / "analysis" / "all_results.csv",
        workspace / "model_tests" / "analysis" / "all_results.csv",
        workspace / "matched_tests" / "analysis" / "all_results.csv",
        workspace / "combined_tests" / "analysis" / "all_results.csv",
        workspace / "analysis" / "all_results.csv",
    ]
    for csv_path in candidates:
        if csv_path.exists():
            return csv_path

    for results_root in [workspace / "model_tests", workspace / "matched_tests", workspace / "combined_tests"]:
        if results_root.exists():
            analysis_dir = results_root / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(workspace / "scripts" / "analyze_all_results.py"), str(results_root), "--output", str(analysis_dir)],
                cwd=workspace,
                check=True,
            )
            csv_path = analysis_dir / "all_results.csv"
            if csv_path.exists():
                return csv_path

    raise FileNotFoundError("Could not find a consolidated analysis CSV under the workspace")


def load_data(workspace: Path) -> pd.DataFrame:
    """Load the consolidated analysis CSV and return a merged frame."""
    csv_path = resolve_analysis_csv(workspace)
    df = pd.read_csv(csv_path)
    frames = []
    for (name, label, kind) in MODELS:
        row = df[df["name"] == name]
        if row.empty:
            print(f"  WARNING: '{name}' not found in {csv_path}")
            continue
        row = row.iloc[0].copy()
        row["short_label"] = label
        row["kind"] = kind
        frames.append(row)
    return pd.DataFrame(frames).reset_index(drop=True)


def fmt(v, is_pct=False):
    """Format a single numeric value for table display."""
    if pd.isna(v):
        return "—"
    if is_pct:
        val = v if v > 1.0 else v * 100
        return f"{val:.1f}%"
    return f"{v:.3f}"


# ─────────────────────────────────────────────
#  7-condition layout - organized by test data
# ─────────────────────────────────────────────

# Row 1: DOC data
_CONDITIONS_ROW1 = [
    {
        "label": "DOC matched\n(validation thresholds)",
        "cols": {
            "macro_f1":     "doc_matched_validation_threshold_macro_f1",
            "micro_f1":     "doc_matched_validation_threshold_micro_f1",
            "overall_acc":  "doc_matched_validation_threshold_acc",
            "labelled_acc": "doc_matched_validation_threshold_acc_labelled",
        },
        "color": "#3498DB",
        "edge":  "#1A5276",
    },
    {
        "label": "DOC matched\n(DOC thresholds)",
        "cols": {
            "macro_f1":     "doc_matched_doc_threshold_macro_f1",
            "micro_f1":     "doc_matched_doc_threshold_micro_f1",
            "overall_acc":  "doc_matched_doc_threshold_acc",
            "labelled_acc": "doc_matched_doc_threshold_acc_labelled",
        },
        "color": "#27AE60",
        "edge":  "#1E8449",
    },
    {
        "label": "DOC matched\n(AviaNZ thresholds)",
        "cols": {
            "macro_f1":     "doc_matched_avianz_threshold_macro_f1",
            "micro_f1":     "doc_matched_avianz_threshold_micro_f1",
            "overall_acc":  "doc_matched_avianz_threshold_acc",
            "labelled_acc": "doc_matched_avianz_threshold_acc_labelled",
        },
        "color": "#E67E22",
        "edge":  "#873600",
    },
]

# Row 2: AviaNZ data
_CONDITIONS_ROW2 = [
    {
        "label": "AviaNZ matched\n(validation thresholds)",
        "cols": {
            "macro_f1":     "avianz_matched_validation_threshold_macro_f1",
            "micro_f1":     "avianz_matched_validation_threshold_micro_f1",
            "overall_acc":  "avianz_matched_validation_threshold_acc",
            "labelled_acc": "avianz_matched_validation_threshold_acc_labelled",
        },
        "color": "#8E44AD",
        "edge":  "#5B2C6F",
    },
    {
        "label": "AviaNZ matched\n(DOC thresholds)",
        "cols": {
            "macro_f1":     "avianz_matched_doc_threshold_macro_f1",
            "micro_f1":     "avianz_matched_doc_threshold_micro_f1",
            "overall_acc":  "avianz_matched_doc_threshold_acc",
            "labelled_acc": "avianz_matched_doc_threshold_acc_labelled",
        },
        "color": "#F1C40F",
        "edge":  "#B8860B",
    },
    {
        "label": "AviaNZ matched\n(AviaNZ thresholds)",
        "cols": {
            "macro_f1":     "avianz_matched_avianz_threshold_macro_f1",
            "micro_f1":     "avianz_matched_avianz_threshold_micro_f1",
            "overall_acc":  "avianz_matched_avianz_threshold_acc",
            "labelled_acc": "avianz_matched_avianz_threshold_acc_labelled",
        },
        "color": "#A569BD",
        "edge":  "#5B2C6F",
    },
]

# Row 3: Validation data
_CONDITIONS_ROW3 = [
    {
        "label": "Validation\n(validation thresholds)",
        "cols": {
            "macro_f1":     "val_validation_threshold_macro_f1",
            "micro_f1":     "val_validation_threshold_micro_f1",
            "overall_acc":  "val_validation_threshold_acc",
            "labelled_acc": "val_validation_threshold_acc_labelled",
        },
        "color": "#16A085",
        "edge":  "#0E6655",
    },
]

# Combine all conditions in order
_CONDITIONS = _CONDITIONS_ROW1 + _CONDITIONS_ROW2 + _CONDITIONS_ROW3

_METRIC_META = {
    "macro_f1":     ("Macro F1",           "Macro F1 Score",               0.85, False, "summary_macro_f1.png"),
    "micro_f1":     ("Micro F1",           "Micro F1 Score",               0.95, False, "summary_micro_f1.png"),
    "overall_acc":  ("Exact Accuracy (%)", "Overall Exact Accuracy",        90,   True,  "summary_overall_acc.png"),
    "labelled_acc": ("Exact Accuracy (%)", "Labelled-only Exact Accuracy",  90,   True,  "summary_labelled_acc.png"),
}


# ─────────────────────────────────────────────
#  Per-metric figure
# ─────────────────────────────────────────────
def make_metric_figure(df: pd.DataFrame, out_dir: Path, metric: str):
    """Figure with 3 rows (DOC data, AviaNZ data, Validation data)."""
    xlabel, suptitle, xmax, is_pct, fname = _METRIC_META[metric]

    n = len(df)
    bar_h = 0.6
    y = np.arange(n)
    
    n_rows = 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(7, n * 0.75) * 1.5),
                             sharey=True)
    axes = axes.flatten()

    # Add row labels - positioned far left to avoid overlap
    row_labels = ["DOC Data", "AviaNZ Data", "Validation Data"]
    for row in range(3):
        ax = axes[row * 3]
        # Move label far left to avoid overlap with subplot
        ax.annotate(row_labels[row], xy=(-0.4, 0.5), xycoords="axes fraction",
                   fontsize=11, fontweight="bold", ha="center", va="center",
                   rotation=90)

    for idx, cond in enumerate(_CONDITIONS):
        ax = axes[idx]
        col = cond["cols"][metric]

        for i, (_, row) in enumerate(df.iterrows()):
            kind = row["kind"]
            shade = {"kaytoo": "#FFF3E0", "birdnet": "#FDECEA"}.get(kind)
            if shade:
                ax.axhspan(i - 0.5, i + 0.5, color=shade, alpha=0.55, zorder=0)

            v = row.get(col, float("nan"))
            if is_pct and not pd.isna(v) and v <= 1.0:
                v *= 100
            if not pd.isna(v):
                ax.barh(i, v, height=bar_h, color=KIND_COLOR[kind],
                        edgecolor=KIND_EDGE[kind], linewidth=0.8, zorder=3)
                ax.text(v + xmax * 0.015, i,
                        f"{v:.2f}" if not is_pct else f"{v:.1f}",
                        ha="left", va="center", fontsize=7.5, zorder=5)

        ax.set_title(cond["label"], fontsize=10, fontweight="bold", pad=7)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xlim(0, xmax * 1.25)
        ax.set_ylim(-0.6, n - 0.4)
        ax.invert_yaxis()
        ax.xaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Only leftmost panel in first row shows model labels
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["short_label"], fontsize=9)

    # Hide unused subplots
    axes[7].set_visible(False)
    axes[8].set_visible(False)

    kind_patches = [
        mpatches.Patch(facecolor=KIND_COLOR[k], edgecolor=KIND_EDGE[k],
                       label={"ours": "Our models",
                              "kaytoo": "Kaytoo (reference)",
                              "birdnet": "BirdNET (reference)"}[k])
        for k in ["birdnet", "kaytoo", "ours"]
    ]
    fig.legend(handles=kind_patches, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=True)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    
    # Adjust the layout to make room for row labels
    plt.tight_layout(rect=[0.08, 0.06, 1, 1])
    
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(out_dir.parent.parent)}")

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=".",
                        help="Root of the AviaNZ workspace (default: current dir)")
    parser.add_argument("--out-dir", default="summary_figure",
                        help="Output directory (default: summary_figure/)")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_dir   = workspace / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = load_data(workspace)
    print(f"  {len(df)} models loaded\n")

    print("Creating outputs…")
    for metric in ("macro_f1", "micro_f1", "overall_acc", "labelled_acc"):
        make_metric_figure(df, out_dir, metric)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()