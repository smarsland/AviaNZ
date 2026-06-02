#!/usr/bin/env python3
"""
Produce publication-ready summary figures comparing 8 key experiments.

The 8 models (in narrative order) are:
  1. BirdNet (pretrained)             – external reference baseline
  2. RegNet Baseline                  – our plain model on corrected DOC labels
  3. RegNet + BgSub                   – +background subtraction augmentation
  4. Kaytoo (pretrained)              – THE model to beat (trained on all DOC, noisy labels)
  5. Kaytoo (finetuned)               – Kaytoo fine-tuned on corrected labels
  6. AST finetuned N=8k               – transformer architecture for comparison
  7. RegNet Scale N=8k                – CNN scales up with noisy DOC data
  8. RegNet Scale N=7k + Finetune     – DOC scaling + fine-tune on corrected labels

Outputs two figures:
  summary_f1.png        – Macro F1 (fixed threshold | tuned thresholds)
  summary_accuracy.png  – Accuracy (overall | labelled-only)

Usage:
    python3 scripts/make_summary_figure.py [--out-dir summary_figure]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Model catalogue
# ─────────────────────────────────────────────
MODELS = [
    # (experiment_name,  source_csv,            short_label,         kind)
    # kind: "birdnet" | "kaytoo" | "ours"
    (
        "birdnet_pretrained_seed0",
        "matched_tests/analysis/all_results.csv",
        "BirdNet\n(pretrained)",
        "birdnet",
    ),
    (
        "regnet_on_doc_baseline",
        "matched_tests/analysis/all_results.csv",
        "RegNet\nBaseline",
        "ours",
    ),
    (
        "regnet_on_doc_bgsub",
        "matched_tests/analysis/all_results.csv",
        "RegNet\n+BgSub",
        "ours",
    ),
    (
        "kaytoo_pretrained_seed0",
        "matched_tests/analysis/all_results.csv",
        "Kaytoo\n(pretrained)",
        "kaytoo",
    ),
    (
        "kaytoo_finetuned_seed0",
        "matched_tests/analysis/all_results.csv",
        "Kaytoo\n(finetuned)",
        "kaytoo",
    ),
    (
        "ast_on_doc_scaling_ft_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "AST\nN=8k+FT",
        "ours",
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "RegNet\nScale N=8k",
        "ours",
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_ft_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "RegNet\nN=8k+FT",
        "ours",
    ),
]

# ─────────────────────────────────────────────
#  Colour scheme
# ─────────────────────────────────────────────
# Color encodes model type; hatch encodes dataset split (AviaNZ=solid, DOC=hatched)
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

def load_data(workspace: Path) -> pd.DataFrame:
    """Load the two all_results CSVs and return a merged frame with deduplication."""
    frames = []
    for (name, csv_rel, label, kind) in MODELS:
        csv_path = workspace / csv_rel
        df = pd.read_csv(csv_path)
        row = df[df["name"] == name]
        if row.empty:
            print(f"  WARNING: '{name}' not found in {csv_rel}")
            continue
        row = row.iloc[0].copy()
        row["short_label"] = label
        row["kind"] = kind
        frames.append(row)
    return pd.DataFrame(frames).reset_index(drop=True)


def pct(v):
    """Convert a fraction value to a percentage string."""
    if pd.isna(v):
        return "N/A"
    return f"{v * 100:.1f}" if v <= 1.0 else f"{v:.1f}"


def fmt(v, is_pct=False):
    """Format a single numeric value for table display."""
    if pd.isna(v):
        return "—"
    if is_pct:
        val = v if v > 1.0 else v * 100
        return f"{val:.1f}%"
    return f"{v:.3f}"


# ─────────────────────────────────────────────
#  Shared bar-drawing helper
# ─────────────────────────────────────────────

def _draw_panel(ax, df, col, ylabel, ymax, title, is_pct=False):
    """Draw one panel: a single bar per model for a given metric column."""
    n = len(df)
    x = np.arange(n)
    w = 0.60

    for i, row in df.iterrows():
        kind  = row["kind"]
        color = KIND_COLOR[kind]
        edge  = KIND_EDGE[kind]

        v = row[col]
        if is_pct and not pd.isna(v) and v <= 1.0:
            v *= 100

        if kind == "kaytoo":
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="#FFF3E0", alpha=0.40, zorder=0)
        elif kind == "birdnet":
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="#FDECEA", alpha=0.40, zorder=0)

        if not pd.isna(v):
            ax.bar(x[i], v, width=w, color=color, edgecolor=edge, linewidth=1.1, zorder=3)
            ax.text(x[i], v + ymax * 0.012,
                    f"{v:.2f}" if not is_pct else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color="#111111", zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["short_label"], fontsize=8.5, ha="center")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_legend(fig, df):
    """Single legend: model type by colour."""
    kind_labels = {
        "ours":    "Our models",
        "kaytoo":  "Kaytoo (reference)",
        "birdnet": "BirdNet (reference)",
    }
    patches = [
        mpatches.Patch(facecolor=KIND_COLOR[k], edgecolor=KIND_EDGE[k], label=kind_labels[k])
        for k in ["birdnet", "kaytoo", "ours"]
    ]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, fontsize=9, title_fontsize=9, frameon=True)


# ─────────────────────────────────────────────
#  Four figures: fixed/tuned × AviaNZ/DOC
# ─────────────────────────────────────────────

# Column sets: (f1_col, acc_col, acc_lab_col)
_COLS = {
    ("fixed", "avianz"): ("test1_macro_f1",    "test1_acc",          "test1_acc_labelled"),
    ("fixed", "doc"):    ("test2_macro_f1",    "test2_acc",          "test2_acc_labelled"),
    ("tuned", "avianz"): ("test1_adaptive_f1", "test1_adaptive_acc", "test1_adaptive_acc_labelled"),
    ("tuned", "doc"):    ("test2_adaptive_f1", "test2_adaptive_acc", "test2_adaptive_acc_labelled"),
}

_SPLIT_LABEL = {"avianz": "AviaNZ", "doc": "DOC"}
_THRESH_LABEL = {"fixed": "Fixed Threshold (0.5)", "tuned": "Per-class Tuned Thresholds"}
_FNAME = {
    ("fixed", "avianz"): "summary_avianz_fixed.png",
    ("fixed", "doc"):    "summary_doc_fixed.png",
    ("tuned", "avianz"): "summary_avianz_tuned.png",
    ("tuned", "doc"):    "summary_doc_tuned.png",
}


def make_split_figure(df: pd.DataFrame, out_dir: Path, threshold: str, split: str):
    """Three-panel figure for one threshold mode and one test split."""
    f1_col, acc_col, acc_lab_col = _COLS[(threshold, split)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

    _draw_panel(ax1, df, f1_col,      "Macro F1",      0.72, "Macro F1")
    _draw_panel(ax2, df, acc_col,     "Exact Accuracy (%)",  90,   "Overall Exact Accuracy",       is_pct=True)
    _draw_panel(ax3, df, acc_lab_col, "Exact Accuracy (%)",  90,   "Labelled-only Exact Accuracy", is_pct=True)

    _add_legend(fig, df)
    fig.suptitle(
        f"Bird Call Classification — {_SPLIT_LABEL[split]} test split"
        f" — {_THRESH_LABEL[threshold]}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = out_dir / _FNAME[(threshold, split)]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.relative_to(out_dir.parent.parent)}")


# ─────────────────────────────────────────────
#  Text table
# ─────────────────────────────────────────────

def make_table(df: pd.DataFrame, out_dir: Path):
    """Write a Markdown table with all key metrics."""
    lines = []
    lines.append("# Bird Classification — Key Experiment Summary\n")
    lines.append(
        "Eight experiments selected to tell the story of training data quality vs quantity,\n"
        "augmentation, and cross-dataset generalisation.\n\n"
        "- **AviaNZ** = Waitākere Ranges data (reliable labels, ~24 species)\n"
        "- **DOC** = Department of Conservation data (noisy labels, ~130 species, 12 tested)\n"
        "- **Kaytoo** = reference model trained on all DOC data (including noisy labels)\n"
        "- **BirdNet** = external pretrained model (Google, not all NZ species)\n\n"
        "> threshold 0.5 = fixed operating point; tuned = per-class thresholds optimised on each split\n"
    )

    header_fixed = (
        "| Model | AviaNZ F1 | DOC F1 | AviaNZ Acc | AviaNZ Acc (lab) | DOC Acc | DOC Acc (lab) |"
    )
    sep_fixed = "| --- | --- | --- | --- | --- | --- | --- |"
    header_tuned = (
        "| Model | AviaNZ F1† | DOC F1† | AviaNZ Acc† | AviaNZ Acc† (lab) | DOC Acc† | DOC Acc† (lab) |"
    )

    def _row_fixed(r):
        a1 = fmt(r["test1_macro_f1"])
        a2 = fmt(r["test2_macro_f1"])
        a3 = fmt(r["test1_acc"], is_pct=True)
        a4 = fmt(r["test1_acc_labelled"], is_pct=True)
        a5 = fmt(r["test2_acc"], is_pct=True)
        a6 = fmt(r["test2_acc_labelled"], is_pct=True)
        return f"| {r['short_label'].replace(chr(10), ' ')} | {a1} | {a2} | {a3} | {a4} | {a5} | {a6} |"

    def _row_tuned(r):
        a1 = fmt(r["test1_adaptive_f1"])
        a2 = fmt(r["test2_adaptive_f1"])
        a3 = fmt(r["test1_adaptive_acc"], is_pct=True)
        a4 = fmt(r["test1_adaptive_acc_labelled"], is_pct=True)
        a5 = fmt(r["test2_adaptive_acc"], is_pct=True)
        a6 = fmt(r["test2_adaptive_acc_labelled"], is_pct=True)
        return f"| {r['short_label'].replace(chr(10), ' ')} | {a1} | {a2} | {a3} | {a4} | {a5} | {a6} |"

    lines.append("## Threshold = 0.5 (fixed)\n")
    lines.append(header_fixed)
    lines.append(sep_fixed)
    for _, r in df.iterrows():
        lines.append(_row_fixed(r))

    lines.append("\n## Per-class tuned thresholds (†)\n")
    lines.append(header_tuned)
    lines.append(sep_fixed)
    for _, r in df.iterrows():
        lines.append(_row_tuned(r))

    lines.append("\n---\n")
    lines.append("*F1 = macro-F1 over species present in the test set.  "
                 "Acc = exact-match accuracy (all files).  "
                 "Acc (lab) = accuracy on labelled files only.*\n")

    out_path = out_dir / "summary_table.md"
    out_path.write_text("\n".join(lines))
    print(f"✓ {out_path.relative_to(out_dir.parent.parent)}")


# ─────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────

def make_csv(df: pd.DataFrame, out_dir: Path):
    """Export a tidy CSV with the metrics for all 8 models."""
    keep = [
        "name", "short_label", "kind", "category",
        "test1_macro_f1", "test1_adaptive_f1", "test1_cross_f1",
        "test1_acc", "test1_acc_labelled",
        "test1_adaptive_acc", "test1_adaptive_acc_labelled",
        "test2_macro_f1", "test2_adaptive_f1", "test2_cross_f1",
        "test2_acc", "test2_acc_labelled",
        "test2_adaptive_acc", "test2_adaptive_acc_labelled",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out.columns = [c.replace("test1_", "avianz_").replace("test2_", "doc_")
                   for c in out.columns]
    out_path = out_dir / "summary_metrics.csv"
    out.to_csv(out_path, index=False)
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
    for threshold in ("fixed", "tuned"):
        for split in ("avianz", "doc"):
            make_split_figure(df, out_dir, threshold, split)
    make_table(df, out_dir)
    make_csv(df, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
