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
        "ast_on_doc_scaling_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "AST +BgSub\nN=8k",
        "ours",
    ),
    (
        "ast_on_doc_scaling_ft_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "AST +BgSub\nN=8k (finetuned)",
        "ours",
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "RegNet +BgSub\nN=8k",
        "ours",
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_ft_N8000_seed0",
        "scaling_tests/analysis/all_results.csv",
        "RegNet +BgSub\nN=8k (finetuned)",
        "ours",
    ),
    (
        "ast_all_species_bgsubtract_seed0",
        "all_species_tests/analysis/all_results.csv",
        "AST +BgSub\nAll Species",
        "ours",
    ),
    (
        "ast_all_species_bgsubtract_ft_seed0",
        "all_species_tests/analysis/all_results.csv",
        "AST +BgSub\nAll Species (finetuned)",
        "ours",
    ),
    (
        "regnet_all_species_bgsubtract_seed0",
        "all_species_tests/analysis/all_results.csv",
        "RegNet +BgSub\nAll Species",
        "ours",
    ),
    (
        "regnet_all_species_bgsubtract_ft_seed0",
        "all_species_tests/analysis/all_results.csv",
        "RegNet +BgSub\nAll Species (finetuned)",
        "ours",
    ),
    (
        "ast_avianz_all_species_bgsubtract_seed0",
        "avianz_all_species_tests/analysis/all_results.csv",
        "AST +BgSub\nAviaNZ All Species",
        "ours",
    ),
    (
        "ast_avianz_all_species_bgsubtract_ft_seed0",
        "avianz_all_species_tests/analysis/all_results.csv",
        "AST +BgSub\nAviaNZ All Species (finetuned)",
        "ours",
    ),
    (
        "regnet_avianz_all_species_bgsubtract_seed0",
        "avianz_all_species_tests/analysis/all_results.csv",
        "RegNet +BgSub\nAviaNZ All Species",
        "ours",
    ),
    (
        "regnet_avianz_all_species_bgsubtract_ft_seed0",
        "avianz_all_species_tests/analysis/all_results.csv",
        "RegNet +BgSub\nAviaNZ All Species (finetuned)",
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
        if not csv_path.exists():
            print(f"  WARNING: '{name}' skipped — CSV not found: {csv_rel}")
            continue
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
#  3-condition layout
#   1. DOC (self-tuned)          – DOC thresholds evaluated on DOC
#   2. AviaNZ (DOC thresholds)   – DOC thresholds evaluated on AviaNZ (cross)
#   3. AviaNZ (self-tuned)       – AviaNZ thresholds evaluated on AviaNZ
# ─────────────────────────────────────────────

_CONDITIONS_3 = [
    {
        "label": "DOC  (self-tuned)",
        "cols": {
            "macro_f1":     "test2_adaptive_f1",
            "overall_acc":  "test2_adaptive_acc",
            "labelled_acc": "test2_adaptive_acc_labelled",
        },
        "color": "#27AE60",
        "edge":  "#1E8449",
    },
    {
        "label": "AviaNZ  (DOC thresholds)",
        "cols": {
            "macro_f1":     "test1_cross_f1",
            "overall_acc":  "test1_cross_acc",
            "labelled_acc": "test1_cross_acc_labelled",
        },
        "color": "#E67E22",
        "edge":  "#873600",
    },
    {
        "label": "AviaNZ  (self-tuned)",
        "cols": {
            "macro_f1":     "test1_adaptive_f1",
            "overall_acc":  "test1_adaptive_acc",
            "labelled_acc": "test1_adaptive_acc_labelled",
        },
        "color": "#3498DB",
        "edge":  "#1A5276",
    },
]

_METRIC_META = {
    "macro_f1":     ("Macro F1",           "Macro F1 Score",               0.85, False, "summary_macro_f1.png"),
    "overall_acc":  ("Exact Accuracy (%)", "Overall Exact Accuracy",        90,   True,  "summary_overall_acc.png"),
    "labelled_acc": ("Exact Accuracy (%)", "Labelled-only Exact Accuracy",  90,   True,  "summary_labelled_acc.png"),
}


# ─────────────────────────────────────────────
#  Per-metric figure (single panel, 3 bars per model)
# ─────────────────────────────────────────────

def make_metric_figure(df: pd.DataFrame, out_dir: Path, metric: str):
    """1×3 figure: one panel per condition, shared y-axis of model names."""
    xlabel, suptitle, xmax, is_pct, fname = _METRIC_META[metric]

    n = len(df)
    bar_h = 0.6
    y = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(7, n * 0.75)),
                             sharey=True)

    for ax, cond in zip(axes, _CONDITIONS_3):
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

    # Only leftmost panel shows model labels
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["short_label"], fontsize=9)

    kind_patches = [
        mpatches.Patch(facecolor=KIND_COLOR[k], edgecolor=KIND_EDGE[k],
                       label={"ours": "Our models",
                              "kaytoo": "Kaytoo (reference)",
                              "birdnet": "BirdNet (reference)"}[k])
        for k in ["birdnet", "kaytoo", "ours"]
    ]
    fig.legend(handles=kind_patches, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=True)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = out_dir / fname
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
        "test1_adaptive_f1", "test1_cross_f1",
        "test1_acc", "test1_acc_labelled",
        "test1_adaptive_acc", "test1_adaptive_acc_labelled",
        "test1_cross_acc", "test1_cross_acc_labelled",
        "test2_adaptive_f1", "test2_cross_f1",
        "test2_acc", "test2_acc_labelled",
        "test2_adaptive_acc", "test2_adaptive_acc_labelled",
        "test2_cross_acc", "test2_cross_acc_labelled",
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
    for metric in ("macro_f1", "overall_acc", "labelled_acc"):
        make_metric_figure(df, out_dir, metric)
    make_table(df, out_dir)
    make_csv(df, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
