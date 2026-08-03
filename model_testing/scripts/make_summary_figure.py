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
    # (experiment_name,  source_csv,            short_label,         kind)
    # kind: "birdnet" | "kaytoo" | "ours"
    (
        "birdnet_pretrained_seed0",
        "matched_tests/analysis/all_results.csv",
        "BirdNET\n(pretrained)",
        "birdnet",
    ),
    (
        "kaytoo_pretrained_seed0",
        "matched_tests/analysis/all_results.csv",
        "Kaytoo\n(pretrained)",
        "kaytoo",
    ),
    (
        "regnet_on_doc_bgsub",
        "matched_tests/analysis/all_results.csv",
        "RegNet +BgSub\n(DOC noisy)",
        "ours",
    ),
    (
        "regnet_combined_bgsubtract_seed0",
        "combined_tests/analysis/all_results.csv",
        "RegNet +BgSub\n(Combined)",
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

def load_data(workspace: Path) -> pd.DataFrame:
    """Load the all_results CSVs and return a merged frame."""
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


def fmt(v, is_pct=False):
    """Format a single numeric value for table display."""
    if pd.isna(v):
        return "—"
    if is_pct:
        val = v if v > 1.0 else v * 100
        return f"{val:.1f}%"
    return f"{v:.3f}"


# ─────────────────────────────────────────────
#  4-condition layout
# ─────────────────────────────────────────────

_CONDITIONS = [
    {
        "label": "AviaNZ  (self-tuned)",
        "cols": {
            "macro_f1":     "test1_adaptive_f1",
            "micro_f1":     "test1_adaptive_micro_f1",
            "overall_acc":  "test1_adaptive_acc",
            "labelled_acc": "test1_adaptive_acc_labelled",
        },
        "color": "#3498DB",
        "edge":  "#1A5276",
    },
    {
        "label": "DOC  (self-tuned)",
        "cols": {
            "macro_f1":     "test2_adaptive_f1",
            "micro_f1":     "test2_adaptive_micro_f1",
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
            "micro_f1":     "test1_cross_micro_f1",
            "overall_acc":  "test1_cross_acc",
            "labelled_acc": "test1_cross_acc_labelled",
        },
        "color": "#E67E22",
        "edge":  "#873600",
    },
    {
        "label": "DOC  (AviaNZ thresholds)",
        "cols": {
            "macro_f1":     "test2_cross_f1",
            "micro_f1":     "test2_cross_micro_f1",
            "overall_acc":  "test2_cross_acc",
            "labelled_acc": "test2_cross_acc_labelled",
        },
        "color": "#8E44AD",
        "edge":  "#5B2C6F",
    },
]

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
    """Figure with one panel per condition."""
    xlabel, suptitle, xmax, is_pct, fname = _METRIC_META[metric]

    n = len(df)
    bar_h = 0.6
    y = np.arange(n)

    fig, axes = plt.subplots(1, 4, figsize=(24, max(7, n * 0.75)),
                             sharey=True)

    for ax, cond in zip(axes, _CONDITIONS):
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
                              "birdnet": "BirdNET (reference)"}[k])
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
        "Four experiments comparing external baselines against our RegNet approach.\n\n"
        "- **AviaNZ** = Waitākere Ranges data (reliable labels, ~24 species)\n"
        "- **DOC** = Department of Conservation data (**noisy** labels, 12 species used for testing)\n"
        "- **Combined** = DOC noisy + all available AviaNZ data\n\n"
        "> threshold 0.5 = fixed operating point; tuned = per-class thresholds optimised on each split\n"
    )

    header = (
        "| Model | AviaNZ F1† | DOC F1† | AviaNZ F1* | DOC F1* | "
        "AviaNZ Acc† | DOC Acc† | AviaNZ Acc* | DOC Acc* |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    def _row(r):
        # Self-tuned (†)
        a_avianz_self = fmt(r["test1_adaptive_f1"])
        a_doc_self = fmt(r["test2_adaptive_f1"])
        # Cross-tuned (*)
        a_avianz_cross = fmt(r["test1_cross_f1"])
        a_doc_cross = fmt(r["test2_cross_f1"])
        # Accuracies
        a_avianz_acc = fmt(r["test1_adaptive_acc"], is_pct=True)
        a_doc_acc = fmt(r["test2_adaptive_acc"], is_pct=True)
        a_avianz_acc_cross = fmt(r["test1_cross_acc"], is_pct=True)
        a_doc_acc_cross = fmt(r["test2_cross_acc"], is_pct=True)
        
        return (f"| {r['short_label'].replace(chr(10), ' ')} | "
                f"{a_avianz_self} | {a_doc_self} | {a_avianz_cross} | {a_doc_cross} | "
                f"{a_avianz_acc} | {a_doc_acc} | {a_avianz_acc_cross} | {a_doc_acc_cross} |")

    lines.append(header)
    lines.append(sep)
    for _, r in df.iterrows():
        lines.append(_row(r))

    lines.append("\n---\n")
    lines.append("*F1 = macro-F1 over species present in the test set.  "
                 "Acc = exact-match accuracy (all files).  "
                 "† = thresholds tuned on same split.  "
                 "* = thresholds tuned on other split.*\n")

    out_path = out_dir / "summary_table.md"
    out_path.write_text("\n".join(lines))
    print(f"✓ {out_path.relative_to(out_dir.parent.parent)}")


# ─────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────

def make_csv(df: pd.DataFrame, out_dir: Path):
    """Export a tidy CSV with the metrics."""
    keep = [
        "name", "short_label", "kind", "category",
        "test1_adaptive_f1", "test2_adaptive_f1",
        "test1_cross_f1", "test2_cross_f1",
        "test1_acc", "test2_acc",
        "test1_acc_labelled", "test2_acc_labelled",
        "test1_adaptive_acc", "test2_adaptive_acc",
        "test1_cross_acc", "test2_cross_acc",
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
    for metric in ("macro_f1", "micro_f1", "overall_acc", "labelled_acc"):
        make_metric_figure(df, out_dir, metric)
    make_table(df, out_dir)
    make_csv(df, out_dir)

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()