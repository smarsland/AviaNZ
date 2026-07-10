#!/usr/bin/env python3
"""
Generate side-by-side confusion matrices (DOC self-tuned | AviaNZ self-tuned)
for every model that appears in the summary figure.

Multi-label confusion matrix definition used here:
  M[i, j] = fraction of segments where true species i is present
             that also have species j predicted (using self-tuned thresholds).

  Diagonal  → per-class recall.
  Off-diag  → co-prediction rate (false positives on class j when class i is present,
               plus legitimate multi-label co-occurrences).

Rows with all-NaN indicate a class absent from the ground truth in that split.

Output (per model):
  <out_dir>/<experiment_name>_confusion.png

Usage:
    python3 scripts/make_confusion_figure.py [--workspace .] [--out-dir summary_figure/confusion_matrices]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Model catalogue  (mirrors make_summary_figure.py MODELS list)
#  (experiment_name,  parent_test_dir,  short_label,                   kind,      pred_remap)
#  pred_remap: dict of prediction-column renames to apply before aligning to true_ columns.
#              Needed when the model was trained on a vocabulary that uses different label
#              names than the test-set ground-truth columns (e.g. combined models where
#              "tui" and "bellbird" must be merged into "tui/bellbird").
# ─────────────────────────────────────────────

# Standard remap for combined-dataset models
_COMBINED_REMAP = {
    "tui":            "tui/bellbird",
    "bellbird":       "tui/bellbird",
    "new zealand kaka": "kaka",
}

_MODELS = [
    (
        "birdnet_pretrained_seed0",
        "matched_tests",
        "BirdNet (pretrained)",
        "birdnet",
        None,
    ),
    (
        "regnet_on_doc_baseline",
        "matched_tests",
        "RegNet Baseline",
        "ours",
        None,
    ),
    (
        "regnet_on_doc_bgsub",
        "matched_tests",
        "RegNet +BgSub",
        "ours",
        None,
    ),
    (
        "kaytoo_pretrained_seed0",
        "matched_tests",
        "Kaytoo (pretrained)",
        "kaytoo",
        None,  # eBird codes; handled via name_map at load time
    ),
    (
        "kaytoo_finetuned_seed0",
        "matched_tests",
        "Kaytoo (finetuned)",
        "kaytoo",
        None,
    ),
    (
        "ast_on_doc_scaling_N8000_seed0",
        "scaling_tests",
        "AST +BgSub  N=8k",
        "ours",
        None,
    ),
    (
        "ast_on_doc_scaling_ft_N8000_seed0",
        "scaling_tests",
        "AST +BgSub  N=8k (finetuned)",
        "ours",
        None,
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_N8000_seed0",
        "scaling_tests",
        "RegNet +BgSub  N=8k",
        "ours",
        None,
    ),
    (
        "regnet_on_doc_scaling_kbird2_bgsubtract_ft_N8000_seed0",
        "scaling_tests",
        "RegNet +BgSub  N=8k (finetuned)",
        "ours",
        None,
    ),
    (
        "ast_full_doc_bgsubtract_seed0",
        "full_doc_tests",
        "AST +BgSub  Full DOC",
        "ours",
        None,
    ),
    (
        "regnet_full_doc_bgsubtract_seed0",
        "full_doc_tests",
        "RegNet +BgSub  Full DOC",
        "ours",
        None,
    ),
    (
        "ast_combined_bgsubtract_seed0",
        "combined_tests",
        "AST +BgSub  Combined",
        "ours",
        _COMBINED_REMAP,
    ),
    (
        "ast_combined_bgsubtract_ft_seed0",
        "combined_tests",
        "AST +BgSub  Combined (finetuned)",
        "ours",
        _COMBINED_REMAP,
    ),
    (
        "regnet_combined_bgsubtract_seed0",
        "combined_tests",
        "RegNet +BgSub  Combined",
        "ours",
        _COMBINED_REMAP,
    ),
    (
        "regnet_combined_bgsubtract_ft_seed0",
        "combined_tests",
        "RegNet +BgSub  Combined (finetuned)",
        "ours",
        _COMBINED_REMAP,
    ),
    (
        "regnet_combined_nobgsub_seed0",
        "combined_tests",
        "RegNet  Combined (no BgSub)",
        "ours",
        _COMBINED_REMAP,
    ),
]

# ─────────────────────────────────────────────
#  Visual settings
# ─────────────────────────────────────────────
KIND_TITLE_COLOR = {
    "ours":    "#1A5276",
    "kaytoo":  "#7D3010",
    "birdnet": "#7B241C",
}
KIND_SPINE_COLOR = {
    "ours":    "#2B7BB9",
    "kaytoo":  "#E07A3A",
    "birdnet": "#C44E52",
}

DOC_CMAP   = "Blues"
AVI_CMAP   = "Greens"


# ─────────────────────────────────────────────
#  eBird → common-name map (for Kaytoo)
# ─────────────────────────────────────────────

_DISPLAY_NORMALIZE = {
    "new zealand kaka":    "kaka",
    "new zealand fantail": "fantail",
    "common chaffinch":    "chaffinch",
    "common blackbird":    "blackbird",
    "grey warbler":        "grey warbler",
    "morepork":            "morepork",
    "new zealand bellbird": "bellbird",
    "tui":                 "tui",
    "silvereye":           "silvereye",
    "tomtit":              "tomtit",
}


def load_ebird_name_map(workspace: Path) -> dict:
    """
    Build a dict: lower-case eBird code → normalised common name.
    Used to convert Kaytoo's eBird-code column labels to human-readable names.
    """
    map_path = workspace / "data" / "DOC_bird_naming_map.csv"
    if not map_path.exists():
        return {}
    df = pd.read_csv(map_path)
    result = {}
    for _, row in df.iterrows():
        ebird  = str(row["eBird"]).strip().lower()
        common = str(row["CommonName"]).strip().lower()
        # Apply display normalization (e.g. "new zealand kaka" → "kaka")
        common = _DISPLAY_NORMALIZE.get(common, common)
        result[ebird] = common
    return result


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────
def load_split(exp_dir: Path, split: str,
               pred_remap: dict = None,
               name_map: dict = None):
    """
    Load predictions and self-tuned thresholds for *split* ('doc_split' or 'avianz_split').

    Parameters
    ----------
    pred_remap : dict, optional
        Map of prediction-column name → merged column name.  Used for combined-dataset
        models where 'tui' and 'bellbird' must be merged into 'tui/bellbird' before
        aligning to the true_ ground-truth columns.
    name_map : dict, optional
        Map of raw class name → display name.  Used to convert eBird codes (Kaytoo)
        to human-readable common names.

    Returns
    -------
    (y_true [N×C int], y_pred [N×C int], class_names [C str])
    restricted to classes that actually appear in the ground truth, or None if files missing.
    """
    pred_path   = exp_dir / f"predictions_{split}.csv"
    thresh_path = exp_dir / f"thresholds_{split}.csv"
    if not pred_path.exists() or not thresh_path.exists():
        return None

    df        = pd.read_csv(pred_path, index_col="filename")
    thresh_df = pd.read_csv(thresh_path)

    class_cols   = [c for c in df.columns if not c.startswith("true_")]
    true_cols    = [c for c in df.columns if c.startswith("true_")]
    true_classes = [c[len("true_"):] for c in true_cols]

    # Apply prediction-column remap (e.g. merge 'tui'+'bellbird' → 'tui/bellbird')
    if pred_remap:
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for col in class_cols:
            groups[pred_remap.get(col, col)].append(col)
        new_pred_df = {}
        for target, sources in groups.items():
            new_pred_df[target] = df[sources].max(axis=1)
        df_pred = pd.DataFrame(new_pred_df, index=df.index)
        class_cols = list(df_pred.columns)
        probs = df_pred.values.astype(np.float32)
    else:
        probs = df[class_cols].values.astype(np.float32)

    y_true = df[true_cols].values.astype(np.int32)

    thresh_map = dict(zip(thresh_df["class"], thresh_df["threshold"].astype(float)))
    pred_idx   = {c: i for i, c in enumerate(class_cols)}

    # Align probabilities to the true-class order; absent → 0 probability
    probs_aligned = np.zeros((len(df), len(true_classes)), dtype=np.float32)
    thresholds    = np.full(len(true_classes), 1.0, dtype=np.float32)
    for j, tc in enumerate(true_classes):
        if tc in pred_idx:
            probs_aligned[:, j] = probs[:, pred_idx[tc]]
        if tc in thresh_map:
            thresholds[j] = thresh_map[tc]

    y_pred = (probs_aligned >= thresholds[np.newaxis, :]).astype(np.int32)

    # ── Restrict to classes that actually appear in ground truth ──────────────
    # Non-present classes (absent from this test split) have all-zero true_ columns
    # and produce NaN rows in the confusion matrix.  Dropping them keeps the
    # matrix compact and focused on the evaluated species only.
    present_mask = y_true.sum(axis=0) > 0
    y_true  = y_true[:, present_mask]
    y_pred  = y_pred[:, present_mask]
    present_classes = [tc for tc, p in zip(true_classes, present_mask) if p]

    # Map raw class names to display names (e.g. eBird codes → common names)
    if name_map:
        present_classes = [name_map.get(c, c) for c in present_classes]

    return y_true, y_pred, present_classes


# ─────────────────────────────────────────────
#  Matrix construction
# ─────────────────────────────────────────────
def build_coprediction_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Build row-normalised co-prediction matrix M (C × C float):

      M[i, j] = fraction of segments where true_i==1 that also have pred_j==1

    Row i is NaN when species i never appears in ground truth.
    Diagonal M[i,i] equals recall for class i.
    """
    n = y_true.shape[1]
    M = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        mask = y_true[:, i] == 1
        if mask.any():
            M[i, :] = y_pred[mask].mean(axis=0)
    return M


# ─────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────
def _draw_cm(ax, cm: np.ndarray, classes: list[str], title: str,
             cmap: str, kind: str, show_ylabel: bool = True):
    """Draw a single confusion matrix on *ax*."""
    n = len(classes)

    # Mask NaN rows for imshow (plot as grey)
    display = np.where(np.isnan(cm), -0.05, cm)
    im = ax.imshow(display, aspect="equal", cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation="nearest")

    # Grey out NaN rows
    for i in range(n):
        if np.all(np.isnan(cm[i, :])):
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1,
                                       color="#CCCCCC", zorder=2))

    # Cell annotations
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > 0.60 else "black"
            weight    = "bold"  if i == j     else "normal"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6.0, color=txt_color, fontweight=weight, zorder=6)

    # Highlight diagonal cells with a thicker border
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   linewidth=1.5, edgecolor="#222222",
                                   facecolor="none", zorder=5))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(classes if show_ylabel else [""] * n, fontsize=7)
    if show_ylabel:
        ax.set_ylabel("True species", fontsize=8)
    ax.set_xlabel("Predicted species", fontsize=8)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=7,
                 color=KIND_TITLE_COLOR.get(kind, "#000000"))

    spine_c = KIND_SPINE_COLOR.get(kind, "#333333")
    for sp in ax.spines.values():
        sp.set_edgecolor(spine_c)
        sp.set_linewidth(1.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction rate", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _recall_row(cm: np.ndarray) -> list[float]:
    """Return the diagonal of cm as a list (recall per class), NaN for absent classes."""
    if cm is None:
        return []
    return [cm[i, i] for i in range(cm.shape[0])]


# ─────────────────────────────────────────────
#  Per-model figure  (DOC | AviaNZ side by side)
# ─────────────────────────────────────────────
def make_model_figure(exp_dir: Path, name: str, label: str, kind: str,
                      out_path: Path,
                      pred_remap: dict = None,
                      name_map: dict = None):
    """Produce a figure with two confusion matrices side by side for one model."""
    doc_data = load_split(exp_dir, "doc_split",   pred_remap=pred_remap, name_map=name_map)
    avi_data = load_split(exp_dir, "avianz_split", pred_remap=pred_remap, name_map=name_map)

    if doc_data is None and avi_data is None:
        print(f"  WARNING: no prediction CSVs found for {name} — skipping.")
        return False

    classes = (doc_data or avi_data)[2]
    cm_doc  = build_coprediction_matrix(*doc_data[:2]) if doc_data else None
    cm_avi  = build_coprediction_matrix(*avi_data[:2]) if avi_data else None

    n_panels = (cm_doc is not None) + (cm_avi is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels + 0.5, 6.2))
    if n_panels == 1:
        axes = [axes]

    ax_iter = iter(axes)
    if cm_doc is not None:
        _draw_cm(next(ax_iter), cm_doc, classes,
                 "DOC  (thresholds tuned on DOC)", DOC_CMAP, kind, show_ylabel=True)
    if cm_avi is not None:
        _draw_cm(next(ax_iter), cm_avi, classes,
                 "AviaNZ  (thresholds tuned on AviaNZ)", AVI_CMAP, kind,
                 show_ylabel=(n_panels == 1))

    kind_labels = {
        "ours":    "Our model",
        "kaytoo":  "Kaytoo (reference)",
        "birdnet": "BirdNet (reference)",
    }
    fig.suptitle(f"{label}  —  {kind_labels.get(kind, kind)}",
                 fontsize=11, fontweight="bold",
                 color=KIND_TITLE_COLOR.get(kind, "#000000"),
                 y=1.01)

    note = ("Rows = true species.  Diagonal = recall.  "
            "Off-diagonal = fraction of true-class segments where that column species is also predicted.")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7,
             color="#555555", style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ─────────────────────────────────────────────
#  Combined overview figure  (all models, compact)
# ─────────────────────────────────────────────
def make_overview_figure(results: list[dict], out_path: Path):
    """
    Produce a tall figure with all models, DOC (left) and AviaNZ (right) CMs side by side.
    Arranged as rows of models, 2 CM panels per row.
    """
    valid = [r for r in results if r["cm_doc"] is not None or r["cm_avi"] is not None]
    n_models = len(valid)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 2,
                             figsize=(14, 6.5 * n_models),
                             squeeze=False)

    for row_idx, r in enumerate(valid):
        ax_doc = axes[row_idx, 0]
        ax_avi = axes[row_idx, 1]
        classes = r["classes"]
        kind    = r["kind"]
        label   = r["label"]

        if r["cm_doc"] is not None:
            _draw_cm(ax_doc, r["cm_doc"], classes,
                     f"{label}\nDOC (self-tuned)", DOC_CMAP, kind, show_ylabel=True)
        else:
            ax_doc.set_visible(False)

        if r["cm_avi"] is not None:
            _draw_cm(ax_avi, r["cm_avi"], classes,
                     f"{label}\nAviaNZ (self-tuned)", AVI_CMAP, kind, show_ylabel=False)
        else:
            ax_avi.set_visible(False)

    fig.suptitle("Confusion matrices — all models  (self-tuned thresholds)",
                 fontsize=14, fontweight="bold", y=1.002)
    note = ("Each cell: fraction of segments where the row species is truly present "
            "that also have the column species predicted.\n"
            "Diagonal = recall.  Off-diagonal = co-prediction rate.")
    fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=8,
             color="#555555", style="italic")

    fig.tight_layout(h_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path.name}  (combined overview, {n_models} models)")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side DOC / AviaNZ confusion matrices for all summary-figure models."
    )
    parser.add_argument("--workspace", default=".",
                        help="Root of the AviaNZ workspace (default: current directory)")
    parser.add_argument("--out-dir", default="summary_figure/confusion_matrices",
                        help="Output directory (relative to workspace or absolute)")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the combined all-models overview figure")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_dir   = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = workspace / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output → {out_dir.relative_to(workspace)}/\n")

    ebird_name_map = load_ebird_name_map(workspace)

    results = []
    for name, test_dir, label, kind, pred_remap in _MODELS:
        exp_dir  = workspace / test_dir / name
        out_path = out_dir / f"{name}_confusion.png"

        # Kaytoo uses eBird codes as column labels; map to common names
        name_map = ebird_name_map if kind == "kaytoo" else None

        print(f"  {label} …")
        doc_data = load_split(exp_dir, "doc_split",   pred_remap=pred_remap, name_map=name_map)
        avi_data = load_split(exp_dir, "avianz_split", pred_remap=pred_remap, name_map=name_map)

        if doc_data is None and avi_data is None:
            print(f"    WARNING: no prediction CSVs found — skipping.")
            results.append({"label": label, "kind": kind,
                             "cm_doc": None, "cm_avi": None, "classes": []})
            continue

        classes = (doc_data or avi_data)[2]
        cm_doc  = build_coprediction_matrix(*doc_data[:2]) if doc_data else None
        cm_avi  = build_coprediction_matrix(*avi_data[:2]) if avi_data else None

        results.append({
            "label": label, "kind": kind,
            "cm_doc": cm_doc, "cm_avi": cm_avi, "classes": classes,
        })

        ok = make_model_figure(exp_dir, name, label, kind, out_path,
                               pred_remap=pred_remap, name_map=name_map)
        if ok:
            print(f"    ✓ {out_path.name}")

    # Combined overview
    if not args.no_overview:
        overview_path = out_dir / "all_models_confusion_overview.png"
        print(f"\nBuilding combined overview …")
        make_overview_figure(results, overview_path)

    n_ok = sum(1 for r in results if r["cm_doc"] is not None or r["cm_avi"] is not None)
    print(f"\nDone — {n_ok}/{len(_MODELS)} models rendered.")


if __name__ == "__main__":
    main()
