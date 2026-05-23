#!/usr/bin/env python3
"""
Ensemble inference: average predictions across all trained sweep models.

Reads prediction CSVs (predictions_{split}.csv) from each experiment directory,
aligns samples by row_id, averages probabilities, evaluates the ensemble, and
saves a JSON report alongside individual experiments.

Usage (zero required arguments — paths read from constants below):
    python scripts/ensemble_inference.py

Filter to one training dataset:
    python scripts/ensemble_inference.py --train-ds avianz
    python scripts/ensemble_inference.py --train-ds doc

Use only the top-N models ranked by a metric from their existing JSON reports:
    python scripts/ensemble_inference.py --top-n 10 --rank-by macro_f1

Custom output folder name:
    python scripts/ensemble_inference.py --name my_ensemble

Output:
    {TESTS_BASE}/{name}/ensemble_test_{split}_multilabel_report.json
        → Picked up by summarize_results.py as "{name}" row in the table.
    {TESTS_BASE}/{name}/predictions_{split}_ensemble.csv
        → Per-sample ensemble predictions (row_id + per-class probability).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluation_utils import EvaluationManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NORM_SUFFIXES  = ["_boxcox", "_pcen", "_log"]
_BG_SUFFIX      = "_bgmed"
_TRAIN_DATASETS = ["avianz", "doc"]
_MODEL_TYPES    = ["regnet", "ast", "cnn"]

# Default paths for the sweep experiments (override with --tests-base / --sweep-base)
_DEFAULT_TESTS_BASE = "/local/scratch/freangi/sweep_tests"
_DEFAULT_SWEEP_BASE = "/local/scratch/freangi/sweep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sweep_paths():
    """Return (TESTS_BASE, SWEEP_BASE) defaults."""
    return _DEFAULT_TESTS_BASE, _DEFAULT_SWEEP_BASE


def parse_exp_name(name: str):
    """Return (model_type, train_ds, slug) or None if not a sweep experiment."""
    name = Path(name).name
    for mt in _MODEL_TYPES:
        if name.startswith(f"{mt}_on_"):
            rest = name[len(f"{mt}_on_"):]
            for ds in _TRAIN_DATASETS:
                if rest.startswith(f"{ds}_"):
                    rest = rest[len(f"{ds}_"):]
                    if rest.endswith(_BG_SUFFIX):
                        rest = rest[:-len(_BG_SUFFIX)]
                    for ns in _NORM_SUFFIXES:
                        if rest.endswith(ns):
                            rest = rest[:-len(ns)]
                            break
                    if rest:
                        return mt, ds, rest
    return None


def get_metric_from_report(exp_dir: Path, split: str, metric: str):
    """
    Read a scalar metric from the experiment's existing JSON report.
    Used to rank experiments for --top-n without re-running inference.
    """
    reports = list(exp_dir.glob(f"*_test_{split}_multilabel_report.json"))
    if not reports:
        return None
    with open(reports[0]) as f:
        data = json.load(f)
    METRIC_MAP = {
        "macro_f1":             lambda d: d.get("macro avg", {}).get("f1-score"),
        "micro_f1":             lambda d: d.get("micro avg", {}).get("f1-score"),
        "exact_match":          lambda d: d.get("exact_match_accuracy"),
        "exact_match_labelled": lambda d: d.get("exact_match_accuracy_labelled"),
        "jaccard":              lambda d: d.get("jaccard_score"),
    }
    fn = METRIC_MAP.get(metric)
    return fn(data) if fn else None


def load_predictions_csv(exp_dir: Path, split: str):
    """
    Load predictions_{split}.csv from an experiment directory.
    Returns a dict with row_ids, probs, true_labels, categories — or None.
    """
    csv_path = exp_dir / f"predictions_{split}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Accept both 'filename' (model_trainer) and legacy 'row_id' column names
    if "filename" in df.columns:
        id_col = "filename"
    elif "row_id" in df.columns:
        id_col = "row_id"
    else:
        print(f"  [warn] {csv_path} has no row_id or filename column — skipping")
        return None
    all_cols  = df.columns.tolist()
    pred_cols = [c for c in all_cols if c not in (id_col,) and not c.startswith("y_")]
    true_cols = [f"y_{c}" for c in pred_cols if f"y_{c}" in all_cols]
    if not pred_cols:
        return None
    return {
        "row_ids":     df[id_col].tolist(),
        "probs":       df[pred_cols].to_numpy(np.float32),
        "true_labels": df[true_cols].to_numpy(np.float32) if true_cols else None,
        "categories":  pred_cols,
        "exp_name":    exp_dir.name,
    }


def align(members: list) -> tuple:
    """
    Inner-join members on row_id.
    Returns (common_ids, probs_stack[M,N,C], true_labels[N,C], categories).
    """
    categories = members[0]["categories"]
    for m in members[1:]:
        if m["categories"] != categories:
            raise ValueError(
                f"Class name mismatch:\n"
                f"  {members[0]['exp_name']}: {categories}\n"
                f"  {m['exp_name']}: {m['categories']}"
            )

    common = sorted(set.intersection(*(set(m["row_ids"]) for m in members)))
    if not common:
        raise ValueError(
            "No common row_ids across members."
        )

    total   = sum(len(m["row_ids"]) for m in members)
    dropped = total - len(common) * len(members)
    if dropped:
        print(f"  Alignment: {len(common)} common samples (dropped {dropped} non-shared)")
    else:
        print(f"  Alignment: {len(common)} samples, perfectly aligned across {len(members)} models")

    C           = len(categories)
    prob_stack  = np.zeros((len(members), len(common), C), dtype=np.float32)
    true_labels = np.zeros((len(common), C), dtype=np.float32)

    for i, m in enumerate(members):
        df = pd.DataFrame(
            {"row_id": m["row_ids"],
             **{f"p{j}": m["probs"][:, j]       for j in range(C)},
             **{f"y{j}": m["true_labels"][:, j]  for j in range(C)}},
        ).set_index("row_id")
        a = df.loc[common]
        prob_stack[i] = a[[f"p{j}" for j in range(C)]].to_numpy(np.float32)
        if i == 0:
            true_labels = a[[f"y{j}" for j in range(C)]].to_numpy(np.float32)

    return common, prob_stack, true_labels, categories


def evaluate_and_save(probs_2d, true_labels, categories, out_dir: str, name: str) -> dict:
    """Evaluate a (N,C) probability array, write reports, return report dict."""
    os.makedirs(out_dir, exist_ok=True)
    EvaluationManager(out_dir, categories, is_multilabel=True) \
        ._evaluate_multilabel(true_labels, probs_2d, name, data={})
    with open(os.path.join(out_dir, f"{name}_multilabel_report.json")) as f:
        return json.load(f)


def summary(report: dict) -> dict:
    macro = report.get("macro avg", {})
    return {
        "exact_match_all":      report.get("exact_match_accuracy"),
        "exact_match_labelled": report.get("exact_match_accuracy_labelled"),
        "exact_match_bg":       report.get("exact_match_accuracy_background"),
        "macro_f1":             macro.get("f1-score"),
        "micro_f1":             report.get("micro avg", {}).get("f1-score"),
        "jaccard":              report.get("jaccard_score"),
        "n_samples":            report.get("num_samples"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        default_tests_base, _ = _load_sweep_paths()
    except Exception:
        default_tests_base = None

    parser = argparse.ArgumentParser(
        description="Ensemble all sweep models (zero required args)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tests-base", default=default_tests_base, metavar="DIR",
                        help=f"Root of experiment results (default: {default_tests_base})")
    parser.add_argument("--splits", nargs="+", default=["avianz_split", "doc_split"],
                        metavar="SPLIT",
                        help="Test splits to ensemble (default: avianz_split doc_split)")
    parser.add_argument("--train-ds", choices=_TRAIN_DATASETS + ["both"], default="both",
                        help="Filter to models trained on a specific dataset (default: both)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Use only top-N models (ranked by --rank-by on --rank-split)")
    parser.add_argument("--rank-by", default="macro_f1",
                        choices=["macro_f1", "micro_f1", "exact_match",
                                 "exact_match_labelled", "jaccard"],
                        help="Metric for --top-n ranking (default: macro_f1)")
    parser.add_argument("--rank-split", default="avianz_split",
                        help="Split whose JSON reports are used for --top-n ranking")
    parser.add_argument("--name", default="ensemble_all",
                        help="Output subfolder name under TESTS_BASE (default: ensemble_all)")

    args = parser.parse_args()

    if not args.tests_base:
        parser.error("--tests-base is required (or set TESTS_BASE in run_sweep.py)")

    tests_base = Path(args.tests_base)
    out_dir    = tests_base / args.name

    # ── Discover sweep experiments ────────────────────────────────────────────
    all_exp = sorted(
        p for p in tests_base.iterdir()
        if p.is_dir() and parse_exp_name(p.name) is not None
    )
    print(f"Found {len(all_exp)} sweep experiments under {tests_base}")

    if args.train_ds != "both":
        all_exp = [p for p in all_exp if parse_exp_name(p.name)[1] == args.train_ds]
        print(f"After --train-ds {args.train_ds}: {len(all_exp)} experiments")

    if not all_exp:
        print("No experiments found. Check --tests-base and --train-ds.")
        sys.exit(1)

    # ── Optional: rank and keep top-N ─────────────────────────────────────────
    if args.top_n:
        scored = []
        for p in all_exp:
            v = get_metric_from_report(p, args.rank_split, args.rank_by)
            scored.append((v if v is not None else -1.0, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"\nTop {args.top_n} by {args.rank_by} on {args.rank_split}:")
        for v, p in scored[:args.top_n]:
            tag = f"{v:.4f}" if v >= 0 else "N/A "
            print(f"  {tag}  {p.name}")
        all_exp = [p for _, p in scored[:args.top_n]]

    print(f"\nEnsembling {len(all_exp)} models → {out_dir}")
    print(f"Splits: {args.splits}")

    # ── Per-split ensemble ────────────────────────────────────────────────────
    for split in args.splits:
        print(f"\n{'='*60}\n  Split: {split}\n{'='*60}")

        # Load prediction CSVs
        members, missing = [], []
        for exp_dir in all_exp:
            m = load_predictions_csv(exp_dir, split)
            if m is None:
                missing.append(exp_dir.name)
            else:
                members.append(m)

        if missing:
            print(f"\n  WARNING: {len(missing)} experiments have no predictions_{split}.csv")
            for nm in missing[:5]:
                print(f"    {nm}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
            print(f"  Re-train missing experiments to generate prediction CSVs.")

        if len(members) < 2:
            print(f"  Need ≥2 members with CSVs for split '{split}', got {len(members)}. Skipping.")
            continue

        print(f"  Loaded {len(members)} prediction CSVs")

        # Align by row_id
        try:
            common_ids, prob_stack, true_labels, categories = align(members)
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        # Average probabilities (mean_probs — CSVs store probs, not logits)
        ens_probs = prob_stack.mean(axis=0)   # (N, C)

        # Evaluate — name pattern must match *_test_{split}_multilabel_report.json
        # so that summarize_results.py picks it up as "ensemble_all" row.
        report_name = f"ensemble_test_{split}"
        report = evaluate_and_save(ens_probs, true_labels, categories,
                                   str(out_dir), report_name)

        # Save per-sample ensemble predictions CSV
        ens_csv = out_dir / f"predictions_{split}_ensemble.csv"
        pd.DataFrame(
            {"row_id": common_ids,
             **{c: ens_probs[:, i] for i, c in enumerate(categories)}},
        ).to_csv(ens_csv, index=False)
        print(f"  Saved ensemble predictions → {ens_csv.name}")

        # Print summary
        s = summary(report)
        print(f"\n  Ensemble ({len(members)} models, {s['n_samples']} samples):")
        print(f"    Exact match (all):      {s['exact_match_all']:.4f}")
        print(f"    Exact match (labelled): {s['exact_match_labelled']:.4f}")
        print(f"    Exact match (bg):       {s['exact_match_bg']:.4f}")
        print(f"    Macro F1:               {s['macro_f1']:.4f}")
        print(f"    Micro F1:               {s['micro_f1']:.4f}")
        print(f"    Jaccard:                {s['jaccard']:.4f}")

        # Save metadata
        meta = {
            "n_members":  len(members),
            "n_samples":  len(common_ids),
            "train_ds_filter": args.train_ds,
            "top_n":      args.top_n,
            "rank_by":    args.rank_by,
            "categories": categories,
            "members":    [m["exp_name"] for m in members],
        }
        with open(out_dir / f"meta_{split}.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"View in summary:  python scripts/summarize_results.py {tests_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
