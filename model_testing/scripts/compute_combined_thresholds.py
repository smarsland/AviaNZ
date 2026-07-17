#!/usr/bin/env python3
"""
Compute per-class thresholds optimised on the *combined* AviaNZ + DOC test sets.

For each class that appears in either test set, we find the threshold that
maximises F1 over all samples from both datasets together.  Classes absent
from both sets get threshold = 1.0 (never fire).

Usage:
    python3 scripts/compute_combined_thresholds.py \
        model_testing/regnet_combined_bgsubtract_seed0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_combined_thresholds(model_dir: Path) -> pd.DataFrame:
    # Prefer predictions_val.csv (covers all training classes with ground truth)
    # over the matched test-split CSVs (only 9 classes have ground truth there).
    val_file = model_dir / "predictions_val.csv"
    if val_file.exists():
        pred_files = [val_file]
        print(f"Using validation predictions (all classes): {val_file.name}")
    else:
        pred_files = sorted(model_dir.glob("predictions_*.csv"))
        print(f"No predictions_val.csv found — using test split CSVs ({len(pred_files)} files).")
        print(f"  Note: only the 9 matched test species will be tuned; the rest")
        print(f"  default to 0.5.  Re-run training/eval to generate predictions_val.csv.")
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.csv found in {model_dir}")

    # ── 1. Load all prediction CSVs ──────────────────────────────────────────
    all_probs = []   # list of DataFrames (one row = one sample, columns = class names)
    all_trues = []   # list of DataFrames (one row = one sample, columns = class names)

    for path in pred_files:
        df = pd.read_csv(path, index_col="filename")
        pred_cols  = [c for c in df.columns if not c.startswith("true_")]
        true_cols  = [c for c in df.columns if c.startswith("true_")]

        probs_df = df[pred_cols].copy()
        trues_df = df[true_cols].copy()
        # Strip the "true_" prefix so columns can be aligned with pred columns
        trues_df.columns = [c[len("true_"):] for c in trues_df.columns]

        all_probs.append(probs_df)
        all_trues.append(trues_df)

    # ── 2. Collect the union of all class names ───────────────────────────────
    all_class_names = []
    seen = set()
    for probs_df in all_probs:
        for c in probs_df.columns:
            if c not in seen:
                all_class_names.append(c)
                seen.add(c)

    n_classes = len(all_class_names)
    class_idx = {c: i for i, c in enumerate(all_class_names)}

    # ── 3. Build combined probability / ground-truth matrices ─────────────────
    total_samples = sum(len(p) for p in all_probs)
    probs_all = np.zeros((total_samples, n_classes), dtype=np.float32)
    trues_all = np.full((total_samples, n_classes), np.nan, dtype=np.float32)

    row = 0
    for probs_df, trues_df in zip(all_probs, all_trues):
        n = len(probs_df)
        for c, col_probs in probs_df.items():
            ci = class_idx[c]
            probs_all[row:row + n, ci] = col_probs.values.astype(np.float32)
        for c, col_trues in trues_df.items():
            if c in class_idx:
                ci = class_idx[c]
                trues_all[row:row + n, ci] = col_trues.values.astype(np.float32)
        row += n

    # ── 4. Per-class threshold optimisation (F1 over labelled classes only) ───
    # A class is "present" if at least one sample has a ground-truth label (0 or 1)
    # i.e. the class appeared in at least one test set.
    candidates = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    # Default 0.5 for classes not covered by any test set.
    # (1.0 was appropriate for evaluation to avoid false metrics, but silences
    #  real predictions in deployment — 0.5 is a neutral operating point.)
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)

    for ci, class_name in enumerate(all_class_names):
        col_true = trues_all[:, ci]
        # Use only samples where this class has a ground-truth annotation
        mask = ~np.isnan(col_true)
        if not mask.any():
            continue  # no ground truth → keep default 0.5

        tc = col_true[mask].astype(np.int32)
        pc = probs_all[mask, ci]

        if tc.sum() == 0:
            # Class is always negative in labelled samples → keep default 0.5
            continue

        # Vectorised F1 over all threshold candidates
        preds_all = (pc[np.newaxis, :] >= candidates[:, np.newaxis]).astype(np.int32)
        pos_mask  = preds_all.sum(axis=1) > 0
        if not pos_mask.any():
            continue

        tp    = (preds_all * tc[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fp    = (preds_all * (1 - tc)[np.newaxis, :]).sum(axis=1).astype(np.float32)
        fn    = ((1 - preds_all) * tc[np.newaxis, :]).sum(axis=1).astype(np.float32)
        denom = 2 * tp + fp + fn
        f1s   = np.where(denom > 0, 2 * tp / denom, 0.0)
        f1s[~pos_mask] = -1.0

        thresholds[ci] = candidates[np.argmax(f1s)]
        best_f1 = f1s[np.argmax(f1s)]
        print(f"  {class_name}: threshold={thresholds[ci]:.4f}  F1={best_f1:.3f}")

    result = pd.DataFrame({"class": all_class_names, "threshold": thresholds})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?",
                        default="model_testing/regnet_combined_bgsubtract_seed0",
                        help="Directory containing predictions_*.csv and model config")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Directory not found: {model_dir}")

    print(f"Computing combined thresholds from: {model_dir}")
    thresholds_df = compute_combined_thresholds(model_dir)

    out_path = model_dir / "thresholds_combined.csv"
    thresholds_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {len(thresholds_df)} thresholds → {out_path}")

    # Summary
    labelled = thresholds_df[thresholds_df["threshold"] < 1.0]
    tuned = thresholds_df[(thresholds_df["threshold"] != 0.5)]
    print(f"  Classes with tuned thresholds (appeared in test sets): {len(tuned)}")
    print(f"  Classes using default 0.5 threshold (no test data):    "
          f"{len(thresholds_df) - len(tuned)}")


if __name__ == "__main__":
    main()
