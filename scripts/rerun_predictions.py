#!/usr/bin/env python3
"""
Re-run every trained sweep model on its test data and save full prediction CSVs.

Loads already-trained model checkpoints — no retraining required.
By default overwrites any existing predictions_{split}.csv files so that
analyze_all_results.py --tune-thresholds and ensemble_inference.py always
have up-to-date probabilities.

Usage (zero required arguments — paths read from run_sweep.py):
    python scripts/rerun_predictions.py

Options:
    --skip-existing  Skip experiment/split pairs whose CSV already exists
    --splits S [S]   Splits to process (default: avianz_split doc_split)
    --tests-base DIR Override TESTS_BASE from run_sweep.py
    --sweep-base DIR Override SWEEP_BASE from run_sweep.py
    --batch-size N   Batch size (default: 32)
    --device X       e.g. cuda, cpu (default: auto)

Output:
    {TESTS_BASE}/{exp_name}/predictions_{split}.csv
    columns: row_id, {class1}, ..., {classN}, y_{class1}, ..., y_{classN}
             where {class} columns are predicted probabilities (sigmoid output)
             and y_{class} columns are true binary labels.

These CSVs are consumed by ensemble_inference.py and
analyze_all_results.py --tune-thresholds.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.predict import ModelPredictor
from src.core.utils import pick_free_gpu

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


def pick_free_device():
    """Pin CUDA_VISIBLE_DEVICES to the least-used GPU and return cuda:0, or cpu."""
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        try:
            chosen = pick_free_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(chosen)
        except RuntimeError as e:
            print(f"    [warn] GPU auto-select failed ({e}); using CPU")
            return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_checkpoint(exp_dir):
    """Return (ckpt_path, config_path, model_type). Prefers _best over _final."""
    p = Path(exp_dir)
    for mt in _MODEL_TYPES:
        cfg = p / f"{mt}_model_config.json"
        if cfg.exists():
            for suffix in ("_model_best.pt", "_model.pt"):
                ckpt = p / f"{mt}{suffix}"
                if ckpt.exists():
                    return str(ckpt), str(cfg), mt
    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


def run_one(exp_dir: Path, test_dir: Path, out_csv: Path,
            batch_size: int, device, force: bool) -> bool:
    """
    Run inference for one (experiment, split) pair and save a CSV.
    Returns True if a CSV now exists (written or already existed), False on error.
    """
    if out_csv.exists() and force:
        print(f"    [skip] {out_csv.name} already exists")
        return True

    try:
        ckpt, cfg, mt = find_checkpoint(exp_dir)
    except FileNotFoundError as e:
        print(f"    [skip] {e}")
        return False

    if not test_dir.is_dir():
        print(f"    [skip] test dir not found: {test_dir}")
        return False

    print(f"    {mt} checkpoint: {Path(ckpt).name}")
    print(f"    test dir:       {test_dir}")

    # If no explicit device requested, pin CUDA_VISIBLE_DEVICES to the free GPU.
    if device:
        resolved_device = torch.device(device)
    else:
        resolved_device = pick_free_device()
    print(f"    device:         {resolved_device}")

    def _make_predictor(dev):
        return ModelPredictor(
            model_path=ckpt,
            model_config=cfg,
            data_folder=str(test_dir),
            output_file=str(out_csv),
            batch_size=batch_size,
            device=dev,
        )

    def _try_run(dev):
        pred = _make_predictor(dev)
        pred.load_model()
        pred.load_data()
        return pred, pred.predict_logits_with_ids()

    try:
        predictor, (row_ids, logits, true_labels) = _try_run(resolved_device)
    except Exception as e:
        err = str(e)
        if "CUDA error" in err or "CUDA-capable" in err or "cudaError" in err:
            print(f"    [warn] CUDA error on {resolved_device}, retrying on CPU: {e}")
            try:
                predictor, (row_ids, logits, true_labels) = _try_run(torch.device("cpu"))
            except Exception as e2:
                print(f"    [ERROR] inference failed on CPU too: {e2}")
                return False
        else:
            print(f"    [ERROR] inference failed: {e}")
            return False

    # logits → probabilities via stable sigmoid
    probs = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    categories = predictor.categories

    # Build DataFrame: row_id | predicted probs | true labels
    df = pd.DataFrame({
        "row_id": row_ids,
        **{c: probs[:, i]                    for i, c in enumerate(categories)},
        **{f"y_{c}": true_labels[:, i].astype(int) for i, c in enumerate(categories)},
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"    Saved {len(df)} samples → {out_csv.name}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        default_tests_base, default_sweep_base = _load_sweep_paths()
    except Exception:
        default_tests_base = default_sweep_base = None

    parser = argparse.ArgumentParser(
        description="Re-run all sweep models and save full prediction CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tests-base", default=default_tests_base, metavar="DIR",
                        help=f"Root of experiment results (default: {default_tests_base})")
    parser.add_argument("--sweep-base", default=default_sweep_base, metavar="DIR",
                        help=f"Root of spectrogram datasets (default: {default_sweep_base})")
    parser.add_argument("--splits", nargs="+", default=["avianz_split", "doc_split"],
                        metavar="SPLIT",
                        help="Test splits to process (default: avianz_split doc_split)")
    parser.add_argument("--skip-existing", action="store_true", dest="force",
                        help="Skip experiment/split pairs whose CSV already exists (default: overwrite)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default: 32)")
    parser.add_argument("--device", default=None,
                        help="Compute device, e.g. cuda / cpu (default: auto)")

    args = parser.parse_args()

    if not args.tests_base:
        parser.error("--tests-base is required (or set TESTS_BASE in run_sweep.py)")
    if not args.sweep_base:
        parser.error("--sweep-base is required (or set SWEEP_BASE in run_sweep.py)")

    tests_base = Path(args.tests_base)
    sweep_base = Path(args.sweep_base)

    # Discover all sweep experiment directories
    exp_dirs = sorted(
        p for p in tests_base.iterdir()
        if p.is_dir() and parse_exp_name(p.name) is not None
    )
    print(f"Found {len(exp_dirs)} sweep experiments under {tests_base}")
    print(f"Splits: {args.splits}")
    print(f"Skip existing: {args.force}")
    print()

    total = written = skipped = failed = 0

    for exp_dir in exp_dirs:
        result = parse_exp_name(exp_dir.name)
        if result is None:
            continue
        _, _, slug = result

        print(f"[{exp_dir.name}]")

        for split in args.splits:
            # Sweep layout: sweep_base/{slug}/{split}/test
            # Matched layout: sweep_base/{split}/test  (no slug subdirectory)
            test_dir = sweep_base / slug / split / "test"
            if not test_dir.is_dir():
                test_dir = sweep_base / split / "test"
            out_csv  = exp_dir / f"predictions_{split}.csv"
            total   += 1

            existed_before = out_csv.exists()
            success = run_one(exp_dir, test_dir, out_csv, args.batch_size, args.device, args.force)

            if success:
                if existed_before and args.force:   # args.force == skip_existing
                    skipped += 1
                else:
                    written += 1
            else:
                failed += 1

        print()

    print("=" * 60)
    print(f"Done: {written} written, {skipped} skipped, {failed} failed  (of {total} total)")
    if failed:
        print(f"  {failed} failed — check that model checkpoints are synced from the server.")
    if skipped:
        print(f"  {skipped} skipped (--skip-existing was set).")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  python scripts/ensemble_inference.py")
    print("  python scripts/analyze_all_results.py RESULTS_DIR --tune-thresholds")


if __name__ == "__main__":
    main()
