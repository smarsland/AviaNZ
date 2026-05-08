#!/usr/bin/env python3
"""
Master sweep: build all 18 unique spectrogram-variant datasets and run all
216 training experiments (18 × 3 norms × 2 bg options × 2 train datasets).

Efficiency:
  Normalization (Log/PCEN/Box-Cox) and bg-subtract/median-filter are applied
  at training time by train.py — not at dataset-build time.  Only the 18 unique
  (sgType × window × scale) combinations require separate dataset builds.

Dataset layout:
  SWEEP_BASE/{slug}/doc_matched/
                   /avianz_matched/
                   /doc_split/{train,test}/
                   /avianz_split/{train,test}/
                   /merged_train/

Training run layout:
  TESTS_BASE/{model}_on_{train}_{slug}_{norm}[_bgmed]/

Usage:
  python run_sweep.py                    # run everything
  python run_sweep.py --build-only       # only build datasets
  python run_sweep.py --train-only       # only run training (datasets must exist)
  python run_sweep.py --dry-run          # print what would be run, no execution
  python run_sweep.py --list             # list all 216 experiment names
"""

import os
import sys
import subprocess
import argparse

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = "/local/scratch/freangi"
DOC_RAW     = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
AVIANZ_RAW  = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
REVIEWED_CSV = "data/doc_reviewed.csv"
MAPPING      = "data/DOC_bird_naming_map.csv"

SWEEP_BASE = f"{BASE}/sweep"        # one sub-dir per raw spectrogram config
TESTS_BASE = f"{BASE}/sweep_tests"  # one sub-dir per training run

# ── Training hyperparams (matching run_experiments.sh) ─────────────────────
MODEL_TYPE  = "ast"
EPOCHS      = 100
PATIENCE    = 15
MIXUP       = 0.25
VIZ_SAMPLES = 3

# ── Parameter grid ────────────────────────────────────────────────────────────
SG_TYPES = ["Standard", "Reassigned", "Multi-tapered"]

# Windows vary for Standard and Reassigned; Multi-tapered uses its own tapers
WINDOWS = ["Hann", "Hamming", "Blackman", "BlackmanHarris"]

SG_SCALES = [
    ("Mel Frequency", "mel"),
    ("Linear",        "linear"),
]

NORMALIZATIONS = [
    ("Log",     "log"),
    ("PCEN",    "pcen"),
    ("Box-Cox", "boxcox"),
]

# (use_bg_subtract_and_median, slug_suffix)
BG_OPTIONS = [
    (False, ""),
    (True,  "_bgmed"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def unique_raw_configs():
    """Yield (sg_type, window, scale_label, scale_slug) for all 18 unique raw configs."""
    for sg_type in SG_TYPES:
        windows = WINDOWS if sg_type != "Multi-tapered" else [None]
        for window in windows:
            for scale_label, scale_slug in SG_SCALES:
                yield sg_type, window, scale_label, scale_slug


def config_slug(sg_type, window, scale_slug):
    type_slug = sg_type.lower().replace("-", "").replace(" ", "_")
    win_slug  = window.lower() if window else "default"
    return f"{type_slug}_{win_slug}_{scale_slug}"


def run_cmd(cmd, dry_run=False):
    """Run a subprocess command, streaming output. Exit on failure."""
    print("  $", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)


# ── Dataset build ─────────────────────────────────────────────────────────────

def build_dataset(sg_type, window, scale_label, scale_slug, matched_base, dry_run=False):
    """Run the 3-step dataset build for one raw spectrogram config."""
    doc_matched   = f"{matched_base}/doc_matched"
    avianz_matched = f"{matched_base}/avianz_matched"
    doc_split     = f"{matched_base}/doc_split"
    avianz_split  = f"{matched_base}/avianz_split"
    merged_train  = f"{matched_base}/merged_train"

    slug = config_slug(sg_type, window, scale_slug)
    print(f"\n{'─'*60}")
    print(f"  Dataset: {slug}")
    print(f"{'─'*60}")

    # Step 1: build matched datasets
    step1_done = (
        os.path.isfile(f"{doc_matched}/labels.json") and
        os.path.isfile(f"{avianz_matched}/labels.json")
    )
    if not step1_done:
        print("  Step 1: building matched datasets...")
        cmd = [
            sys.executable, "src/experiments/build_matched_datasets.py",
            "--reviewed-csv", REVIEWED_CSV,
            "--doc-raw",      DOC_RAW,
            "--avianz-raw",   AVIANZ_RAW,
            "--output",       matched_base,
            "--mapping",      MAPPING,
            "--fixed-length",
            "--spec-type",    sg_type,
            "--sg-scale",     scale_label,
        ]
        if window:
            cmd += ["--window-type", window]
        run_cmd(cmd, dry_run)
    else:
        print("  Step 1: [skip] matched datasets already exist")

    # Step 2: split into train/test
    splits_done = all(
        os.path.isfile(f"{matched_base}/{ds}_split/{part}/labels.json")
        for ds in ("doc", "avianz")
        for part in ("train", "test")
    )
    if not splits_done or not step1_done:
        print("  Step 2: splitting datasets...")
        run_cmd([
            sys.executable, "src/experiments/split_matched_datasets.py",
            avianz_matched, doc_matched, matched_base,
            "--test-ratio", "0.25",
            "--seed", "42",
            "--overwrite",
        ], dry_run)
    else:
        print("  Step 2: [skip] splits already exist")

    # Step 3: merge training sets
    if not os.path.isfile(f"{merged_train}/labels.json") or not splits_done:
        print("  Step 3: merging training datasets...")
        run_cmd([
            sys.executable, "src/experiments/merge_datasets.py",
            f"{doc_split}/train",
            f"{avianz_split}/train",
            merged_train,
            "--symlink",
            "--no-audio",
        ], dry_run)
    else:
        print("  Step 3: [skip] merged_train already exists")


# ── Training ──────────────────────────────────────────────────────────────────

def run_experiment(sg_type, window, scale_slug, norm_label, norm_slug,
                   use_bg, bg_suffix, matched_base, dry_run=False):
    """Run training for one (train_dataset, normalization, bg_option) triple."""
    slug = config_slug(sg_type, window, scale_slug)

    for train_name, train_dir in [
        ("doc",    f"{matched_base}/doc_split/train"),
        ("avianz", f"{matched_base}/avianz_split/train"),
    ]:
        avianz_test = f"{matched_base}/avianz_split/test"
        doc_test    = f"{matched_base}/doc_split/test"

        run_name = f"{MODEL_TYPE}_on_{train_name}_{slug}_{norm_slug}{bg_suffix}"
        out_dir  = f"{TESTS_BASE}/{run_name}"

        # Skip if already trained
        if (os.path.isfile(f"{out_dir}/model.pt") or
                os.path.isfile(f"{out_dir}/model.pth")):
            print(f"  [skip] {run_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train.py",
            train_dir, out_dir,
            "--test-folder",  avianz_test,
            "--test-folder2", doc_test,
            "--visualize-attention",
            "--viz-samples",  str(VIZ_SAMPLES),
            "--epochs",       str(EPOCHS),
            "--patience",     str(PATIENCE),
            "--mixup",        str(MIXUP),
            "--model-type",   MODEL_TYPE,
            "--spec-transform", norm_label,
        ]
        if use_bg:
            cmd += ["--bg-subtract", "--median-filter"]

        run_cmd(cmd, dry_run)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full spectrogram sweep: 18 dataset builds × 216 training runs"
    )
    parser.add_argument("--build-only",  action="store_true",
                        help="Only build datasets, skip training")
    parser.add_argument("--train-only",  action="store_true",
                        help="Only run training (datasets must already exist)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--list",        action="store_true",
                        help="List all experiment names and exit")
    args = parser.parse_args()

    raw_configs = list(unique_raw_configs())
    n_builds    = len(raw_configs)                               # 18
    n_runs      = n_builds * len(NORMALIZATIONS) * len(BG_OPTIONS) * 2  # 216

    if args.list:
        print(f"{'─'*60}")
        print(f"  {n_builds} dataset builds  →  {n_runs} training runs")
        print(f"{'─'*60}")
        for sg_type, window, scale_label, scale_slug in raw_configs:
            slug = config_slug(sg_type, window, scale_slug)
            for norm_label, norm_slug in NORMALIZATIONS:
                for use_bg, bg_suffix in BG_OPTIONS:
                    for train_name in ("doc", "avianz"):
                        print(f"  {MODEL_TYPE}_on_{train_name}_{slug}_{norm_slug}{bg_suffix}")
        return

    os.makedirs(SWEEP_BASE, exist_ok=True)
    os.makedirs(TESTS_BASE, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Sweep: {n_builds} dataset builds, {n_runs} training runs")
    if args.dry_run:
        print("  DRY RUN — no commands will be executed")
    print(f"{'#'*60}")

    for sg_type, window, scale_label, scale_slug in raw_configs:
        slug         = config_slug(sg_type, window, scale_slug)
        matched_base = f"{SWEEP_BASE}/{slug}"

        if not args.train_only:
            build_dataset(sg_type, window, scale_label, scale_slug,
                          matched_base, dry_run=args.dry_run)

        if not args.build_only:
            for norm_label, norm_slug in NORMALIZATIONS:
                for use_bg, bg_suffix in BG_OPTIONS:
                    run_experiment(
                        sg_type, window, scale_slug,
                        norm_label, norm_slug,
                        use_bg, bg_suffix,
                        matched_base,
                        dry_run=args.dry_run,
                    )

    print(f"\nSweep complete.")
    print(f"  Datasets : {SWEEP_BASE}")
    print(f"  Results  : {TESTS_BASE}")


if __name__ == "__main__":
    main()
