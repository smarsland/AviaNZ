"""
Scan a dataset folder for .npy files that are not float32, and reprocess
them from their original source audio using the same spectrogram settings.

Reads labels.json for source_file paths, checks each .npy, and overwrites
only the bad ones.

Usage:
    PYTHONPATH=. python3 scripts/repair_dataset_dtypes.py \
        --dataset /local/scratch/freangi/scaling/doc_large \
        --spec-type Standard \
        --window-type Hamming \
        --sg-scale "Mel Frequency"
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import config
from src.experiments.build_matched_datasets import make_spec_processor, trim_spectrogram_to_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset folder (contains labels.json and data/)')
    parser.add_argument('--spec-type', default='Standard')
    parser.add_argument('--window-type', default='Hamming')
    parser.add_argument('--sg-scale', default='Mel Frequency')
    parser.add_argument('--dry-run', action='store_true', help='Report bad files but do not reprocess')
    args = parser.parse_args()

    labels_file = os.path.join(args.dataset, 'labels.json')
    data_dir = os.path.join(args.dataset, 'data')

    with open(labels_file) as f:
        label_data = json.load(f)

    files = label_data['files']
    target_time_bins = config.DEFAULT_TIME_BINS

    bad_castable = []   # numeric dtype (e.g. float64) — just cast in-place
    bad_reprocess = []  # unloadable or object dtype — need reprocessing from source
    missing = []
    for entry in files:
        npy_path = os.path.join(data_dir, entry['filename'])
        if not os.path.exists(npy_path):
            missing.append(entry['filename'])
            continue
        try:
            arr = np.load(npy_path)
            if arr.dtype == np.float32:
                continue
            if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
                bad_castable.append(entry)
            else:
                bad_reprocess.append(entry)
        except Exception as e:
            print(f"Cannot load {entry['filename']}: {e}")
            bad_reprocess.append(entry)

    total_bad = len(bad_castable) + len(bad_reprocess)
    print(f"Scanned {len(files)} files: {len(bad_castable)} castable, {len(bad_reprocess)} need reprocessing, {len(missing)} missing")
    if total_bad == 0:
        print("Nothing to repair.")
        return

    if args.dry_run:
        for entry in bad_castable + bad_reprocess:
            npy_path = os.path.join(data_dir, entry['filename'])
            try:
                arr = np.load(npy_path)
                print(f"  {entry['filename']}  dtype={arr.dtype}  source={entry.get('source_file','?')}")
            except Exception as e:
                print(f"  {entry['filename']}  UNLOADABLE: {e}  source={entry.get('source_file','?')}")
        return

    # --- Pass 1: in-place cast for numeric dtypes (fast) ---
    cast_ok = 0
    cast_fail = 0
    for entry in bad_castable:
        npy_path = os.path.join(data_dir, entry['filename'])
        try:
            arr = np.load(npy_path).astype(np.float32)
            np.save(npy_path, arr)
            cast_ok += 1
        except Exception as e:
            print(f"  CAST FAIL {entry['filename']}: {e}")
            cast_fail += 1

    print(f"Cast pass: {cast_ok} fixed, {cast_fail} failed")

    # --- Pass 2: reprocess from source for unloadable / object-dtype files ---
    if bad_reprocess:
        spec_proc = make_spec_processor(args.spec_type, args.window_type, args.sg_scale)
        reprocess_ok = 0
        reprocess_fail = 0
        for entry in bad_reprocess:
            source_file = entry.get('source_file')
            npy_path = os.path.join(data_dir, entry['filename'])
            if not source_file or not os.path.exists(source_file):
                print(f"  SKIP {entry['filename']}: source missing ({source_file})")
                reprocess_fail += 1
                continue
            sg = spec_proc.process_audio_file(source_file)
            if sg is None:
                print(f"  FAIL {entry['filename']}: spec processing returned None")
                reprocess_fail += 1
                continue
            if sg.shape[1] > target_time_bins:
                sg = trim_spectrogram_to_length(sg, target_time_bins)
                if sg is None:
                    print(f"  FAIL {entry['filename']}: trim returned None")
                    reprocess_fail += 1
                    continue
            np.save(npy_path, np.asarray(sg, dtype=np.float32))
            reprocess_ok += 1
        print(f"Reprocess pass: {reprocess_ok} fixed, {reprocess_fail} failed")

    print(f"\nDone. Total fixed: {cast_ok + (reprocess_ok if bad_reprocess else 0)}/{total_bad}")


if __name__ == '__main__':
    main()
