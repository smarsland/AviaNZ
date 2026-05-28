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

    bad = []
    missing = []
    for entry in files:
        npy_path = os.path.join(data_dir, entry['filename'])
        if not os.path.exists(npy_path):
            missing.append(entry['filename'])
            continue
        try:
            arr = np.load(npy_path)
            if arr.dtype != np.float32:
                bad.append(entry)
        except Exception as e:
            print(f"Cannot load {entry['filename']}: {e}")
            bad.append(entry)

    print(f"Scanned {len(files)} files: {len(bad)} bad dtype, {len(missing)} missing")
    if not bad:
        print("Nothing to repair.")
        return

    if args.dry_run:
        for entry in bad:
            npy_path = os.path.join(data_dir, entry['filename'])
            try:
                arr = np.load(npy_path)
                print(f"  {entry['filename']}  dtype={arr.dtype}  source={entry.get('source_file','?')}")
            except Exception as e:
                print(f"  {entry['filename']}  UNLOADABLE: {e}  source={entry.get('source_file','?')}")
        return

    spec_proc = make_spec_processor(args.spec_type, args.window_type, args.sg_scale)

    repaired = 0
    failed = 0
    for entry in bad:
        source_file = entry.get('source_file')
        npy_path = os.path.join(data_dir, entry['filename'])

        if not source_file or not os.path.exists(source_file):
            print(f"  SKIP {entry['filename']}: source_file missing or not found ({source_file})")
            failed += 1
            continue

        sg = spec_proc.process_audio_file(source_file)
        if sg is None:
            print(f"  FAIL {entry['filename']}: spec processing returned None for {source_file}")
            failed += 1
            continue

        if sg.shape[1] > target_time_bins:
            sg = trim_spectrogram_to_length(sg, target_time_bins)
            if sg is None:
                print(f"  FAIL {entry['filename']}: trim returned None")
                failed += 1
                continue

        # save_spectrogram now enforces float32, but be explicit here too
        np.save(npy_path, np.asarray(sg, dtype=np.float32))
        repaired += 1

    print(f"\nRepaired {repaired}/{len(bad)} files. Failed: {failed}")


if __name__ == '__main__':
    main()
