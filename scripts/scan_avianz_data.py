#!/usr/bin/env python3
"""
Scan a root directory for AviaNZ-style data and report counts per top-level subfolder.

AviaNZ-style data = WAV files that have a corresponding .wav.data annotation file.

Usage:
    python scripts/scan_avianz_data.py /media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/
    python scripts/scan_avianz_data.py /path/to/root --show-species
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def count_annotations_in_data_file(data_path):
    """Return number of annotated segments in a .wav.data file (excludes header row)."""
    try:
        with open(data_path, 'r', errors='replace') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 1:
            return len(data) - 1  # first element is file-level metadata
        return 0
    except Exception:
        return 0


def collect_species_in_data_file(data_path):
    """Return set of species names found in a .wav.data file."""
    species = set()
    try:
        with open(data_path, 'r', errors='replace') as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) < 2:
            return species
        for seg in data[1:]:
            # Segment format: [t0, t1, freq_lo, freq_hi, labels]
            if isinstance(seg, list) and len(seg) >= 5:
                labels = seg[4]
                if isinstance(labels, list):
                    for lab in labels:
                        if isinstance(lab, dict) and 'species' in lab:
                            species.add(lab['species'])
                        elif isinstance(lab, list) and len(lab) > 0:
                            species.add(str(lab[0]))
    except Exception:
        pass
    return species


def scan_folder(top_folder, show_species=False):
    """Walk a top-level folder and return stats."""
    wav_count = 0
    annotated_count = 0
    segment_count = 0
    species_set = set()

    for root, dirs, files in os.walk(top_folder, followlinks=False):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            if fname.lower().endswith('.wav') and not fname.endswith('.backup'):
                wav_count += 1
                data_path = os.path.join(root, fname + '.data')
                if os.path.isfile(data_path):
                    annotated_count += 1
                    segs = count_annotations_in_data_file(data_path)
                    segment_count += segs
                    if show_species:
                        species_set |= collect_species_in_data_file(data_path)

    return wav_count, annotated_count, segment_count, species_set


def main():
    parser = argparse.ArgumentParser(description="Find AviaNZ-style annotated data in a root directory.")
    parser.add_argument('root', help='Root directory to scan')
    parser.add_argument('--show-species', action='store_true',
                        help='Also list unique species per folder (slower)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel worker threads (default: 8)')
    args = parser.parse_args()

    root = os.path.realpath(args.root)
    if not os.path.isdir(root):
        print(f"Error: {root!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Collect top-level entries (both dirs and loose WAV/data files)
    try:
        entries = sorted(os.listdir(root))
    except PermissionError as e:
        print(f"Error reading root: {e}", file=sys.stderr)
        sys.exit(1)

    top_dirs = []
    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full) and not entry.startswith('.'):
            top_dirs.append((entry, full))

    print(f"Scanning {root}")
    print(f"Found {len(top_dirs)} top-level directories.\n")

    results = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_name = {
            pool.submit(scan_folder, full, args.show_species): name
            for name, full in top_dirs
        }
        done = 0
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            done += 1
            print(f"  [{done}/{len(top_dirs)}] {name}", end='\r', flush=True)
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = (0, 0, 0, set())
                print(f"\n  Warning: error scanning {name}: {e}", file=sys.stderr)

    print()  # clear progress line

    # Sort by annotated WAV count descending
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)

    # Print table
    col_w = max((len(n) for n, _ in sorted_results), default=30)
    col_w = max(col_w, 30)
    header = f"{'Folder':<{col_w}}  {'WAVs':>8}  {'Annotated':>10}  {'Segments':>10}"
    print(header)
    print('-' * len(header))

    total_wav = total_ann = total_seg = 0
    for name, (wav_count, annotated_count, segment_count, species_set) in sorted_results:
        total_wav += wav_count
        total_ann += annotated_count
        total_seg += segment_count
        line = f"{name:<{col_w}}  {wav_count:>8,}  {annotated_count:>10,}  {segment_count:>10,}"
        if annotated_count == 0:
            line += "  (no AviaNZ data)"
        print(line)
        if args.show_species and species_set:
            sp_sorted = sorted(species_set)
            print(f"  {'Species:':<{col_w-2}} {', '.join(sp_sorted)}")

    print('-' * len(header))
    print(f"{'TOTAL':<{col_w}}  {total_wav:>8,}  {total_ann:>10,}  {total_seg:>10,}")


if __name__ == '__main__':
    main()
