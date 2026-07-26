#!/usr/bin/env python3
"""
Patch an already-built dataset's labels.json in place — no spectrogram
re-extraction.

Two operations, both label-only, so they are cheap even on a dataset that took
hours to build:

  --label-remap           merge/rename classes (e.g. 'bellbird' → 'tui/bellbird')
  --exclude-source-files  drop every sample that came from a held-out recording

Use this to retrofit an existing combined_large/ instead of rebuilding it:

    python3 scripts/patch_dataset_labels.py \\
        /local/scratch/freangi/combined_dataset/combined_large \\
        --label-remap "new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird" \\
        --exclude-source-files /local/scratch/freangi/matched/avianz_split/test/labels.json

The original labels.json is copied to labels.json.bak (once — an existing .bak
is never overwritten, so repeated runs keep the pristine original).

Orphaned .npy files are left on disk; they are simply no longer referenced.
Note that dropping samples happens *after* the build-time per-species cap, so
excluded species end up slightly below the cap rather than being topped up.
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def load_exclusion_list(paths):
    """Return a set of realpath'd wav paths from labels.json / plain text lists."""
    excluded = set()
    for p in paths or []:
        if not os.path.exists(p):
            raise SystemExit(f"--exclude-source-files: not found: {p}")
        if p.endswith(".json"):
            with open(p) as f:
                payload = json.load(f)
            entries = payload["files"] if isinstance(payload, dict) else payload
            found = {e["source_file"] for e in entries if e.get("source_file")}
        else:
            with open(p) as f:
                found = {line.strip() for line in f if line.strip()}
        print(f"  exclusion list {p}: {len(found)} unique source recordings")
        excluded.update(os.path.realpath(x) for x in found)
    return excluded


def parse_remap(spec):
    if not spec:
        return {}
    remap = {}
    for pair in spec.split(","):
        old, new = pair.split(":", 1)
        remap[old.strip()] = new.strip()
    return remap


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_dir", help="Directory containing labels.json")
    parser.add_argument("--label-remap", default=None,
                        help='Comma-separated old:new pairs, e.g. "tui:tui/bellbird"')
    parser.add_argument("--exclude-source-files", default=None, action="append",
                        help="labels.json or text file listing recordings to drop. Repeatable.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels_path = dataset_dir / "labels.json"
    if not labels_path.exists():
        raise SystemExit(f"labels.json not found in {dataset_dir}")

    remap = parse_remap(args.label_remap)
    excluded = load_exclusion_list(args.exclude_source_files)

    if not remap and not excluded:
        raise SystemExit("Nothing to do — pass --label-remap and/or --exclude-source-files")

    with open(labels_path) as f:
        payload = json.load(f)
    entries = payload["files"]

    print(f"\nLoaded {len(entries)} samples, "
          f"{len(payload.get('categories', []))} classes from {labels_path}")

    # ── 1. Drop samples from held-out recordings ──────────────────────────────
    n_before = len(entries)
    if excluded:
        entries = [e for e in entries
                   if os.path.realpath(e.get("source_file", "")) not in excluded]
        n_dropped = n_before - len(entries)
        print(f"\nExclusion: dropped {n_dropped} / {n_before} samples "
              f"({n_dropped / n_before * 100:.1f}%)")
        if n_dropped == 0:
            print("  WARNING: nothing matched. Either this dataset genuinely contains "
                  "none of those recordings, or the 'source_file' paths do not resolve "
                  "to the same locations as the exclusion list. Check before trusting "
                  "any test metric computed against those recordings.")

    # ── 2. Remap class names ──────────────────────────────────────────────────
    if remap:
        n_touched = 0
        for e in entries:
            names = e.get("class_names") or []
            new_names = []
            for n in names:
                n = remap.get(n, n)
                if n not in new_names:      # merging can create duplicates
                    new_names.append(n)
            if new_names != names:
                n_touched += 1
            e["class_names"] = new_names
        print(f"\nRemap {remap}: rewrote labels on {n_touched} samples")

    # ── 3. Recompute the class vocabulary ─────────────────────────────────────
    old_categories = list(payload.get("categories", []))
    categories = sorted({c for e in entries for c in (e.get("class_names") or [])})
    removed = [c for c in old_categories if c not in categories]
    added = [c for c in categories if c not in old_categories]

    print(f"\nClasses: {len(old_categories)} → {len(categories)}")
    if removed:
        print(f"  removed: {removed}")
    if added:
        print(f"  added:   {added}")

    counts = Counter(c for e in entries for c in (e.get("class_names") or []))
    empty = [c for c in categories if counts[c] == 0]
    if empty:
        print(f"  WARNING: {len(empty)} classes now have 0 samples: {empty}")

    payload["files"] = entries
    payload["categories"] = categories
    payload["num_classes"] = len(categories)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup = labels_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(labels_path, backup)
        print(f"\nBacked up original → {backup}")
    else:
        print(f"\nKeeping existing backup at {backup}")

    with open(labels_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(entries)} samples, {len(categories)} classes → {labels_path}")
    print("\nThe class count changed — any model trained on the old labels.json "
          "is no longer compatible. Retrain.")


if __name__ == "__main__":
    main()
