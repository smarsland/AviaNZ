#!/usr/bin/env python3
"""
Count how many raw audio samples are available per species in the DOC and AviaNZ
datasets, restricted to the species that appear in the matched/reviewed CSV.

For labels with "/" (e.g. "tui/bellbird") both sides are counted independently.

Usage:
    python3 scripts/count_available_species.py \
        --doc-raw  /path/to/NZBirds \
        --avianz-raw /path/to/Joe_MoDone \
        --reviewed-csv data/doc_reviewed.csv \
        --mapping data/DOC_bird_naming_map.csv

Output: a table sorted by DOC count, showing counts in both datasets.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.analyze_dataset_quality import (
    build_group_cache,
    is_poor_quality,
    load_bird_name_mapping,
    normalize_species_name_to_codes,
)

# Inline lightweight AviaNZ helpers to avoid pulling in pyflac via dataset_builder.
def _find_wav_files(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.wav') and not f.endswith('.backup'):
                yield os.path.join(root, f)


def _load_annotation_file(data_file):
    try:
        with open(data_file) as fh:
            data = json.load(fh)
    except Exception:
        return []
    if not isinstance(data, list) or len(data) < 2:
        return []
    segments = []
    for seg in data[1:]:
        try:
            # seg format: [start, end, freq_low, freq_high, labels]
            if isinstance(seg, list) and len(seg) >= 5 and isinstance(seg[4], list):
                segments.append(seg[4])  # list of label dicts
        except Exception:
            continue
    return segments


def _normalize_to_ebird(species_name, name_mapping):
    if not species_name or species_name in ('Empty Sample', 'Tree Weta', 'Spy Bird', "Don't Know"):
        return species_name
    if species_name in name_mapping:
        return name_mapping[species_name]
    lower = species_name.lower()
    for k, v in name_mapping.items():
        if k.lower() == lower:
            return v
    if '(' in species_name:
        base = species_name.split('(')[0].strip()
        if base in name_mapping:
            return name_mapping[base]
    return species_name


def get_matched_species(reviewed_csv, mapping_csv):
    """
    Return the set of eBird codes that appear in the reviewed CSV (species 1 column).
    For "/" labels both sides are included. Poor-quality and uncertain rows are skipped.
    Also returns a dict: ebird_code -> set of raw label strings that map to it.
    """
    ebird_to_common, common_to_ebird = load_bird_name_mapping(mapping_csv)
    group_cache = build_group_cache(ebird_to_common)

    df = pd.read_csv(reviewed_csv, header=0)

    codes_seen = set()
    code_to_labels = defaultdict(set)  # ebird_code -> human label strings

    for _, row in df.iterrows():
        if is_poor_quality(row.get('Note', '')):
            continue

        species1 = str(row.get('Species 1', '')).strip()
        if not species1 or pd.isna(row.get('Species 1', '')):
            continue
        if species1.startswith('?'):
            continue
        if 'unknown' in species1.lower():
            continue

        codes = normalize_species_name_to_codes(
            species1, common_to_ebird, ebird_to_common, group_cache
        )
        for code in codes:
            codes_seen.add(code)
            code_to_labels[code].add(species1.lower())

        # Also scan Species 2+
        species2 = str(row.get('Species 2+', '')).strip()
        if species2 and not pd.isna(row.get('Species 2+', '')):
            for part in species2.split(','):
                part = part.strip()
                if not part or part.startswith('?') or 'unknown' in part.lower():
                    continue
                extra_codes = normalize_species_name_to_codes(
                    part, common_to_ebird, ebird_to_common, group_cache
                )
                for code in extra_codes:
                    codes_seen.add(code)
                    code_to_labels[code].add(part.lower())

    return codes_seen, code_to_labels, ebird_to_common


def count_doc_samples(doc_raw, target_codes):
    """
    Count audio files in DOC raw folder per eBird code.
    Structure: {doc_raw}/{site}/{site}/train_audio/{ebird_code}/*.flac
    """
    counts = defaultdict(int)
    audio_exts = {'.wav', '.mp3', '.flac'}

    if not os.path.isdir(doc_raw):
        print(f"WARNING: DOC raw folder not found: {doc_raw}")
        return counts

    for site in os.listdir(doc_raw):
        site_path = os.path.join(doc_raw, site, site, 'train_audio')
        if not os.path.isdir(site_path):
            continue
        for code_folder in os.listdir(site_path):
            if code_folder not in target_codes:
                continue
            folder_path = os.path.join(site_path, code_folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1].lower() in audio_exts:
                    counts[code_folder] += 1

    return counts


def count_avianz_segments(avianz_raw, target_codes, mapping_csv):
    """
    Count AviaNZ annotated segments per eBird code by scanning all .wav.data files.
    """
    name_mapping = _load_avianz_name_mapping(mapping_csv)
    counts = defaultdict(int)

    if not os.path.isdir(avianz_raw):
        print(f"WARNING: AviaNZ raw folder not found: {avianz_raw}")
        return counts

    wav_files = list(_find_wav_files(avianz_raw))
    print(f"AviaNZ: scanning {len(wav_files)} wav files...")

    for i, wav_file in enumerate(wav_files):
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(wav_files)}")
        data_file = wav_file + '.data'
        if not os.path.exists(data_file):
            continue
        for labels in _load_annotation_file(data_file):
            for lab in labels:
                if not isinstance(lab, dict):
                    continue
                if lab.get('certainty', 0) < 50:
                    continue
                code = _normalize_to_ebird(lab.get('species', ''), name_mapping)
                if code in target_codes:
                    counts[code] += 1

    return counts


def _load_avianz_name_mapping(mapping_csv):
    """Mirror of the same helper in build_matched_datasets.py."""
    df = pd.read_csv(mapping_csv)
    mapping = {}
    for _, row in df.iterrows():
        ebird = row['eBird']
        if pd.isna(ebird):
            continue
        for col in ['CommonName', 'ExtraName', 'ListDOCBirds']:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                mapping[str(val).strip()] = str(ebird).strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Count available raw samples per species')
    parser.add_argument('--doc-raw',     required=True, help='Path to raw DOC audio (NZBirds)')
    parser.add_argument('--avianz-raw',  required=True, help='Path to raw AviaNZ audio (Joe_MoDone)')
    parser.add_argument('--reviewed-csv', default='data/doc_reviewed.csv')
    parser.add_argument('--mapping',      default='data/DOC_bird_naming_map.csv')
    parser.add_argument('--min-count',    type=int, default=0,
                        help='Only show species with at least this many samples in DOC')
    args = parser.parse_args()

    print("=== Extracting matched species from reviewed CSV ===")
    target_codes, code_to_labels, ebird_to_common = get_matched_species(
        args.reviewed_csv, args.mapping
    )
    print(f"Found {len(target_codes)} distinct eBird codes in the reviewed CSV\n")

    print("=== Counting DOC raw samples ===")
    doc_counts = count_doc_samples(args.doc_raw, target_codes)

    print("\n=== Counting AviaNZ segments ===")
    avianz_counts = count_avianz_segments(args.avianz_raw, target_codes, args.mapping)

    # Build display table
    rows = []
    for code in sorted(target_codes):
        common = ebird_to_common.get(code, code)
        doc_n = doc_counts.get(code, 0)
        avianz_n = avianz_counts.get(code, 0)
        labels = ', '.join(sorted(code_to_labels.get(code, {code})))
        rows.append((code, common or code, doc_n, avianz_n, labels))

    # Sort by DOC count descending
    rows.sort(key=lambda r: r[2], reverse=True)

    if args.min_count > 0:
        rows = [r for r in rows if r[2] >= args.min_count]

    # Print table
    print()
    print(f"{'Code':<12} {'Common Name':<30} {'DOC':>6} {'AviaNZ':>8}  Labels in CSV")
    print("-" * 90)
    for code, common, doc_n, avianz_n, labels in rows:
        print(f"{code:<12} {common:<30} {doc_n:>6} {avianz_n:>8}  {labels}")

    print()
    total_doc = sum(r[2] for r in rows)
    total_avianz = sum(r[3] for r in rows)
    print(f"{'TOTAL':<12} {'':<30} {total_doc:>6} {total_avianz:>8}")
    print(f"\n{len(rows)} species shown")


if __name__ == '__main__':
    main()
