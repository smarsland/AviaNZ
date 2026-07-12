"""
Build large DOC and AviaNZ datasets without record-level matching.

Unlike build_matched_datasets.py this script:
  1. Does NOT require a reviewed/corrections CSV.
  2. Takes up to --max-per-species samples from every species in each dataset.
  3. Uses folder structure (DOC) and annotation files (AviaNZ) directly.
  4. Ensures both datasets share the same class vocabulary (intersection).
  5. Saves audio alongside spectrograms so Kaytoo/BirdNET can be evaluated.
  6. Uses the best-performing model's spectrogram settings by default:
       Reassigned spectrogram, Hamming window, Linear frequency scale.

DOC folder layout expected:
    doc_raw/*/{name}/train_audio/{eBird_code}/*.{wav,mp3,flac,ogg}

AviaNZ folder layout expected:
    avianz_raw/**/*.wav  +  *.wav.data  (annotation files)

Usage:
    python src/experiments/build_large_datasets.py \\
        --doc-raw /path/to/NZBirds \\
        --avianz-raw /path/to/Joe_MoDone \\
        --output /path/to/large \\
        --mapping data/DOC_bird_naming_map.csv
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import soundfile as sf

from src.core import config
from src.data.dataset_builder import AviaNZDataProcessor
from src.experiments.analyze_dataset_quality import load_bird_name_mapping, norm_key
from src.experiments.build_matched_datasets import (
    filter_to_common_classes,
    load_avianz_name_mapping,
    make_spec_processor,
    trim_spectrogram_to_length,
    write_labels_json,
)
from src.experiments.split_matched_datasets import split_avianz_by_file


# Species labels to skip in AviaNZ annotations.
_SKIP_SPECIES = frozenset({
    'Empty Sample', 'Tree Weta', 'Spy Bird', "Don't Know", None, '',
    'Noise', 'Background Noise',
})


# ---------------------------------------------------------------------------
# DOC scanning
# ---------------------------------------------------------------------------

def scan_doc_by_species(doc_raw):
    """
    Walk the DOC folder tree and collect audio files keyed by eBird code.

    Expected structure:
        doc_raw/<id>/<id>/train_audio/<eBird_code>/<audio_file>

    Returns:
        dict  {eBird_code: [absolute_audio_path, ...]}
    """
    species_files = defaultdict(list)
    audio_exts = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a')
    for root, _dirs, files in os.walk(doc_raw):
        parent_name = os.path.basename(os.path.dirname(root))
        if parent_name == 'train_audio':
            species_code = os.path.basename(root)
            for fname in files:
                if fname.lower().endswith(audio_exts):
                    species_files[species_code].append(os.path.join(root, fname))
    return dict(species_files)


# ---------------------------------------------------------------------------
# DOC dataset builder
# ---------------------------------------------------------------------------

def build_doc_large(
    doc_raw, output_folder, ebird_to_common,
    max_per_species=1000, seed=42,
    fixed_length=True, target_time_bins=None,
    with_audio=True,
    sg_type=None, window_type=None, sg_scale=None,
    restrict_classes=None,
    label_remap=None,
):
    """
    Scan the DOC folder structure and extract up to max_per_species spectrograms
    per eBird code.  Labels are normalised common names (lowercase).

    Args:
        restrict_classes: If provided (set/list of normalised common names), skip
                          any species that doesn't map to one of these labels.
                          Applied AFTER label_remap so remap targets can be used.
        label_remap: Optional dict mapping normalised label → new label, applied
                     before restrict_classes.  Use to merge classes or rename:
                     e.g. {"tui": "tui/bellbird", "bellbird": "tui/bellbird",
                            "new zealand kaka": "kaka"}

    Returns list of label dicts compatible with labels.json.
    """
    if target_time_bins is None:
        target_time_bins = config.DEFAULT_TIME_BINS

    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    data_dir = os.path.join(output_folder, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(output_folder, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    species_files = scan_doc_by_species(doc_raw)
    print(f'DOC: found {len(species_files)} species codes in folder structure')

    rng = random.Random(seed)
    labels = []
    skipped_unmapped = []
    restrict_set = set(restrict_classes) if restrict_classes else None
    # Per-species sample counts after remapping (for max_per_species cap)
    label_sample_counts = defaultdict(int)

    for species_code, file_list in sorted(species_files.items()):
        # Map eBird code to normalised common name
        common_name = ebird_to_common.get(norm_key(species_code))
        if not common_name:
            skipped_unmapped.append(species_code)
            continue
        label = norm_key(common_name)

        # Apply label remapping (merge or rename)
        if label_remap:
            label = label_remap.get(label, label)

        # Skip species not in the allowed set (if a filter is given)
        if restrict_set is not None and label not in restrict_set:
            continue

        # Respect per-label cap even when multiple eBird codes map to same label
        already_saved = label_sample_counts[label]
        remaining_cap = max_per_species - already_saved
        if remaining_cap <= 0:
            print(f'  {species_code:12s} ({label:20s}): skipped — cap already reached ({max_per_species})')
            continue

        # Randomly sample up to remaining_cap from this eBird code's files
        if len(file_list) > remaining_cap:
            sampled = rng.sample(file_list, remaining_cap)
        else:
            sampled = list(file_list)

        saved = 0
        failed = 0
        for audio_path in sampled:
            sg = spec_proc.process_audio_file(audio_path)
            if sg is None:
                failed += 1
                continue

            if fixed_length and sg.shape[1] > target_time_bins:
                sg = trim_spectrogram_to_length(sg, target_time_bins)
                if sg is None:
                    failed += 1
                    continue

            basename = f'file_{len(labels):08d}'
            spec_proc.save_spectrogram(sg, data_dir, basename)

            if with_audio:
                audio_data, audio_sr = sf.read(audio_path)
                sf.write(os.path.join(audio_dir, f'{basename}.wav'), audio_data, audio_sr)

            labels.append({
                'filename': f'{basename}.npy',
                'class_names': [label],
                'source_file': audio_path,
            })
            label_sample_counts[label] += 1
            saved += 1

        print(f'  {species_code:12s} ({label:20s}): '
              f'saved {saved}/{len(file_list)}  (failed={failed})')

    if skipped_unmapped:
        print(f'\nDOC: skipped {len(skipped_unmapped)} unmapped eBird codes: '
              f'{sorted(skipped_unmapped)[:10]}{"..." if len(skipped_unmapped) > 10 else ""}')

    print(f'\nDOC large: total {len(labels)} samples')
    return labels


# ---------------------------------------------------------------------------
# AviaNZ dataset builder
# ---------------------------------------------------------------------------

def build_avianz_large(
    avianz_raw, output_folder, mapping_csv, ebird_to_common,
    max_per_species=1000, seed=42,
    fixed_length=True, target_time_bins=None,
    with_audio=True,
    sg_type=None, window_type=None, sg_scale=None,
):
    """
    Scan AviaNZ annotation files and extract up to max_per_species spectrograms
    per species.

    Two-phase approach:
      Phase 1 – collect all unique segments with their label sets.
      Phase 2 – for each species independently sample up to max_per_species
                 segment keys; take the union of selected keys.
      Phase 3 – process and save each unique segment exactly once.

    Returns list of label dicts compatible with labels.json.
    """
    if target_time_bins is None:
        target_time_bins = config.DEFAULT_TIME_BINS

    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    name_mapping = load_avianz_name_mapping(mapping_csv)
    proc = AviaNZDataProcessor(name_mapping=name_mapping)

    data_dir = os.path.join(output_folder, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(output_folder, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: collect unique segments
    # -----------------------------------------------------------------------
    # key = (wav_file, start_time)
    # value = (end_time, [common_name_label, ...], freq_low, freq_high)
    all_segments = {}
    # ebird_code -> set of segment keys that contain that species
    species_to_keys = defaultdict(set)

    if isinstance(avianz_raw, (list, tuple)):
        avianz_raw_list = avianz_raw
    else:
        avianz_raw_list = [avianz_raw]
    wav_files = []
    for raw_dir in avianz_raw_list:
        wav_files.extend(proc.find_wav_files(raw_dir))
    print(f'AviaNZ: scanning {len(wav_files)} wav files across {len(avianz_raw_list)} source folder(s) ...')

    for wav_file in wav_files:
        data_file = wav_file + '.data'
        if not os.path.exists(data_file):
            continue
        segments = proc.load_annotation_file(data_file)
        for seg in segments:
            # Collect eBird codes with certainty >= 50, skip non-birds
            seg_codes = []
            for lab in seg.labels:
                if lab['certainty'] < 50:
                    continue
                sp = lab['species']
                if sp in _SKIP_SPECIES:
                    continue
                code = proc.normalize_to_ebird(sp)
                if code and code not in seg_codes:
                    seg_codes.append(code)

            if not seg_codes:
                continue

            # Map eBird codes to normalised common names
            common_labels = []
            for code in seg_codes:
                common = ebird_to_common.get(norm_key(code))
                if common:
                    lbl = norm_key(common)
                    if lbl not in common_labels:
                        common_labels.append(lbl)

            if not common_labels:
                continue

            key = (wav_file, seg.start_time)
            if key not in all_segments:
                all_segments[key] = (
                    seg.end_time, common_labels, seg.freq_low, seg.freq_high
                )
                for code in seg_codes:
                    if ebird_to_common.get(norm_key(code)):
                        species_to_keys[norm_key(code)].add(key)

    print(f'AviaNZ: found {len(all_segments)} unique segments '
          f'covering {len(species_to_keys)} mappable species')

    # -----------------------------------------------------------------------
    # Phase 2: per-species sampling
    # -----------------------------------------------------------------------
    rng = random.Random(seed)
    selected_keys = set()
    for code, keys in species_to_keys.items():
        keys_list = sorted(keys)  # deterministic order before shuffle
        rng.shuffle(keys_list)
        selected_keys.update(keys_list[:max_per_species])

    print(f'AviaNZ: selected {len(selected_keys)} unique segments after per-species cap')

    # -----------------------------------------------------------------------
    # Phase 3: process and save
    # -----------------------------------------------------------------------
    labels = []
    failed = 0
    trimmed = 0

    for key in sorted(selected_keys):
        wav_file, start_time = key
        end_time, label_names, freq_low, freq_high = all_segments[key]

        sg = spec_proc.process_audio_segment(wav_file, start_time, end_time)
        if sg is None:
            failed += 1
            continue

        if fixed_length and sg.shape[1] > target_time_bins:
            sg = trim_spectrogram_to_length(sg, target_time_bins)
            if sg is None:
                failed += 1
                continue
            trimmed += 1

        basename = f'file_{len(labels):08d}'
        spec_proc.save_spectrogram(sg, data_dir, basename)

        if with_audio:
            info = sf.info(wav_file)
            start_frame = int(start_time * info.samplerate)
            stop_frame = int(end_time * info.samplerate)
            seg_data, seg_sr = sf.read(wav_file, start=start_frame, stop=stop_frame)
            sf.write(os.path.join(audio_dir, f'{basename}.wav'), seg_data, seg_sr)

        labels.append({
            'filename': f'{basename}.npy',
            'class_names': label_names,
            'source_file': wav_file,
            'start_time': start_time,
            'end_time': end_time,
        })

        if len(labels) % 500 == 0 and len(labels) > 0:
            print(f'  ... {len(labels)} saved so far')

    print(f'\nAviaNZ large: total {len(labels)} samples '
          f'(failed={failed}, trimmed={trimmed})')
    return labels


# ---------------------------------------------------------------------------
# Combined (DOC + AviaNZ) dataset builder
# ---------------------------------------------------------------------------

def build_combined(
    doc_labels, avianz_labels,
    doc_out, avianz_out,
    combined_out,
    max_per_species=5000,
    seed=42,
):
    """
    Pool DOC and AviaNZ label lists, apply a combined per-species cap, and
    save the result as a single dataset under combined_out/combined_large/.

    Filenames are renumbered to avoid collisions between the two source
    datasets.  Source .npy files are symlinked (falling back to copy) so no
    extra disk space is used.

    Returns (combined_labels, combined_large_path).
    """
    rng = random.Random(seed)

    # --- pool and cap ---
    by_species = defaultdict(list)
    for entry in doc_labels + avianz_labels:
        primary = entry['class_names'][0] if entry.get('class_names') else '__bg__'
        by_species[primary].append(entry)

    capped = []
    for species, entries in sorted(by_species.items()):
        rng.shuffle(entries)
        kept = entries[:max_per_species]
        capped.extend(kept)
        if len(entries) > max_per_species:
            print(f'  {species:<30s}: {len(entries):>6,} → {max_per_species:>6,} (capped)')
        else:
            print(f'  {species:<30s}: {len(entries):>6,}')

    rng.shuffle(capped)

    # --- renumber and symlink ---
    combined_large = os.path.join(combined_out, 'combined_large')
    data_dir = os.path.join(combined_large, 'data')
    os.makedirs(data_dir, exist_ok=True)

    src_data_dirs = [
        os.path.join(doc_out, 'data'),
        os.path.join(avianz_out, 'data'),
    ]

    combined_labels = []
    for i, entry in enumerate(capped):
        new_basename = f'file_{i:08d}'
        old_fname = entry['filename']

        linked = False
        for src_dir in src_data_dirs:
            src_npy = os.path.join(src_dir, old_fname)
            if os.path.exists(src_npy):
                dst_npy = os.path.join(data_dir, f'{new_basename}.npy')
                if not os.path.exists(dst_npy):
                    try:
                        os.symlink(src_npy, dst_npy)
                    except OSError:
                        shutil.copy2(src_npy, dst_npy)
                linked = True
                break

        if not linked:
            print(f'  WARNING: source file not found for {old_fname}, skipping')
            continue

        combined_labels.append({**entry, 'filename': f'{new_basename}.npy'})

    categories = sorted({c for e in combined_labels for c in e.get('class_names', [])})
    payload = {
        'files': combined_labels,
        'categories': categories,
        'num_classes': len(categories),
        'dataset': 'combined_large',
    }
    labels_path = os.path.join(combined_large, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'\nCombined dataset: {len(combined_labels)} samples, '
          f'{len(categories)} classes → {labels_path}')
    return combined_labels, combined_large


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_doc_stratified(labels, test_ratio=0.25, seed=42):
    """
    Stratified random split of DOC data by primary species class.
    Each species gets at least 1 test sample when it has more than 1 total.

    Returns (train_files, test_files).
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for entry in labels:
        primary = entry['class_names'][0] if entry.get('class_names') else '__bg__'
        by_class[primary].append(entry)

    train_files, test_files = [], []
    for cls, entries in sorted(by_class.items()):
        rng.shuffle(entries)
        if len(entries) <= 1:
            train_files.extend(entries)
            continue
        n_test = max(1, round(len(entries) * test_ratio))
        n_test = min(n_test, len(entries) - 1)
        test_files.extend(entries[:n_test])
        train_files.extend(entries[n_test:])

    rng.shuffle(train_files)
    rng.shuffle(test_files)
    return train_files, test_files


def save_split(entries, src_folder, output_base, split_name, categories):
    """
    Save split: write labels.json and symlink (or copy) data/ and audio/ files.

    Args:
        entries       : list of label dicts for this split
        src_folder    : folder that holds data/ and audio/ subdirs
        output_base   : parent folder (e.g. doc_split/)
        split_name    : 'train' or 'test'
        categories    : sorted list of class names for this dataset
    """
    out = os.path.join(output_base, split_name)
    data_out = os.path.join(out, 'data')
    os.makedirs(data_out, exist_ok=True)

    src_data = os.path.join(src_folder, 'data')
    src_audio = os.path.join(src_folder, 'audio')
    has_audio = os.path.isdir(src_audio)
    if has_audio:
        audio_out = os.path.join(out, 'audio')
        os.makedirs(audio_out, exist_ok=True)

    print(f'\n  Linking {len(entries)} {split_name} files ...')
    for i, entry in enumerate(entries):
        fname = entry['filename']
        src_npy = os.path.join(src_data, fname)
        dst_npy = os.path.join(data_out, fname)
        if os.path.exists(src_npy) and not os.path.exists(dst_npy):
            try:
                os.symlink(src_npy, dst_npy)
            except OSError:
                shutil.copy2(src_npy, dst_npy)

        if has_audio:
            wav_name = fname.replace('.npy', '.wav')
            src_w = os.path.join(src_audio, wav_name)
            dst_w = os.path.join(audio_out, wav_name)
            if os.path.exists(src_w) and not os.path.exists(dst_w):
                try:
                    os.symlink(src_w, dst_w)
                except OSError:
                    shutil.copy2(src_w, dst_w)

        if (i + 1) % 1000 == 0:
            print(f'    {i + 1}/{len(entries)}')

    payload = {
        'files': entries,
        'categories': sorted(categories),
        'num_classes': len(categories),
        'dataset': split_name,
    }
    labels_path = os.path.join(out, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'  Saved {labels_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build large (unmatched) DOC + AviaNZ datasets with audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--doc-raw', default=None,
                        help='Raw DOC dataset root (NZBirds folder). Required unless --avianz-only.')
    parser.add_argument('--avianz-raw', default=None, action='append',
                        help='Raw AviaNZ dataset folder. Repeat to include multiple folders. '
                             'Not required when --doc-only.')
    parser.add_argument('--doc-only', action='store_true',
                        help='Build DOC dataset only (skip AviaNZ and skip common-class filtering).')
    parser.add_argument('--avianz-only', action='store_true',
                        help='Build AviaNZ dataset only (skip DOC and common-class filtering). '
                             'Requires --avianz-raw; --doc-raw is not needed.')
    parser.add_argument('--restrict-classes', default=None,
                        help='Comma-separated list of class names to keep (after label-remap). '
                             'Use with --doc-only to restrict to a fixed class vocabulary.')
    parser.add_argument('--label-remap', default=None,
                        help='Comma-separated old:new pairs to rename/merge labels before filtering. '
                             'e.g. "tui:tui/bellbird,bellbird:tui/bellbird,new zealand kaka:kaka"')
    parser.add_argument('--output', required=True,
                        help='Output base directory')
    parser.add_argument('--mapping', default='data/DOC_bird_naming_map.csv',
                        help='Path to DOC_bird_naming_map.csv')
    parser.add_argument('--max-per-species', type=int, default=1000,
                        help='Maximum samples per species per dataset (default: 1000)')
    parser.add_argument('--min-per-class', type=int, default=50,
                        help='Minimum samples per class in BOTH datasets to keep the class '
                             '(default: 50)')
    parser.add_argument('--test-ratio', type=float, default=0.25,
                        help='Fraction of data to use for test (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-audio', action='store_true',
                        help='Skip saving audio files (Kaytoo/BirdNET eval will not work)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-build even if output folders already exist')
    parser.add_argument('--combined-out', default=None,
                        help='If set, merge DOC and AviaNZ labels into a single combined dataset '
                             'saved under <combined-out>/combined_large/ with a per-species cap of '
                             '--max-per-species applied across both sources.  Requires both '
                             '--doc-raw and --avianz-raw.  The Trainer does its own 80/20 split.')
    parser.add_argument('--class-filter', action='store_true',
                        help='Filter to the intersection of DOC and AviaNZ classes '
                             '(only keep species present in both with >= min-per-class samples). '
                             'Default: keep ALL DOC species.')
    parser.add_argument(
        '--spec-type', default='Reassigned',
        choices=['Standard', 'Multi-tapered', 'Reassigned', 'Bandpass'],
        help='Spectrogram type (default: Reassigned — best model setting)',
    )
    parser.add_argument(
        '--window-type', default='Hamming',
        choices=['Hann', 'Hamming', 'Blackman', 'BlackmanHarris'],
        help='Window function (default: Hamming — best model setting)',
    )
    parser.add_argument(
        '--sg-scale', default='Linear',
        choices=['Linear', 'Mel Frequency', 'Bark Frequency'],
        help='Frequency scale (default: Linear — best model setting)',
    )
    args = parser.parse_args()

    if args.avianz_only and args.doc_only:
        parser.error('--avianz-only and --doc-only are mutually exclusive')
    if args.combined_out and (args.doc_only or args.avianz_only):
        parser.error('--combined-out cannot be used with --doc-only or --avianz-only')
    if args.combined_out and (args.doc_raw is None or args.avianz_raw is None):
        parser.error('--combined-out requires both --doc-raw and --avianz-raw')
    if not args.avianz_only and args.doc_raw is None:
        parser.error('--doc-raw is required unless --avianz-only is specified')
    if not args.doc_only and not args.avianz_only and args.avianz_raw is None:
        parser.error('--avianz-raw is required unless --doc-only or --avianz-only is specified')

    with_audio = not args.no_audio
    target_time_bins = config.DEFAULT_TIME_BINS  # 1024

    doc_out = os.path.join(args.output, 'doc_large')
    avianz_out = os.path.join(args.output, 'avianz_large')
    doc_split_base = os.path.join(args.output, 'doc_split')
    avianz_split_base = os.path.join(args.output, 'avianz_split')

    restrict_classes = None
    if args.restrict_classes:
        restrict_classes = [c.strip() for c in args.restrict_classes.split(',') if c.strip()]

    label_remap = None
    if args.label_remap:
        label_remap = {}
        for pair in args.label_remap.split(','):
            old, new = pair.split(':', 1)
            label_remap[old.strip()] = new.strip()
        print(f'  Label remap    : {label_remap}')

    print('=' * 70)
    print(' Build large unmatched datasets')
    print('=' * 70)
    print(f'  DOC raw        : {args.doc_raw}')
    print(f'  AviaNZ raw     : {args.avianz_raw}')
    print(f'  Doc only       : {args.doc_only}')
    print(f'  AviaNZ only    : {args.avianz_only}')
    if restrict_classes:
        print(f'  Restrict to    : {restrict_classes}')
    print(f'  Output         : {args.output}')
    print(f'  Max/species    : {args.max_per_species}')
    print(f'  Min/class      : {args.min_per_class}')
    print(f'  Test ratio     : {args.test_ratio}')
    print(f'  With audio     : {with_audio}')
    print(f'  Spec type      : {args.spec_type}')
    print(f'  Window         : {args.window_type}')
    print(f'  Scale          : {args.sg_scale}')
    print(f'  Overwrite      : {args.overwrite}')
    print('=' * 70)

    # Load name mapping
    ebird_to_common, _common_to_ebird = load_bird_name_mapping(args.mapping)

    # -----------------------------------------------------------------------
    # AviaNZ-only path: build AviaNZ dataset and exit.
    # (No DOC, no common-class filtering — Trainer does its own 80/20 split.)
    # -----------------------------------------------------------------------
    if args.avianz_only:
        if args.overwrite or not os.path.exists(os.path.join(avianz_out, 'labels.json')):
            print('\n=== AviaNZ-only: building AviaNZ large dataset ===')
            os.makedirs(avianz_out, exist_ok=True)
            avianz_labels = build_avianz_large(
                args.avianz_raw, avianz_out, args.mapping, ebird_to_common,
                max_per_species=args.max_per_species,
                seed=args.seed,
                fixed_length=True, target_time_bins=target_time_bins,
                with_audio=with_audio,
                sg_type=args.spec_type,
                window_type=args.window_type,
                sg_scale=args.sg_scale,
            )
        else:
            print('\n=== AviaNZ-only: dataset already exists, loading ===')
            with open(os.path.join(avianz_out, 'labels.json')) as f:
                avianz_labels = json.load(f)['files']

        if restrict_classes:
            restrict_set = set(restrict_classes)
            before = len(avianz_labels)
            avianz_labels = [e for e in avianz_labels
                             if any(c in restrict_set for c in e.get('class_names', []))]
            print(f'\n--restrict-classes: kept {len(avianz_labels)} / {before} samples '
                  f'matching {sorted(restrict_set)}')

        write_labels_json(avianz_out, avianz_labels, 'AviaNZ_large')

        categories = sorted({c for e in avianz_labels for c in e.get('class_names', [])})
        print('\n' + '=' * 70)
        print(' Done (avianz-only mode).')
        print(f'  AviaNZ dataset : {avianz_out}')
        print(f'  Samples        : {len(avianz_labels)}')
        print(f'  Classes ({len(categories)}): {categories}')
        print('=' * 70)
        return

    # -----------------------------------------------------------------------
    # Step 1: build raw DOC dataset
    # -----------------------------------------------------------------------
    if args.overwrite or not os.path.exists(os.path.join(doc_out, 'labels.json')):
        print('\n=== Step 1: build DOC large dataset ===')
        os.makedirs(doc_out, exist_ok=True)
        doc_labels = build_doc_large(
            args.doc_raw, doc_out, ebird_to_common,
            max_per_species=args.max_per_species,
            seed=args.seed,
            fixed_length=True, target_time_bins=target_time_bins,
            with_audio=with_audio,
            sg_type=args.spec_type,
            window_type=args.window_type,
            sg_scale=args.sg_scale,
            restrict_classes=restrict_classes,
            label_remap=label_remap,
        )
        # Write labels.json immediately so it is never missing even if a later
        # step (AviaNZ build, combined merge) is interrupted.
        write_labels_json(doc_out, doc_labels, 'DOC_large')
    else:
        print('\n=== Step 1: DOC large dataset already exists, loading ===')
        with open(os.path.join(doc_out, 'labels.json')) as f:
            doc_labels = json.load(f)['files']

    # -----------------------------------------------------------------------
    # Doc-only path: filter, write labels.json and exit.
    # (No AviaNZ, no train/test split — the Trainer does its own 80/20 split.)
    # -----------------------------------------------------------------------
    if args.doc_only:
        if restrict_classes:
            restrict_set = set(restrict_classes)
            before = len(doc_labels)
            doc_labels = [e for e in doc_labels
                          if any(c in restrict_set for c in e.get('class_names', []))]
            print(f'\n--restrict-classes: kept {len(doc_labels)} / {before} samples '
                  f'matching {sorted(restrict_set)}')

        write_labels_json(doc_out, doc_labels, 'DOC_large')

        categories = sorted({c for e in doc_labels for c in e.get('class_names', [])})
        print('\n' + '=' * 70)
        print(' Done (doc-only mode).')
        print(f'  DOC dataset : {doc_out}')
        print(f'  Samples     : {len(doc_labels)}')
        print(f'  Classes ({len(categories)}): {categories}')
        print('=' * 70)
        return

    # -----------------------------------------------------------------------
    # Step 2: build raw AviaNZ dataset
    # -----------------------------------------------------------------------
    if args.overwrite or not os.path.exists(os.path.join(avianz_out, 'labels.json')):
        print('\n=== Step 2: build AviaNZ large dataset ===')
        os.makedirs(avianz_out, exist_ok=True)
        avianz_labels = build_avianz_large(
            args.avianz_raw, avianz_out, args.mapping, ebird_to_common,  # type: ignore[arg-type]
            max_per_species=args.max_per_species,
            seed=args.seed,
            fixed_length=True, target_time_bins=target_time_bins,
            with_audio=with_audio,
            sg_type=args.spec_type,
            window_type=args.window_type,
            sg_scale=args.sg_scale,
        )
        # Write labels.json immediately so it is never missing if a later step
        # is interrupted.
        write_labels_json(avianz_out, avianz_labels, 'AviaNZ_large')
    else:
        print('\n=== Step 2: AviaNZ large dataset already exists, loading ===')
        with open(os.path.join(avianz_out, 'labels.json')) as f:
            avianz_labels = json.load(f)['files']

    # -----------------------------------------------------------------------
    # Combined-out path: merge DOC + AviaNZ into one dataset and exit.
    # (No domain-split logic — the Trainer does its own 80/20 split.)
    # -----------------------------------------------------------------------
    if args.combined_out:
        os.makedirs(args.combined_out, exist_ok=True)
        combined_large = os.path.join(args.combined_out, 'combined_large')
        if args.overwrite or not os.path.exists(os.path.join(combined_large, 'labels.json')):
            print('\n=== Combined: merging DOC + AviaNZ into single dataset ===')
            combined_labels, combined_large = build_combined(
                doc_labels, avianz_labels,
                doc_out, avianz_out,
                args.combined_out,
                max_per_species=args.max_per_species,
                seed=args.seed,
            )
        else:
            print('\n=== Combined: dataset already exists, skipping ===')
            with open(os.path.join(combined_large, 'labels.json')) as f:
                combined_labels = json.load(f)['files']

        categories = sorted({c for e in combined_labels for c in e.get('class_names', [])})
        print('\n' + '=' * 70)
        print(' Done (combined mode).')
        print(f'  Combined dataset : {combined_large}')
        print(f'  Samples          : {len(combined_labels)}')
        print(f'  Classes ({len(categories)}): {categories}')
        print('=' * 70)
        return

    # -----------------------------------------------------------------------
    # Step 3: (optionally) filter to common classes
    # -----------------------------------------------------------------------

    # Always save the FULL DOC labels before any intersection filtering so that
    # the DOC-only training path (step 7) can use all species.
    full_doc_cats = sorted({c for e in doc_labels for c in e.get('class_names', [])})
    full_labels_path = os.path.join(doc_out, 'labels_all.json')
    with open(full_labels_path, 'w') as f:
        json.dump({
            'files': doc_labels,
            'categories': full_doc_cats,
            'num_classes': len(full_doc_cats),
            'dataset': 'DOC_large_all',
        }, f, indent=2)
    print(f'\nSaved full (unfiltered) DOC labels: {len(doc_labels)} samples, '
          f'{len(full_doc_cats)} classes → {full_labels_path}')

    if args.class_filter:
        print(f'\n=== Step 3: filter to common classes (min {args.min_per_class}/class) ===')
        doc_labels, avianz_labels = filter_to_common_classes(
            doc_labels, avianz_labels,
            min_samples_per_class=args.min_per_class,
        )
        all_categories = set()
        for e in doc_labels + avianz_labels:
            all_categories.update(e.get('class_names', []))
        categories = sorted(all_categories)
        print(f'  Final category set ({len(categories)} classes): {categories}')
        write_labels_json(doc_out, doc_labels, 'DOC_large')
        write_labels_json(avianz_out, avianz_labels, 'AviaNZ_large')
    else:
        print('\n=== Step 3: keeping all DOC classes (default — use --class-filter to restrict) ===')
        print(f'  Keeping all {len(full_doc_cats)} DOC classes for training')
        categories = full_doc_cats
        write_labels_json(doc_out, doc_labels, 'DOC_large')
        write_labels_json(avianz_out, avianz_labels, 'AviaNZ_large')

    # -----------------------------------------------------------------------
    # Step 4: split into train / test
    # -----------------------------------------------------------------------
    print('\n=== Step 4: split datasets ===')

    # AviaNZ: file-level stratified split (prevents data leakage)
    print('\nAviaNZ file-level split ...')
    avianz_train, avianz_test, _ = split_avianz_by_file(
        avianz_labels, args.test_ratio, random_state=args.seed
    )
    print(f'  AviaNZ train: {len(avianz_train)}, test: {len(avianz_test)}')

    # DOC: stratified random split by primary species
    print('\nDOC stratified split ...')
    doc_train, doc_test = split_doc_stratified(
        doc_labels, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f'  DOC train: {len(doc_train)}, test: {len(doc_test)}')

    # -----------------------------------------------------------------------
    # Step 5: save splits
    # -----------------------------------------------------------------------
    print('\n=== Step 5: save splits ===')

    print('\nSaving AviaNZ splits ...')
    save_split(avianz_train, avianz_out, avianz_split_base, 'train', categories)
    save_split(avianz_test, avianz_out, avianz_split_base, 'test', categories)

    print('\nSaving DOC splits ...')
    save_split(doc_train, doc_out, doc_split_base, 'train', categories)
    save_split(doc_test, doc_out, doc_split_base, 'test', categories)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(' Done.')
    print(f'  DOC    raw       : {doc_out}')
    print(f'  AviaNZ raw       : {avianz_out}')
    print(f'  DOC    train/test: {doc_split_base}')
    print(f'  AviaNZ train/test: {avianz_split_base}')
    print(f'  Classes ({len(categories)}): {categories}')
    print('=' * 70)


if __name__ == '__main__':
    main()
