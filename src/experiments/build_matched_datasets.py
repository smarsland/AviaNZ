"""
Build matched DOC and AviaNZ datasets for clean domain-shift testing.

For each of the ~1000 human-reviewed DOC samples (non-poor-quality, with a
resolvable species label), extract a spectrogram from the raw DOC audio and
label it with the human truth (normalized: lowercase, no leading '?').

Then find one AviaNZ segment annotated with the same species (or any of the
ambiguous candidates if the human wrote "Tui/bellbird") and label it with
the same normalized human label.

Result: ~1000 DOC samples and ~1000 AviaNZ samples, matched record-for-record
by human label.  Same number, same species distribution, different domain.

Species 1  = what the human primarily heard.  "/" means uncertainty (pick any).
Species 2+ = other species also present.  Included in the DOC label but NOT
             used as the AviaNZ search key.

Usage:
    python build_matched_datasets.py \\
        --reviewed-csv doc_reviewed.csv \\
        --doc-raw /path/to/raw/NZBirds \\
        --avianz-raw /path/to/raw/Joe_MoDone \\
        --output /path/to/output \\
        --mapping DOC_bird_naming_map.csv
"""

import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd
import soundfile as sf
import numpy as np

from src.core import config
from src.experiments.analyze_dataset_quality import (
    build_group_cache,
    is_poor_quality,
    load_bird_name_mapping,
    normalize_species_name_to_codes,
    parse_species_list_to_codes,
)
from src.data.dataset_builder import AviaNZDataProcessor
from src.data.spectrogram_utils import SpectrogramProcessor, CQTProcessor, apply_freq_mask


def normalize_label(label):
    """
    Normalize human labels to be consistent:
    - Lowercase
    - Strip whitespace
    - Standardize spacing around slashes
    
    Returns None if label should be skipped (uncertain/empty/unknown).
    """
    if not label or pd.isna(label):
        return None
    
    label = str(label).strip()
    
    # Skip uncertain labels (starting with ? or just ?)
    if label.startswith('?') or label == '?':
        return None
    
    # Skip "unknown" classes - these don't exist in AviaNZ data
    if 'unknown' in label.lower():
        return None
    
    # Skip empty after stripping
    if not label:
        return None
    
    # Lowercase
    label = label.lower()
    
    # Normalize spacing around slashes (e.g., "tui / bellbird" -> "tui/bellbird")
    label = label.replace(' / ', '/')
    label = label.replace('/ ', '/')
    label = label.replace(' /', '/')
    
    # Standardize hyphenation in common names
    label = label.replace('long tailed', 'long-tailed')
    
    return label


def get_audio_duration(audio_path):
    """Get audio file duration in seconds without loading entire file."""
    try:
        info = sf.info(audio_path)
        return info.frames / info.samplerate
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return None


def trim_spectrogram_to_length(sg, target_time_bins):
    """
    Trim spectrogram to exactly target_time_bins columns by selecting the region
    with highest energy (where the actual vocalization likely is).
    
    This is much better than center or random cropping for long AviaNZ segments
    where the annotation spans a long period but the actual call is shorter.
    
    Args:
        sg: Spectrogram array of shape (freq_bins, time_bins)
        target_time_bins: Target number of time bins (columns)
        
    Returns:
        Trimmed spectrogram of shape (freq_bins, target_time_bins)
    """
    if sg.shape[1] >= target_time_bins:
        # Calculate energy per time bin (sum over frequency axis)
        energy_per_bin = sg.sum(axis=0)  # Shape: (time_bins,)
        
        # Use sliding window to find highest-energy region
        best_energy = -1
        best_start = 0
        
        for start in range(sg.shape[1] - target_time_bins + 1):
            window_energy = energy_per_bin[start:start + target_time_bins].sum()
            if window_energy > best_energy:
                best_energy = window_energy
                best_start = start
        
        return sg[:, best_start:best_start + target_time_bins]
    else:
        # Should not happen if we filter properly, but just in case
        return None


def load_avianz_name_mapping(mapping_csv):
    """
    Build a CommonName -> eBird dict from DOC_bird_naming_map.csv.
    Includes CommonName, ExtraName, and ListDOCBirds variants.
    """
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


def make_spec_processor(sg_type=None, window_type=None, sg_scale=None):
    if sg_type == 'Bandpass':
        return CQTProcessor(
            n_bins=config.DEFAULT_FREQ_BINS,
            hop_length=int(config.DEFAULT_HOP_SECONDS * config.DEFAULT_SAMPLE_RATE),
            fs=config.DEFAULT_SAMPLE_RATE,
        )
    spec_params = config.SPECTROGRAM_PARAMS.copy()
    if sg_type is not None:
        spec_params['sgType'] = sg_type
    if window_type is not None:
        spec_params['windowType'] = window_type
    if sg_scale is not None:
        spec_params['sgScale'] = sg_scale
    return SpectrogramProcessor(
        window_seconds=config.DEFAULT_WINDOW_SECONDS,
        hop_seconds=config.DEFAULT_HOP_SECONDS,
        freq_bins=config.DEFAULT_FREQ_BINS,
        fs=config.DEFAULT_SAMPLE_RATE,
        spec_params=spec_params,
    )


def parse_reviewed_csv(csv_path, mapping_csv):
    """
    Returns a list of records, one per usable DOC sample.  Each record has:
      species1_raw   : normalized human label string for Species 1 (e.g. "bellbird/tui").
                       Used as the class label in BOTH datasets.
      human_labels   : [species1_raw] + comma-split Species 2+ items (normalized).
      species1_codes : eBird codes resolved from Species 1 — used ONLY as AviaNZ
                       search keys, never stored as class labels.
      folder, predicted_code, audio_filename : for locating the raw audio.
    """
    ebird_to_common, common_to_ebird = load_bird_name_mapping(mapping_csv)
    group_cache = build_group_cache(ebird_to_common)

    df = pd.read_csv(csv_path, header=0)
    col_folder = df.columns[0]
    col_predicted = df.columns[3]
    col_file = df.columns[4]

    records = []
    skipped_poor = 0
    skipped_no_label = 0
    skipped_uncertain = 0

    for _, row in df.iterrows():
        if is_poor_quality(row.get('Note', '')):
            skipped_poor += 1
            continue

        species1_raw = row.get('Species 1', '')
        if pd.isna(species1_raw):
            species1_raw = ''
        species1_raw = species1_raw.strip()

        # Normalize and check if we should skip this label
        species1_normalized = normalize_label(species1_raw)
        if species1_normalized is None:
            skipped_uncertain += 1
            continue

        species2_raw = row.get('Species 2+', '')
        if pd.isna(species2_raw):
            species2_raw = ''

        # eBird codes: ONLY used internally to search AviaNZ for matching segments.
        # "/" means the human was uncertain — we'll accept a segment matching ANY candidate.
        # Use the ORIGINAL (pre-normalized) label to look up eBird codes
        species1_codes = normalize_species_name_to_codes(
            species1_raw, common_to_ebird, ebird_to_common, group_cache
        )
        # If we can't resolve to any code at all we still keep the record for
        # DOC, but AviaNZ matching will fail and the pair will be dropped.

        # Normalize secondary species labels
        species2_items = [normalize_label(s) for s in species2_raw.split(',') if s.strip()] if species2_raw else []
        species2_items = [s for s in species2_items if s is not None]  # Filter out uncertain ones
        human_labels = list(dict.fromkeys([species1_normalized] + species2_items))

        records.append({
            'folder': str(row[col_folder]).strip(),
            'predicted_code': str(row[col_predicted]).strip(),
            'audio_filename': str(row[col_file]).strip(),
            'species1_raw': species1_normalized,       # DOC 'Species 1' column (normalized)
            'human_labels': human_labels,               # full label strings (normalized, no uncertain)
            'species1_codes': species1_codes,           # eBird codes for AviaNZ search only
        })

    print(f'Parsed {len(records)} usable DOC samples from {len(df)} rows')
    print(f'  skipped poor quality : {skipped_poor}')
    print(f'  skipped uncertain    : {skipped_uncertain}')
    print(f'  skipped no label     : {skipped_no_label}')
    return records


def build_doc_dataset(records, doc_raw, output_folder, fixed_length=False, target_time_bins=None, with_audio=False, sg_type=None, window_type=None, sg_scale=None):
    """
    Extract a spectrogram for each record from the raw DOC audio.
    Labels each sample with the full human_codes.
    Returns (labels, records_with_spectrograms) — the latter is needed so
    AviaNZ matching uses only records that actually produced a spectrogram.
    
    Args:
        fixed_length: If True, filter out spectrograms with fewer than target_time_bins and trim to target_time_bins
        target_time_bins: Target number of spectrogram time bins (only used if fixed_length=True)
        with_audio: If True, also save the source audio as a .wav file alongside the spectrogram
        sg_type: Spectrogram type override ('Standard', 'Multi-tapered', or 'Reassigned')
        window_type: Window function override (e.g. 'Hann', 'Hamming', 'Blackman', 'BlackmanHarris')
        sg_scale: Frequency scale override ('Mel Frequency', 'Linear', 'Bark Frequency')
    """
    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    data_dir = os.path.join(output_folder, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(output_folder, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    labels = []
    kept_records = []
    missing = 0
    failed = 0
    too_short = 0
    trimmed = 0

    for i, rec in enumerate(records):
        audio_path = os.path.join(
            doc_raw,
            rec['folder'],
            rec['folder'],
            'train_audio',
            rec['predicted_code'],
            rec['audio_filename'],
        )

        if not os.path.exists(audio_path):
            missing += 1
            continue

        sg = spec_proc.process_audio_file(audio_path)
        if sg is None:
            failed += 1
            continue

        # Trim to fixed length if enabled
        if fixed_length:
            min_bins = 0  # Minimum acceptable time bins
            # Reject if too short
            if sg.shape[1] < min_bins:
                too_short += 1
                continue
            # Trim if too long using energy-based selection
            if sg.shape[1] > target_time_bins:
                sg = trim_spectrogram_to_length(sg, target_time_bins)
                if sg is None:
                    failed += 1
                    continue
                trimmed += 1

        basename = f'file_{len(labels):08d}'
        spec_proc.save_spectrogram(sg, data_dir, basename)

        if with_audio:
            audio_data, audio_sr = sf.read(audio_path)
            sf.write(os.path.join(audio_dir, f'{basename}.wav'), audio_data, audio_sr)

        labels.append({
            'filename': f'{basename}.npy',
            'class_names': rec['human_labels'],
            'source_file': audio_path,
        })
        kept_records.append(rec)

        if (i + 1) % 100 == 0:
            print(f'  DOC: processed {i+1}/{len(records)}, saved {len(labels)}')

    summary = f'DOC: saved {len(labels)} spectrograms  (missing={missing}, failed={failed}, too_short={too_short}'
    if fixed_length:
        summary += f', trimmed={trimmed}'
    summary += ')'
    print(summary)
    return labels, kept_records


def build_avianz_dataset(records, avianz_raw, output_folder, seed, mapping_csv, fixed_length=False, target_time_bins=None, freq_mask=False, with_audio=False, sg_type=None, window_type=None, sg_scale=None):
    """
    For each DOC record, find one AviaNZ segment whose annotation includes
    ANY species from that record's species1_codes (from DOC's 'Species 1' column,
    which may contain uncertain '/' separated options).  If no candidate exists
    for any of those codes, the record is skipped (and the corresponding DOC
    sample should be dropped too).

    Returns (avianz_labels, matched_mask) where matched_mask[i] is True if
    record i was successfully matched.
    
    Args:
        fixed_length: If True, filter out spectrograms with fewer than target_time_bins and trim to target_time_bins
        target_time_bins: Target number of spectrogram time bins (only used if fixed_length=True)
        with_audio: If True, also save the matched audio segment as a .wav file alongside the spectrogram
        sg_type: Spectrogram type override ('Standard', 'Multi-tapered', or 'Reassigned')
        window_type: Window function override (e.g. 'Hann', 'Hamming', 'Blackman', 'BlackmanHarris')
        sg_scale: Frequency scale override ('Mel Frequency', 'Linear', 'Bark Frequency')
    """
    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    name_mapping = load_avianz_name_mapping(mapping_csv)
    proc = AviaNZDataProcessor(name_mapping=name_mapping)

    data_dir = os.path.join(output_folder, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(output_folder, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    # Collect all species codes we'll ever need to search for
    all_search_codes = set()
    for rec in records:
        all_search_codes.update(rec['species1_codes'])

    candidates = defaultdict(list)
    wav_files = proc.find_wav_files(avianz_raw)
    print(f'AviaNZ: scanning {len(wav_files)} wav files...')

    for wav_file in wav_files:
        data_file = wav_file + '.data'
        if not os.path.exists(data_file):
            continue
        segments = proc.load_annotation_file(data_file)
        for seg in segments:
            seg_codes = [
                proc.normalize_to_ebird(lab['species'])
                for lab in seg.labels
                if lab['certainty'] >= 50
            ]
            seg_codes = list(dict.fromkeys(seg_codes))
            for code in seg_codes:
                if code in all_search_codes:
                    candidates[code].append((wav_file, seg.start_time, seg.end_time, seg_codes, seg.freq_low, seg.freq_high))

    rng = random.Random(seed)
    avianz_labels = []
    matched_mask = []
    trimmed = 0

    for rec in records:
        # Pool all AviaNZ candidates for any of this record's Species 1 codes
        pool = []
        for code in rec['species1_codes']:
            pool.extend(candidates.get(code, []))
        # Deduplicate by (wav_file, start_time)
        seen = set()
        deduped = []
        for entry in pool:
            key = (entry[0], entry[1])
            if key not in seen:
                seen.add(key)
                deduped.append(entry)

        if not deduped:
            matched_mask.append(False)
            continue

        # Try candidates until we find one that works
        rng.shuffle(deduped)
        success = False
        
        for wav_file, start, end, seg_codes, freq_low, freq_high in deduped:
            sg = spec_proc.process_audio_segment(wav_file, start, end)
            if sg is None:
                continue

            # Trim to fixed length if enabled
            if fixed_length:
                min_bins = 0  # Minimum acceptable time bins
                # Reject if too short
                if sg.shape[1] < min_bins:
                    continue  # Try next candidate
                # Trim if too long using energy-based selection
                if sg.shape[1] > target_time_bins:
                    sg = trim_spectrogram_to_length(sg, target_time_bins)
                    if sg is None:
                        continue  # Try next candidate
                    trimmed += 1

            if freq_mask:
                sg = apply_freq_mask(sg, freq_low, freq_high, config.DEFAULT_SAMPLE_RATE)

            basename = f'file_{len(avianz_labels):08d}'
            spec_proc.save_spectrogram(sg, data_dir, basename)

            if with_audio:
                info = sf.info(wav_file)
                start_frame = int(start * info.samplerate)
                stop_frame = int(end * info.samplerate)
                seg_data, seg_sr = sf.read(wav_file, start=start_frame, stop=stop_frame)
                sf.write(os.path.join(audio_dir, f'{basename}.wav'), seg_data, seg_sr)

            # Use the DOC human label — both datasets must have identical class names.
            avianz_labels.append({
                'filename': f'{basename}.npy',
                'class_names': rec['human_labels'],
                'source_file': wav_file,
                'start_time': start,
                'end_time': end,
                'freq_low': freq_low,
                'freq_high': freq_high,
            })
            success = True
            break  # Found a good one
        
        matched_mask.append(success)

    matched = sum(matched_mask)
    unmatched = len(records) - matched
    summary = f'AviaNZ: matched {matched} / {len(records)} records  (unmatched={unmatched}'
    if fixed_length:
        summary += f', trimmed={trimmed}'
    summary += ')'
    print(summary)
    return avianz_labels, matched_mask


def filter_to_common_classes(doc_labels, avianz_labels, min_samples_per_class=20):
    """
    Filter both datasets to keep only species that appear with sufficient samples in BOTH.
    This ensures train/test splits will have the same classes available.
    
    With test_ratio=0.25 and min=20, each class gets ~15 train and ~5 test samples.
    This is enough to ensure robust train/test splits with guaranteed class overlap.
    """
    doc_counts = defaultdict(int)
    avianz_counts = defaultdict(int)
    
    for e in doc_labels:
        for cls in e['class_names']:
            doc_counts[cls] += 1
    
    for e in avianz_labels:
        for cls in e['class_names']:
            avianz_counts[cls] += 1
    
    common_classes = set()
    for species in set(doc_counts.keys()) & set(avianz_counts.keys()):
        if doc_counts[species] >= min_samples_per_class and avianz_counts[species] >= min_samples_per_class:
            common_classes.add(species)
    
    # Filter entries to those with at least one common class, AND remove rare classes from class_names
    doc_filtered = []
    for e in doc_labels:
        filtered_classes = [c for c in e['class_names'] if c in common_classes]
        if filtered_classes:
            e_copy = e.copy()
            e_copy['class_names'] = filtered_classes
            doc_filtered.append(e_copy)
    
    avianz_filtered = []
    for e in avianz_labels:
        filtered_classes = [c for c in e['class_names'] if c in common_classes]
        if filtered_classes:
            e_copy = e.copy()
            e_copy['class_names'] = filtered_classes
            avianz_filtered.append(e_copy)
    
    print(f'\n=== Filtering to common classes (min {min_samples_per_class} samples each) ===')
    print(f'  Before: DOC={len(doc_labels)}, AviaNZ={len(avianz_labels)}')
    print(f'  Common classes: {len(common_classes)} species')
    print(f'  After: DOC={len(doc_filtered)}, AviaNZ={len(avianz_filtered)}')
    print(f'  Classes: {sorted(common_classes)}')
    
    removed_doc = set(doc_counts.keys()) - common_classes
    removed_avianz = set(avianz_counts.keys()) - common_classes
    if removed_doc:
        print(f'  Removed from DOC: {removed_doc}')
    if removed_avianz:
        print(f'  Removed from AviaNZ: {removed_avianz}')
    
    return doc_filtered, avianz_filtered


def add_background_samples_doc(doc_labels_final, doc_raw, doc_out, n=1000, seed=42,
                               fixed_length=False, target_time_bins=None, with_audio=False,
                               all_search_codes=None, sg_type=None, window_type=None, sg_scale=None):
    """
    Add up to n background (no positive class) samples from DOC audio files that were
    not included in the matched dataset.  Each sample gets class_names=[].

    Scans doc_raw recursively for all audio files, skips those already used,
    and skips files whose immediate parent directory name is a known eBird code
    (those folders contain target-species recordings and must not be labelled as
    background — doing so produces directly contradictory training signal).
    """
    if n == 0:
        return []

    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    data_dir = os.path.join(doc_out, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(doc_out, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    used_files = {e['source_file'] for e in doc_labels_final}
    species_folders = set(all_search_codes) if all_search_codes else set()

    all_audio = []
    skipped_species_folder = 0
    for root, _, files in os.walk(doc_raw):
        folder_name = os.path.basename(root)
        if folder_name in species_folders:
            skipped_species_folder += len([f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))])
            continue
        for fname in files:
            if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                path = os.path.join(root, fname)
                if path not in used_files:
                    all_audio.append(path)
    if species_folders:
        print(f'DOC background: skipped {skipped_species_folder} files in target-species folders')

    rng = random.Random(seed + 1)
    rng.shuffle(all_audio)

    # Offset so filenames don't clash with existing files on disk
    existing = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    offset = existing

    extra_labels = []
    failed = 0
    for audio_path in all_audio:
        if len(extra_labels) >= n:
            break

        sg = spec_proc.process_audio_file(audio_path)
        if sg is None:
            failed += 1
            continue

        if fixed_length and sg.shape[1] > target_time_bins:
            sg = trim_spectrogram_to_length(sg, target_time_bins)
            if sg is None:
                failed += 1
                continue

        basename = f'file_{offset + len(extra_labels):08d}'
        spec_proc.save_spectrogram(sg, data_dir, basename)

        if with_audio:
            audio_data, audio_sr = sf.read(audio_path)
            sf.write(os.path.join(audio_dir, f'{basename}.wav'), audio_data, audio_sr)

        extra_labels.append({
            'filename': f'{basename}.npy',
            'class_names': [],
            'source_file': audio_path,
        })

    print(f'DOC background: added {len(extra_labels)} samples  '
          f'(candidates={len(all_audio)}, failed={failed})')
    return extra_labels


def add_background_samples_avianz(avianz_labels, avianz_raw, avianz_out, all_search_codes,
                                   mapping_csv, n=1000, seed=42, fixed_length=False,
                                   target_time_bins=None, freq_mask=False, with_audio=False,
                                   sg_type=None, window_type=None, sg_scale=None):
    """
    Add up to n background (no positive class) samples from AviaNZ segments that were
    not included in the matched dataset AND do not contain any matched-species annotation.
    Each sample gets class_names=[].
    """
    if n == 0:
        return []

    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    name_mapping = load_avianz_name_mapping(mapping_csv)
    proc = AviaNZDataProcessor(name_mapping=name_mapping)

    data_dir = os.path.join(avianz_out, 'data')
    os.makedirs(data_dir, exist_ok=True)
    if with_audio:
        audio_dir = os.path.join(avianz_out, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

    # Keys of segments already used in the matched dataset
    used_segments = {(e['source_file'], e.get('start_time')) for e in avianz_labels}

    # Collect candidate segments: annotated, not already used, no matched-species overlap
    background_candidates = []
    wav_files = proc.find_wav_files(avianz_raw)
    print(f'AviaNZ background: scanning {len(wav_files)} wav files...')

    for wav_file in wav_files:
        data_file = wav_file + '.data'
        if not os.path.exists(data_file):
            continue
        try:
            segments = proc.load_annotation_file(data_file)
        except Exception:
            continue
        for seg in segments:
            if (wav_file, seg.start_time) in used_segments:
                continue
            seg_codes = [
                proc.normalize_to_ebird(lab['species'])
                for lab in seg.labels
                if lab['certainty'] >= 50
            ]
            # Skip if the segment contains any species from the matched set
            if any(c in all_search_codes for c in seg_codes):
                continue
            background_candidates.append(
                (wav_file, seg.start_time, seg.end_time, seg.freq_low, seg.freq_high)
            )

    rng = random.Random(seed + 2)
    rng.shuffle(background_candidates)

    existing = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    offset = existing

    extra_labels = []
    failed = 0
    for wav_file, start, end, freq_low, freq_high in background_candidates:
        if len(extra_labels) >= n:
            break

        sg = spec_proc.process_audio_segment(wav_file, start, end)
        if sg is None:
            failed += 1
            continue

        if fixed_length and sg.shape[1] > target_time_bins:
            sg = trim_spectrogram_to_length(sg, target_time_bins)
            if sg is None:
                failed += 1
                continue

        if freq_mask and freq_low is not None and freq_high is not None:
            sg = apply_freq_mask(sg, freq_low, freq_high, config.DEFAULT_SAMPLE_RATE)

        basename = f'file_{offset + len(extra_labels):08d}'
        spec_proc.save_spectrogram(sg, data_dir, basename)

        if with_audio:
            info = sf.info(wav_file)
            start_frame = int(start * info.samplerate)
            stop_frame = int(end * info.samplerate)
            seg_data, seg_sr = sf.read(wav_file, start=start_frame, stop=stop_frame)
            sf.write(os.path.join(audio_dir, f'{basename}.wav'), seg_data, seg_sr)

        extra_labels.append({
            'filename': f'{basename}.npy',
            'class_names': [],
            'source_file': wav_file,
            'start_time': start,
            'end_time': end,
            'freq_low': freq_low,
            'freq_high': freq_high,
        })

    print(f'AviaNZ background: added {len(extra_labels)} samples  '
          f'(candidates={len(background_candidates)}, failed={failed})')
    return extra_labels


def write_labels_json(output_folder, labels, dataset_name):
    species_counts = defaultdict(int)
    all_codes = set()
    for e in labels:
        # Count each species in this multilabel entry
        for species in e['class_names']:
            species_counts[species] += 1
        all_codes.update(e['class_names'])

    payload = {
        'files': labels,
        'categories': sorted(all_codes),
        'num_classes': len(all_codes),
        'dataset': dataset_name,
        'species_counts': dict(species_counts),
    }
    out = os.path.join(output_folder, 'labels.json')
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Wrote {len(labels)} entries -> {out}')


def compute_spectrogram_stats(output_folder, dataset_name):
    """
    Compute statistics about spectrograms in the dataset.
    
    Returns dict with:
        - num_files: number of spectrogram files
        - freq_bins: frequency bins (height) - should be constant
        - total_time_bins: sum of all time bins (columns)
        - min_time_bins: minimum time bins
        - max_time_bins: maximum time bins
        - mean_time_bins: mean time bins
    """
    data_dir = os.path.join(output_folder, 'data')
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not npy_files:
        return {
            'dataset': dataset_name,
            'num_files': 0,
            'freq_bins': 0,
            'total_time_bins': 0,
            'min_time_bins': 0,
            'max_time_bins': 0,
            'mean_time_bins': 0,
        }
    
    time_bins_list = []
    freq_bins = None
    
    for npy_file in npy_files:
        sg = np.load(os.path.join(data_dir, npy_file))
        if freq_bins is None:
            freq_bins = sg.shape[0]  # height
        time_bins_list.append(sg.shape[1])  # width (columns)
    
    stats = {
        'dataset': dataset_name,
        'num_files': len(npy_files),
        'freq_bins': freq_bins,
        'total_time_bins': sum(time_bins_list),
        'min_time_bins': min(time_bins_list),
        'max_time_bins': max(time_bins_list),
        'mean_time_bins': sum(time_bins_list) / len(time_bins_list),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Build matched DOC + AviaNZ datasets')
    parser.add_argument('--reviewed-csv', default='data/doc_reviewed.csv')
    parser.add_argument('--doc-raw', required=True,
                        help='Raw DOC dataset root (NZBirds folder)')
    parser.add_argument('--avianz-raw', required=True,
                        help='Raw AviaNZ dataset root (Joe_MoDone folder)')
    parser.add_argument('--output', required=True,
                        help='Output base; doc_matched/ and avianz_matched/ created inside')
    parser.add_argument('--mapping', default='data/DOC_bird_naming_map.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fixed-length', action='store_true',
                        help='Filter out files shorter than model input size and trim all to same length')
    parser.add_argument('--freq-mask', action='store_true',
                        help='Zero out AviaNZ spectrogram values outside the annotated frequency limits')
    parser.add_argument('--with-audio', action='store_true',
                        help='Also save audio files alongside spectrograms (required for Kaytoo and BirdNET evaluation)')
    parser.add_argument('--background-n', type=int, default=1000,
                        help='Number of background (no positive class) samples to add to each dataset '
                             'from rejected/unmatched files. Set to 0 to disable. (default: 1000)')
    parser.add_argument('--spec-type', default=None,
                        choices=['Standard', 'Multi-tapered', 'Reassigned', 'Bandpass'],
                        help='Spectrogram type to use (default: Standard, from config). '
                             'Bandpass uses a Constant-Q Transform (log-frequency filterbank).')
    parser.add_argument('--window-type', default=None,
                        choices=['Hann', 'Hamming', 'Blackman', 'BlackmanHarris'],
                        help='Window function to use (default: Hann, from config)')
    parser.add_argument('--sg-scale', default=None,
                        choices=['Linear', 'Mel Frequency', 'Bark Frequency'],
                        help='Frequency scale to use (default: Mel Frequency, from config)')
    args = parser.parse_args()

    doc_out = os.path.join(args.output, 'doc_matched')
    avianz_out = os.path.join(args.output, 'avianz_matched')
    os.makedirs(doc_out, exist_ok=True)
    os.makedirs(avianz_out, exist_ok=True)

    # Compute target duration and time bins based on model config
    target_duration = config.DEFAULT_TIME_BINS * config.DEFAULT_HOP_SECONDS
    target_time_bins = config.DEFAULT_TIME_BINS
    
    if args.fixed_length:
        min_bins = 0
        print(f'\n=== Fixed-length mode enabled ===')
        print(f'  Target time bins: {target_time_bins}')
        print(f'  Minimum time bins: {min_bins}')
        print(f'  Spectrograms with fewer than {min_bins} bins will be filtered out')
        print(f'  Spectrograms longer than {target_time_bins} bins will be trimmed')
        print(f'  Spectrograms between {min_bins}-{target_time_bins} bins will be kept as-is')
        print('='*50 + '\n')

    print('=== Step 1: parse reviewed CSV ===')
    records = parse_reviewed_csv(args.reviewed_csv, args.mapping)

    if args.with_audio:
        print('\n=== Audio saving enabled: .wav files will be saved alongside spectrograms ===')
    if args.spec_type is not None:
        print(f'\n=== Spectrogram type override: {args.spec_type} ===')

    print('\n=== Step 2: build DOC dataset (human labels, raw audio) ===')
    doc_labels, kept_records = build_doc_dataset(
        records, args.doc_raw, doc_out, 
        fixed_length=args.fixed_length,
        target_time_bins=target_time_bins,
        with_audio=args.with_audio,
        sg_type=args.spec_type,
        window_type=args.window_type,
        sg_scale=args.sg_scale,
    )

    print('\n=== Step 3: find matching AviaNZ sample for each DOC record ===')
    if args.freq_mask:
        print('  Frequency masking enabled: zeroing spectrogram values outside annotated limits')
    avianz_labels, matched_mask = build_avianz_dataset(
        kept_records, args.avianz_raw, avianz_out, args.seed, args.mapping,
        fixed_length=args.fixed_length,
        target_time_bins=target_time_bins,
        freq_mask=args.freq_mask,
        with_audio=args.with_audio,
        sg_type=args.spec_type,
        window_type=args.window_type,
        sg_scale=args.sg_scale,
    )

    # Drop DOC samples that had no AviaNZ match
    doc_labels_final = [l for l, m in zip(doc_labels, matched_mask) if m]

    print(f'\nMatched dataset size: {len(doc_labels_final)} DOC  /  {len(avianz_labels)} AviaNZ')

    # Filter to common classes present in both datasets with sufficient samples
    # With test_ratio=0.25, min=20 gives ~15 train, ~5 test per class
    # This ensures robust splits even with multilabel (where splitting by first class may miss some labels)
    doc_labels_final, avianz_labels = filter_to_common_classes(doc_labels_final, avianz_labels, min_samples_per_class=20)

    # Add background (no positive class) samples from rejected/unmatched files
    if args.background_n > 0:
        # Compute all species codes that were searched for (needed to exclude matched-species segments)
        all_search_codes = set()
        for rec in kept_records:
            all_search_codes.update(rec['species1_codes'])

        print(f'\n=== Step 4b: adding background samples (doc={args.background_n}, avianz={args.background_n}) ===')
        doc_background = add_background_samples_doc(
            doc_labels_final, args.doc_raw, doc_out,
            n=args.background_n, seed=args.seed,
            fixed_length=args.fixed_length,
            target_time_bins=target_time_bins,
            with_audio=args.with_audio,
            all_search_codes=all_search_codes,
            sg_type=args.spec_type,
            window_type=args.window_type,
            sg_scale=args.sg_scale,
        )
        doc_labels_final = doc_labels_final + doc_background

        avianz_background = add_background_samples_avianz(
            avianz_labels, args.avianz_raw, avianz_out,
            all_search_codes=all_search_codes,
            mapping_csv=args.mapping,
            n=args.background_n, seed=args.seed,
            fixed_length=args.fixed_length,
            target_time_bins=target_time_bins,
            freq_mask=args.freq_mask,
            with_audio=args.with_audio,
            sg_type=args.spec_type,
            window_type=args.window_type,
            sg_scale=args.sg_scale,
        )
        avianz_labels = avianz_labels + avianz_background

    print('\n=== Step 4: write labels.json ===')
    write_labels_json(doc_out, doc_labels_final, 'DOC_matched')
    write_labels_json(avianz_out, avianz_labels, 'AviaNZ_matched')

    # Compute and report spectrogram statistics
    print('\n=== Step 5: compute spectrogram statistics ===')
    doc_stats = compute_spectrogram_stats(doc_out, 'DOC_matched')
    avianz_stats = compute_spectrogram_stats(avianz_out, 'AviaNZ_matched')
    
    print(f'\nDOC dataset:')
    print(f'  Files: {doc_stats["num_files"]}')
    print(f'  Shape: ({doc_stats["freq_bins"]}, {doc_stats["min_time_bins"]}-{doc_stats["max_time_bins"]})')
    print(f'  Time bins: min={doc_stats["min_time_bins"]}, max={doc_stats["max_time_bins"]}, mean={doc_stats["mean_time_bins"]:.1f}')
    print(f'  Total time bins (columns): {doc_stats["total_time_bins"]}')
    
    print(f'\nAviaNZ dataset:')
    print(f'  Files: {avianz_stats["num_files"]}')
    print(f'  Shape: ({avianz_stats["freq_bins"]}, {avianz_stats["min_time_bins"]}-{avianz_stats["max_time_bins"]})')
    print(f'  Time bins: min={avianz_stats["min_time_bins"]}, max={avianz_stats["max_time_bins"]}, mean={avianz_stats["mean_time_bins"]:.1f}')
    print(f'  Total time bins (columns): {avianz_stats["total_time_bins"]}')
    
    # Save stats to JSON
    stats_file = os.path.join(args.output, 'dataset_stats.json')
    stats_output = {
        'doc': doc_stats,
        'avianz': avianz_stats,
        'fixed_length': args.fixed_length,
        'freq_mask': args.freq_mask,
    }
    if args.fixed_length:
        stats_output['target_time_bins'] = target_time_bins
    
    with open(stats_file, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f'\n✓ Saved statistics to: {stats_file}')

    print(f'\nDone.\n  {doc_out}\n  {avianz_out}')


if __name__ == '__main__':
    main()
