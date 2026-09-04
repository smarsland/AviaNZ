#!/usr/bin/env python3
"""
Build a noise dataset for the --noise-folder / --noise mixing augmentation
(see model_testing/train.py and src/data/data_utils.py::_mix_noise_2d).

Combines two sources of "not a bird call" audio:

1. Environmental noise (wind, rain, etc.) from a freefield-style archive of
   zipped recordings, e.g. the VUW freefield share:
       /media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/freefield
   Zip files are extracted automatically (NoiseDataProcessor handles this).

2. AviaNZ "true silence" background: audio taken from the GAPS between
   annotated segments in reviewed AviaNZ recordings (i.e. time ranges with
   NO annotation at all - not just an annotation for a non-target species).
   This captures the actual field-recording noise floor/microphone/wind
   characteristics of the AviaNZ deployments, which is the real source of
   domain shift we want the model to become invariant to.

Output: a single merged noise folder with the same {labels.json, data/}
schema as NoiseDataProcessor, suitable for `train.py --noise-folder`.

Usage:
    python build_noise_dataset.py \\
        --freefield-dir /media/.../freefield \\
        --avianz-raw /media/.../AviaNZ_drive1 --avianz-raw /media/.../AviaNZ_drive2 \\
        --output /local/scratch/freangi/noise_dataset \\
        --num-environmental 2000 --num-avianz-background 2000
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.src.core import config
from model_testing.src.data.dataset_builder import AviaNZDataProcessor, NoiseDataProcessor
from model_testing.src.experiments.build_matched_datasets import make_spec_processor


def build_environmental_noise(freefield_dir, output_folder, num_samples, sg_type, window_type,
                               sg_scale, with_audio, overwrite):
    """Extract noise spectrograms from a (possibly zipped) freefield archive."""
    print(f"\n=== Environmental noise: {freefield_dir} ===")
    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    proc = NoiseDataProcessor(spec_proc, segment_extractor=None, output_format='spectrogram',
                               with_audio=with_audio)
    return proc.process(freefield_dir, output_folder, num_samples=num_samples, overwrite=overwrite)


def find_annotation_gaps(segments, duration, min_gap_seconds):
    """Return [(start, end), ...] time ranges in [0, duration] with NO annotation at all."""
    if not segments:
        return [(0.0, duration)] if duration >= min_gap_seconds else []

    intervals = sorted((s.start_time, s.end_time) for s in segments)
    gaps = []
    cursor = 0.0
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))

    return [(s, e) for s, e in gaps if e - s >= min_gap_seconds]


def build_avianz_background_noise(avianz_raw_folders, output_folder, num_samples, clip_seconds,
                                   min_gap_seconds, sg_type, window_type, sg_scale, with_audio,
                                   overwrite, seed=42):
    """Extract noise spectrograms from unannotated gaps in reviewed AviaNZ recordings."""
    print(f"\n=== AviaNZ background noise (unannotated gaps): {avianz_raw_folders} ===")
    spec_proc = make_spec_processor(sg_type, window_type, sg_scale)
    proc = AviaNZDataProcessor()

    candidates = []  # (wav_file, start, end)
    for folder in avianz_raw_folders:
        wav_files = proc.find_wav_files(folder)
        print(f"  Scanning {len(wav_files)} wav files in {folder}...")
        for wav_file in wav_files:
            data_file = wav_file + '.data'
            if not os.path.exists(data_file):
                continue
            try:
                segments = proc.load_annotation_file(data_file)
                duration = sf.info(wav_file).frames / sf.info(wav_file).samplerate
            except Exception:
                continue
            for gap_start, gap_end in find_annotation_gaps(segments, duration, min_gap_seconds):
                # One candidate clip per gap (sampled uniformly within the gap)
                candidates.append((wav_file, gap_start, gap_end))

    print(f"  Found {len(candidates)} unannotated gaps >= {min_gap_seconds}s")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if os.path.exists(output_folder):
        if overwrite:
            shutil.rmtree(output_folder)
        else:
            print(f"  Output folder {output_folder} already exists, skipping (use --overwrite)")
            return 0

    data_dir = os.path.join(output_folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    audio_dir = None
    if with_audio:
        audio_dir = os.path.join(output_folder, "audio")
        os.makedirs(audio_dir, exist_ok=True)

    file_labels = []
    failed = 0
    for wav_file, gap_start, gap_end in candidates:
        if len(file_labels) >= num_samples:
            break

        gap_len = gap_end - gap_start
        clip_dur = min(clip_seconds, gap_len)
        max_offset = gap_len - clip_dur
        start = gap_start + (rng.uniform(0, max_offset) if max_offset > 0 else 0.0)
        end = start + clip_dur

        sg = spec_proc.process_audio_segment(wav_file, start, end)
        if sg is None:
            failed += 1
            continue

        basename = f'avianz_bg_{len(file_labels):06d}'
        spec_proc.save_spectrogram(sg, data_dir, basename)

        audio_filename = None
        if with_audio:
            try:
                info = sf.info(wav_file)
                start_frame = int(start * info.samplerate)
                stop_frame = int(end * info.samplerate)
                seg_data, seg_sr = sf.read(wav_file, start=start_frame, frames=stop_frame - start_frame)
                sf.write(os.path.join(audio_dir, f'{basename}.wav'), seg_data, seg_sr)
                audio_filename = f'{basename}.wav'
            except Exception as e:
                print(f"  Warning: could not save audio for {basename}: {e}")

        label_entry = {'filename': f'{basename}.npy', 'source_file': wav_file,
                        'start_time': start, 'end_time': end}
        if audio_filename:
            label_entry['audio_file'] = audio_filename
        file_labels.append(label_entry)

    labels_data = {'files': file_labels, 'num_files': len(file_labels), 'source_type': 'noise'}
    with open(os.path.join(output_folder, "labels.json"), 'w') as f:
        json.dump(labels_data, f, indent=2)

    print(f"  Saved {len(file_labels)} AviaNZ background-gap noise clips "
          f"(candidates={len(candidates)}, failed={failed})")
    return len(file_labels)


def merge_noise_folders(folders, output_folder, symlink=True):
    """Merge several noise-schema folders ({labels.json, data/[, audio/]}) into one."""
    print(f"\n=== Merging {len(folders)} noise folders -> {output_folder} ===")
    data_dir = os.path.join(output_folder, "data")
    os.makedirs(data_dir, exist_ok=True)

    all_files = []
    for folder in folders:
        labels_path = os.path.join(folder, "labels.json")
        if not os.path.exists(labels_path):
            print(f"  Skipping {folder} (no labels.json)")
            continue
        with open(labels_path) as f:
            labels = json.load(f)
        src_data_dir = os.path.join(folder, "data")
        for entry in labels['files']:
            src_path = os.path.join(src_data_dir, entry['filename'])
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(data_dir, entry['filename'])
            if os.path.exists(dst_path):
                continue  # already merged (filenames are prefixed uniquely per source)
            if symlink:
                os.symlink(os.path.abspath(src_path), dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            all_files.append({'filename': entry['filename'], 'source_file': entry.get('source_file')})
        print(f"  {folder}: {len(labels['files'])} files")

    with open(os.path.join(output_folder, "labels.json"), 'w') as f:
        json.dump({'files': all_files, 'num_files': len(all_files), 'source_type': 'noise'}, f, indent=2)

    print(f"  Merged noise dataset: {len(all_files)} files -> {output_folder}")
    return len(all_files)


def main():
    parser = argparse.ArgumentParser(
        description="Build a merged noise dataset (freefield environmental + AviaNZ "
                     "unannotated-gap background) for --noise-folder mixing augmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument('--freefield-dir', default=None,
                        help="Freefield noise archive (dir of .zip files or extracted audio). Optional.")
    parser.add_argument('--avianz-raw', action='append', default=None,
                        help="Raw AviaNZ recordings folder (with .wav + .wav.data files). "
                             "Repeat to include several folders. Optional.")
    parser.add_argument('--output', required=True, help="Output base directory")
    parser.add_argument('--num-environmental', type=int, default=2000,
                        help="Max environmental (freefield) noise clips (default: 2000)")
    parser.add_argument('--num-avianz-background', type=int, default=2000,
                        help="Max AviaNZ unannotated-gap noise clips (default: 2000)")
    parser.add_argument('--min-gap-seconds', type=float, default=3.0,
                        help="Minimum unannotated gap length to sample from (default: 3.0)")
    parser.add_argument('--clip-seconds', type=float, default=5.0,
                        help="Duration of each extracted AviaNZ background clip (default: 5.0)")
    parser.add_argument('--spec-type', default='Standard')
    parser.add_argument('--window-type', default='Hamming')
    parser.add_argument('--sg-scale', default='Mel Frequency')
    parser.add_argument('--with-audio', action='store_true')
    parser.add_argument('--symlink', action='store_true', default=True,
                        help="Symlink files into the merged folder instead of copying (default: True)")
    parser.add_argument('--copy', dest='symlink', action='store_false',
                        help="Copy files into the merged folder instead of symlinking")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not args.freefield_dir and not args.avianz_raw:
        parser.error("Provide at least one of --freefield-dir or --avianz-raw")

    os.makedirs(args.output, exist_ok=True)
    sources = []

    if args.freefield_dir:
        env_out = os.path.join(args.output, 'noise_environmental')
        build_environmental_noise(args.freefield_dir, env_out, args.num_environmental,
                                   args.spec_type, args.window_type, args.sg_scale,
                                   args.with_audio, args.overwrite)
        sources.append(env_out)

    if args.avianz_raw:
        avianz_out = os.path.join(args.output, 'noise_avianz_background')
        build_avianz_background_noise(args.avianz_raw, avianz_out, args.num_avianz_background,
                                       args.clip_seconds, args.min_gap_seconds, args.spec_type,
                                       args.window_type, args.sg_scale, args.with_audio,
                                       args.overwrite, seed=args.seed)
        sources.append(avianz_out)

    combined_out = os.path.join(args.output, 'noise_combined')
    merge_noise_folders(sources, combined_out, symlink=args.symlink)

    print(f"\nDone. Pass this to train.py:")
    print(f"  --noise-folder {combined_out} --noise 0.2")


if __name__ == "__main__":
    main()
