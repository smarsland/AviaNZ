#!/usr/bin/env python3
"""
Build a noise dataset (same {labels.json, data/} schema as build_noise_dataset.py)
from a single long dawn-chorus recording, for train.py --noise-folder.

The recording is cut into fixed-length chunks first: the trainer holds out 10%
of the noise files, so a single file would leave no noise for training.

Usage:
    python build_chorus_noise_dataset.py \\
        --input "Sound Files/UK_dawn_chorus.wav" \\
        --output /local/scratch/freangi/noise_dataset/uk_dawn_chorus \\
        --chunk-seconds 30
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_testing.src.data.dataset_builder import NoiseDataProcessor
from model_testing.src.experiments.build_matched_datasets import make_spec_processor


def split_into_chunks(input_file, chunk_seconds, chunk_folder):
    audio, sr = sf.read(input_file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    chunk_len = int(chunk_seconds * sr)
    n_chunks = len(audio) // chunk_len
    for i in range(n_chunks):
        sf.write(os.path.join(chunk_folder, f"chorus_{i:04d}.wav"),
                 audio[i * chunk_len:(i + 1) * chunk_len], sr)
    return n_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk-seconds', type=float, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    chunk_folder = tempfile.mkdtemp(prefix='chorus_chunks_')
    n_chunks = split_into_chunks(args.input, args.chunk_seconds, chunk_folder)
    print(f"Split {args.input} into {n_chunks} chunks of {args.chunk_seconds}s")

    spec_proc = make_spec_processor()
    proc = NoiseDataProcessor(spec_proc, segment_extractor=None, output_format='spectrogram',
                              with_audio=False)
    n_built = proc.process(chunk_folder, args.output, num_samples=n_chunks,
                           overwrite=args.overwrite)
    shutil.rmtree(chunk_folder)
    print(f"Built {n_built} chorus noise spectrograms in {args.output}")


if __name__ == '__main__':
    main()
