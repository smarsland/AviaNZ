import argparse
import json
import os
import shutil

import config
from spectrogram_utils import SpectrogramProcessor, AudioSetFbankProcessor


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        metadata = json.load(f)
    if 'files' not in metadata:
        raise ValueError("labels.json missing 'files' list")
    return metadata['files']


def resolve_audio_path(output_folder, label):
    audio_folder = os.path.join(output_folder, "audio")
    if 'audio_file' in label:
        return os.path.join(audio_folder, label['audio_file'])

    filename = label.get('filename', '')
    base, ext = os.path.splitext(filename)
    if ext == '.npy':
        audio_name = f"{base}.wav"
    else:
        audio_name = f"{filename}.wav"
    return os.path.join(audio_folder, audio_name)


def build_processor(args):
    if args.audioset_fbank:
        return AudioSetFbankProcessor(
            target_sample_rate=16000,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            num_mel_bins=128,
        )

    window = args.window if args.window is not None else config.DEFAULT_WINDOW_SECONDS
    hop = args.hop if args.hop is not None else config.DEFAULT_HOP_SECONDS
    freq_bins = args.freq_bins if args.freq_bins is not None else config.SPECTROGRAM_PARAMS['nfilters']
    fs = args.fs if args.fs is not None else config.DEFAULT_SAMPLE_RATE

    return SpectrogramProcessor(
        window,
        hop,
        freq_bins,
        fs,
        config.SPECTROGRAM_PARAMS,
    )


def rebuild_spectrograms(args):
    labels_path = os.path.join(args.output_folder, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found at {labels_path}")

    labels = load_labels(labels_path)

    data_folder = os.path.join(args.output_folder, "data")
    if os.path.exists(data_folder):
        if not args.overwrite:
            raise FileExistsError(
                f"Data folder already exists: {data_folder}. Use --overwrite to replace it."
            )
        shutil.rmtree(data_folder)

    os.makedirs(data_folder, exist_ok=True)

    processor = build_processor(args)

    for idx, label in enumerate(labels, start=1):
        audio_path = resolve_audio_path(args.output_folder, label)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if 'start_time' in label and 'end_time' in label:
            sg_raw = processor.process_audio_segment(
                audio_path,
                float(label['start_time']),
                float(label['end_time']),
            )
        else:
            sg_raw = processor.process_audio_file(audio_path)

        if sg_raw is None:
            raise ValueError(f"Failed to build spectrogram for {audio_path}")

        filename = label.get('filename')
        if not filename:
            raise ValueError("Label entry missing 'filename'")

        base_name = os.path.splitext(filename)[0]
        processor.save_spectrogram(sg_raw, data_folder, base_name)

        if idx % 50 == 0 or idx in [1, 10]:
            print(f"Rebuilt {idx}/{len(labels)} spectrograms")

    print(f"Rebuilt {len(labels)} spectrograms in {data_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild spectrograms from audio/ and labels.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rebuild_spectrograms.py "Sound Files/GSK_spec" \
    --window 0.025 --hop 0.010 --freq-bins 128 --fs 48000 --overwrite

  python rebuild_spectrograms.py "Sound Files/GSK_spec" \
    --audioset-fbank --overwrite
        """,
    )

    parser.add_argument(
        'output_folder',
        type=str,
        help="Folder that contains labels.json and audio/",
    )
    parser.add_argument(
        '--window',
        type=float,
        help="Window width in seconds",
    )
    parser.add_argument(
        '--hop',
        type=float,
        help="Hop length in seconds",
    )
    parser.add_argument(
        '--freq-bins',
        type=int,
        help="Number of frequency bins",
    )
    parser.add_argument(
        '--fs',
        type=int,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        '--audioset-fbank',
        action='store_true',
        help="Use AudioSet-style Kaldi fbank features (fixed params)",
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing data/ folder",
    )

    rebuild_spectrograms(parser.parse_args())
