#!/usr/bin/env python3
"""Compare DOC, AviaNZ, and long-recording test spectrograms exactly as training sees them.

This does not reimplement any preprocessing math. It calls the exact same code
the dataset builders and trainer use:
  1. SpectrogramProcessor.process_audio_segment() + save_spectrogram() - identical
     to build_matched_datasets.py / build_large_datasets.py (raw .npy on disk).
  2. SpectrogramDataset.__getitem__() from src/data/data_utils.py - the literal
     class model_trainer.py uses to load training/eval data (log transform,
     background subtraction, padding, orientation - all untouched, none of it
     reimplemented here).

Example:
    python scripts/compare_audio_test_spectrograms.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.src.core import config
from model_testing.src.data.data_utils import SpectrogramDataset
from model_testing.src.data.spectrogram_utils import SpectrogramProcessor


def audio_files(folder):
    extensions = {".wav", ".flac", ".mp3", ".ogg"}
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in extensions)


def make_processor():
    params = config.SPECTROGRAM_PARAMS.copy()
    params["sgType"] = "Standard"
    params["windowType"] = "Hamming"
    params["sgScale"] = "Mel Frequency"
    return SpectrogramProcessor(
        window_seconds=config.DEFAULT_WINDOW_SECONDS,
        hop_seconds=config.DEFAULT_HOP_SECONDS,
        freq_bins=config.DEFAULT_FREQ_BINS,
        fs=config.DEFAULT_SAMPLE_RATE,
        spec_params=params,
    )


def dataset_array(processor, path, start_seconds, duration_seconds, time_bins, bg_subtract, tmp_dir):
    """Run one file through the exact on-disk format + SpectrogramDataset the trainer uses."""
    info = sf.info(str(path))
    start_seconds = max(0.0, min(float(start_seconds), max(0.0, info.duration - 0.01)))
    duration_seconds = min(float(duration_seconds), max(0.01, info.duration - start_seconds))
    sg = processor.process_audio_segment(str(path), start_seconds, start_seconds + duration_seconds)
    if sg is None:
        raise RuntimeError(f"Could not process {path}")

    # Save exactly as the dataset builders do: raw linear-power .npy on disk.
    processor.save_spectrogram(sg, tmp_dir, "sample")
    npy_path = str(Path(tmp_dir) / "sample.npy")

    # Use the literal SpectrogramDataset class the trainer loads data with -
    # no hand-rolled log/normalize/pad here.
    ds = SpectrogramDataset(
        filenames=[npy_path],
        labels=[[1.0]],
        img_height=config.SPECTROGRAM_PARAMS["nfilters"],
        img_width=time_bins,
        channels=1,
        cropping_mode="center",
        noise_filenames=None,
        noise_ratio=0.0,
        spec_transform="Log",
        training=False,
        bg_subtract=bg_subtract,
        apply_reverb=False,
        use_temporal_roll=False,
        background_prob=0.0,
    )
    x, _ = ds[0]
    array = x.numpy()[0]  # (C,H,W) -> (H,W), untouched orientation from the dataset class
    return array, info, start_seconds, duration_seconds


def process_for_model(processor, path, start_seconds, duration_seconds, time_bins, bg_subtract, tmp_dir):
    original, info, actual_start, actual_duration = dataset_array(
        processor, path, start_seconds, duration_seconds, time_bins, False, tmp_dir
    )
    transformed = original
    if bg_subtract:
        transformed, _, _, _ = dataset_array(
            processor, path, start_seconds, duration_seconds, time_bins, True, tmp_dir
        )
    return original, transformed, info, actual_start, actual_duration


def plot_panel(ax, sg, title, info, start_seconds, duration_seconds, vmin=None, vmax=None):
    # Display only: SpectrogramDataset's array has row 0 = highest frequency
    # (proven with pure-tone tests), so flip for a normal low-freq-at-bottom
    # view. The array fed to the model below is never flipped.
    sg = np.flipud(sg)
    finite = sg[np.isfinite(sg)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    if vmin is None:
        vmin = low
    if vmax is None:
        vmax = high
    if vmax <= vmin:
        vmax = vmin + 1.0
    if high <= low:
        high = low + 1.0
    image = ax.imshow(sg, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time frames (10 ms)")
    ax.set_ylabel("Mel bin")
    ax.text(
        0.01,
        0.98,
        f"{info.samplerate / 1000:.1f} kHz source\n"
        f"chunk {start_seconds:.1f}-{start_seconds + duration_seconds:.1f} s",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )
    return image


def main():
    default_root = Path.home() / "Desktop" / "audio_test"
    doc_dir = default_root / "doc"
    avianz_dir = default_root / "avianz_examples"
    test_dir = default_root / "test_examples"
    output_dir = default_root / "processed_spectrogram_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 20
    chunk_seconds = 10.24
    bg_subtract = True

    folders = [("DOC", doc_dir), ("AviaNZ", avianz_dir), ("New test", test_dir)]
    files_by_label = {}
    for label, folder in folders:
        files = audio_files(folder)
        if not files:
            raise FileNotFoundError(f"No audio files found under {folder}")
        files_by_label[label] = files

    image_count = batch_size
    if any(len(files_by_label[label]) == 0 for label, _ in folders):
        raise RuntimeError("No complete DOC/AviaNZ/test comparisons are available")

    processor = make_processor()
    time_bins = config.DEFAULT_TIME_BINS
    print(f"Pipeline: 32 kHz / 64 ms / 10 ms / 224 Mel / Log / "
          f"background_subtract={bg_subtract}; generating {image_count} images")

    tmp_dir = tempfile.mkdtemp(prefix="spec_compare_")
    try:
        for index in range(image_count):
            processed = []
            for label, _ in folders:
                path = files_by_label[label][index % len(files_by_label[label])]
                start = 0.0
                original, transformed, info, actual_start, actual_duration = process_for_model(
                    processor, path, start, chunk_seconds, time_bins, bg_subtract, tmp_dir
                )
                processed.append((label, path, original, transformed, info, actual_start, actual_duration))
                print(f"{index + 1:02d}/{image_count} {label:8s}: {path.name} "
                      f"({info.duration:.1f}s, {info.samplerate} Hz) -> "
                      f"chunk {actual_start:.1f}s, {actual_duration:.1f}s, {transformed.shape}")

            fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
            for column, (label, path, original, transformed, info, actual_start, actual_duration) in enumerate(processed):
                title = f"{label}\n{path.name}"
                plot_panel(axes[0, column], original, title, info, actual_start, actual_duration)
                plot_panel(axes[1, column], transformed, "Background-subtracted", info,
                           actual_start, actual_duration)
            axes[0, 0].set_ylabel("Original\nMel bin")
            axes[1, 0].set_ylabel("Transformed\nMel bin")
            fig.suptitle(f"Original (top) vs background-subtracted (bottom) {index + 1:02d}",
                         fontsize=15, fontweight="bold")
            out = output_dir / f"comparison_{index + 1:02d}.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.close(fig)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved {image_count} images in {output_dir}")


if __name__ == "__main__":
    main()
