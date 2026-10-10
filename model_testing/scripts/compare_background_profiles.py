#!/usr/bin/env python3
"""Compare per-band background levels between the audio_test folders.

Computes mel spectrograms with the exact dataset-builder processor
(SpectrogramProcessor with config.py defaults: 32 kHz, 64 ms Hamming window,
10 ms hop, 224 mel filters, RMS-normalised audio, linear power) and stores
them as .npy files - the same format build_matched_datasets.py writes.
Already-computed .npy files are reused on re-runs.

Two per-folder profile pairs are then plotted (value on x, frequency on y):
  1. mean spectrogram value per frequency band
  2. std of spectrogram values per frequency band
  3. per-band background mean estimated from the bottom 50% of each row,
     exactly as normalizer.split_spectrogram_components computes it (true_mu)
  4. per-band background spread from the same estimate (true_sigma)

Usage:
    python model_testing/scripts/compare_background_profiles.py
"""
import sys
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
from model_testing.src.data.normalizer import split_spectrogram_components
from model_testing.src.data.spectrogram_utils import SpectrogramProcessor

AUDIO_ROOT = Path.home() / "Desktop" / "audio_test"
OUT_ROOT = AUDIO_ROOT / "background_profiles"
FOLDERS = [("DOC", "doc"), ("AviaNZ", "avianz_examples"), ("New test", "test_examples")]


def make_processor():
    return SpectrogramProcessor(
        window_seconds=config.DEFAULT_WINDOW_SECONDS,
        hop_seconds=config.DEFAULT_HOP_SECONDS,
        freq_bins=config.DEFAULT_FREQ_BINS,
        fs=config.DEFAULT_SAMPLE_RATE,
        spec_params=config.SPECTROGRAM_PARAMS.copy(),
    )


def folder_spectrograms(processor, folder, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    spectrograms = []
    rates = {}
    total_seconds = 0.0
    for path in sorted(Path(folder).glob("*.wav")):
        npy_path = out_dir / f"{path.stem}.npy"
        info = sf.info(str(path))
        rates[info.samplerate] = rates.get(info.samplerate, 0) + 1
        total_seconds += info.duration
        if npy_path.exists():
            spectrograms.append(np.load(npy_path))
            continue
        sg = processor.process_audio_file(str(path))
        if sg is None:
            print(f"  {path.name}: processing failed, skipped")
            continue
        processor.save_spectrogram(sg, str(out_dir), path.stem)
        spectrograms.append(np.asarray(sg, dtype=np.float32))
        print(f"  {path.name}: {info.duration:.1f}s -> {sg.shape}")
    return spectrograms, rates, total_seconds


def band_profiles(spectrograms):
    mean_profile = np.mean([sg.mean(axis=1) for sg in spectrograms], axis=0)
    std_profile = np.mean([sg.std(axis=1) for sg in spectrograms], axis=0)
    bg_mu = np.mean(
        [split_spectrogram_components(sg, stochastic=False)["true_mu"][:, 0]
         for sg in spectrograms],
        axis=0,
    )
    bg_sigma = np.mean(
        [split_spectrogram_components(sg, stochastic=False)["true_sigma"][:, 0]
         for sg in spectrograms],
        axis=0,
    )
    return mean_profile, std_profile, bg_mu, bg_sigma


def band_frequencies(processor):
    n = config.DEFAULT_FREQ_BINS
    nyquist = config.DEFAULT_SAMPLE_RATE / 2
    points = np.linspace(processor.sp.convertHztoMel(0),
                         processor.sp.convertHztoMel(nyquist), n + 2)
    centres = processor.sp.convertMeltoHz(points)[1:-1]
    # Saved .npy row 0 is the HIGHEST mel band (np.rot90 in process_audio_file)
    return centres[::-1]


def main():
    processor = make_processor()
    profiles = {}
    for label, folder in FOLDERS:
        print(f"{label} ({folder})")
        spectrograms, rates, total_seconds = folder_spectrograms(
            processor, AUDIO_ROOT / folder, OUT_ROOT / "spectrograms" / folder)
        print(f"  {len(spectrograms)} files, sample rates={rates}, "
              f"total {total_seconds / 60:.1f} min")
        profiles[label] = band_profiles(spectrograms)

    freqs_khz = band_frequencies(processor) / 1000
    fig, axes = plt.subplots(1, 4, figsize=(20, 9), sharey=True)
    titles = ["Mean value per band", "Std of values per band",
              "Background mean (bottom-50% estimate,\nnormalizer true_mu)",
              "Background spread (bottom-50% estimate,\nnormalizer true_sigma)"]
    for label, _ in FOLDERS:
        for ax, profile in zip(axes, profiles[label]):
            ax.plot(profile, freqs_khz, lw=1.5, label=label)
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel("Value (linear power, log scale)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Frequency (kHz)")
    fig.suptitle("audio_test folder comparison - pipeline mel spectrograms "
                 "(32 kHz, 64 ms Hamming, 10 ms hop, 224 mel)", fontsize=12)
    out_path = OUT_ROOT / "background_profiles.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved {out_path}")

    reference = profiles[FOLDERS[0][0]]
    print("\nAll-band averages (linear power), ratio vs DOC:")
    for label, _ in FOLDERS:
        mean_profile, std_profile, bg_mu, bg_sigma = profiles[label]
        print(f"  {label:8s} mean={mean_profile.mean():.6g} "
              f"(x{mean_profile.mean() / reference[0].mean():.3f})  "
              f"std={std_profile.mean():.6g} "
              f"(x{std_profile.mean() / reference[1].mean():.3f})  "
              f"bg_mu={bg_mu.mean():.6g} "
              f"(x{bg_mu.mean() / reference[2].mean():.3f})  "
              f"bg_sigma={bg_sigma.mean():.6g} "
              f"(x{bg_sigma.mean() / reference[3].mean():.3f})")


if __name__ == "__main__":
    main()
