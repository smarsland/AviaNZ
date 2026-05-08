#!/usr/bin/env python3
"""Generate spectrogram images for all reasonable parameter combinations.

Usage:
    python scripts/visualize_spectrograms.py path/to/sound.wav
    python scripts/visualize_spectrograms.py path/to/sound.wav --output_dir my_images
"""

import sys
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the project root is on the path regardless of where the script is invoked from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.spectrogram import Spectrogram

# ── Fixed window / hop parameters ──────────────────────────────────────────
WINDOW_WIDTH = 256
INCR = 128
NFILTERS = 128

# ── Parameter space ─────────────────────────────────────────────────────────
# Spectrogram types
SG_TYPES = ['Standard', 'Reassigned', 'Multi-tapered']

# Windows to vary — only meaningful for Standard and Reassigned
WINDOWS = ['Hann', 'Hamming', 'Blackman', 'BlackmanHarris']

# Frequency scales
SG_SCALES = [
    ('Linear',        'linear'),
    ('Mel Frequency', 'mel'),
]

# Normalisations (Sigmoid has a TODO in the source, so it is excluded)
NORMALIZATIONS = [
    ('Log',     'log'),
    ('PCEN',    'pcen'),
    ('Box-Cox', 'boxcox'),
]


def save_spectrogram_image(sg, title, output_path):
    """Save a 2-D spectrogram array as a PNG image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(sg.T, aspect='auto', origin='lower', cmap='inferno')
    ax.set_title(title, fontsize=8, wrap=True)
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Frequency bins')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_all(sound_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    errors = []

    for sg_type in SG_TYPES:
        # Multi-tapered uses its own internal tapers — no user-facing window choice
        window_list = WINDOWS if sg_type in ('Standard', 'Reassigned') else [None]

        for window in window_list:
            for scale_label, scale_slug in SG_SCALES:
                for norm_label, norm_slug in NORMALIZATIONS:

                    # Build a safe, descriptive filename
                    type_slug  = sg_type.lower().replace('-', '').replace(' ', '_')
                    win_slug   = window.lower() if window else 'default'
                    filename   = f"{type_slug}_{win_slug}_{scale_slug}_{norm_slug}.png"
                    output_path = os.path.join(output_dir, filename)

                    try:
                        sp = Spectrogram(window_width=WINDOW_WIDTH, incr=INCR)
                        sp.readSoundFile(sound_file, silent=True)

                        sp.spectrogram(
                            window_width=WINDOW_WIDTH,
                            incr=INCR,
                            window=window if window else 'Hann',
                            sgType=sg_type,
                            sgScale=scale_label,
                            nfilters=NFILTERS,
                        )

                        sg_norm = sp.normalisedSpec(tr=norm_label)

                        if sg_type == 'Multi-tapered':
                            title = f"{sg_type} | {scale_label} | {norm_label}"
                        else:
                            title = f"{sg_type} ({window}) | {scale_label} | {norm_label}"

                        save_spectrogram_image(sg_norm, title, output_path)
                        count += 1
                        print(f"  [{count:3d}] {filename}")

                    except Exception as exc:
                        msg = f"{filename}: {exc}"
                        errors.append(msg)
                        print(f"  [ERR] {msg}")

    print(f"\nSaved {count} spectrograms to '{output_dir}'")
    if errors:
        print(f"{len(errors)} failed:")
        for e in errors:
            print(f"  {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate spectrogram images for all parameter combinations'
    )
    parser.add_argument('sound_file', help='Path to the input sound file')
    parser.add_argument(
        '--output_dir', default='spectrogram_images',
        help='Directory in which to save the images (default: spectrogram_images)'
    )
    args = parser.parse_args()

    sound_file = os.path.abspath(args.sound_file)
    if not os.path.isfile(sound_file):
        print(f"ERROR: File not found: {sound_file}", file=sys.stderr)
        sys.exit(1)

    generate_all(sound_file, args.output_dir)


if __name__ == '__main__':
    main()
