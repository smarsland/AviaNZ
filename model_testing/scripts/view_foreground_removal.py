#!/usr/bin/env python3
"""Interactive viewer: flip through matched AviaNZ/DOC spectrograms with foreground removal.

Shows ONE example at a time. Press Right/Left to step forward/backward through the
interleaved sequence AviaNZ 0 -> DOC 0 -> AviaNZ 1 -> DOC 1 -> ...  For each example
the top panel is the original spectrogram and the bottom panel is the same clip after
foreground removal - the exact transform used by train.py's --background-prob
augmentation (normalizer.get_background_spectrogram - the single canonical
implementation, re-exported through data_utils): per frequency row the background
is estimated from the middle 80% of pixels, and any pixel more than 4 sigma from it
- the actual call energy - is replaced with resampled background pixels from that
same row, leaving only the ambient noise floor.

Like compare_foreground_removal_spectrograms.py, this reimplements no preprocessing
math. It loads the raw linear-power .npy files that build_matched_datasets.py already
saved, then runs them through the literal SpectrogramDataset (model_trainer.py's data
class) for the Log transform, at each file's natural width so padding/cropping are
no-ops.

Controls:
    Right / Left : next / previous example (AviaNZ n <-> DOC n interleaved)
    Prev / Next buttons at the bottom do the same with the mouse
    Q / Escape   : quit

The x-axis is always a fixed 10-second window (columns are 10 ms), so short
clips leave blank space on the right and durations are directly comparable.

Example:
    python scripts/view_foreground_removal.py
    python scripts/view_foreground_removal.py --start 10
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# The default navigation toolbar shortcuts bind 'left'/'right' to back/forward
# view navigation (rcParams['keymap.back'/'forward']) and can swallow the arrow
# keys before user handlers see them. Remove those bindings so the keys reliably
# reach our key_press handler.
for _keymap_name, _arrow in (("keymap.back", "left"), ("keymap.forward", "right")):
    _keys = mpl.rcParams.get(_keymap_name, [])
    if _arrow in _keys:
        _keys.remove(_arrow)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.src.data.data_utils import SpectrogramDataset
from model_testing.src.data.normalizer import get_background_spectrogram, swap_foregrounds

DEFAULT_ROOT = REPO_ROOT / "Sound Files" / "data100" / "matched"

HOP_SECONDS = 0.01      # one spectrogram column = 10 ms
WINDOW_SECONDS = 10.0   # fixed x-axis span for every displayed example


def load_dataset_folder(folder):
    """Load entries from a matched-dataset folder (labels.json + data/*.npy)."""
    folder = Path(folder)
    with open(folder / "labels.json") as f:
        label_data = json.load(f)
    entries = []
    for info in label_data["files"]:
        npy_path = folder / "data" / info["filename"]
        if npy_path.exists():
            entries.append({"npy_path": npy_path, **info})
    if not entries:
        raise FileNotFoundError(f"No .npy files listed in {folder / 'labels.json'} exist on disk")
    return entries


def log_transform_display(sg_raw, tmp_dir, basename):
    """Run a raw linear-power spectrogram through the literal SpectrogramDataset the
    trainer uses (Log transform), at the file's natural height/width so the dataset's
    padding and center-crop are exact no-ops. No bg-subtract, no augmentation, no reverb."""
    npy_path = Path(tmp_dir) / f"{basename}.npy"
    np.save(npy_path, sg_raw)
    height, width = sg_raw.shape
    ds = SpectrogramDataset(
        filenames=[str(npy_path)],
        labels=[[1.0]],
        img_height=height,
        img_width=width,
        channels=1,
        cropping_mode="center",
        noise_filenames=None,
        noise_ratio=0.0,
        spec_transform="Log",
        training=False,
        bg_subtract=False,
        apply_reverb=False,
        use_temporal_roll=False,
        background_prob=0.0,
    )
    x, _ = ds[0]
    return x.numpy()[0]  # (C,H,W) -> (H,W), untouched orientation from the dataset class


def compute_panels(entry, tmp_dir):
    """Return (original, foreground_removed) log-transformed arrays for one .npy file."""
    sg_raw = np.load(entry["npy_path"])
    # Foreground removal happens on the raw linear-power spectrogram, before any log
    # transform - exactly where the trainer's --background-prob path applies it
    # (via the data_utils re-export of this same function).
    # get_background_spectrogram MUTATES its input in place, so hand it a copy -
    # otherwise sg_raw is rewritten too and the "original" panel shows the transform.
    sg_bg_only = get_background_spectrogram(sg_raw.copy())
    stem = entry["npy_path"].stem
    original = log_transform_display(sg_raw, tmp_dir, f"{stem}_orig")
    removed = log_transform_display(sg_bg_only, tmp_dir, f"{stem}_bg")
    return original, removed


class Viewer:
    def __init__(self, datasets, order, tmp_dir, start=0):
        self.datasets = datasets          # {"AviaNZ": [entries], "DOC": [entries]}
        self.order = order                # interleaved [(label, pair_index), ...]
        self.tmp_dir = tmp_dir
        self.pos = start % len(order)
        self.cache = {}                   # (label, pair_index) -> (orig, removed)

        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.14, top=0.90, hspace=0.35)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Mouse fallback: some backends/window managers don't give the canvas
        # keyboard focus, so key presses never arrive. Buttons always work.
        ax_prev = self.fig.add_axes([0.30, 0.015, 0.12, 0.045])
        ax_next = self.fig.add_axes([0.58, 0.015, 0.12, 0.045])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(lambda _event: self.step(-1))
        self.btn_next.on_clicked(lambda _event: self.step(1))

        # Persistent artists, created ONCE. Navigation only swaps their pixel data
        # (set_data/set_extent) and never clears an axes, so the figure and canvas
        # are left structurally untouched and keyboard focus can never be lost no
        # matter how many times the arrow keys are pressed. Clearing/recreating
        # artists on every step is what killed the keys after a few presses.
        self.images = []
        for ax, subtitle in zip(self.axes, ("Original", "Foreground removed (background only)")):
            image = ax.imshow(np.zeros((1, 1)), origin="lower", aspect="auto",
                              cmap="magma", extent=(0.0, HOP_SECONDS, 0.0, 1.0))
            ax.set_xlim(0.0, WINDOW_SECONDS)
            ax.set_title(subtitle, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel bin")
            self.images.append(image)
        self.title = self.fig.suptitle("", fontsize=13, fontweight="bold")

        self.redraw()

    def get_panels(self, label, pair_index):
        key = (label, pair_index)
        if key not in self.cache:
            # Cache also pins down the random resampling inside
            # get_background_spectrogram, so a panel never changes when revisited.
            self.cache[key] = compute_panels(self.datasets[label][pair_index], self.tmp_dir)
        return self.cache[key]

    def redraw(self):
        label, pair_index = self.order[self.pos]
        entry = self.datasets[label][pair_index]
        original, removed = self.get_panels(label, pair_index)

        # Shared color scale from the ORIGINAL panel, so the bottom panel shows
        # honestly what disappeared (the call) and what remained (the noise floor).
        finite = original[np.isfinite(original)]
        low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0

        duration = entry.get("end_time", 0.0) - entry.get("start_time", 0.0)
        classes = ", ".join(entry.get("class_names", [])) or "?"
        for image, sg in zip(self.images, (original, removed)):
            # Display only: SpectrogramDataset's array has row 0 = highest frequency,
            # so flip for a normal low-freq-at-bottom view. The array fed to the model
            # is never flipped.
            height, width = sg.shape
            image.set_data(np.flipud(sg))
            image.set_extent((0.0, width * HOP_SECONDS, 0.0, float(height)))
            image.set_clim(low, high)
        # Fixed 10-second x-axis for every example, so clip durations are directly
        # comparable (shorter clips leave blank space on the right).
        for ax in self.axes:
            ax.set_xlim(0.0, WINDOW_SECONDS)

        step = self.pos + 1
        self.title.set_text(
            f"{label} #{pair_index}  ({step}/{len(self.order)})   {entry['filename']}"
            f"   classes: {classes}   [{duration:.2f} s]"
        )
        self.fig.canvas.draw_idle()
        # Re-assert keyboard focus on the canvas after every redraw: with some Qt
        # window managers the canvas silently loses focus (which is what made the
        # arrow keys appear to "stop working"). This is harmless when focus is
        # already there.
        try:
            self.fig.canvas.setFocus()
        except (AttributeError, RuntimeError):
            pass
        print(f"[{step:3d}/{len(self.order)}] {label:6s} #{pair_index}: "
              f"{entry['filename']} ({classes})")

    def step(self, delta):
        self.pos = (self.pos + delta) % len(self.order)
        self.redraw()

    def on_key(self, event):
        if event.key == "right":
            self.step(1)
        elif event.key == "left":
            self.step(-1)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avianz", type=Path, default=DEFAULT_ROOT / "avianz_matched",
                        help="AviaNZ matched dataset folder (labels.json + data/)")
    parser.add_argument("--doc", type=Path, default=DEFAULT_ROOT / "doc_matched",
                        help="DOC matched dataset folder (labels.json + data/)")
    parser.add_argument("--start", type=int, default=0,
                        help="Position in the interleaved sequence to start at "
                             "(0 = AviaNZ 0, 1 = DOC 0, 2 = AviaNZ 1, ...)")
    args = parser.parse_args()

    datasets = {
        "AviaNZ": load_dataset_folder(args.avianz),
        "DOC": load_dataset_folder(args.doc),
    }
    n_pairs = min(len(datasets["AviaNZ"]), len(datasets["DOC"]))
    order = [(label, i) for i in range(n_pairs) for label in ("AviaNZ", "DOC")]

    print(f"AviaNZ: {len(datasets['AviaNZ'])} files, DOC: {len(datasets['DOC'])} files "
          f"-> {n_pairs} interleaved pairs ({len(order)} steps)")
    print("Controls: Right/Left arrows or Prev/Next buttons = navigate "
          "(AviaNZ n <-> DOC n), Q = quit")

    tmp_dir = tempfile.mkdtemp(prefix="fg_removal_view_")
    try:
        Viewer(datasets, order, tmp_dir, start=args.start)
        plt.show()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
