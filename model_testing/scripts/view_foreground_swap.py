#!/usr/bin/env python3
"""Interactive viewer: flip through matched AviaNZ/DOC pairs showing all four
background/foreground cross-combinations from generate_spectrogram_combinations.

Shows ONE PAIR at a time as a 2x2 matrix - backgrounds as rows, foregrounds as
columns:

                   Foreground: AviaNZ              Foreground: DOC
    Background:    +----------------------------+---------------------------+
    AviaNZ         | bg AviaNZ + fg AviaNZ      | bg AviaNZ + fg DOC        |
                   | (reconstructed original)   | (DOC call on AviaNZ floor)|
                   +----------------------------+---------------------------+
    Background:    | bg DOC + fg AviaNZ         | bg DOC + fg DOC           |
    DOC            | (AviaNZ call on DOC floor) | (reconstructed original)  |
                   +----------------------------+---------------------------+

The combinations are the literal generate_spectrogram_combinations from
model_testing/src/data/normalizer.py (the single canonical implementation -
currently NOT wired into train.py; this viewer previews what such an augmentation
would look like). Each spectrogram is split per frequency row into a normalized
background (bottom-10% quietest pixels, truncated-normal recovery, foreground
pixels |z| > 3 resampled as background noise) and a foreground difference
(residual on the foreground mask). A combination adds the FOREGROUND clip's
difference onto the BACKGROUND clip's noise floor, then rescales to the background
clip's own per-row level. Diagonal panels reconstruct the near-original clips;
off-diagonal panels carry one recorder's call on the other's noise floor.

Like view_foreground_removal.py, this reimplements no preprocessing math. It loads
the raw linear-power .npy files that build_matched_datasets.py already saved, then
runs them through the literal SpectrogramDataset (model_trainer.py's data class)
for the Log transform, at each file's natural width so padding/cropping are
no-ops.

Matched pairs differ in duration. Each combination takes the FOREGROUND clip's
width: the background clip's noise floor is tiled (repeated) as many times as
needed to fill it. So "bg AviaNZ + fg DOC" is as wide as the DOC clip, with the
AviaNZ noise floor looping underneath. Diagonal panels are as wide as their own
clip (their background tiles exactly once - no visible repetition).

Controls:
    Right / Left : next / previous pair
    Prev / Next buttons at the bottom do the same with the mouse
    Q / Escape   : quit

The x-axis is a fixed 10-second window (columns are 10 ms), so short clips leave
blank space on the right and durations are directly comparable.

Example:
    python scripts/view_foreground_swap.py
    python scripts/view_foreground_swap.py --start 10
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
from model_testing.src.data.normalizer import generate_spectrogram_combinations

DEFAULT_ROOT = REPO_ROOT / "Sound Files" / "data100" / "matched"

HOP_SECONDS = 0.01      # one spectrogram column = 10 ms
WINDOW_SECONDS = 10.0   # fixed x-axis span for every displayed example

# (grid position, cache key, subtitle) for the 2x2 matrix.
# Row = background clip, column = foreground clip.
PANELS = [
    ((0, 0), "av_av", "bg AviaNZ + fg AviaNZ\n(reconstructed original)"),
    ((0, 1), "av_doc", "bg AviaNZ + fg DOC\n(DOC call on AviaNZ floor)"),
    ((1, 0), "doc_av", "bg DOC + fg AviaNZ\n(AviaNZ call on DOC floor)"),
    ((1, 1), "doc_doc", "bg DOC + fg DOC\n(reconstructed original)"),
]


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


def compute_panels(avianz_entry, doc_entry, tmp_dir):
    """Return (panels_dict, None) for one pair.

    panels_dict keys: 'av_orig', 'doc_orig' (full originals, for the shared color
    scale) plus the four combination panels 'av_av', 'av_doc', 'doc_av', 'doc_doc'.
    Each combination panel takes its foreground clip's width: the background
    clip's noise floor is tiled (repeated) to fill it. Combinations are computed
    on the raw linear-power spectrograms, before any log transform.
    """
    sg_av = np.load(avianz_entry["npy_path"])
    sg_doc = np.load(doc_entry["npy_path"])

    av_stem = avianz_entry["npy_path"].stem
    doc_stem = doc_entry["npy_path"].stem

    # Full originals (log-transformed) for the shared color scale.
    av_orig = log_transform_display(sg_av, tmp_dir, f"{av_stem}_orig")
    doc_orig = log_transform_display(sg_doc, tmp_dir, f"{doc_stem}_orig")

    # The 4 combinations, in the order generate_spectrogram_combinations returns
    # them: (bg A + fg A, bg A + fg B, bg B + fg A, bg B + fg B).
    # split_spectrogram_components copies internally, so the function does not
    # mutate its inputs - no need to copy here.
    bg_a_fg_a, bg_a_fg_b, bg_b_fg_a, bg_b_fg_b = generate_spectrogram_combinations(
        sg_av, sg_doc)

    panels = {
        "av_orig": av_orig,
        "doc_orig": doc_orig,
        "av_av": log_transform_display(bg_a_fg_a, tmp_dir, f"{av_stem}_avav"),
        "av_doc": log_transform_display(bg_a_fg_b, tmp_dir, f"{av_stem}_avdoc"),
        "doc_av": log_transform_display(bg_b_fg_a, tmp_dir, f"{doc_stem}_docav"),
        "doc_doc": log_transform_display(bg_b_fg_b, tmp_dir, f"{doc_stem}_docdoc"),
    }
    return panels, None


class Viewer:
    def __init__(self, avianz, doc, tmp_dir, start=0):
        self.avianz = avianz            # [entries]
        self.doc = doc                  # [entries]
        self.n_pairs = min(len(avianz), len(doc))
        self.tmp_dir = tmp_dir
        self.pos = start % self.n_pairs
        self.cache = {}                 # pair_index -> compute_panels(...) result

        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
        self.fig.subplots_adjust(bottom=0.14, top=0.84, hspace=0.45, wspace=0.15)
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
        self.images = {}
        for (row, col), key, subtitle in PANELS:
            ax = self.axes[row, col]
            image = ax.imshow(np.zeros((1, 1)), origin="lower", aspect="auto",
                              cmap="magma", extent=(0.0, HOP_SECONDS, 0.0, 1.0))
            ax.set_xlim(0.0, WINDOW_SECONDS)
            ax.set_title(subtitle, fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel bin")
            self.images[key] = image
        self.title = self.fig.suptitle("", fontsize=12, fontweight="bold")

        self.redraw()

    def get_panels(self, pair_index):
        if pair_index not in self.cache:
            self.cache[pair_index] = compute_panels(
                self.avianz[pair_index], self.doc[pair_index], self.tmp_dir)
        return self.cache[pair_index]

    def redraw(self):
        av_entry = self.avianz[self.pos]
        doc_entry = self.doc[self.pos]
        panels, _ = self.get_panels(self.pos)

        # Shared color scale from BOTH reconstructed-original diagonal panels, so
        # the off-diagonal combinations show honestly what the call looks like on
        # the other recorder's noise floor.
        finite = np.concatenate([panels["av_av"][np.isfinite(panels["av_av"])],
                                 panels["doc_doc"][np.isfinite(panels["doc_doc"])]])
        low, high = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0

        for _pos, key, _subtitle in PANELS:
            sg = panels[key]
            # Display only: SpectrogramDataset's array has row 0 = highest frequency,
            # so flip for a normal low-freq-at-bottom view. The array fed to the model
            # is never flipped.
            height, width = sg.shape
            image = self.images[key]
            image.set_data(np.flipud(sg))
            image.set_extent((0.0, width * HOP_SECONDS, 0.0, float(height)))
            image.set_clim(low, high)
        # Fixed 10-second x-axis for every panel, so clip durations are directly
        # comparable (shorter clips leave blank space on the right).
        for ax in self.axes.flat:
            ax.set_xlim(0.0, WINDOW_SECONDS)

        step = self.pos + 1
        av_classes = ", ".join(av_entry.get("class_names", [])) or "?"
        doc_classes = ", ".join(doc_entry.get("class_names", [])) or "?"
        av_duration = av_entry.get("end_time", 0.0) - av_entry.get("start_time", 0.0)
        doc_duration = doc_entry.get("end_time", 0.0) - doc_entry.get("start_time", 0.0)
        self.title.set_text(
            f"Pair #{self.pos}  ({step}/{self.n_pairs})   rows: background - columns: foreground\n"
            f"AviaNZ: {av_entry['filename']}  ({av_classes}, {av_duration:.2f} s)      "
            f"DOC: {doc_entry['filename']}  ({doc_classes}, {doc_duration:.2f} s)"
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
        print(f"[{step:3d}/{self.n_pairs}] pair #{self.pos}: "
              f"AviaNZ {av_entry['filename']} ({av_classes})  <->  "
              f"DOC {doc_entry['filename']} ({doc_classes})")

    def step(self, delta):
        self.pos = (self.pos + delta) % self.n_pairs
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
                        help="Pair index to start at (0-based)")
    args = parser.parse_args()

    avianz = load_dataset_folder(args.avianz)
    doc = load_dataset_folder(args.doc)
    n_pairs = min(len(avianz), len(doc))
    if n_pairs == 0:
        raise RuntimeError("No matched pairs to show")

    print(f"AviaNZ: {len(avianz)} files, DOC: {len(doc)} files -> {n_pairs} pairs")
    print("Controls: Right/Left arrows or Prev/Next buttons = navigate pairs, Q = quit")

    tmp_dir = tempfile.mkdtemp(prefix="fg_swap_view_")
    try:
        Viewer(avianz, doc, tmp_dir, start=args.start)
        plt.show()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
