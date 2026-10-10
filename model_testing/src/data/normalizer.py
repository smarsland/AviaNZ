"""
Spectrogram normalization and mixing utilities.
Provides background normalization and foreground swapping for spectrograms to reduce noise and enhance bird calls.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.ndimage import label

def split_spectrogram_components(img, stochastic=True):
    """
    Splits a spectrogram into its normalized background, foreground difference,
    foreground mask, and row-wise scaling parameters (mu, sigma).

    stochastic: if True (default, training behaviour) foreground pixels in norm_bg
    are replaced with freshly SAMPLED N(0,1) noise, so the background estimate is
    stochastic. If False (evaluation/inference), foreground pixels are replaced with
    the background MEAN 0 instead - deterministic, so a model sees the same
    reconstruction every time for a given input (the only residual difference from
    the original at foreground pixels is the 0-mean vs resampled-noise background).
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape

    # Sort each row to estimate background statistics from the bottom 50%
    sorted_indices = np.argsort(img, axis=1)
    p = 0.5
    # Guard against the estimation window going empty: int(W*p) is 0 for W < 10,
    # which makes np.mean/np.std over the empty slice return NaN and poison the
    # entire reconstruction (the experiment-9-style NaN collapse). Use at least
    # one column, and at least 2 for a meaningful std; for very short rows fall
    # back to the whole row so stats are always defined.
    n_bg = max(2, int(W * p))
    n_bg = min(n_bg, W)
    bg_indices_per_band = sorted_indices[:, :n_bg]

    bg_mean = np.mean(np.take_along_axis(img, bg_indices_per_band, axis=1), axis=1, keepdims=True)
    bg_std = np.std(np.take_along_axis(img, bg_indices_per_band, axis=1), axis=1, keepdims=True)

    # Recovery Logic for Bottom Truncation
    z = norm.ppf(p)         # Standard normal quantile (~ -0.6745)
    w = norm.pdf(z) / p     # Truncation ratio term (~ 1.2714)
    var_factor = 1.0 - (z * w) - (w ** 2)

    true_sigma = bg_std / np.sqrt(var_factor)
    true_mu = bg_mean + (w * true_sigma)

    # Row-normalized image
    row_normalized_image = (img - true_mu) / (true_sigma + 1e-8)
    fg_mask = np.abs(row_normalized_image) > 3

    # Final safety net: a degenerate row (e.g. all-identical values) can still yield a
    # non-finite sigma. Replace any non-finite normalized values with 0 (background)
    # so a single bad row/column can never inject NaN into the batch.
    if not np.isfinite(row_normalized_image).all():
        bad = ~np.isfinite(row_normalized_image)
        row_normalized_image = np.where(bad, 0.0, row_normalized_image)
        fg_mask = fg_mask & ~bad
        true_mu = np.where(np.isfinite(true_mu), true_mu, 0.0)
        true_sigma = np.where(np.isfinite(true_sigma) & (true_sigma > 0), true_sigma, 1.0)

    # Estimate normalized background by replacing foreground pixels with normal noise
    norm_bg = row_normalized_image.copy()
    for i in range(H):
        if np.any(fg_mask[i]):
            if stochastic:
                norm_bg[i, fg_mask[i]] = np.random.normal(0, 1, size=np.sum(fg_mask[i]))
            else:
                # Deterministic reconstruction (evaluation): fill foreground pixels
                # with the background mean, which is 0 in normalized space.
                norm_bg[i, fg_mask[i]] = 0.0

    # Foreground difference relative to the background in normalized space
    fg_diff = np.zeros_like(row_normalized_image)
    fg_diff[fg_mask] = row_normalized_image[fg_mask]

    return {
        'norm_bg': norm_bg,
        'fg_diff': fg_diff,
        'fg_mask': fg_mask,
        'true_mu': true_mu,
        'true_sigma': true_sigma
    }


def generate_spectrogram_combinations(img1, img2, reverb_fn=None):
    """
    Takes two spectrograms (A and B) and returns all 4 background/foreground
    cross-combinations:
    1. Background A + Foreground A (Reconstructed A)
    2. Background A + Foreground B
    3. Background B + Foreground A
    4. Background B + Foreground B (Reconstructed B)

    The two spectrograms may differ in WIDTH (time). Each combination takes the
    FOREGROUND clip's width: the background clip's normalized noise floor (and its
    per-row mu/sigma) is tiled (repeated) as many times as needed to match. So
    "bg A + fg B" is as wide as B, with A's noise floor looping underneath; the
    diagonal reconstructions are exactly as wide as their source clip and tile
    that clip's own background an integral number of times (1x, i.e. no tiling).
    Heights (mel bins) must still match - they index the same frequency rows.

    reverb_fn: optional callable applied to each foreground difference (normalized
    space, zero outside the foreground mask) BEFORE it is added to a background -
    e.g. apply_foreground_reverb, so the echo tail becomes part of the foreground
    and trails past the mask onto the background it is placed on.
    """
    if img1.shape[0] != img2.shape[0]:
        raise ValueError(
            f"Spectrograms must have equal heights (mel bins), got {img1.shape[0]} "
            f"and {img2.shape[0]}. Widths may differ - the background is tiled to "
            "match the foreground clip's width."
        )

    comp_a = split_spectrogram_components(img1)
    comp_b = split_spectrogram_components(img2)

    def _reconstruct(bg_comp, fg_comp):
        # Width follows the FOREGROUND clip; the background clip's noise floor is
        # tiled (repeated) to fill it.
        W_fg = fg_comp['fg_diff'].shape[1]
        reps = int(np.ceil(W_fg / bg_comp['norm_bg'].shape[1]))

        # Start with the background's normalized base, tiled to the fg width.
        combined_norm = np.tile(bg_comp['norm_bg'], (1, reps))[:, :W_fg].copy()

        # Add the target foreground. fg_diff is zero outside the foreground mask,
        # so add it whole rather than via the mask: a reverb tail (if reverb_fn
        # was applied) then extends past the mask onto the background.
        fg_diff = fg_comp['fg_diff']
        if reverb_fn is not None:
            fg_diff = reverb_fn(fg_diff)
        combined_norm += fg_diff

        # Reverse row normalization and clip. true_mu/true_sigma are (H, 1) column
        # vectors - constant across time - so they broadcast directly onto the
        # (H, W_fg) image; no tiling needed for them.
        result = combined_norm * (bg_comp['true_sigma'] + 1e-8) + bg_comp['true_mu']
        return np.clip(result, 0, None)

    # Generate the 4 combinations
    bg_a_fg_a = _reconstruct(comp_a, comp_a)
    bg_a_fg_b = _reconstruct(comp_a, comp_b)
    bg_b_fg_a = _reconstruct(comp_b, comp_a)
    bg_b_fg_b = _reconstruct(comp_b, comp_b)

    return bg_a_fg_a, bg_a_fg_b, bg_b_fg_a, bg_b_fg_b


def normalize_spectrogram(img):
    """
    Apply background subtraction preprocessing to a single spectrogram.
    """
    comp = split_spectrogram_components(img)
    bg = comp['norm_bg'] * (comp['true_sigma'] + 1e-8) + comp['true_mu']
    bg = np.clip(bg, 0, None)
    return img - bg


def get_background_spectrogram(img):
    """Background-only estimate of a spectrogram (foreground pixels replaced with
    resampled per-row background noise). Canonical home for train.py's
    --background-prob augmentation - data_utils.py re-exports this.
    W<2 guard: the bottom-10% estimation slice is empty for a 1-column-wide
    spectrogram, so return the image unchanged rather than producing NaNs."""
    img = np.asarray(img, dtype=np.float32)
    if img.shape[1] < 2:
        return img
    comp = split_spectrogram_components(img)
    #bg = comp['norm_bg'] * (comp['true_sigma'] + 1e-8) + comp['true_mu']
    #return np.clip(bg, 0, None)
    fg = comp['fg_diff']# * (comp['true_sigma'] + 1e-8) + comp['true_mu']
    return fg


def swap_foregrounds(img1, img2):
    """Bidirectionally swap the foreground content of two equal-shaped spectrograms.
    Pixels that are foreground in EITHER image exchange their z-scored values, then
    each image is rescaled to its own per-row background level."""
    if img1.shape != img2.shape:
        raise ValueError(
            f"swap_foregrounds requires equal-shaped inputs, got {img1.shape} and "
            f"{img2.shape}. Crop/pad to a common window first (e.g. the pair's "
            "overlapping time region)."
        )
    comp1 = split_spectrogram_components(img1)
    comp2 = split_spectrogram_components(img2)

    swap_map = comp1['fg_mask'] | comp2['fg_mask']
    # Snapshot BOTH source regions before writing either one: writing img1 first and
    # then reading img1 back as img2's source would hand img2 its own pixels,
    # making the "swap" one-directional.
    norm1 = (np.asarray(img1, dtype=np.float32) - comp1['true_mu']) / (comp1['true_sigma'] + 1e-8)
    norm2 = (np.asarray(img2, dtype=np.float32) - comp2['true_mu']) / (comp2['true_sigma'] + 1e-8)
    fg_pixels1 = norm1[swap_map].copy()
    fg_pixels2 = norm2[swap_map].copy()
    norm1[swap_map] = fg_pixels2
    norm2[swap_map] = fg_pixels1

    img1_swapped = np.clip(norm1 * (comp1['true_sigma'] + 1e-8) + comp1['true_mu'], 0, None)
    img2_swapped = np.clip(norm2 * (comp2['true_sigma'] + 1e-8) + comp2['true_mu'], 0, None)
    return img1_swapped, img2_swapped