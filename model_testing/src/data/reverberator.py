"""
Spectrogram reverb generator.
Provides reverberation effects for spectrograms to simulate acoustic environments.
We try to only apply reverberation to loud noises, keeping noise alone. 
"""

import numpy as np

def apply_reverb(img, freq_start=0, freq_end=500, delay_mean=5, delay_std=4, length=30):
    H, W = img.shape

    # Estimate background independently for each frequency bin.
    n_bg_pixels = max(1, W // 10)

    sorted_pixels = np.sort(img, axis=0)
    bg_pixels = sorted_pixels[:n_bg_pixels, :]

    mu0 = np.mean(bg_pixels, axis=0, keepdims=True)
    var0 = np.var(bg_pixels, axis=0, keepdims=True)

    # Normalize relative to the estimated background.
    sg_normalized = (img - mu0) / (np.sqrt(var0) + 1e-6)

    # Keep only interesting pixels.
    sg_only_outliers = np.where(
        sg_normalized > 4,
        sg_normalized,
        0
    )

    # Convert requested log-normal mean/std into the parameters
    # of the underlying normal distribution.
    sigma = np.sqrt(np.log(1 + (delay_std / delay_mean) ** 2))
    mu = np.log(delay_mean) - 0.5 * sigma**2

    # Log-normal delay distribution.
    x = np.arange(length)

    reverb_kernel = np.zeros(length, dtype=float)
    mask = x > 0

    reverb_kernel[mask] = (
        np.exp(
            -(np.log(x[mask]) - mu) ** 2
            / (2 * sigma**2)
        )
        / (
            x[mask]
            * sigma
            * np.sqrt(2 * np.pi)
        )
    )

    # Apply the reverb only to the requested frequency range.
    full_kernel = np.zeros((W, length))
    full_kernel[freq_start:freq_end, :] = reverb_kernel

    # Causal temporal reverb.
    reverb = np.zeros_like(sg_normalized)

    for delay in range(length):
        reverb[delay:] += (
            sg_only_outliers[:-delay or None]
            * full_kernel[:, delay]
        )

    return img + reverb