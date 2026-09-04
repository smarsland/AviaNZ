import numpy as np
from scipy.ndimage import maximum_filter1d

def apply_reverb(img, delay_mean=5, delay_std=4, length=30, decay=0.35, threshold=2.5, freq_smooth=5):
    """Simulate exponential-decay reverberation on a linear-power spectrogram.

    Loud (foreground) time-frequency cells are detected via a per-row background
    z-score, then an echo of those cells is convolved forward in time with a
    log-normal decay kernel and added back to the spectrogram.

    `threshold` (lowered from the original 4.0 sigma) makes more of a call's
    energy count as "loud" rather than only its single peak, and `freq_smooth`
    dilates the loud-cell mask across neighbouring frequency bins so a whole
    call reverberates coherently instead of only its single loudest bin -
    both changes make the augmentation's effect much more visible/audible.
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape

    # Estimate background and loud pixels
    n_bg_pixels = max(1, W // 10)
    sorted_pixels = np.sort(img, axis=1)
    bg_pixels = sorted_pixels[:, :n_bg_pixels]
    mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    std0 = np.sqrt(np.var(bg_pixels, axis=1, keepdims=True))
    z = (img - mu0) / (std0 + 1e-6)
    mask = z > threshold

    # Spread the "loud" mask across neighbouring frequency bins: a single
    # reflecting surface reverberates the whole call, not just its peak bin.
    if freq_smooth > 1 and H > 1:
        mask = maximum_filter1d(mask.astype(np.float32), size=min(freq_smooth, H), axis=0) > 0

    # Make log-normal delay distribution
    sigma = np.sqrt(np.log(1.0 + (delay_std / delay_mean) ** 2))
    mu = (np.log(delay_mean) - 0.5 * sigma ** 2)

    # Make the main reverb kernel
    delays = np.arange(1, length + 1, dtype=np.float32)
    kernel = (np.exp(-(np.log(delays) - mu) ** 2 / (2.0 * sigma ** 2)) / (delays * sigma * np.sqrt(2.0 * np.pi)))
    kernel /= kernel.sum()
    kernel *= decay
    reverb = np.zeros_like(img, dtype=np.float32)

    # Apply
    for i, delay in enumerate(range(1, length + 1)):
        amount = kernel[i]
        source = img[:, :-delay]
        source_mask = mask[:, :-delay]
        reverb[:, delay:] += (np.where(source_mask, source, 0.0) * amount)

    return img + reverb