"""
Spectrogram normalization utilities.
Provides background normalization for spectrograms to reduce noise and enhance bird calls.
"""

import numpy as np
from scipy.ndimage import median_filter as median_filter_func

def get_background_spectrogram(img):
    # Assume for any frequency band no more than 10% of the pixels are interesting
    # Therefore take the bottom 10% as non-interesting to estimate the background
    H, W = img.shape
    
    sorted_pixels = np.sort(img, axis=1)
    bg_pixels = sorted_pixels[:, :W//10]
    
    # Calculate mean and variance of background pixels per frequency band
    mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    var0 = np.var(bg_pixels, axis=1, keepdims=True)

    # Normalize: z-score normalization per frequency band
    sg_normalized = (img - mu0) / (np.sqrt(var0) + 1e-6)

    # Vectorized outlier replacement: replace values > 4 std with the per-row
    # median of non-outlier values (avoids a slow Python loop over H=224 rows).
    outlier_mask = sg_normalized > 4  # (H, W)
    sg_no_outliers = np.where(outlier_mask, np.nan, sg_normalized)
    row_fill = np.nanmedian(sg_no_outliers, axis=1, keepdims=True)  # (H, 1)
    row_fill = np.where(np.isnan(row_fill), 0.0, row_fill)
    sg_normalized = np.where(outlier_mask, row_fill, sg_normalized)

    img = (sg_normalized * (np.sqrt(var0) + 1e-6)) + mu0

    return img

def normalize_spectrogram(img, median_filter=False, bg_subtract=False):
    """
    Apply preprocessing to a spectrogram.
    
    Args:
        img: Input spectrogram (2D numpy array)
        median_filter: Whether to apply temporal median filtering (default: False)
        bg_subtract: Whether to apply background subtraction (default: False)
    
    Both options work independently and can be combined.
    """
    # Ensure input is float
    img = np.asarray(img, dtype=np.float32)

    if median_filter:
        img = median_filter_func(img, size=(1, 5))
    
    if bg_subtract:
        img = img - get_background_spectrogram(img)

    return img