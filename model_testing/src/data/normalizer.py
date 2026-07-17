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

    for row in range(H):
        outliers = sg_normalized[row, :] > 4
        not_outliers = sg_normalized[row, :] <= 4
        sg_normalized[row, outliers] = np.random.choice(sg_normalized[row, not_outliers], size=np.sum(outliers), replace=True)

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