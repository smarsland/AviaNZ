"""
Spectrogram normalization utilities.
Provides background normalization for spectrograms to reduce noise and enhance bird calls.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d


def normalize_spectrogram(img, gaussian_sigma=3.0, eps=1e-6):
    """
    Apply background normalization to a spectrogram.
    
    This method estimates background noise from the bottom half of pixels in each
    frequency band and normalizes the spectrogram accordingly. This enhances the
    contrast of bird calls against background noise.
    
    Args:
        img: Input spectrogram (2D numpy array, shape: [freq_bins, time_bins])
        gaussian_sigma: Sigma for Gaussian smoothing of background estimates (default: 3.0)
        eps: Small epsilon to prevent division by zero (default: 1e-6)
    
    Returns:
        Normalized spectrogram (same shape as input)
    """
    # Ensure input is float
    img = np.asarray(img, dtype=np.float32)
    
    # Assume for any frequency band no more than half of the pixels are interesting
    # Therefore take the bottom half as non-interesting to estimate the background
    H, W = img.shape
    sorted_pixels = np.sort(img, axis=1)
    bg_pixels = sorted_pixels[:, :W//2]
    
    # Calculate mean and variance of background pixels per frequency band
    mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    var0 = np.var(bg_pixels, axis=1, keepdims=True)
    
    # Smooth background estimates across frequency bands
    mu0 = gaussian_filter1d(mu0.flatten(), sigma=gaussian_sigma).reshape(H, 1)
    var0 = gaussian_filter1d(var0.flatten(), sigma=gaussian_sigma).reshape(H, 1)
    
    # Normalize: z-score normalization per frequency band
    sg_normalized = (img - mu0) / (np.sqrt(var0) + eps)
    
    return sg_normalized


def visualize_normalization(img, gaussian_sigma=3.0, eps=1e-6):
    """
    Visualize original and normalized spectrograms side by side.
    
    Args:
        img: Input spectrogram (2D numpy array)
        gaussian_sigma: Sigma for Gaussian smoothing (default: 3.0)
        eps: Small epsilon value (default: 1e-6)
    """
    sg_normalized = normalize_spectrogram(img, gaussian_sigma, eps)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original Spectrogram")
    plt.imshow(img, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity (dB)')
    plt.subplot(2, 1, 2)
    plt.title("Background Normalized Spectrogram")
    plt.imshow(sg_normalized, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Normalized Intensity (dB)')
    plt.tight_layout()
    plt.show()


def visualize_folder(folder_path="NZ_bird_spec/data", gaussian_sigma=3.0, eps=1e-6):
    """
    Visualize normalization for all spectrograms in a folder.
    
    Args:
        folder_path: Path to folder containing .npy spectrogram files
        gaussian_sigma: Sigma for Gaussian smoothing (default: 3.0)
        eps: Small epsilon value (default: 1e-6)
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    files = os.listdir(folder_path)
    
    for file in files:
        if file.endswith('.npy'):
            print(f"Processing: {file}")
            img = np.load(os.path.join(folder_path, file))
            img = img[1:-1]  # Remove first and last frequency bins
            
            LOG_OFFSET = 1e-7
            img = img + LOG_OFFSET
            minsg = np.min(img)
            img = 10*(np.log10(img)-np.log10(minsg))
            img = np.abs(img)
            
            visualize_normalization(img, gaussian_sigma, eps)


if __name__ == "__main__":
    # Example usage for standalone visualization
    visualize_folder()