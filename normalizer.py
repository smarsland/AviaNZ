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
        
    # Normalize: z-score normalization per frequency band
    sg_normalized = (img - mu0) / (np.sqrt(var0) + eps)

    flat_order = np.argsort(sg_normalized, axis=None)

    ranks = np.empty_like(flat_order)
    ranks[flat_order] = np.arange(flat_order.size)

    result = ranks.reshape(sg_normalized.shape)
    
    sg_normalized = result / (sg_normalized.shape[0] * sg_normalized.shape[1])

    return sg_normalized

def normalize_spectrogram_old(img, gaussian_sigma=3.0, eps=1e-6):
    """
    Apply background normalization to a spectrogram.
   
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