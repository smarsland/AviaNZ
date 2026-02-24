"""
Spectrogram normalization utilities.
Provides background normalization for spectrograms to reduce noise and enhance bird calls.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def normalize_spectrogram(img):
    """
    Apply background normalization to a spectrogram.
    KDE is computed using only the lowest 1/2 of points in each row.
    """
    # Ensure input is float
    img = np.asarray(img, dtype=np.float32)
    
    H, W = img.shape

    # sorted_pixels = np.sort(img, axis=1)
    # bg_pixels = sorted_pixels[:, :W//4]
    # mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    # var0 = np.var(bg_pixels, axis=1, keepdims=True)
    # img = (img - mu0) / (np.sqrt(var0) + 1e-6)
    
    # for c in range(img.shape[1]):
    #     col = img[:,c]
    #     distances = np.abs(np.linspace(0,1,len(col)).reshape(-1,1)-np.linspace(0,1,len(col)).reshape(1,-1))
    #     kernel = np.exp(-20 * distances**2)
    #     contributions = kernel / np.sum(kernel,axis=0)
    #     estimates = np.sum(contributions * col.reshape(-1,1),axis=0)
    #     img[:,c] = img[:,c] - estimates
    
    # img = np.asarray(img, dtype=np.float32)

    flat_order = np.argsort(img, axis=1)
    ranks = np.empty_like(flat_order)

    # Assign ranks row by row
    for i in range(img.shape[0]):
        ranks[i, flat_order[i]] = np.arange(img.shape[1])

    # Normalize ranks to [0, 1]
    img = ranks / (img.shape[1] - 1)

    return img

def normalize_spectrogram_old(img):
    """
    Apply background normalization to a spectrogram.
   
    """
    # Ensure input is float
    img = np.asarray(img, dtype=np.float32)

    flat_order = np.argsort(img, axis=1)
    ranks = np.empty_like(flat_order)

    # Assign ranks row by row
    for i in range(img.shape[0]):
        ranks[i, flat_order[i]] = np.arange(img.shape[1])

    # Normalize ranks to [0, 1]
    sg_normalized = ranks / (img.shape[1] - 1)
    return sg_normalized

def normalize_spectrogram_old_old(img):
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

def visualize_normalization(img):
    """
    Visualize original and normalized spectrograms side by side.
    
    Args:
        img: Input spectrogram (2D numpy array)
        gaussian_sigma: Sigma for Gaussian smoothing (default: 3.0)
        eps: Small epsilon value (default: 1e-6)
    """
    sg_normalized = normalize_spectrogram(img)
    sg_normalized = sg_normalized / (np.max(sg_normalized) + 1e-6)
    
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


def visualize_folder(folder_path="NZ_bird_spec/data"):
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
            
            visualize_normalization(img)


if __name__ == "__main__":
    # Example usage for standalone visualization
    visualize_folder()