"""
Spectrogram normalization utilities.
Provides background normalization for spectrograms to reduce noise and enhance bird calls.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def gaussianity_score(x):
    """
    Jarque-Bera statistic (lower = more Gaussian).
    """
    n = len(x)
    if n < 3:
        return np.inf

    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return np.inf

    z = (x - mean) / std
    skew = np.mean(z**3)
    kurt = np.mean(z**4)

    jb = (n / 6.0) * (skew**2 + 0.25 * (kurt - 3)**2)
    return jb


def top_tail_gaussian_split(x, min_frac=0.5, max_frac=0.99, step=1):
    """
    Find the threshold that separates background (Gaussian noise) from signal.
    Stops when adding more points makes the distribution less Gaussian.
    """
    x = np.sort(np.asarray(x))
    n = len(x)

    min_size = max(5, int(np.ceil(n * min_frac)))
    max_size = int(np.floor(n * max_frac))
    
    # Adaptive step size for efficiency
    if step == 1 and (max_size - min_size) > 100:
        step = max(1, (max_size - min_size) // 50)

    prev_score = gaussianity_score(x[:min_size])
    best_split = min_size

    # Find where the distribution stops improving (starts getting less Gaussian)
    for split in range(min_size + step, max_size + 1, step):
        x_low = x[:split]
        score = gaussianity_score(x_low)

        # If score increased (got worse), we've passed the background
        if score > prev_score * 1.2:  # 20% worse
            return x[best_split]
        
        # If score improved, update best
        if score < prev_score:
            best_split = split
        
        prev_score = score

    return x[best_split]

def normalize_spectrogram(img, method='gaussian_split', robust=True):
    """
    Apply background normalization to a spectrogram using per-frequency-band statistics.
    
    For each frequency band (row), estimates background noise distribution and normalizes
    to z-scores, making signals stand out from noise.
    
    Args:
        img: Input spectrogram (H x W array), typically in dB scale
        method: Background estimation method
            - 'gaussian_split': Use Gaussian fitting to find background (default)
            - 'percentile': Use bottom 50th percentile as background
        robust: If True, use robust statistics (median/MAD) instead of mean/std
    
    Returns:
        Normalized spectrogram (H x W array) where background noise ~ N(0, 1)
    """
    # Make a copy to avoid modifying input
    img = np.asarray(img, dtype=np.float32).copy()
    
    H, W = img.shape
    normalized = np.zeros_like(img)

    for row in range(H):
        row_data = img[row, :]
        
        # Estimate background region
        if method == 'gaussian_split':
            threshold = top_tail_gaussian_split(row_data)
            bg_mask = row_data < threshold
        elif method == 'percentile':
            threshold = np.percentile(row_data, 50)
            bg_mask = row_data < threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle edge case: no background pixels
        if np.sum(bg_mask) < 3:
            # Fall back to global statistics
            bg_data = row_data
        else:
            bg_data = row_data[bg_mask]
        
        # Compute background statistics
        if robust:
            # Median Absolute Deviation (more robust to outliers)
            loc = np.median(bg_data)
            mad = np.median(np.abs(bg_data - loc))
            scale = mad * 1.4826  # Scale factor to match std for Gaussian
        else:
            # Standard mean and standard deviation
            loc = np.mean(bg_data)
            scale = np.std(bg_data)
        
        # Normalize to z-scores
        normalized[row, :] = (row_data - loc) / (scale + 1e-8)
    
    normalized[normalized>3] = np.random.normal(0, scale=1, size=np.sum(normalized>3))

    return normalized

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