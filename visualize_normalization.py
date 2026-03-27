"""
Create spectrogram visualizations showing the effect of different normalization methods.
This generates Figure comparing Log, Log+normalize, and PCEN for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import json
import random

# Import our normalizer
from normalizer import normalize_spectrogram


def compute_pcen(sg, eps=1e-6):
    """Compute PCEN transform on a spectrogram."""
    from scipy import signal
    
    # Standard PCEN parameters
    gain = 0.8
    bias = 10
    power = 0.25
    t = 0.060
    
    fs = 16000
    hop_samples = int(0.010 * fs)
    s = 1 - np.exp(-hop_samples / (t * fs))
    
    # Apply IIR filtering to get smoothed version
    M = signal.lfilter([s], [1, s-1], sg, axis=1)
    smooth = (eps + M)**(-gain)
    pcen = (sg * smooth + bias)**power - bias**power
    
    return pcen


def compute_boxcox(sg, lam=0.25):
    """Compute Box-Cox transform on a spectrogram."""
    from scipy.special import boxcox as boxcox_transform
    
    # Flatten, transform, reshape
    sg_flat = sg.flatten()
    sg_transformed = boxcox_transform(sg_flat, lam)
    return sg_transformed.reshape(sg.shape)


def plot_spectrogram_comparison(dataset_path, output_path='figures/spectrogram_comparison.pdf', species=None):
    """
    Create comparison of all 5 normalization methods tested in experiments.
    
    Args:
        dataset_path: Path to dataset folder (with data/ and labels.json)
        output_path: Where to save the figure
        species: Optional species name to filter for
    """
    dataset_path = Path(dataset_path)
    
    # Load labels
    with open(dataset_path / 'labels.json') as f:
        data = json.load(f)
    
    files = data['files']
    if species:
        files = [f for f in files if species in f['class_names']]
    
    # Pick a random file
    file_info = random.choice(files)
    filename = file_info['filename']
    species_name = file_info['class_names'][0]
    
    print(f"Using file: {filename} (species: {species_name})")
    
    # Load pre-computed spectrogram (linear power)
    spec_path = dataset_path / 'data' / filename
    S_linear = np.load(spec_path)
    
    # Crop to reasonable size for visualization - use more frames for better resolution
    if S_linear.shape[1] < 200:
        print(f"Warning: Spectrogram too short ({S_linear.shape[1]} frames), using what we have")
        S_cropped = S_linear
    else:
        # Use central portion - wider crop for better horizontal resolution
        start_frame = S_linear.shape[1] // 2 - 100
        end_frame = start_frame + 200
        S_cropped = S_linear[:, start_frame:end_frame]
    
    # 1. Log transform (baseline) - convert to dB
    S_log = 10 * np.log10(S_cropped + 1e-10)
    
    # 2. Log + Background Normalization (with median filter)
    S_normalized = normalize_spectrogram(S_log.copy(), use_median_filter=True)
    
    # 3. Log + Background Normalization (no median filter - ablation)
    S_normalized_no_median = normalize_spectrogram(S_log.copy(), use_median_filter=False)
    
    # 4. PCEN (on original magnitude)
    S_pcen = compute_pcen(S_cropped)
    
    # 5. Box-Cox
    S_boxcox = compute_boxcox(S_cropped)
    
    # Create figure - 5 rows, wider for better resolution
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    
    # Common colormap settings
    cmap = 'viridis'
    aspect = 'auto'
    origin = 'lower'
    interpolation = 'none'
    
    # Compact font for single-column layout
    title_size = 10
    
    # Plot 1: Log baseline
    axes[0].imshow(S_log, aspect=aspect, origin=origin, cmap=cmap, interpolation=interpolation)
    axes[0].set_title('(a) Log', fontsize=title_size, fontweight='bold', pad=8)
    axes[0].axis('off')
    
    # Plot 2: Log + Background Normalization (default - no median)
    axes[1].imshow(S_normalized_no_median, aspect=aspect, origin=origin, cmap=cmap, interpolation=interpolation)
    axes[1].set_title('(b) Log+normalize', fontsize=title_size, fontweight='bold', pad=8)
    axes[1].axis('off')
    
    # Plot 3: Log + Background Normalization (with median filter added)
    axes[2].imshow(S_normalized, aspect=aspect, origin=origin, cmap=cmap, interpolation=interpolation)
    axes[2].set_title('(c) Log+normalize (with median)', fontsize=title_size, fontweight='bold', pad=8)
    axes[2].axis('off')
    
    # Plot 4: PCEN
    axes[3].imshow(S_pcen, aspect=aspect, origin=origin, cmap=cmap, interpolation=interpolation)
    axes[3].set_title('(d) PCEN', fontsize=title_size, fontweight='bold', pad=8)
    axes[3].axis('off')
    
    # Plot 5: Box-Cox
    axes[4].imshow(S_boxcox, aspect=aspect, origin=origin, cmap=cmap, interpolation=interpolation)
    axes[4].set_title('(e) Box-Cox', fontsize=title_size, fontweight='bold', pad=8)
    axes[4].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved spectrogram comparison to {output_path}")
    plt.close()


def find_example_dataset():
    """Find a good example dataset with spectrograms."""
    # Look for test datasets
    search_paths = [
        "../test/data",
        "test/data",
        "test/doc_split/test",
        "test/joe_mo_split/test",
        "/local/scratch/freangi/matched/doc_split/test",
        "/local/scratch/freangi/matched/avianz_split/test",
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            # Check if it has labels.json or if it's just a data folder
            if os.path.exists(os.path.join(search_path, 'labels.json')):
                return search_path
            # If it's a data folder, check parent for labels.json
            parent = os.path.dirname(search_path)
            if os.path.exists(os.path.join(parent, 'labels.json')):
                return parent
    
    return None


if __name__ == "__main__":
    # Find example dataset
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        species = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        dataset_path = find_example_dataset()
        if dataset_path is None:
            print("Error: Could not find any dataset with spectrograms.")
            print("Usage: python visualize_normalization.py <dataset_path> [species_name]")
            print("  dataset_path: folder containing data/ subfolder and labels.json")
            print("  species_name: optional, e.g., 'nztui1' or 'nezbel1'")
            sys.exit(1)
        species = None
    
    print(f"Using dataset: {dataset_path}")
    
    # Create visualization
    plot_spectrogram_comparison(dataset_path, species=species)
    
    print("\nVisualization complete!")
    print("Add to paper with:")
    print("\\begin{figure}[t]")
    print("\\centering")
    print("\\includegraphics[width=\\columnwidth]{figures/spectrogram_comparison.pdf}")
    print("\\caption{Comparison of all 5 spectrogram transformations tested:")
    print("(a) Log baseline shows persistent background noise;")
    print("(b) Log+normalize removes background, enhancing signal;")
    print("(c) Log+normalize (with median) adds median filtering;")
    print("(d) PCEN over-smooths temporal structure;")
    print("(e) Box-Cox alternative transform.}")
    print("\\label{fig:spectrograms}")
    print("\\end{figure}")
