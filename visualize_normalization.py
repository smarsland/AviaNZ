"""
Create spectrogram visualizations showing the effect of different normalization methods.
This generates Figure comparing Log, Log+normalize, and PCEN for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from pathlib import Path
import sys

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


def plot_spectrogram_comparison(audio_file, output_path='figures/spectrogram_comparison.pdf'):
    """
    Create side-by-side comparison of Log, Log+Normalize, and PCEN.
    
    Args:
        audio_file: Path to audio file to visualize
        output_path: Where to save the figure
    """
    # Load audio and compute spectrogram
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Compute mel spectrogram (same parameters as training)
    n_fft = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)  # 10ms
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        fmin=0,
        fmax=8000
    )
    
    # Ensure we have enough time frames for visualization
    if S.shape[1] < 100:
        print(f"Warning: Audio too short ({S.shape[1]} frames), using what we have")
        end_frame = S.shape[1]
    else:
        # Use central portion
        start_frame = S.shape[1] // 2 - 50
        end_frame = start_frame + 100
    
    S_cropped = S[:, start_frame:end_frame]
    
    # 1. Log transform (baseline)
    LOG_OFFSET = 1e-7
    S_log = np.log(S_cropped + LOG_OFFSET)
    
    # 2. Log + Background Normalization
    S_normalized = normalize_spectrogram(S_log.copy())
    
    # 3. PCEN (on original magnitude)
    S_pcen = compute_pcen(S_cropped)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    # Time axis (in seconds)
    hop_s = hop_length / sr
    times = np.arange(S_cropped.shape[1]) * hop_s
    
    # Frequency axis (in kHz)
    freqs = np.linspace(0, 8, 128)
    
    # Common colormap settings
    cmap = 'viridis'
    aspect = 'auto'
    origin = 'lower'
    
    # Plot 1: Log baseline
    im0 = axes[0].imshow(S_log, aspect=aspect, origin=origin, cmap=cmap, 
                          extent=[times[0], times[-1], freqs[0], freqs[-1]])
    axes[0].set_ylabel('Frequency (kHz)', fontsize=11)
    axes[0].set_title('(a) Log Transform (Baseline)', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    plt.colorbar(im0, ax=axes[0], label='Log Magnitude')
    
    # Plot 2: Log + Background Normalization
    im1 = axes[1].imshow(S_normalized, aspect=aspect, origin=origin, cmap=cmap,
                          extent=[times[0], times[-1], freqs[0], freqs[-1]])
    axes[1].set_ylabel('Frequency (kHz)', fontsize=11)
    axes[1].set_title('(b) Log + Background Normalization (Proposed)', fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    plt.colorbar(im1, ax=axes[1], label='Normalized Magnitude')
    
    # Plot 3: PCEN
    im2 = axes[2].imshow(S_pcen, aspect=aspect, origin=origin, cmap=cmap,
                          extent=[times[0], times[-1], freqs[0], freqs[-1]])
    axes[2].set_ylabel('Frequency (kHz)', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_title('(c) PCEN (Catastrophic Failure)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], label='PCEN Value')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved spectrogram comparison to {output_path}")
    plt.close()


def find_example_audio():
    """Find a good example audio file from the dataset."""
    # Look for a tui or bellbird recording (distinctive calls)
    search_paths = [
        "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds/DOC_001_Tier1/DOC_001_Tier1/train_audio/nztui1/",
        "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds/DOC_001_Tier1/DOC_001_Tier1/train_audio/nezbel1/",
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            files = list(Path(search_path).glob("*.flac"))
            if files:
                return str(files[0])
    
    # If mounted paths don't exist, look in any spectrograms we have
    print("Mounted paths not found, searching for any audio file...")
    for root, dirs, files in os.walk("/home/giotto/Desktop/AviaNZ"):
        for f in files:
            if f.endswith(('.flac', '.wav', '.mp3')):
                return os.path.join(root, f)
    
    return None


if __name__ == "__main__":
    # Find example audio
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = find_example_audio()
        if audio_file is None:
            print("Error: Could not find any audio files.")
            print("Usage: python visualize_normalization.py <path_to_audio_file>")
            sys.exit(1)
    
    print(f"Using audio file: {audio_file}")
    
    # Create visualization
    plot_spectrogram_comparison(audio_file)
    
    print("\nVisualization complete!")
    print("Add to paper with:")
    print("\\begin{figure}[t]")
    print("\\centering")
    print("\\includegraphics[width=0.48\\textwidth]{figures/spectrogram_comparison.pdf}")
    print("\\caption{Spectrogram comparison for identical bird call: (a) Log baseline shows")
    print("persistent background noise floor; (b) Log+normalize removes background, enhancing")
    print("vocalization contrast; (c) PCEN over-smooths temporal structure critical for")
    print("species discrimination.}")
    print("\\label{fig:spectrograms}")
    print("\\end{figure}")
