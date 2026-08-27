#!/usr/bin/env python3
"""Generate log mel spectrogram images for all .wav files in a folder structure.

Usage:
    python scripts/generate_mel_spectrograms.py path/to/folder
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the project root is on the path regardless of where the script is invoked from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.spectrogram import Spectrogram
from src.data.reverberator import apply_reverb

# ── Fixed parameters for log mel spectrogram ──────────────────────────────
WINDOW_WIDTH = 2048  # Standard for mel spectrograms
INCR = 512          # Hop length
NFILTERS = 128      # Number of mel bands
WINDOW = 'Hann'     # Window type
SG_TYPE = 'Standard'  # Spectrogram type
SG_SCALE = 'Mel Frequency'  # Use mel scale
NORMALIZATION = 'Log'  # Log normalization


def save_spectrogram_image(original_sg, reverb_sg, output_path):
    """Save original and reverberated log mel spectrograms stacked vertically."""
    # Create a figure with twice the height
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot original spectrogram
    im1 = ax1.imshow(original_sg, aspect='auto', origin='lower', cmap='inferno')
    ax1.set_xlabel('Time frames')
    ax1.set_ylabel('Mel frequency bins')
    ax1.set_title('Original Spectrogram')
    
    # Plot reverberated spectrogram
    im2 = ax2.imshow(reverb_sg, aspect='auto', origin='lower', cmap='inferno')
    ax2.set_xlabel('Time frames')
    ax2.set_ylabel('Mel frequency bins')
    ax2.set_title('Reverberated Spectrogram')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def process_folder(input_folder):
    """Process all .wav files in male/female subfolders."""
    input_path = Path(input_folder)
    output_path = input_path / 'images'  # Create images folder in the same location
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_files = 0
    successful = 0
    errors = []
    
    # Look for male and female subfolders
    for subfolder in ['male', 'female']:
        sub_input = input_path / subfolder
        sub_output = output_path / subfolder
        
        if not sub_input.exists() or not sub_input.is_dir():
            print(f"Warning: Subfolder '{subfolder}' not found in {input_path}")
            continue
            
        # Create output subfolder
        sub_output.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {subfolder} folder: {sub_input}")
        
        # Find all .wav files in this subfolder
        wav_files = list(sub_input.glob('*.wav'))
        if not wav_files:
            print(f"  No .wav files found in {subfolder}")
            continue
            
        print(f"  Found {len(wav_files)} .wav files")
        
        # Process each .wav file
        for i, wav_file in enumerate(wav_files, 1):
            total_files += 1
            # Use the base filename without extension for the output image
            output_filename = wav_file.stem + '.png'
            output_path_file = sub_output / output_filename
            
            try:
                # Create spectrogram object
                sp = Spectrogram(window_width=WINDOW_WIDTH, incr=INCR)
                sp.readSoundFile(str(wav_file), silent=True)
                
                # Generate log mel spectrogram
                sp.spectrogram(
                    window_width=WINDOW_WIDTH,
                    incr=INCR,
                    window=WINDOW,
                    sgType=SG_TYPE,
                    sgScale=SG_SCALE,
                    nfilters=NFILTERS,
                )
                
                # Apply log normalization
                sg_norm = sp.normalisedSpec(tr=NORMALIZATION)

                # Transpose to get frequency bins along the first dimension and time bins along the second
                sg_norm = sg_norm.T
                
                # IMPORTANT: Create a deep copy for the original
                original_sg = sg_norm.copy()
                
                # Apply reverb to a copy (not the original)
                delay_mean = np.random.randint(1, 20)
                delay_std = delay_mean * np.random.uniform(0.5, 1.5)
                length = 50
                decay = np.random.uniform(0.0, 1.0)

                # Create a copy for reverb processing
                reverb_sg = sg_norm.copy()
                reverb_sg = apply_reverb(reverb_sg, delay_mean, delay_std, length, decay)
                
                # Save the image with both spectrograms
                save_spectrogram_image(original_sg, reverb_sg, str(output_path_file))
                
                successful += 1
                print(f"  [{i:3d}/{len(wav_files)}] {output_filename} ✓")
                
            except Exception as exc:
                msg = f"{wav_file.name}: {exc}"
                errors.append(msg)
                print(f"  [{i:3d}/{len(wav_files)}] {output_filename} ✗ ERROR")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {len(errors)}")
    print(f"Output saved to: {output_path}")
    
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate log mel spectrogram images for all .wav files in a folder with male/female subfolders'
    )
    parser.add_argument('input_folder', help='Path to the folder containing male/female subfolders with .wav files')
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    if not os.path.isdir(input_folder):
        print(f"ERROR: Folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    process_folder(input_folder)


if __name__ == '__main__':
    main()