"""
Extract Shape Sequences from Raw Spectrograms

This script processes raw spectrograms, extracts signal regions, and saves
shape sequences with actual intensity values (not just binary masks).

Combines masking + sequence extraction into one efficient pipeline.
"""

import argparse
import os
import json
import numpy as np
import shutil
from tqdm import tqdm
from skimage import measure, transform
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def process_spectrogram(spec_linear, log_offset=1e-7, eps=1e-6, thresh_percentile=90.0, min_component_size=20):
    """Process a spectrogram and return normalized spec + signal region mask"""
    spec_linear = spec_linear + log_offset
    min_val = np.min(spec_linear)
    spec_db = np.abs(10 * (np.log10(spec_linear) - np.log10(min_val)))
    
    H, W = spec_db.shape
    sorted_rows = np.sort(spec_db, axis=1)
    background_pixels = sorted_rows[:, : W // 2]
    
    background_mean = np.mean(background_pixels, axis=1, keepdims=True)
    background_var = np.var(background_pixels, axis=1, keepdims=True)
    
    spec_normalized = (spec_db - background_mean) / (np.sqrt(background_var) + eps)
    
    energy_threshold = np.percentile(spec_normalized, thresh_percentile)
    high_energy_mask = spec_normalized > energy_threshold
    
    labels, _ = label(high_energy_mask)
    component_sizes = np.bincount(labels.ravel())
    
    keep_component = component_sizes >= min_component_size
    keep_component[0] = False
    
    # Map labels: keep only components that meet size threshold
    labeled_regions = np.where(keep_component[labels], labels, 0)
    
    return spec_normalized, labeled_regions


def extract_shape_sequence(spec_normalized, labeled_regions, target_size=(32, 32)):
    """Extract shape sequence using actual intensity values from normalized spectrogram"""
    H, W = labeled_regions.shape
    
    # Get unique label IDs (excluding 0 which is background)
    unique_labels = np.unique(labeled_regions)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return []
    
    shapes_data = []
    
    for label_id in unique_labels:
        coords = np.argwhere(labeled_regions == label_id)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        # Extract mask for this component
        shape_mask = labeled_regions[y0:y1, x0:x1] == label_id
        
        # Extract actual intensity values where mask is True
        intensity_patch = spec_normalized[y0:y1, x0:x1]
        shape_values = intensity_patch * shape_mask  # Zero out non-shape pixels
        
        # Resize to target size preserving intensity information
        shape_resized = transform.resize(shape_values, target_size, anti_aliasing=True, preserve_range=True)
        shape_vec = shape_resized.flatten()
        
        # Normalize
        shape_vec = shape_vec / (np.linalg.norm(shape_vec) + 1e-8)
        
        time_pos = (x0 + x1) / 2 / W
        freq_pos = (y0 + y1) / 2 / H
        duration = (x1 - x0) / W
        bandwidth = (y1 - y0) / H
        
        # Additional intensity statistics
        mean_intensity = float(np.mean(intensity_patch[shape_mask]))
        max_intensity = float(np.max(intensity_patch[shape_mask]))
        
        shapes_data.append({
            'shape_vector': shape_vec.tolist(),
            'time_pos': float(time_pos),
            'freq_pos': float(freq_pos),
            'duration': float(duration),
            'bandwidth': float(bandwidth),
            'mean_intensity': mean_intensity,
            'max_intensity': max_intensity,
        })
    
    shapes_data.sort(key=lambda x: x['time_pos'])
    
    return shapes_data


def visualize_extraction(spec_normalized, regions, shapes_data, output_path):
    """Create visualization of extracted shapes"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top: normalized spectrogram with bounding boxes
    ax = axes[0]
    ax.imshow(spec_normalized, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_title(f'Normalized Spectrogram with Detected Regions ({len(shapes_data)} shapes extracted)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    
    H, W = spec_normalized.shape
    
    # Draw bounding boxes from already-computed shape data
    for i, shape in enumerate(shapes_data):
        x_center = shape['time_pos'] * W
        y_center = shape['freq_pos'] * H
        w = shape['duration'] * W
        h = shape['bandwidth'] * H
        
        rect = patches.Rectangle(
            (x_center - w/2, y_center - h/2), w, h,
            linewidth=1, edgecolor='red', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)
    
    # Bottom: binary mask (show what regions passed the size filter)
    ax = axes[1]
    ax.imshow(regions, aspect='auto', origin='lower', cmap='gray', interpolation='nearest')
    n_pixels = np.sum(regions)
    ax.set_title(f'Detected Signal Regions (Binary Mask) - {n_pixels} pixels in {len(shapes_data)} components')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_shapes(shapes_data, output_path):
    """Visualize individual extracted shapes"""
    n_show = len(shapes_data)
    if n_show == 0:
        return
    
    grid_size = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    if grid_size == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    
    for i in range(n_show):
        shape_vec = np.array(shapes_data[i]['shape_vector']).reshape(32, 32)
        axes[i].imshow(shape_vec, cmap='viridis', interpolation='nearest', origin='lower')
        axes[i].axis('off')
        axes[i].set_title(f't={shapes_data[i]["time_pos"]:.2f}', fontsize=6)
    
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Extracted Shape Vectors (with intensity)', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def extract_sequences_for_folder(data_folder, target_size=(32, 32), visualize=False, max_files=None, with_audio=False):
    data_dir = os.path.join(data_folder, 'data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Missing data folder: {data_dir}")
    
    # Check for audio folder if with_audio is enabled
    audio_dir = os.path.join(data_folder, 'audio')
    if with_audio and not os.path.exists(audio_dir):
        print(f"Warning: --with-audio specified but audio folder not found: {audio_dir}")
        print("Audio files will not be copied.")
        with_audio = False
    
    # Load labels to organize visualizations by class
    labels_file = os.path.join(data_folder, 'labels.json')
    filename_to_classes = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        filename_to_classes = {item['filename']: item['class_names'] for item in labels_data['files']}
    
    # Scan for all .npy files
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    if not all_files:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    # Limit number of files if max_files is specified
    if max_files is not None and max_files < len(all_files):
        print(f"⚠ Limiting to {max_files} files (from {len(all_files)} total)")
        np.random.shuffle(all_files)
        all_files = all_files[:max_files]
    
    print(f"Found {len(all_files)} spectrogram files in {data_dir}")
    
    sequences_dir = os.path.join(data_folder, 'sequences')
    os.makedirs(sequences_dir, exist_ok=True)
    
    viz_dir = None
    if visualize:
        viz_dir = os.path.join(data_folder, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Will save visualizations to {viz_dir}")
        if with_audio:
            print(f"Will save audio files to {viz_dir}/[class_folders]/")
    
    print("\nProcessing spectrograms and extracting sequences...")
    for idx, filename in enumerate(tqdm(all_files, desc="Processing")):
        spec_path = os.path.join(data_dir, filename)
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Missing spectrogram file: {spec_path}")
        
        # Load and process spectrogram
        spec_linear = np.load(spec_path)
        spec_normalized, labeled_regions = process_spectrogram(spec_linear)
        
        # Extract shapes with intensity values
        shapes = extract_shape_sequence(spec_normalized, labeled_regions, target_size=target_size)
        
        # Save sequence
        output_path = os.path.join(sequences_dir, filename.replace('.npy', '.json'))
        sequence_data = {
            'filename': filename,
            'num_shapes': len(shapes),
            'shapes': shapes
        }
        
        with open(output_path, 'w') as f:
            json.dump(sequence_data, f)
        
        # Create visualizations
        if visualize:
            base_name = filename.replace('.npy', '')
            
            # Determine class folder
            classes = filename_to_classes.get(filename, [])
            if len(classes) == 0:
                class_folder = 'unlabeled'
            else:
                class_folder = classes[0].replace('/', '_').replace(' ', '_')
            
            class_viz_dir = os.path.join(viz_dir, class_folder)
            os.makedirs(class_viz_dir, exist_ok=True)
            
            # Copy audio file to visualization folder if requested
            if with_audio:
                audio_source = os.path.join(audio_dir, filename.replace('.npy', '.wav'))
                if os.path.exists(audio_source):
                    audio_dest = os.path.join(class_viz_dir, filename.replace('.npy', '.wav'))
                    shutil.copy2(audio_source, audio_dest)
            
            viz_spec_path = os.path.join(class_viz_dir, f'{base_name}_detection.png')
            viz_shapes_path = os.path.join(class_viz_dir, f'{base_name}_shapes.png')
            
            # Create binary mask for visualization
            binary_mask = (labeled_regions > 0)
            visualize_extraction(spec_normalized, binary_mask, shapes, viz_spec_path)
            visualize_shapes(shapes, viz_shapes_path)
    
    print(f"\n✓ Saved {len(all_files)} sequence files to {sequences_dir}")
    if visualize:
        print(f"✓ Saved visualizations to {viz_dir}")
        if with_audio:
            # Count audio files in viz directory
            audio_count = 0
            for root, dirs, files in os.walk(viz_dir):
                audio_count += len([f for f in files if f.endswith('.wav')])
            print(f"✓ Saved {audio_count} audio files to visualization folders")
    
    return sequences_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract shape sequences from raw spectrograms (with intensity values)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python shape_extract_sequences.py train_data/ test_data/
  
  # With visualizations to verify extraction
  python shape_extract_sequences.py train_data/ --visualize
  
  # With audio files for listening to samples
  python shape_extract_sequences.py train_data/ --with-audio --visualize
  
  # Limit to first 1000 files for testing
  python shape_extract_sequences.py train_data/ --max-files 1000
        """
    )
    
    parser.add_argument('data_folders', nargs='+',
                       help="Folders containing data/ with .npy spectrograms")
    parser.add_argument('--target-size', type=int, nargs=2, default=(32, 32),
                       metavar=('H', 'W'), help="Shape resize target (default: 32 32)")
    parser.add_argument('--visualize', action='store_true',
                       help="Save visualizations of extracted shapes")
    parser.add_argument('--with-audio', action='store_true',
                       help="Copy corresponding audio files to sequences_audio/ folder")
    parser.add_argument('--max-files', type=int, default=None,
                       help="Max files to process (default: None = all)")
    
    args = parser.parse_args()
    
    for folder in args.data_folders:
        print(f"\n{'='*60}")
        print(f"Dataset: {folder}")
        print('='*60)
        extract_sequences_for_folder(
            folder, 
            target_size=tuple(args.target_size),
            visualize=args.visualize,
            max_files=args.max_files,
            with_audio=args.with_audio
        )


if __name__ == '__main__':
    main()
