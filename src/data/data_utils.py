"""
Data utilities for loading and processing bird sound spectrograms.
This module handles data loading, train/test splitting, and PyTorch data generation.
"""

import numpy as np
import os
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, WeightedRandomSampler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.ndimage import label
from .normalizer import normalize_spectrogram


def get_background_spectrogram(img):
    H, W = img.shape
    sorted_pixels = np.sort(img, axis=1)
    bg_pixels = sorted_pixels[:, :W//2]
    
    mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    var0 = np.var(bg_pixels, axis=1, keepdims=True)

    sg_normalized = (img - mu0) / (np.sqrt(var0) + 1e-6)

    for row in range(H):
        outliers = sg_normalized[row, :] > 3
        not_outliers = sg_normalized[row, :] <= 3
        sg_normalized[row, outliers] = np.random.choice(sg_normalized[row, not_outliers], size=np.sum(outliers), replace=True)

    sg_fixed = (sg_normalized * (np.sqrt(var0) + 1e-6)) + mu0

    return sg_fixed


class DataLoader:
    """Handles loading and splitting of spectrogram data."""
    
    def __init__(self, folder, noise_folder=None):
        """
        Initialize DataLoader.
        
        Args:
            folder: Path to folder containing labels.json and data/
            noise_folder: Optional path to folder containing noise data (if different from folder)
        """
        self.folder = folder
        self.noise_folder = noise_folder if noise_folder else folder
        
    def load_data(self, use_multilabel=True, validation_share=0.2):
        """
        Load files and labels from standardized JSON format.
        
        Args:
            use_multilabel: If True, return multi-label vectors; if False, return single-label vectors
            validation_share: Fraction of data to use for validation
        
        Returns:
            Dictionary containing all loaded data
        """
        labels_file = os.path.join(self.folder, "labels.json")
        data_folder = os.path.join(self.folder, "data")
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            label_data = json.load(f)
        
        if 'categories' in label_data:
            categories = label_data['categories']
        elif 'species_list' in label_data:
            categories = label_data['species_list']
        else:
            raise KeyError(f"Neither 'categories' nor 'species_list' found in {labels_file}. Available keys: {list(label_data.keys())}")
        
        filenames = []
        labels = []
        
        # Create category to index mapping
        category_to_idx = {category: idx for idx, category in enumerate(categories)}

        source_files = []
        files_checked = 0
        files_found = 0
        for file_info in label_data['files']:
            filename = file_info['filename']
            file_path = os.path.join(data_folder, filename)
            files_checked += 1

            if os.path.exists(file_path):
                files_found += 1
                filenames.append(file_path)
                source_files.append(file_info.get('source_file'))

                if use_multilabel:
                    label_vector = [0.0] * len(categories)
                    if 'class_names' in file_info:
                        for class_name in file_info['class_names']:
                            if class_name in category_to_idx:
                                label_vector[category_to_idx[class_name]] = 1.0
                    labels.append(label_vector)
        
        # Load noise data if available
        noise_filenames = self._load_noise_data()
        
        labels = np.array(labels, dtype=np.float32)
        mode_str = "multi-label" if use_multilabel else "single-label"
        
        if len(filenames) == 0:
            raise ValueError(
                f"No data files found in {self.folder}.\n"
                f"  Labels file: {labels_file}\n"
                f"  Data folder: {data_folder}\n"
                f"  Files in labels.json: {files_checked}\n"
                f"  Files found on disk: {files_found}\n"
                f"Check that labels.json exists and that .npy files exist in data/ folder."
            )
        
        if len(labels.shape) == 1 or labels.shape[0] == 0:
            raise ValueError(f"No valid labels loaded from {self.folder}. Found {len(filenames)} files but labels array is empty.")
        
        print(f"Loaded {mode_str} data: {len(filenames)} files, {labels.shape[1]} classes")
        
        split_data = self.split_data(
            filenames, labels, noise_filenames, validation_share
        )
        
        # Get class names with proper mapping
        class_names = self._get_class_names(categories)
        
        # Combine all data into a single dictionary
        data = {
            'train_filenames': split_data[0],
            'train_labels': split_data[1],
            'test_filenames': split_data[2],
            'test_labels': split_data[3],
            'train_noise_filenames': split_data[4],
            'test_noise_filenames': split_data[5],
            'categories': categories,
            'class_names': class_names,
            'nclasses': len(categories)
        }
        
        return data
    
    def _load_noise_data(self):
        """Load noise data if available."""
        noise_filenames = []
        noise_labels_file = os.path.join(self.noise_folder, "labels.json")
        
        if os.path.exists(noise_labels_file):
            with open(noise_labels_file, 'r') as f:
                noise_data = json.load(f)
            
            data_folder = os.path.join(self.noise_folder, "data")
            for file_info in noise_data['files']:
                filename = file_info['filename']
                file_path = os.path.join(data_folder, filename)
                if os.path.exists(file_path):
                    noise_filenames.append(file_path)
            print(f"Loaded {len(noise_filenames)} noise files from {self.noise_folder}")
        else:
            print(f"No noise data found in {self.noise_folder}")
        
        return noise_filenames
    
    def split_data(self, filenames, labels, noise_filenames, validation_share):
        """Split data into training and test sets using random split."""
        
        # Handle case where validation is disabled (validation_share == 0)
        if validation_share == 0.0 or validation_share is None:
            print(f"Validation disabled: using all {len(filenames)} files for training")
            return (filenames, labels, [], np.array([]), noise_filenames, [])
        
        # Simple random split
        train_filenames, test_filenames, train_labels, test_labels = train_test_split(
            filenames, labels, test_size=validation_share, random_state=42
        )
        
        # Split noise data if available (for augmentation)
        train_noise_filenames, test_noise_filenames = None, None
        if noise_filenames and len(noise_filenames) > 0:
            train_noise_size = int(len(noise_filenames) * (1 - validation_share))
            train_noise_filenames = noise_filenames[:train_noise_size]
            test_noise_filenames = noise_filenames[train_noise_size:]
            print(f"Split noise data for augmentation: {len(train_noise_filenames)} training and {len(test_noise_filenames)} validation noise files")
        
        print(f"Split data: {len(train_filenames)} training and {len(test_filenames)} validation files")
        return train_filenames, train_labels, test_filenames, test_labels, train_noise_filenames, test_noise_filenames
    
    def _get_class_names(self, categories, name_mapping_path="data/DOC_bird_naming_map.csv"):
        """Get human-readable class names."""
        class_names = []
        name_mapping = {}
        
        # Load bird name mapping if it exists
        if os.path.exists(name_mapping_path):
            name_mapping = self._load_bird_name_mapping(name_mapping_path)
        
        for category_code in categories:
            # If we have a mapping from eBird code to common name, use it
            if category_code in name_mapping:
                class_names.append(name_mapping[category_code])
            else:
                # Use category code as-is, replacing underscores with spaces
                class_names.append(category_code.replace('_', ' '))
        
        return class_names
    
    def _load_bird_name_mapping(self, csv_path):
        """Load the bird name mapping from CSV file."""
        try:
            ebird_to_common = {}
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ebird_to_common[row['eBird']] = row['CommonName']
            return ebird_to_common
        except Exception as e:
            print(f"Warning: Could not load bird name mapping from {csv_path}: {e}")
            return {}


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for loading and augmenting spectrogram data.
    
    The spectrograms are stored in linear domain. Log transformation and noise mixing
    are applied on-the-fly during training for maximum flexibility.
    """
    
    def __init__(self, filenames, labels, img_height, img_width, channels=1, 
                 cropping_mode="center", noise_filenames=None, noise_ratio=0.3, 
                 spec_transform="Log", training=True, width_downsizing=None, normalize=False,
                 normalize_median_filter=True, median_only=False, 
                 use_temporal_roll=True, noise_mode='full', background_prob=0.0):
        """
        Initialize SpectrogramDataset.
        
        Args:
            filenames: List of file paths
            labels: Array of labels
            img_height: Target image height
            img_width: Target image width  
            channels: Number of channels
            cropping_mode: How to crop images ('center' or 'random')
            noise_filenames: List of noise file paths for augmentation
            noise_ratio: Expected noise mixing ratio (samples uniformly from [0, min(2×noise_ratio, 1.0)], clipped to valid range)
            spec_transform: Transform to apply ("Log", "PCEN", "Box-Cox", "Sigmoid", None)
            training: Whether this is training data (affects augmentation)
            width_downsizing: Stride for width downsampling (e.g., 4 means [:, ::4])
            normalize: Whether to apply background normalization
            use_temporal_roll: If True, randomly shift spectrogram along time axis (circular) during training
            noise_mode: How to extract noise - 'full' (mix entire spectrogram), 'background' (extract quiet segments), 'both' (random 50/50)
            background_prob: Probability of replacing sample with its background spectrogram (zeros labels)
        
        Note: For time axis, uses RANDOM SAMPLING (samples from per-frequency distribution)
        instead of zero-padding or tiling to avoid creating distinguishable artifacts.
        """
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)
        self.nclasses = len(labels[0])
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.cropping_mode = cropping_mode
        self.noise_filenames = noise_filenames if noise_filenames else []
        self.noise_ratio = noise_ratio if training else 0.0  # Expected noise ratio (samples from [0, 2×ratio])
        self.spec_transform = spec_transform
        self.training = training
        self.width_downsizing = width_downsizing
        self.normalize = normalize
        self.normalize_median_filter = normalize_median_filter
        self.median_only = median_only
        self.use_temporal_roll = use_temporal_roll if training else False  # Only roll during training
        self.noise_mode = noise_mode
        self.background_prob = background_prob if training else 0.0
        self.rng = np.random.RandomState(21390)
        
        # Cache noise data in memory (WAY faster than loading from disk every time)
        self.noise_cache = []
        if self.noise_filenames:
            print(f"Loading {len(self.noise_filenames)} noise files into memory...")
            for noise_file in self.noise_filenames:
                noise_data = np.load(noise_file)
                if not np.isfinite(noise_data).all():
                    print(f"WARNING: Skipping noise file with NaN/Inf: {noise_file}")
                    continue
                self.noise_cache.append(noise_data)
            print(f"✓ Cached {len(self.noise_cache)} noise files in memory")
        
        # Calculate final dimensions after downsampling
        final_width = img_width // width_downsizing if width_downsizing else img_width
        
        # Calculate patch count (assuming patch size 16x16 for AST/ViT)
        patch_size = 16
        height_patches = img_height // patch_size
        width_patches = final_width // patch_size
        total_patches = height_patches * width_patches
        
        print(f"Dataset initialized with {len(self.filenames)} data files and {len(self.noise_filenames)} noise files")
        print(f"Spectrogram transform: {self.spec_transform}")
        if self.noise_ratio > 0:
            print(f"Training mode: {self.training} (expected noise ratio: {self.noise_ratio}, uniformly sampled [0, min({2*self.noise_ratio:.1f}, 1.0)], clipped)")
        else:
            print(f"Training mode: {self.training} (noise augmentation: disabled)")
        if width_downsizing:
            print(f"Width downsampling: stride={width_downsizing} ({img_width} -> {final_width})")
        if normalize:
            print(f"Background normalization: enabled")
        if self.background_prob > 0:
            print(f"⚡ Background replacement: {self.background_prob*100:.1f}% of samples replaced with background (labels zeroed)")
        print(f"Time-axis padding: RANDOM SAMPLING (samples from per-frequency distribution, no repetition/silence artifacts)")
        print(f"Final image size: {img_height}x{final_width}")
        print(f"AST patches (16x16): {height_patches}x{width_patches} = {total_patches} total patches")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load spectrogram data
        file_path = self.filenames[idx]
        try:
            data = np.load(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load spectrogram file {file_path}: {e}")
        
        # Robust NaN/Inf checking
        if not np.isfinite(data).all():
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            raise ValueError(f"Invalid values in {file_path}: {nan_count} NaNs, {inf_count} Infs")
        
        # Check for extreme values that could cause numerical issues
        if np.abs(data).max() > 1e10:
            print(f"⚠️  WARNING: Extremely large values in {file_path} (max: {np.abs(data).max():.2e})")
        
        # Ensure data is 2D (H, W)
        while data.ndim > 2:
            data = np.squeeze(data)
        assert data.ndim == 2, f"Data should be 2D after squeeze, got {data.shape}"
        
        # Background replacement augmentation: replace with background spectrogram and zero labels
        replace_with_background = False
        if self.training and self.background_prob > 0 and self.rng.rand() < self.background_prob:
            data = get_background_spectrogram(data)
            replace_with_background = True
        
        # Process the spectrogram
        x = self.apply_padding_and_add_channels(data)
        assert x.ndim == 3, f"After padding should be 3D (H,W,C), got {x.shape}"
        
        # Apply temporal roll (random circular shift along time axis) during training
        if self.use_temporal_roll and self.training:
            shift_amount = self.rng.randint(0, x.shape[1])
            x = np.roll(x, shift_amount, axis=1)
        
        x = self.apply_crop(x)
        assert x.ndim == 3, f"After crop should be 3D (H,W,C), got {x.shape}"
        
        # Apply width downsampling if specified
        if self.width_downsizing and self.width_downsizing > 1:
            x = x[:, ::self.width_downsizing, :]
            assert x.ndim == 3, f"After downsampling should be 3D (H,W,C), got {x.shape}"
        
        # Apply noise mixing if training
        # Do not noise-mix explicit background/noise samples (all-zero labels)
        # They should remain pure negatives to teach rejection.
        is_all_zero_label = bool((self.labels[idx].sum() == 0).item())
        if self.training and (not is_all_zero_label) and len(self.noise_filenames) > 0 and self.noise_ratio > 0:
            x = self.mix_with_noise(x)
            assert x.ndim == 3, f"After noise mix should be 3D (H,W,C), got {x.shape}"
        
        # Apply spectrogram transformation
        x = self.apply_spec_transform(x)
        assert x.ndim == 3, f"After transform should be 3D (H,W,C), got {x.shape}"
        
        # Apply background normalization if enabled (normalize OR median_only)
        if self.normalize or getattr(self, 'median_only', False):
            # Convert (H, W, C) to (H, W) for normalization
            x_2d = x[:, :, 0] if x.shape[2] == 1 else x[:, :, 0]  # Take first channel
            use_median = getattr(self, 'normalize_median_filter', True)
            median_only = getattr(self, 'median_only', False)
            x_2d = normalize_spectrogram(x_2d, use_median_filter=use_median, median_only=median_only)
            x[:, :, 0] = x_2d  # Put back in the channel
        
        # SpecAugment (time/frequency masking) during training
        if self.training:
            x = self.apply_specaugment(x)
            assert x.ndim == 3, f"After specaugment should be 3D (H,W,C), got {x.shape}"
        
        # Convert to tensor and ensure correct format: (C, H, W)
        x = torch.FloatTensor(x).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        assert x.ndim == 3, f"After permute should be 3D (C,H,W), got {x.shape}"
        
        # Get label
        y = self.labels[idx]
        if replace_with_background:
            y = torch.zeros_like(y)
        
        return x, y

    def apply_spec_transform(self, sg):
        """Apply spectrogram transformation (log, PCEN, etc.).
        
        Returns transformed spectrogram.
        For PCEN, implements Per-Channel Energy Normalization (arXiv 1607.05666, arXiv 1905.08352v2).
        """
        if self.spec_transform == "None" or self.spec_transform is None:
            return sg
        
        LOG_OFFSET = 1e-7
        
        if self.spec_transform == "Log":
            sg = np.maximum(sg, 0.0)
            sg_safe = sg + LOG_OFFSET
            return np.log(sg_safe)
        
        elif self.spec_transform == "PCEN":
            from scipy import signal
            # Per Channel Energy Normalisation (non-trained version) arXiv 1607.05666, arXiv 1905.08352v2
            gain = 0.8
            bias = 10
            power = 0.25
            t = 0.060
            eps = 1e-6
            
            fs = 16000
            hop_samples = int(0.010 * fs)
            s = 1 - np.exp(-hop_samples / (t * fs))
            
            sg_2d = sg[:, :, 0] if sg.ndim == 3 else sg
            M = signal.lfilter([s], [1, s-1], sg_2d, axis=1)
            smooth = (eps + M)**(-gain)
            pcen = (sg_2d * smooth + bias)**power - bias**power
            
            if sg.ndim == 3:
                result = np.zeros_like(sg)
                result[:, :, 0] = pcen
                return result
            return pcen
        
        elif self.spec_transform == "Box-Cox":
            # boxcox already imported from scipy.stats at top of file
            size = sg.shape
            sg_flat = np.maximum(sg.flatten() + LOG_OFFSET, 1e-10)
            lam = 0.5
            sg_transformed = boxcox(sg_flat, lam)
            return np.reshape(sg_transformed, size)
        
        else:
            print(f"Warning: Unknown transform {self.spec_transform}, using linear")
            return sg

    def apply_padding_and_add_channels(self, array, is_noise=False):
        """Apply padding and ensure correct number of channels.
        
        Time axis padding: Sample from per-frequency-band distribution
        - For each frequency row, randomly sample from existing values to fill padding
        - Preserves per-frequency statistics without creating repetition or silence artifacts
        - Creates statistically-similar but temporally-incoherent padding
        - Distribution is consistent across training/validation/test (RNG state doesn't matter)
        
        Frequency axis: zero-padding (different recordings may have different freq ranges)
        """
        if len(array.shape) == 2:
            array = np.expand_dims(array, axis=-1)

        h, w, c = array.shape
        
        # Frequency axis (height): zero-pad if needed
        if h < self.img_height:
            pad_h = self.img_height - h
            array = np.concatenate([array, np.zeros((pad_h, w, c))], axis=0)
        
        # Time axis (width): PAD WITH RANDOM SAMPLES FROM PER-FREQUENCY DISTRIBUTION
        if w < self.img_width:
            pad_w = self.img_width - w
            # For each frequency band, sample from existing values
            padded_section = np.zeros((array.shape[0], pad_w, c))
            for row in range(array.shape[0]):
                for ch in range(c):
                    # Sample from this frequency band's existing values
                    padded_section[row, :, ch] = self.rng.choice(
                        array[row, :, ch], 
                        size=pad_w, 
                        replace=True
                    )
            array = np.concatenate([array, padded_section], axis=1)

        # Channel axis: pad with zeros if needed
        if c < self.channels:
            pad_c = self.channels - c
            array = np.concatenate([array, np.zeros((*array.shape[:2], pad_c))], axis=-1)
        elif c > self.channels:
            array = array[:, :, :self.channels]

        return array

    def mix_with_noise(self, bird_spectrogram):
        """Mix bird spectrogram with noise spectrogram.
        
        Uses a fixed noise ratio for deterministic, interpretable augmentation:
        - noise_ratio=0.4 means 40% noise + 60% signal on every sample
        - noise_ratio capped at <1.0 to avoid pure noise (label-contradicting) samples
        
        Supports two modes:
        - 'full': Mix entire noise spectrogram (traditional approach)
        - 'background': Extract quiet segments from noise/training files (smart noise)
        - 'both': Randomly choose between full and background (50/50)
        """
        if not self.noise_cache or self.noise_ratio <= 0:
            return bird_spectrogram
        
        # Determine which mode to use
        use_background_mode = False
        if self.noise_mode == 'background':
            use_background_mode = True
        elif self.noise_mode == 'both':
            use_background_mode = self.rng.rand() < 0.5
        # else: mode is 'full', keep use_background_mode = False
        
        # Use fixed noise ratio directly (deterministic per sample)
        # Clamp to [0, 0.95] to prevent pure noise while allowing high-noise training
        actual_noise_ratio = np.clip(self.noise_ratio, 0.0, 0.95)
        
        # Use cached noise data (much faster than loading from disk!)
        noise_idx = self.rng.randint(0, len(self.noise_cache))
        noise_data = self.noise_cache[noise_idx]
        
        # Process noise with zero-padding for height, tiling for width
        noise_processed = self.apply_padding_and_add_channels(noise_data, is_noise=True)
        
        if use_background_mode:
            # BACKGROUND MODE: Extract quiet segments only
            # Get target dimensions from bird spectrogram
            target_width = bird_spectrogram.shape[1]
            
            # Extract background noise from quiet regions
            noise_cropped = self.extract_background_noise(
                noise_processed, 
                target_width, 
                quietest_percentile=20
            )
        else:
            # FULL MODE: Use entire noise spectrogram (traditional)
            # Apply random crop to noise (always random for noise)
            original_cropping_mode = self.cropping_mode
            self.cropping_mode = "random"
            noise_cropped = self.apply_crop(noise_processed)
            self.cropping_mode = original_cropping_mode
        
        # Apply width downsampling to noise if specified (to match bird spectrogram)
        if self.width_downsizing and self.width_downsizing > 1:
            noise_cropped = noise_cropped[:, ::self.width_downsizing, :]
        
        # Ensure positive values (magnitude spectrograms)
        bird_linear = np.maximum(bird_spectrogram, 0)
        noise_linear = np.maximum(noise_cropped, 0)
        
        # Scale noise to match bird's energy level (RMS-based scaling for volume matching)
        bird_energy = np.sqrt(np.mean(bird_linear**2))
        noise_energy = np.sqrt(np.mean(noise_linear**2))
        
        energy_scale = bird_energy / noise_energy
        noise_linear_scaled = noise_linear * energy_scale
        
        # Mix with randomly sampled ratio
        mixed_linear = (1.0 - actual_noise_ratio) * bird_linear + actual_noise_ratio * noise_linear_scaled
        
        return mixed_linear

    def apply_crop(self, array):
        h, w, c = array.shape
        assert self.cropping_mode in ["center", "random"]
        
        if self.cropping_mode == "center":
            start_row = (h - self.img_height) // 2 if h > self.img_height else 0
            start_col = (w - self.img_width) // 2 if w > self.img_width else 0
        elif self.cropping_mode == "random":
            start_row = self.rng.randint(0, h - self.img_height + 1) if h > self.img_height else 0
            start_col = self.rng.randint(0, w - self.img_width + 1) if w > self.img_width else 0
        
        return array[start_row:start_row + self.img_height, start_col:start_col + self.img_width]
    
    def extract_background_noise(self, source_spectrogram, target_width, quietest_percentile=20):
        """Extract background noise from quiet regions of a spectrogram.
        
        This identifies low-energy time columns (likely background/silence) and 
        extracts them to use as realistic noise augmentation.
        
        Args:
            source_spectrogram: Spectrogram to extract noise from (H, W, C)
            target_width: Target width for the output noise
            quietest_percentile: Percentile threshold for "quiet" columns (default: 20 = bottom 20%)
        
        Returns:
            Extracted background noise spectrogram (H, target_width, C)
        """
        h, w, c = source_spectrogram.shape
        
        # Remove channel dimension for energy calculation
        spec_2d = source_spectrogram[:, :, 0] if c == 1 else source_spectrogram.mean(axis=2)
        
        # Calculate energy per time column (sum over frequency axis)
        energy_per_column = spec_2d.sum(axis=0)  # Shape: (w,)
        
        # Find columns below energy threshold (quietest X%)
        threshold = np.percentile(energy_per_column, quietest_percentile)
        quiet_indices = np.where(energy_per_column <= threshold)[0]
        
        if len(quiet_indices) == 0:
            # Fallback: if no quiet columns found, use columns with minimum energy
            num_quiet = max(1, int(w * quietest_percentile / 100))
            quiet_indices = np.argsort(energy_per_column)[:num_quiet]
        
        # Strategy: Randomly sample from quiet columns with replacement
        # This creates more variety than simple tiling
        sampled_indices = self.rng.choice(quiet_indices, size=target_width, replace=True)
        
        # Extract columns
        background_noise = source_spectrogram[:, sampled_indices, :]  # (H, target_width, C)
        
        return background_noise

    def apply_specaugment(self, x):
        """Apply spectrogram augmentations: time stretch, frequency shift, and time/frequency masking.
        x is (H, W, C)
        
        Augmentations applied:
        1. Time stretching (horizontal resize with interpolation)
        2. Frequency shifting (vertical shift with wrapping)
        3. SpecAugment masking (time and frequency masks)
        """
        h, w, c = x.shape
        
        # 1. Time stretching (with 50% probability)
        if self.rng.rand() < 0.5:
            from src.core import config
            from scipy.ndimage import zoom
            
            stretch_factor = self.rng.uniform(config.DEFAULT_TIME_STRETCH_RANGE[0], 
                                             config.DEFAULT_TIME_STRETCH_RANGE[1])
            
            # Stretch along time axis
            x_stretched = zoom(x, (1.0, stretch_factor, 1.0), order=1)
            
            # Ensure exact output width by cropping or padding
            stretched_w = x_stretched.shape[1]
            if stretched_w > w:
                start = (stretched_w - w) // 2
                x = x_stretched[:, start:start+w, :]
            elif stretched_w < w:
                pad = w - stretched_w
                pad_left = pad // 2
                pad_right = pad - pad_left
                x = np.pad(x_stretched, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
            else:
                x = x_stretched
            
            # Final safety check: ensure exact dimensions
            if x.shape[1] != w:
                x = x[:, :w, :]
        
        # 2. Frequency shifting (with 50% probability)
        if self.rng.rand() < 0.5:
            from src.core import config
            freq_shift = self.rng.randint(config.DEFAULT_FREQ_SHIFT_RANGE[0], 
                                         config.DEFAULT_FREQ_SHIFT_RANGE[1] + 1)
            if freq_shift != 0:
                x = np.roll(x, freq_shift, axis=0)
        
        # 3. SpecAugment masking
        freq_mask_param = min(48, h)
        time_mask_param = min(192, max(1, int(0.2 * w)))
        nm = 2
        
        for _ in range(nm):
            # Frequency masking
            f = self.rng.randint(0, freq_mask_param + 1)
            if f > 0 and h - f > 0:
                f0 = self.rng.randint(0, h - f + 1)
                x[f0:f0 + f, :, :] = 0
            
            # Time masking
            t = self.rng.randint(0, time_mask_param + 1)
            if t > 0 and w - t > 0:
                t0 = self.rng.randint(0, w - t + 1)
                x[:, t0:t0 + t, :] = 0
        
        return x


def sparse_collate_fn(batch):
    """
    Custom collate function for sparse patch data.
    
    Batches the sparse patches from multiple samples.
    
    Args:
        batch: List of dicts with keys: patches, positions, mask, label
    
    Returns:
        Dictionary with batched tensors
    """
    patches_list = [item['patches'] for item in batch]  # List of (K, 16, 16)
    positions_list = [item['positions'] for item in batch]  # List of (K, 2)
    masks_list = [item['mask'] for item in batch]  # List of (K,)
    labels_list = [item['label'] for item in batch]  # List of label tensors
    
    # Stack into batches
    patches = torch.stack(patches_list, dim=0)  # (B, K, 16, 16)
    
    # Verify shape before adding channel dimension
    assert patches.dim() == 4, f"Expected 4D after stack, got {patches.dim()}D with shape {patches.shape}"
    
    # Add channel dimension: (B, K, 16, 16) -> (B, K, 1, 16, 16)
    B, K, H, W = patches.shape
    patches = patches.view(B, K, 1, H, W)
    
    positions = torch.stack(positions_list, dim=0)  # (B, K, 2)
    masks = torch.stack(masks_list, dim=0)  # (B, K)
    labels = torch.stack(labels_list, dim=0)  # (B, num_classes)
    
    return {
        'patches': patches,  # (B, K, 1, 16, 16)
        'positions': positions,  # (B, K, 2)
        'mask': masks,  # (B, K)
        'label': labels  # (B, num_classes)
    }


def create_data_loaders(data, batch_size, img_height, img_width, channels=1, 
                       cropping_mode="center", noise_ratio=0.3, spec_transform=None, 
                       num_workers=4, width_downsizing=None, mixup_alpha=0.0,
                       use_class_balancing=False, normalize=False, normalize_median_filter=True,
                       median_only=False, use_temporal_roll=True,
                       mixup_mode='mixup', noise_mode='full', background_prob=0.0):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        data: Dictionary containing train/test filenames and labels
        batch_size: Batch size for training
        img_height: Target image height
        img_width: Target image width
        channels: Number of channels
        cropping_mode: How to crop images ('center' or 'random')
        noise_ratio: Expected noise mixing ratio (samples uniformly from [0, min(2×noise_ratio, 1.0)], clipped to valid range)
        spec_transform: Spectrogram transformation (None uses config default)
        num_workers: Number of workers for data loading
        width_downsizing: Stride for width downsampling (e.g., 4 means [:, ::4])
        mixup_alpha: Mixup alpha parameter (0 = disabled, 0.3-0.5 recommended)
        use_class_balancing: If True, balance classes using WeightedRandomSampler
        normalize: If True, apply background normalization to spectrograms
        normalize_median_filter: If True (default), use median filter during normalization
        median_only: If True, apply only median filter without background subtraction
        use_temporal_roll: If True, randomly shift spectrogram along time axis (circular) during training
        mixup_mode: Augmentation mode when mixup_alpha > 0: 'mixup', 'cutmix', or 'both' (default: 'mixup')
        noise_mode: Noise extraction mode: 'full' (mix entire spectrogram), 'background' (extract quiet segments), 'both' (random 50/50)
        background_prob: Probability of replacing sample with background spectrogram and zeroing labels
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Use default from config if not specified
    if spec_transform is None:
        from src.core import config
        spec_transform = config.DEFAULT_SPEC_TRANSFORM
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        data['train_filenames'], data['train_labels'], 
        img_height, img_width, channels, cropping_mode,
        noise_filenames=data['train_noise_filenames'], 
        noise_ratio=noise_ratio,
        spec_transform=spec_transform,
        training=True,
        width_downsizing=width_downsizing,
        normalize=normalize,
        normalize_median_filter=normalize_median_filter,
        median_only=median_only,
        use_temporal_roll=use_temporal_roll,
        noise_mode=noise_mode,
        background_prob=background_prob
    )
    
    # Only create validation dataset if validation data exists
    if len(data['test_filenames']) > 0:
        val_dataset = SpectrogramDataset(
            data['test_filenames'], data['test_labels'], 
            img_height, img_width, channels, 'center',  # Always use center crop for validation
            noise_filenames=None,  # No noise for validation
            noise_ratio=0.0,
            spec_transform=spec_transform,
            training=False,
            width_downsizing=width_downsizing,
            normalize=normalize,
            normalize_median_filter=normalize_median_filter,
            median_only=median_only,
            use_temporal_roll=False,  # Never roll validation data
            noise_mode='full',  # Not used (no noise in validation)
            background_prob=0.0  # No background replacement for validation
        )
    else:
        val_dataset = None
    
    # Class balancing setup
    sampler = None
    shuffle_train = True
    
    if use_class_balancing:
        # Calculate class weights for balanced sampling
        train_labels_array = np.array(data['train_labels'])
        
        # For single-label: get class indices
        if train_labels_array.ndim == 2 and not np.all((train_labels_array == 0) | (train_labels_array == 1)):
            # Multi-hot or soft labels - use argmax for balancing
            class_indices = np.argmax(train_labels_array, axis=1)
        else:
            class_indices = np.argmax(train_labels_array, axis=1)
        
        # Count samples per class
        unique_classes, class_counts = np.unique(class_indices, return_counts=True)
        
        # Check if any classes are missing
        total_classes = train_labels_array.shape[1] if train_labels_array.ndim == 2 else max(class_indices) + 1
        missing_classes = set(range(total_classes)) - set(unique_classes)
        if missing_classes:
            print(f"⚠️  WARNING: {len(missing_classes)} classes have no samples in training set!")
            print(f"  Missing class indices: {sorted(missing_classes)}")
            if 'class_names' in data:
                missing_names = [data['class_names'][i] for i in sorted(missing_classes)]
                print(f"  Missing class names: {missing_names[:5]}{'...' if len(missing_names) > 5 else ''}")
            print(f"  These classes will be ignored during training.")
        
        # Calculate class weights (inverse frequency)
        class_weights = 1.0 / class_counts
        
        # Cap extreme weights to prevent NaN issues (max 10x the median)
        median_weight = np.median(class_weights)
        max_weight = 10.0 * median_weight
        num_capped = (class_weights > max_weight).sum()
        if num_capped > 0:
            print(f"  ⚠️  Capping {num_capped} extreme class weights (>{max_weight:.2f}) to prevent instability")
            class_weights = np.clip(class_weights, None, max_weight)
        
        # Normalize so the sum equals number of classes (keeps relative importance)
        class_weights = class_weights * len(unique_classes) / class_weights.sum()
        
        # Create sample weights
        sample_weights = np.zeros(len(class_indices))
        for cls_idx, cls_weight in zip(unique_classes, class_weights):
            sample_weights[class_indices == cls_idx] = cls_weight
        
        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow sampling with replacement
        )
        shuffle_train = False  # Can't use shuffle with sampler
        
        print(f"Class balancing enabled:")
        print(f"  Classes found: {len(unique_classes)}/{total_classes} total")
        print(f"  Sample counts: min={class_counts.min()}, max={class_counts.max()}, mean={class_counts.mean():.1f}")
        print(f"  Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
        
        # Extra warning for very imbalanced datasets
        if class_counts.max() / class_counts.min() > 100:
            print(f"  ⚠️  Highly imbalanced dataset! Max/min ratio: {class_counts.max() / class_counts.min():.1f}x")
            print(f"      Consider using --confusion-sampling instead of --balance for better stability")
    
    # Create data loaders
    # Determine collate function based on mode
    if mixup_alpha > 0:
        # Standard mode with mixup/cutmix
        if mixup_mode == 'mixup':
            print(f"Mixup enabled with alpha={mixup_alpha}")
            train_collate_fn = MixupCollate(mixup_alpha)
        elif mixup_mode == 'cutmix':
            print(f"CutMix enabled with alpha={mixup_alpha}")
            train_collate_fn = CutMixCollate(mixup_alpha)
        elif mixup_mode == 'both':
            print(f"Mixup + CutMix enabled with alpha={mixup_alpha} (50/50 mix)")
            train_collate_fn = MixupCutMixCollate(mixup_alpha, mixup_prob=0.5)
        else:
            raise ValueError(f"Invalid mixup_mode: {mixup_mode}. Must be 'mixup', 'cutmix', or 'both'")
        val_collate_fn = None
    else:
        # Standard mode without mixup
        train_collate_fn = None
        val_collate_fn = None
    
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    # Only create validation loader if validation dataset exists
    if val_dataset is not None:
        val_loader = TorchDataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_collate_fn
        )
    else:
        val_loader = None
    
    return train_loader, val_loader


class MixupCollate:
    """Collate function that applies mixup augmentation to batches.
    
    Mixup is applied at SPECTROGRAM level (linear domain before log transform),
    which is computationally efficient and preserves spectrogram structure.
    This is much cheaper than audio-level mixing which requires regenerating spectrograms.
    """
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Mixup interpolation strength (beta distribution parameter)
                   Recommended: 0.2-0.3 for spectrograms (lower than audio-level)
        """
        self.alpha = alpha
    
    def __call__(self, batch):
        """Apply mixup to a batch of samples.
        
        Args:
            batch: List of (data, label) tuples from dataset
            
        Returns:
            Mixed batch (data_tensor, label_tensor)
            
        Note: Spectrograms are already in log domain when they arrive here,
              so we're mixing log-spectrograms. This works well in practice.
        """
        # Default collate
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        
        if self.alpha > 0:
            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Random permutation of batch
            batch_size = data.size(0)
            index = torch.randperm(batch_size)
            
            # Identify which samples are all-zero (noise samples)
            # Don't mixup noise samples - they need to stay exactly zero
            zero_mask = (labels.sum(dim=1) == 0)  # Shape: (batch_size,)
            zero_mask_permuted = zero_mask[index]
            
            # Only mix if BOTH samples are non-zero (both have birds)
            can_mix = ~zero_mask & ~zero_mask_permuted  # Shape: (batch_size,)
            
            # Mix inputs and labels only where allowed
            mixed_data = data.clone()
            mixed_labels = labels.clone()
            
            if can_mix.any():
                mixed_data[can_mix] = lam * data[can_mix] + (1 - lam) * data[index][can_mix]
                mixed_labels[can_mix] = lam * labels[can_mix] + (1 - lam) * labels[index][can_mix]
            
            return mixed_data, mixed_labels
        else:
            return data, labels


class CutMixCollate:
    """Collate function that applies CutMix augmentation to batches.
    
    CutMix cuts rectangular regions from one spectrogram and pastes them onto another,
    unlike Mixup which blends entire spectrograms. This preserves local structure better
    and forces models to recognize birds from partial spectrograms.
    
    Reference: CutMix: Regularization Strategy to Train Strong Classifiers with 
               Localizable Features (https://arxiv.org/abs/1905.04899)
    """
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: CutMix interpolation strength (beta distribution parameter)
                   Controls the size of the cut region. Same semantics as Mixup.
        """
        self.alpha = alpha
    
    def __call__(self, batch):
        """Apply CutMix to a batch of samples.
        
        Args:
            batch: List of (data, label) tuples from dataset
            
        Returns:
            Mixed batch (data_tensor, label_tensor)
        """
        # Default collate
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        
        if self.alpha > 0:
            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Random permutation of batch
            batch_size = data.size(0)
            index = torch.randperm(batch_size)
            
            # Identify which samples are all-zero (noise samples)
            # Don't cutmix noise samples - they need to stay exactly zero
            zero_mask = (labels.sum(dim=1) == 0)  # Shape: (batch_size,)
            zero_mask_permuted = zero_mask[index]
            
            # Only mix if BOTH samples are non-zero (both have birds)
            can_mix = ~zero_mask & ~zero_mask_permuted  # Shape: (batch_size,)
            
            if can_mix.any():
                # Generate random bounding box
                # Box area should be proportional to (1 - lam) so that lambda represents
                # the proportion of the ORIGINAL image kept
                _, _, H, W = data.shape
                cut_ratio = np.sqrt(1.0 - lam)  # Square root for 2D area
                cut_h = int(H * cut_ratio)
                cut_w = int(W * cut_ratio)
                
                # Random center point for the box
                cx = np.random.randint(W)
                cy = np.random.randint(H)
                
                # Bounding box coordinates (with clipping)
                x1 = np.clip(cx - cut_w // 2, 0, W)
                x2 = np.clip(cx + cut_w // 2, 0, W)
                y1 = np.clip(cy - cut_h // 2, 0, H)
                y2 = np.clip(cy + cut_h // 2, 0, H)
                
                # Actual lambda based on the realized box area
                actual_lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (H * W)
                
                # Apply CutMix: paste cut region from permuted batch onto original
                mixed_data = data.clone()
                mixed_data[can_mix, :, y1:y2, x1:x2] = data[index][can_mix, :, y1:y2, x1:x2]
                
                # Mix labels according to actual area ratio
                mixed_labels = actual_lam * labels.clone()
                mixed_labels[can_mix] = actual_lam * labels[can_mix] + (1.0 - actual_lam) * labels[index][can_mix]
                
                return mixed_data, mixed_labels
            else:
                return data, labels
        else:
            return data, labels


class MixupCutMixCollate:
    """Collate function that randomly applies either Mixup or CutMix.
    
    Combines both augmentation strategies for maximum regularization.
    """
    
    def __init__(self, alpha=0.3, mixup_prob=0.5):
        """
        Args:
            alpha: Interpolation strength for both methods
            mixup_prob: Probability of using mixup (vs cutmix)
        """
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.mixup_collate = MixupCollate(alpha)
        self.cutmix_collate = CutMixCollate(alpha)
    
    def __call__(self, batch):
        """Randomly apply either Mixup or CutMix."""
        if np.random.rand() < self.mixup_prob:
            return self.mixup_collate(batch)
        else:
            return self.cutmix_collate(batch)


def get_dataset_info(folder):
    """Get dataset information from labels file."""
    labels_file = os.path.join(folder, "labels.json")
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    return {
        'num_classes': data['num_classes'],
        'class_names': data['categories']
    }


def extract_signal_regions(spec_linear, log_offset=1e-7, eps=1e-6, thresh_percentile=90.0, min_component_size=20):
    """
    Extract signal regions from a spectrogram (same as shape_extract_sequences.py).
    
    Args:
        spec_linear: Raw linear spectrogram
        log_offset: Offset for log conversion
        eps: Small epsilon for numerical stability
        thresh_percentile: Percentile threshold for energy
        min_component_size: Minimum component size in pixels
    
    Returns:
        spec_normalized: Normalized spectrogram (H, W)
        labeled_regions: Integer array where each connected component has a unique label (H, W)
    """
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
    
    labels_array, _ = label(high_energy_mask)
    component_sizes = np.bincount(labels_array.ravel())
    
    keep_component = component_sizes >= min_component_size
    keep_component[0] = False
    
    labeled_regions = np.where(keep_component[labels_array], labels_array, 0)
    
    return spec_normalized, labeled_regions

