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
from normalizer import normalize_spectrogram


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
        primary_species = []
        
        # Create category to index mapping
        category_to_idx = {category: idx for idx, category in enumerate(categories)}

        source_files = []
        for file_info in label_data['files']:
            filename = file_info['filename']
            file_path = os.path.join(data_folder, filename)

            if os.path.exists(file_path):
                filenames.append(file_path)
                primary_species.append(file_info['primary_class'])
                source_files.append(file_info.get('source_file'))

                if use_multilabel:
                    label_vector = [0.0] * len(categories)
                    for class_name in file_info['class_names']:
                        if class_name in category_to_idx:
                            label_vector[category_to_idx[class_name]] = 1.0
                    labels.append(label_vector)
                else:
                    label_vector = [0.0] * len(categories)
                    if file_info['primary_class'] in category_to_idx:
                        label_vector[category_to_idx[file_info['primary_class']]] = 1.0
                    else:
                        label_vector[0] = 1.0
                    labels.append(label_vector)
        
        # Load noise data if available
        noise_filenames = self._load_noise_data()
        
        labels = np.array(labels, dtype=np.float32)
        mode_str = "multi-label" if use_multilabel else "single-label"
        print(f"Loaded {mode_str} data: {len(filenames)} files, {labels.shape[1]} classes")
        
        # Random stratified split
        split_data = self.split_data(
            filenames, labels, primary_species, noise_filenames, validation_share
        )
        
        # Get class names with proper mapping
        class_names = self._get_class_names(categories)
        
        # Combine all data into a single dictionary
        data = {
            'train_filenames': split_data[0],
            'train_labels': split_data[1],
            'test_filenames': split_data[2],
            'test_labels': split_data[3],
            'train_primary_species': split_data[4],
            'test_primary_species': split_data[5],
            'train_noise_filenames': split_data[6],
            'test_noise_filenames': split_data[7],
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
    
    def split_data(self, filenames, labels, primary_species, noise_filenames, validation_share):
        """Split data into training and test sets using random stratified split."""
        if primary_species is not None:
            train_filenames, test_filenames, train_labels, test_labels, train_primary_species, test_primary_species = train_test_split(
                filenames, labels, primary_species, test_size=validation_share, random_state=42, stratify=primary_species
            )
        else:
            train_filenames, test_filenames, train_labels, test_labels = train_test_split(
                filenames, labels, test_size=validation_share, random_state=42
            )
            train_primary_species, test_primary_species = None, None
        
        # Split noise data if available
        train_noise_filenames, test_noise_filenames = None, None
        if noise_filenames and len(noise_filenames) > 0:
            train_noise_size = int(len(noise_filenames) * (1 - validation_share))
            train_noise_filenames = noise_filenames[:train_noise_size]
            test_noise_filenames = noise_filenames[train_noise_size:]
            print(f"Split noise data: {len(train_noise_filenames)} training and {len(test_noise_filenames)} validation noise files")
        
        print(f"Split data: {len(train_filenames)} training and {len(test_filenames)} validation files")
        return train_filenames, train_labels, test_filenames, test_labels, train_primary_species, test_primary_species, train_noise_filenames, test_noise_filenames
    
    def _get_class_names(self, categories, name_mapping_path="DOC_bird_naming_map.csv"):
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
                 use_sparse_patches=False, num_sparse_patches=20):
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
            noise_ratio: Ratio for mixing noise
            spec_transform: Transform to apply ("Log", "PCEN", "Box-Cox", "Sigmoid", None)
            training: Whether this is training data (affects augmentation)
            width_downsizing: Stride for width downsampling (e.g., 4 means [:, ::4])
            normalize: Whether to apply background normalization
            use_sparse_patches: If True, only extract patches with signal (sparse attention)
            num_sparse_patches: Number of patches to extract in sparse mode (K)
        """
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)
        self.nclasses = len(labels[0])
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.cropping_mode = cropping_mode
        self.noise_filenames = noise_filenames if noise_filenames else []
        self.noise_ratio = noise_ratio if training else 0.0  # No noise during validation
        self.spec_transform = spec_transform
        self.training = training
        self.width_downsizing = width_downsizing
        self.normalize = normalize
        self.use_sparse_patches = use_sparse_patches
        self.num_sparse_patches = num_sparse_patches
        self.rng = np.random.RandomState(21390)
        
        # Calculate final dimensions after downsampling
        final_width = img_width // width_downsizing if width_downsizing else img_width
        
        # Calculate patch count (assuming patch size 16x16 for AST/ViT)
        patch_size = 16
        height_patches = img_height // patch_size
        width_patches = final_width // patch_size
        total_patches = height_patches * width_patches
        
        print(f"Dataset initialized with {len(self.filenames)} data files and {len(self.noise_filenames)} noise files")
        print(f"Spectrogram transform: {self.spec_transform}")
        print(f"Training mode: {self.training} (noise ratio: {self.noise_ratio})")
        if width_downsizing:
            print(f"Width downsampling: stride={width_downsizing} ({img_width} -> {final_width})")
        if normalize:
            print(f"Background normalization: enabled")
        if use_sparse_patches:
            print(f"⚡ Sparse patch mode: extracting top {num_sparse_patches} patches by signal density")
            print(f"   Standard mode would use {total_patches} patches, sparse uses {num_sparse_patches} ({100*num_sparse_patches/total_patches:.1f}%)")
        else:
            print(f"Final image size: {img_height}x{final_width}")
            print(f"AST patches (16x16): {height_patches}x{width_patches} = {total_patches} total patches")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Load spectrogram data
        file_path = self.filenames[idx]
        data = np.load(file_path)
        
        # Ensure data is 2D (H, W)
        while data.ndim > 2:
            data = np.squeeze(data)
        assert data.ndim == 2, f"Data should be 2D after squeeze, got {data.shape}"
        
        # Sparse patch mode: extract signal-rich patches only
        if self.use_sparse_patches:
            # Keep original linear data for signal extraction
            data_linear = data.copy()
            
            # Extract signal regions using the same method as shape_extract_sequences.py
            spec_normalized, labeled_regions = extract_signal_regions(data_linear)
            
            # Resize to target dimensions
            from scipy.ndimage import zoom
            if spec_normalized.shape != (self.img_height, self.img_width):
                h_ratio = self.img_height / spec_normalized.shape[0]
                w_ratio = self.img_width / spec_normalized.shape[1]
                spec_normalized = zoom(spec_normalized, (h_ratio, w_ratio), order=1)
                labeled_regions = zoom(labeled_regions, (h_ratio, w_ratio), order=0)  # Nearest neighbor for labels
            
            # Extract sparse patches (top-K by signal density)
            patches, positions, mask = extract_sparse_patches(
                spec_normalized, labeled_regions, 
                num_patches=self.num_sparse_patches, 
                patch_size=16
            )
            
            # patches: (K, 16, 16), positions: (K, 2), mask: (K,)
            # Convert to tensors - keep patches as (K, 16, 16) for now
            # Channel dimension will be added during batching
            patches_tensor = torch.FloatTensor(patches)  # (K, 16, 16)
            positions_tensor = torch.LongTensor(positions)  # (K, 2)
            mask_tensor = torch.BoolTensor(mask)  # (K,)
            
            # Get label
            y = self.labels[idx]
            
            # Return patches, positions, mask, and label
            # Custom collate function will handle batching
            return {
                'patches': patches_tensor,  # (K, 16, 16)
                'positions': positions_tensor,  # (K, 2)
                'mask': mask_tensor,  # (K,)
                'label': y
            }
        
        # Standard mode: process full spectrogram
        # Process the spectrogram
        x = self.apply_padding_and_add_channels(data)
        assert x.ndim == 3, f"After padding should be 3D (H,W,C), got {x.shape}"
        
        x = self.apply_crop(x)
        assert x.ndim == 3, f"After crop should be 3D (H,W,C), got {x.shape}"
        
        # Apply width downsampling if specified
        if self.width_downsizing and self.width_downsizing > 1:
            x = x[:, ::self.width_downsizing, :]
            assert x.ndim == 3, f"After downsampling should be 3D (H,W,C), got {x.shape}"
        
        # Apply noise mixing if training
        if self.training and len(self.noise_filenames) > 0 and self.noise_ratio > 0:
            x = self.mix_with_noise(x)
            assert x.ndim == 3, f"After noise mix should be 3D (H,W,C), got {x.shape}"
        
        # Apply spectrogram transformation
        x = self.apply_spec_transform(x)
        assert x.ndim == 3, f"After transform should be 3D (H,W,C), got {x.shape}"
        
        # Apply background normalization if enabled
        if self.normalize:
            # Convert (H, W, C) to (H, W) for normalization
            x_2d = x[:, :, 0] if x.shape[2] == 1 else x[:, :, 0]  # Take first channel
            x_2d = normalize_spectrogram(x_2d)
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
        
        return x, y

    def apply_spec_transform(self, sg):
        """Apply spectrogram transformation (log, PCEN, etc.).
        
        Returns log-mel spectrogram without normalization.
        Normalization happens in the model forward pass using AudioSet statistics.
        """
        if self.spec_transform == "None" or self.spec_transform is None:
            # Return linear magnitude as-is (not recommended for AST)
            return sg
        else:
            LOG_OFFSET = 1e-7
            
            if self.spec_transform == "Log":
                sg_safe = sg + LOG_OFFSET
                return np.log(sg_safe)
                
            else:
                print(f"Warning: Unknown transform {self.spec_transform}, using linear")
                return sg

    def apply_padding_and_add_channels(self, array, is_noise=False):
        """Apply padding and ensure correct number of channels. For noise, tile horizontally but pad vertically."""
        if len(array.shape) == 2:
            array = np.expand_dims(array, axis=-1)

        h, w, c = array.shape
        
        if is_noise:
            # For noise files, zero-pad height (frequencies) but tile width (time)
            if h < self.img_height:
                pad_h = self.img_height - h
                array = np.concatenate([array, np.zeros((pad_h, w, c))], axis=0)
            if w < self.img_width:
                tiles_w = int(np.ceil(self.img_width / w))
                array = np.tile(array, (1, tiles_w, 1))
        else:
            # For bird spectrograms, use zero-padding as before
            if h < self.img_height:
                pad_h = self.img_height - h
                array = np.concatenate([array, np.zeros((pad_h, w, c))], axis=0)
            if w < self.img_width:
                pad_w = self.img_width - w
                array = np.concatenate([array, np.zeros((array.shape[0], pad_w, c))], axis=1)

        if c < self.channels:
            pad_c = self.channels - c
            array = np.concatenate([array, np.zeros((*array.shape[:2], pad_c))], axis=-1)
        elif c > self.channels:
            array = array[:, :, :self.channels]

        return array

    def mix_with_noise(self, bird_spectrogram):
        """Mix bird spectrogram with noise spectrogram."""
        if not self.noise_filenames or self.noise_ratio <= 0:
            return bird_spectrogram
        
        noise_file = self.rng.choice(self.noise_filenames)
        noise_data = np.load(noise_file)
        
        # Process noise with zero-padding for height, tiling for width
        noise_processed = self.apply_padding_and_add_channels(noise_data, is_noise=True)
        
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
        
        mixed_linear = (1.0 - self.noise_ratio) * bird_linear + self.noise_ratio * noise_linear_scaled
        
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

    def apply_specaugment(self, x):
        """Apply time and frequency masking (SpecAugment) on numpy array.
        x is (H, W, C)
        
        Uses stronger masking parameters to prevent overfitting on small datasets.
        Frequency masking: up to 48 bins (paper value) capped by height.
        Time masking: up to min(192, 0.2 * width) (scaled for shorter clips).
        """
        h, w, c = x.shape
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
                       use_class_balancing=False, normalize=False,
                       use_sparse_patches=False, num_sparse_patches=20):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        data: Dictionary containing train/test filenames and labels
        batch_size: Batch size for training
        img_height: Target image height
        img_width: Target image width
        channels: Number of channels
        cropping_mode: How to crop images ('center' or 'random')
        noise_ratio: Ratio for mixing noise during training
        spec_transform: Spectrogram transformation (None uses config default)
        num_workers: Number of workers for data loading
        width_downsizing: Stride for width downsampling (e.g., 4 means [:, ::4])
        mixup_alpha: Mixup alpha parameter (0 = disabled, 0.3-0.5 recommended)
        use_class_balancing: If True, balance classes using WeightedRandomSampler
        normalize: If True, apply background normalization to spectrograms
        use_sparse_patches: If True, only extract patches with signal (sparse attention)
        num_sparse_patches: Number of patches to extract in sparse mode (K)
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Use default from config if not specified
    if spec_transform is None:
        import config
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
        use_sparse_patches=use_sparse_patches,
        num_sparse_patches=num_sparse_patches
    )
    
    val_dataset = SpectrogramDataset(
        data['test_filenames'], data['test_labels'], 
        img_height, img_width, channels, 'center',  # Always use center crop for validation
        noise_filenames=None,  # No noise for validation
        noise_ratio=0.0,
        spec_transform=spec_transform,
        training=False,
        width_downsizing=width_downsizing,
        normalize=normalize,
        use_sparse_patches=use_sparse_patches,
        num_sparse_patches=num_sparse_patches
    )
    
    # Class balancing setup
    sampler = None
    shuffle_train = True
    
    if use_class_balancing:
        # Calculate class weights for balanced sampling
        train_labels_array = np.array(data['train_labels'])
        
        # For single-label: get class indices
        if train_labels_array.ndim == 2 and not np.all((train_labels_array == 0) | (train_labels_array == 1)):
            # Multi-hot or soft labels - use primary class
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
    if use_sparse_patches:
        # Sparse patches with optional mixup at embedding level
        if mixup_alpha > 0:
            print(f"Sparse mode: mixup will be applied at embedding level (alpha={mixup_alpha})")
        train_collate_fn = sparse_collate_fn
        val_collate_fn = sparse_collate_fn
    elif mixup_alpha > 0:
        # Standard mode with mixup
        print(f"Mixup enabled with alpha={mixup_alpha}")
        train_collate_fn = MixupCollate(mixup_alpha)
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
    
    val_loader = TorchDataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collate_fn
    )
    
    return train_loader, val_loader


class MixupCollate:
    """Collate function that applies mixup augmentation to batches.
    
    This is the proper place for mixup - in the data loading pipeline,
    not scattered in the training loop.
    """
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Mixup interpolation strength (beta distribution parameter)
        """
        self.alpha = alpha
    
    def __call__(self, batch):
        """Apply mixup to a batch of samples.
        
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
            
            # Mix inputs and labels
            mixed_data = lam * data + (1 - lam) * data[index]
            mixed_labels = lam * labels + (1 - lam) * labels[index]
            
            return mixed_data, mixed_labels
        else:
            return data, labels


def get_dataset_info(folder):
    """Get dataset information from labels file."""
    labels_file = os.path.join(folder, "labels.json")
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    return {
        'num_classes': data['num_classes'],
        'class_names': data['categories']
    }


def compute_confusion_weights(model, data_loader, train_labels, num_classes, device, 
                               boost_factor=2.0, top_k=None, max_samples=10000):
    """
    Evaluate model on training data to find confused classes and reweight samples.
    
    Args:
        model: Trained model to evaluate
        data_loader: DataLoader for training set (without mixup/augmentation)
        train_labels: Array of training labels (num_samples, num_classes)
        num_classes: Number of classes
        device: torch device
        boost_factor: How much to upweight confused classes (2.0 = double)
        top_k: If set, only boost top-k most confused classes
        max_samples: Maximum number of samples to evaluate (for speed)
        
    Returns:
        sample_weights: New sample weights based on confusion
        class_error_rates: Per-class error rates for logging
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    total_batches = len(data_loader)
    max_batches = min(total_batches, max(1, max_samples // data_loader.batch_size))
    
    print(f"  Evaluating {max_batches}/{total_batches} batches (~{max_batches * data_loader.batch_size} samples)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            # Progress indicator
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Progress: {batch_idx}/{max_batches} batches", end='\r')
            
            # Handle both sparse and standard formats
            if isinstance(batch, dict):
                # Sparse patches mode
                patches = batch['patches'].to(device)
                positions = batch['positions'].to(device)
                mask = batch['mask'].to(device)
                target = batch['label'].to(device)
                output = model(patches, sparse_mode=True, positions=positions, mask=mask)
            else:
                # Standard mode
                data, target = batch
                data = data.to(device)
                output = model(data)
            
            if target.dim() == 2:
                pred = output.argmax(dim=1)
                target_idx = target.argmax(dim=1)
            else:
                pred = output.argmax(dim=1)
                target_idx = target
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target_idx.cpu().numpy())
    
    print()  # Clear progress line
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    class_error_rates = np.zeros(num_classes)
    for cls_idx in range(num_classes):
        mask = all_targets == cls_idx
        if mask.sum() > 0:
            class_error_rates[cls_idx] = (all_preds[mask] != all_targets[mask]).mean()
    
    if train_labels.ndim == 2:
        class_indices = np.argmax(train_labels, axis=1)
    else:
        class_indices = train_labels
    
    sample_weights = np.ones(len(class_indices))
    
    # Apply confusion-based weighting
    if top_k is not None:
        confused_classes = np.argsort(class_error_rates)[-top_k:]
        for cls_idx in confused_classes:
            mask = class_indices == cls_idx
            sample_weights[mask] *= boost_factor
    else:
        for cls_idx in range(num_classes):
            if class_error_rates[cls_idx] > 0:
                mask = class_indices == cls_idx
                weight_multiplier = 1.0 + (boost_factor - 1.0) * class_error_rates[cls_idx]
                sample_weights[mask] *= weight_multiplier
    
    # Normalize weights and add safeguards to prevent weight explosion
    sample_weights = sample_weights / sample_weights.mean()
    
    # Clip extreme weights to prevent NaN losses (max 20x the mean)
    max_weight = 20.0
    num_clipped = (sample_weights > max_weight).sum()
    if num_clipped > 0:
        print(f"  Warning: Clipped {num_clipped} extreme weights (>{max_weight:.1f}x mean) to prevent instability")
        sample_weights = np.clip(sample_weights, 0.1, max_weight)
        # Re-normalize after clipping
        sample_weights = sample_weights / sample_weights.mean()
    
    print(f"  Weight stats: min={sample_weights.min():.2f}, mean=1.00, max={sample_weights.max():.2f}")
    
    return sample_weights, class_error_rates


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


def extract_sparse_patches(spec, labeled_regions, num_patches=20, patch_size=16):
    """
    Extract top-K patches by signal density for sparse attention.
    
    Args:
        spec: Spectrogram (H, W) - can be log-transformed or normalized
        labeled_regions: Binary or labeled mask indicating signal regions (H, W)
        num_patches: Number of patches to extract (K)
        patch_size: Size of each patch (default 16 for AST)
    
    Returns:
        patches: Selected patches (K, patch_size, patch_size)
        positions: Grid positions (row, col) for each patch (K, 2)
        mask: Boolean mask indicating which patches are real vs padding (K,)
    """
    H, W = spec.shape
    
    # Create binary signal mask
    signal_mask = (labeled_regions > 0).astype(np.float32)
    
    # Calculate patch grid dimensions
    num_rows = H // patch_size
    num_cols = W // patch_size
    
    # Calculate signal density for each patch
    patch_scores = []
    patch_positions = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            y_start = i * patch_size
            x_start = j * patch_size
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Count signal pixels in this patch
            signal_count = signal_mask[y_start:y_end, x_start:x_end].sum()
            patch_scores.append(signal_count)
            patch_positions.append((i, j))
    
    patch_scores = np.array(patch_scores)
    patch_positions = np.array(patch_positions)
    
    # Select top-K patches
    if len(patch_scores) <= num_patches:
        # If we have fewer patches than requested, take all
        top_indices = np.arange(len(patch_scores))
        actual_num_patches = len(patch_scores)
    else:
        # Select top K by signal density
        top_indices = np.argsort(patch_scores)[::-1][:num_patches]
        actual_num_patches = num_patches
    
    # Extract patches and positions
    selected_patches = []
    selected_positions = []
    
    for idx in top_indices:
        i, j = patch_positions[idx]
        y_start = i * patch_size
        x_start = j * patch_size
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        patch = spec[y_start:y_end, x_start:x_end]
        selected_patches.append(patch)
        selected_positions.append([i, j])
    
    # Pad if necessary
    mask = np.ones(actual_num_patches, dtype=bool)
    if actual_num_patches < num_patches:
        # Pad with zeros
        padding_needed = num_patches - actual_num_patches
        zero_patch = np.zeros((patch_size, patch_size))
        for _ in range(padding_needed):
            selected_patches.append(zero_patch)
            selected_positions.append([0, 0])  # Dummy position
        
        # Update mask
        mask = np.concatenate([mask, np.zeros(padding_needed, dtype=bool)])
    
    patches = np.stack(selected_patches, axis=0)  # (K, patch_size, patch_size)
    positions = np.array(selected_positions)  # (K, 2)
    
    return patches, positions, mask


class InterestPixelDataset(Dataset):
    
    def __init__(self, spec_filenames, interest_map_filenames, img_height, img_width, 
                 spec_transform="Log", training=True):
        self.spec_filenames = spec_filenames
        self.interest_map_filenames = interest_map_filenames
        self.img_height = img_height
        self.img_width = img_width
        self.spec_transform = spec_transform
        self.training = training
        
        print(f"InterestPixelDataset initialized with {len(self.spec_filenames)} files")
        print(f"Image size: {img_height}x{img_width}")
        print(f"Spectrogram transform: {self.spec_transform}")
        print(f"Training mode: {self.training}")
    
    def __len__(self):
        return len(self.spec_filenames)
    
    def __getitem__(self, idx):
        spec = np.load(self.spec_filenames[idx])
        interest_map = np.load(self.interest_map_filenames[idx])
        
        while spec.ndim > 2:
            spec = np.squeeze(spec)
        while interest_map.ndim > 2:
            interest_map = np.squeeze(interest_map)
        
        spec = self.apply_spec_transform(spec)
        
        if spec.shape[0] != self.img_height or spec.shape[1] != self.img_width:
            spec = self.resize_array(spec, self.img_height, self.img_width)
        if interest_map.shape[0] != self.img_height or interest_map.shape[1] != self.img_width:
            interest_map = self.resize_array(interest_map, self.img_height, self.img_width)
        
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        interest_tensor = torch.FloatTensor(interest_map).unsqueeze(0)
        
        return spec_tensor, interest_tensor
    
    def apply_spec_transform(self, sg):
        if self.spec_transform == "None" or self.spec_transform is None:
            return sg
        else:
            LOG_OFFSET = 1e-7
            if self.spec_transform == "Log":
                sg_safe = sg + LOG_OFFSET
                return np.log(sg_safe)
            else:
                return sg
    
    def resize_array(self, array, target_h, target_w):
        from scipy.ndimage import zoom
        h, w = array.shape
        zoom_h = target_h / h
        zoom_w = target_w / w
        return zoom(array, (zoom_h, zoom_w), order=1)


