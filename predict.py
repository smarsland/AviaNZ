"""
Generate predictions from a trained model on spectrogram data.

This script loads a trained AST or CNN model and generates predictions
in the same format as kaytoo's output for evaluation comparison.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from models import AST, CNNModel
from data_utils import SpectrogramDataset
from normalizer import normalize_spectrogram
import config


class ModelPredictor:
    def __init__(self, model_path, model_config, data_folder, output_file, batch_size=32, device=None, normalize=False, inference_time_bins=None):
        self.model_path = model_path
        self.model_config = model_config
        self.data_folder = data_folder
        self.output_file = output_file
        self.batch_size = batch_size
        self.normalize = normalize
        self.inference_time_bins = inference_time_bins  # For resizing model to different input size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.categories = None
        self.dataset = None
    
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        # The .pt file contains only state_dict, need to load config separately
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # Load model configuration from provided JSON file
        if not os.path.exists(self.model_config):
            raise FileNotFoundError(f"Model config file not found: {self.model_config}")
        
        with open(self.model_config, 'r') as f:
            model_config = json.load(f)
        
        num_classes = model_config['num_classes']
        multilabel = model_config.get('multilabel', False)
        model_type = model_config.get('model_type', 'AST').lower()
        
        # Get class names from model config (these are the training classes)
        self.categories = model_config['class_names']
        
        # Get input dimensions from config
        self.expected_freq_bins = model_config.get('freq_bins', config.DEFAULT_FREQ_BINS)
        training_time_bins = model_config.get('time_bins', config.DEFAULT_TIME_BINS)
        
        # Determine inference time_bins (use training size by default, override with --time-bins if specified)
        inference_time_bins = self.inference_time_bins if self.inference_time_bins is not None else training_time_bins
        
        # Show resize info if different from training
        if inference_time_bins != training_time_bins:
            print(f"⚡ Resizing model input: {training_time_bins} → {inference_time_bins} time bins")
        
        self.use_sparse_patches = model_config.get('use_sparse_patches', False)
        self.num_sparse_patches = model_config.get('num_sparse_patches', 20)
        
        # Create model at INFERENCE size
        inference_input_size = (self.expected_freq_bins, inference_time_bins)
        
        print(f"Model type: {model_type}")
        print(f"Number of classes: {num_classes}")
        print(f"Multi-label: {multilabel}")
        print(f"Inference input size: {inference_input_size}")
        
        use_reconstruction = model_config.get('use_reconstruction', False)
        
        if model_type == 'ast':
            self.model = AST(num_classes, multilabel, input_size=inference_input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'multiscaleast':
            from models import MultiScaleAST
            self.model = MultiScaleAST(num_classes, multilabel, input_size=inference_input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'cnn':
            self.model = CNNModel(num_classes, multilabel, dropout=0.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # For sparse models, the patch_projection layer is created dynamically
        if 'patch_projection.weight' in state_dict:
            if not hasattr(self.model, 'patch_projection'):
                import torch.nn as nn
                embed_dim = 768
                patch_size = 16
                self.model.patch_projection = nn.Linear(patch_size * patch_size, embed_dim)
        
        # Handle position embeddings separately if resizing
        pos_embed_key = 'ast.embeddings.position_embeddings'
        pos_embed_checkpoint = None
        
        if inference_time_bins != training_time_bins and pos_embed_key in state_dict:
            # Extract checkpoint position embeddings (at training size)
            pos_embed_checkpoint = state_dict.pop(pos_embed_key)
            
            # Interpolate from training size to inference size
            device = self.device
            dtype = pos_embed_checkpoint.dtype
            B, N_old, C = pos_embed_checkpoint.shape
            
            # Infer old grid from checkpoint size
            n_special = 2
            num_old_patches = N_old - n_special
            
            # Calculate grid dimensions
            projection = self.model.ast.embeddings.patch_embeddings.projection
            patch_size = projection.kernel_size
            stride = projection.stride
            
            h_old = (training_time_bins - patch_size[1]) // stride[1] + 1 if hasattr(self.model.ast, 'embeddings') else None
            w_old = (self.expected_freq_bins - patch_size[0]) // stride[0] + 1 if hasattr(self.model.ast, 'embeddings') else None
            
            # If can't compute from config, factor from num_patches
            if h_old is None:
                for h in range(16, 7, -1):
                    if num_old_patches % h == 0:
                        w = num_old_patches // h
                        if 50 <= w <= 200:
                            h_old, w_old = h, w
                            break
            
            h_new = (inference_time_bins - patch_size[1]) // stride[1] + 1
            w_new = (self.expected_freq_bins - patch_size[0]) // stride[0] + 1
            
            print(f"Interpolating position embeddings from {h_old}x{w_old} to {h_new}x{w_new}")
            
            # Split and interpolate
            special_tokens = pos_embed_checkpoint[:, :n_special, :]
            pos_tokens = pos_embed_checkpoint[:, n_special:, :]
            pos_tokens = pos_tokens.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(h_new, w_new), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_new * w_new, C)
            
            pos_embed_checkpoint = torch.cat([special_tokens, pos_tokens], dim=1)
        
        # Load checkpoint weights (position embeddings may have been removed/modified above)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        # If we interpolated position embeddings, load them now
        if pos_embed_checkpoint is not None:
            self.model.ast.embeddings.position_embeddings.data = pos_embed_checkpoint
        
        # Store the final inference time bins
        self.expected_time_bins = inference_time_bins
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_data(self):
        """Load spectrogram data and labels."""
        print(f"Loading data from {self.data_folder}")
        
        labels_file = os.path.join(self.data_folder, "labels.json")
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        files = labels_data['files']
        dataset_type = labels_data.get('dataset', 'Unknown')
        
        print(f"Dataset type: {dataset_type}")
        print(f"Found {len(files)} files")
        print(f"Model expects {len(self.categories)} categories")
        
        self.file_specs = []
        self.file_metadata = []
        
        for file_info in files:
            spec_path = os.path.join(self.data_folder, "data", file_info['filename'])
            if os.path.exists(spec_path):
                self.file_specs.append(spec_path)
                self.file_metadata.append(file_info)
        
        print(f"Found {len(self.file_specs)} valid spectrogram files")
    
    def predict(self):
        """Generate predictions for all data."""
        print("Generating predictions...")
        
        all_predictions = []
        all_source_files = []
        all_row_ids = []
        
        with torch.no_grad():
            for spec_path, file_info in tqdm(zip(self.file_specs, self.file_metadata), 
                                            total=len(self.file_specs), desc="Predicting"):
                spec = np.load(spec_path)
                
                if spec.ndim == 2:
                    freq_bins, time_bins = spec.shape
                else:
                    raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")
                
                # Pad or crop to expected dimensions from model config
                # Use TILING for padding (matches training: data_utils.py uses np.tile, not np.pad)
                if time_bins < self.expected_time_bins:
                    # Tile/repeat the signal instead of zero-padding
                    tiles_needed = int(np.ceil(self.expected_time_bins / time_bins))
                    spec = np.tile(spec, (1, tiles_needed))
                    spec = spec[:, :self.expected_time_bins]  # Crop to exact size
                elif time_bins > self.expected_time_bins:
                    spec = spec[:, :self.expected_time_bins]
                
                if freq_bins < self.expected_freq_bins:
                    pad_height = self.expected_freq_bins - freq_bins
                    spec = np.pad(spec, ((0, pad_height), (0, 0)), mode='constant')
                elif freq_bins > self.expected_freq_bins:
                    spec = spec[:self.expected_freq_bins, :]
                
                LOG_OFFSET = 1e-7
                spec = np.log(spec + LOG_OFFSET)
                
                # Apply background normalization if enabled
                if self.normalize:
                    spec = normalize_spectrogram(spec)
                
                # Handle sparse patches mode
                if self.use_sparse_patches:
                    from data_utils import extract_signal_regions, extract_sparse_patches
                    
                    # Extract signal regions from linear spectrogram (before log)
                    spec_linear = np.load(spec_path)  # Reload linear version
                    
                    # Resize if needed
                    if spec_linear.shape != (self.expected_freq_bins, self.expected_time_bins):
                        from scipy.ndimage import zoom
                        h_ratio = self.expected_freq_bins / spec_linear.shape[0]
                        w_ratio = self.expected_time_bins / spec_linear.shape[1]
                        spec_linear = zoom(spec_linear, (h_ratio, w_ratio), order=1)
                    
                    spec_normalized, labeled_regions = extract_signal_regions(spec_linear)
                    
                    # Extract sparse patches
                    patches, positions, mask = extract_sparse_patches(
                        spec_normalized, labeled_regions,
                        num_patches=self.num_sparse_patches,
                        patch_size=16
                    )
                    
                    # Convert to tensors with correct shape: (1, K, 1, 16, 16)
                    patches_tensor = torch.from_numpy(patches).float()  # (K, 16, 16)
                    patches_tensor = patches_tensor.unsqueeze(0).unsqueeze(2)  # (1, K, 1, 16, 16)
                    positions_tensor = torch.from_numpy(positions).long().unsqueeze(0)  # (1, K, 2)
                    mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)  # (1, K)
                    
                    patches_tensor = patches_tensor.to(self.device)
                    positions_tensor = positions_tensor.to(self.device)
                    mask_tensor = mask_tensor.to(self.device)
                    
                    outputs = self.model(patches_tensor, sparse_mode=True, positions=positions_tensor, mask=mask_tensor)
                else:
                    # Standard mode
                    spec_tensor = torch.from_numpy(spec).float()
                    spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
                    spec_tensor = spec_tensor.to(self.device)
                    
                    outputs = self.model(spec_tensor)
                
                # Handle reconstruction output if present
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = torch.sigmoid(outputs)
                
                all_predictions.append(probs.cpu().numpy()[0])
                
                filename = file_info['filename']
                source_file = file_info.get('source_file', filename)
                
                if 'row_id' in file_info:
                    row_id = file_info['row_id']
                else:
                    row_id = filename
                
                all_source_files.append(source_file)
                all_row_ids.append(row_id)
        
        all_predictions = np.vstack(all_predictions)
        
        print(f"Generated {len(all_predictions)} predictions from {len(self.file_specs)} files")
        print(f"Predictions shape: {all_predictions.shape}")
        
        self.source_files = all_source_files
        self.row_ids = all_row_ids
        
        return all_predictions
    
    def save_predictions(self, predictions):
        """Save predictions to CSV in kaytoo-compatible format."""
        print(f"Saving predictions to {self.output_file}")
        
        df = pd.DataFrame(predictions, columns=self.categories)
        
        df.insert(0, 'File_Path', self.source_files)
        df.insert(1, 'row_id', self.row_ids)
        
        os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
        df.to_csv(self.output_file, index=False)
        
        print(f"Saved {len(df)} predictions ({len(set(self.source_files))} unique files)")
    
    def run(self):
        """Run full prediction pipeline."""
        self.load_model()
        self.load_data()
        predictions = self.predict()
        self.save_predictions(predictions)
        
        print("\n✓ Prediction complete!")
        print(f"  Model: {self.model_path}")
        print(f"  Data: {self.data_folder}")
        print(f"  Output: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions from trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate predictions on test data
  python predict.py trained_model.pt ast_model_config.json Sound_Files/AviaNZ_spec predictions.csv
  
  # With custom batch size
  python predict.py model.pt ast_model_config.json data/ output.csv --batch-size 64
  
  # With normalization (if model was trained with --normalize)
  python predict.py model.pt ast_model_config.json data/ output.csv --normalize
  
  # Fast 5-second inference (train on 10-sec, run on 5-sec, model resizes automatically)
  python predict.py model.pt ast_model_config.json data_5sec/ output_5sec.csv --time-bins 512
        """
    )
    
    parser.add_argument('model_path', type=str,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument('model_config', type=str,
                       help="Path to model config JSON file (e.g., ast_model_config.json)")
    parser.add_argument('data_folder', type=str,
                       help="Path to folder containing spectrograms and labels.json")
    parser.add_argument('output_file', type=str,
                       help="Path to output CSV file for predictions")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size for inference (default: 32)")
    parser.add_argument('--time-bins', type=int, default=None,
                       help="Override model input time bins (e.g., 512 for 5-sec inference on 10-sec trained model). Model resizes automatically via positional embedding interpolation. Default: uses trained size.")
    parser.add_argument('--device', type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization to spectrograms (use if model was trained with --normalize)")
    
    args = parser.parse_args()
    
    predictor = ModelPredictor(
        model_path=args.model_path,
        model_config=args.model_config,
        data_folder=args.data_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None,
        normalize=args.normalize,
        inference_time_bins=args.time_bins
    )
    
    predictor.run()


if __name__ == '__main__':
    main()
