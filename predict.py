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
from torch.utils.data import DataLoader as TorchDataLoader
from models import AST, CNNModel
from data_utils import SpectrogramDataset
from normalizer import normalize_spectrogram
import config


class ModelPredictor:
    def __init__(self, model_path, model_config, data_folder, output_file, batch_size=32, device=None, normalize=False, remove_baseline=None, inference_time_bins=None):
        self.model_path = model_path
        self.model_config = model_config
        self.data_folder = data_folder
        self.output_file = output_file
        self.batch_size = batch_size
        self.normalize = normalize
        self.remove_baseline = remove_baseline  # None = auto-detect from model config
        self.inference_time_bins = inference_time_bins  # For resizing model to different input size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.categories = None
        self.test_dataset = None
        self.test_loader = None
        self.multilabel = False
    
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        if not os.path.exists(self.model_config):
            raise FileNotFoundError(f"Model config file not found: {self.model_config}")
        
        with open(self.model_config, 'r') as f:
            model_config = json.load(f)
        
        num_classes = model_config['num_classes']
        multilabel = model_config.get('multilabel', False)
        model_type = model_config.get('model_type', 'AST').lower()

        self.multilabel = multilabel
        self.model_type = model_type
        self.categories = model_config['class_names']
        
        self.expected_freq_bins = model_config.get('freq_bins', config.DEFAULT_FREQ_BINS)
        training_time_bins = model_config.get('time_bins', config.DEFAULT_TIME_BINS)
        
        inference_time_bins = self.inference_time_bins if self.inference_time_bins is not None else training_time_bins
        self.expected_time_bins = inference_time_bins  # Save for use in load_data()
        
        if inference_time_bins != training_time_bins:
            print(f"⚡ Resizing model input: {training_time_bins} → {inference_time_bins} time bins")
        
        self.use_sparse_patches = model_config.get('use_sparse_patches', False)
        self.num_sparse_patches = model_config.get('num_sparse_patches', 20)
        
        if model_config.get('normalize', False) and not self.normalize:
            print(f"⚠️  Model was trained with normalization but --normalize flag not set")
            print(f"   Auto-enabling normalization for consistency")
            self.normalize = True
        
        # Auto-detect baseline removal from model config if not explicitly set
        if self.remove_baseline is None:
            if 'remove_baseline' in model_config:
                self.remove_baseline = model_config['remove_baseline']
                status = "enabled" if self.remove_baseline else "disabled"
                print(f"Baseline removal: {status} (from model config)")
            else:
                self.remove_baseline = False  # Default False (backwards compat)
                print(f"Baseline removal: disabled (old model, no config found - assuming False)")
        
        training_input_size = (self.expected_freq_bins, training_time_bins)
        
        print(f"Model type: {model_type}")
        print(f"Number of classes: {num_classes}")
        print(f"Multi-label: {multilabel}")
        print(f"Training input size: {training_input_size}")
        print(f"Inference input size: ({self.expected_freq_bins}, {inference_time_bins})")
        
        use_reconstruction = model_config.get('use_reconstruction', False)
        
        if model_type == 'ast':
            self.model = AST(num_classes, multilabel, input_size=training_input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'multiscaleast':
            from models import MultiScaleAST
            self.model = MultiScaleAST(num_classes, multilabel, input_size=training_input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'cnn':
            if inference_time_bins != training_time_bins:
                raise ValueError(
                    f"CNN checkpoints are input-size dependent; can't resize time bins {training_time_bins} -> {inference_time_bins}. "
                    f"Re-generate spectrograms to the trained size or re-train the CNN."
                )
            self.model = CNNModel(self.expected_freq_bins, training_time_bins, num_classes)
        elif model_type == 'birdclef_finetuned':
            print("Loading fine-tuned BirdClef model...")
            from finetune_birdclef import BirdClefFineTuneModel
            self.model = BirdClefFineTuneModel(
                num_classes=num_classes,
                pretrained_path=None,
                freeze_backbone=False
            )
        elif model_type == 'dann':
            print("Loading DANN (domain adaptation) model...")
            from train_domain_adaptation import DANNModel
            self.model = DANNModel(
                num_classes=num_classes,
                architecture=model_config.get('architecture', 'resnet18'),
                pretrained_path=None
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if 'patch_projection.weight' in state_dict:
            if not hasattr(self.model, 'patch_projection'):
                import torch.nn as nn
                embed_dim = 768
                patch_size = 16
                self.model.patch_projection = nn.Linear(patch_size * patch_size, embed_dim)
        
        if model_type == 'ast':
            pos_embed_key = 'ast.embeddings.position_embeddings'
            if pos_embed_key in state_dict:
                saved_pos_embed = state_dict[pos_embed_key]
                current_pos_embed = self.model.ast.embeddings.position_embeddings
                
                if saved_pos_embed.shape != current_pos_embed.shape:
                    print(f"Position embedding mismatch: checkpoint {saved_pos_embed.shape} vs model {current_pos_embed.shape}")
                    print(f"Using position embeddings from checkpoint (already interpolated during training)")
                    self.model.ast.embeddings.position_embeddings = torch.nn.Parameter(saved_pos_embed)
                    state_dict.pop(pos_embed_key)
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if inference_time_bins != training_time_bins and hasattr(self.model, 'interpolate_pos_embed'):
            if model_type == 'ast':
                self.model.interpolate_pos_embed((self.expected_freq_bins, inference_time_bins))
        
        self.expected_time_bins = inference_time_bins
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_data(self):
        """Load spectrogram data and labels using SpectrogramDataset (matches training)."""
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
        
        # Build file paths and labels in the format SpectrogramDataset expects
        filenames = []
        labels = []
        self.file_metadata = []  # Keep metadata for saving predictions
        
        for file_info in files:
            spec_path = os.path.join(self.data_folder, "data", file_info['filename'])
            if not os.path.exists(spec_path):
                print(f"⚠️  Skipping missing file: {spec_path}")
                continue
            
            filenames.append(spec_path)
            
            # Extract label (handle different label formats)
            if 'label' in file_info:
                if isinstance(file_info['label'], list):
                    labels.append(file_info['label'])
                else:
                    # Single-class label: convert to one-hot
                    label_vec = [0] * len(self.categories)
                    if isinstance(file_info['label'], int):
                        label_vec[file_info['label']] = 1
                    else:
                        # String label: find index
                        label_idx = self.categories.index(file_info['label'])
                        label_vec[label_idx] = 1
                    labels.append(label_vec)
            else:
                # No label provided: dummy label
                labels.append([0] * len(self.categories))
            
            self.file_metadata.append(file_info)
        
        print(f"Found {len(filenames)} valid spectrogram files")
        
        # Create SpectrogramDataset (same parameters as validation in training)
        img_height = self.expected_freq_bins
        img_width = self.expected_time_bins
        
        self.test_dataset = SpectrogramDataset(
            filenames,
            labels,
            img_height,
            img_width,
            config.DEFAULT_CHANNELS,
            cropping_mode='center',
            noise_filenames=None,
            noise_ratio=0.0,
            spec_transform=None,
            training=False,
            width_downsizing=None,
            normalize=self.normalize,
            use_sparse_patches=self.use_sparse_patches,
            num_sparse_patches=self.num_sparse_patches,
            use_temporal_roll=False,
            remove_baseline=self.remove_baseline
        )
        
        # Create DataLoader (same as validation)
        self.test_loader = TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 2,
            pin_memory=True
        )
        
        print(f"Created DataLoader with batch_size={self.batch_size}")
    
    def predict(self):
        """Generate predictions using DataLoader (matches training/validation)."""
        print("Generating predictions...")
        
        all_predictions = []
        
        self.model.eval()
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Predicting"):
                data = data.to(self.device)
                
                # DANN models have separate predict() method for inference
                if self.model_type == 'dann':
                    outputs = self.model.predict(data)
                else:
                    outputs = self.model(data)
                
                # Handle reconstruction output if present
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if self.multilabel:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
                all_predictions.append(probs.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        
        print(f"Generated {len(all_predictions)} predictions")
        print(f"Predictions shape: {all_predictions.shape}")
        
        return all_predictions
    
    def save_predictions(self, predictions):
        """Save predictions to CSV in kaytoo-compatible format."""
        print(f"Saving predictions to {self.output_file}")
        
        # Extract source_files and row_ids from metadata
        source_files = []
        row_ids = []
        
        for file_info in self.file_metadata:
            filename = file_info['filename']
            source_file = file_info.get('source_file', filename)
            
            if 'row_id' in file_info:
                row_id = file_info['row_id']
            else:
                row_id = filename
            
            source_files.append(source_file)
            row_ids.append(row_id)
        
        df = pd.DataFrame(predictions, columns=self.categories)
        
        df.insert(0, 'File_Path', source_files)
        df.insert(1, 'row_id', row_ids)
        
        os.makedirs(os.path.dirname(self.output_file) or '.', exist_ok=True)
        df.to_csv(self.output_file, index=False)
        
        print(f"Saved {len(df)} predictions ({len(set(source_files))} unique files)")
    
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
                       help="Apply background normalization to spectrograms (auto-detected from config, but can override)")
    parser.add_argument('--no-baseline-removal', action='store_true',
                       help="Disable baseline removal (default: auto-detect from model config)")
    
    args = parser.parse_args()
    
    predictor = ModelPredictor(
        model_path=args.model_path,
        model_config=args.model_config,
        data_folder=args.data_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None,
        normalize=args.normalize,
        remove_baseline=False if args.no_baseline_removal else None,  # None = auto-detect
        inference_time_bins=args.time_bins
    )
    
    predictor.run()


if __name__ == '__main__':
    main()
