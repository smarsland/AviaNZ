"""
Generate pseudo-labels from a trained model.

This script loads a trained model and generates pseudo-labels for the training data,
saving them to a new labels.json file. This enables pseudo-labeling/self-training workflows
where you can iteratively refine a model by:
1. Training a model on ground truth labels
2. Generating pseudo-labels on the same (or different) data
3. Training a new model on the pseudo-labels, optionally initialized from the first model

The pseudo-labels can use soft labels (probabilities) or hard labels (thresholded predictions).
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models import AST, CNNModel
from normalizer import normalize_spectrogram
import config


class PseudoLabeler:
    """Generate pseudo-labels from a trained model."""
    
    def __init__(self, model_path, model_config, data_folder, output_labels_file, 
                 threshold=0.5, use_soft_labels=False, top_k=None, device=None,
                 normalize=False, remove_baseline=None):
        """
        Initialize pseudo-labeler.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            model_config: Path to model config JSON file
            data_folder: Path to folder containing spectrograms and original labels.json
            output_labels_file: Path to save new labels.json with pseudo-labels
            threshold: Probability threshold for hard labels (default: 0.5)
            use_soft_labels: If True, save probabilities as soft labels; if False, use hard thresholded labels
            top_k: If set, only keep top-k predictions per sample (for multi-label)
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.model_config = model_config
        self.data_folder = data_folder
        self.output_labels_file = output_labels_file
        self.threshold = threshold
        self.use_soft_labels = use_soft_labels
        self.top_k = top_k
        self.normalize = normalize
        self.remove_baseline = remove_baseline  # None = auto-detect from model config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Threshold: {self.threshold}")
        print(f"Soft labels: {self.use_soft_labels}")
        if self.top_k:
            print(f"Top-k filtering: {self.top_k}")
        
        self.model = None
        self.categories = None
        self.original_labels = None
    
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        # Load model weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # Load model configuration
        if not os.path.exists(self.model_config):
            raise FileNotFoundError(f"Model config file not found: {self.model_config}")
        
        with open(self.model_config, 'r') as f:
            model_config = json.load(f)
        
        num_classes = model_config['num_classes']
        multilabel = model_config.get('multilabel', False)
        model_type = model_config.get('model_type', 'AST').lower()

        self.multilabel = multilabel

        # Get class names from model config
        self.categories = model_config['class_names']
        
        # Get input dimensions
        self.expected_freq_bins = model_config.get('freq_bins', config.DEFAULT_FREQ_BINS)
        self.expected_time_bins = model_config.get('time_bins', config.DEFAULT_TIME_BINS)
        input_size = (self.expected_freq_bins, self.expected_time_bins)
        
        print(f"Model type: {model_type}")
        print(f"Number of classes: {num_classes}")
        print(f"Multi-label: {multilabel}")
        print(f"Input size: {input_size}")
        print(f"Classes: {', '.join(self.categories)}")
        
        # Auto-detect normalization / baseline removal from model config if not explicitly set
        if model_config.get('normalize', False) and not self.normalize:
            print(f"\u26a0\ufe0f  Model was trained with normalization but --normalize flag not set")
            print(f"   Auto-enabling normalization for consistency")
            self.normalize = True

        if self.remove_baseline is None:
            self.remove_baseline = model_config.get('remove_baseline', False)
            if self.remove_baseline:
                print(f"\u26a1 Baseline removal: enabled (from model config)")
            else:
                print(f"Baseline removal: disabled (not in model config / old model)")

        use_reconstruction = model_config.get('use_reconstruction', False)

        # Create model
        if model_type == 'ast':
            self.model = AST(num_classes, multilabel, input_size=input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'multiscaleast':
            from models import MultiScaleAST
            self.model = MultiScaleAST(num_classes, multilabel, input_size=input_size, dropout=0.0, use_reconstruction=use_reconstruction)
        elif model_type == 'cnn':
            self.model = CNNModel(self.expected_freq_bins, self.expected_time_bins, num_classes)
        elif model_type == 'birdclef_finetuned':
            print("Loading fine-tuned BirdClef model...")
            from finetune_birdclef import BirdClefFineTuneModel
            self.model = BirdClefFineTuneModel(
                num_classes=num_classes,
                pretrained_path=None,
                freeze_backbone=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load weights (allow minor mismatches for AST positional embeddings)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading checkpoint (showing up to 5): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading checkpoint (showing up to 5): {unexpected_keys[:5]}")

        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_original_labels(self):
        """Load original labels.json to preserve metadata."""
        labels_file = os.path.join(self.data_folder, "labels.json")
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            self.original_labels = json.load(f)
        
        print(f"Loaded original labels with {len(self.original_labels['files'])} files")
    
    def generate_pseudo_labels(self):
        """Generate pseudo-labels for all files."""
        print("Generating pseudo-labels...")
        
        data_dir = os.path.join(self.data_folder, "data")
        new_files = []
        
        with torch.no_grad():
            for file_info in tqdm(self.original_labels['files'], desc="Generating pseudo-labels"):
                spec_path = os.path.join(data_dir, file_info['filename'])
                
                if not os.path.exists(spec_path):
                    print(f"Warning: Skipping missing file {spec_path}")
                    continue
                
                # Load and preprocess spectrogram
                spec = np.load(spec_path)
                
                if spec.ndim == 2:
                    freq_bins, time_bins = spec.shape
                else:
                    raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")
                
                # Pad or crop to expected dimensions
                # Use tiling (repeat) for time padding to avoid silence artifacts
                if time_bins < self.expected_time_bins:
                    tiles_needed = int(np.ceil(self.expected_time_bins / time_bins))
                    spec = np.tile(spec, (1, tiles_needed))
                    spec = spec[:, :self.expected_time_bins]
                elif time_bins > self.expected_time_bins:
                    spec = spec[:, :self.expected_time_bins]
                
                if freq_bins < self.expected_freq_bins:
                    pad_height = self.expected_freq_bins - freq_bins
                    spec = np.pad(spec, ((0, pad_height), (0, 0)), mode='constant')
                elif freq_bins > self.expected_freq_bins:
                    spec = spec[:self.expected_freq_bins, :]

                # Remove baseline offset before log transform
                if self.remove_baseline:
                    baseline = np.percentile(spec, 10)
                    spec = np.maximum(spec - baseline, 0)
                
                # Normalize (log transform)
                LOG_OFFSET = 1e-7
                spec = np.log(spec + LOG_OFFSET)

                if self.normalize:
                    spec = normalize_spectrogram(spec)
                
                # Convert to tensor
                spec_tensor = torch.from_numpy(spec).float()
                spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
                spec_tensor = spec_tensor.to(self.device)
                
                # Get predictions
                outputs = self.model(spec_tensor)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                if self.multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()[0]
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Create new file entry with pseudo-labels
                new_file_info = {
                    'filename': file_info['filename'],
                    'source_file': file_info.get('source_file', file_info['filename'])
                }
                
                # Preserve row_id if present
                if 'row_id' in file_info:
                    new_file_info['row_id'] = file_info['row_id']
                
                # Generate pseudo-labels based on configuration
                if self.use_soft_labels:
                    # Soft labels: use probabilities directly
                    class_probs = {}
                    for idx, category in enumerate(self.categories):
                        if probs[idx] > 0.01:  # Only save non-trivial probabilities
                            class_probs[category] = float(probs[idx])
                    
                    new_file_info['class_probabilities'] = class_probs
                    
                    # Also set class_names for compatibility (threshold for hard assignment)
                    class_names = [self.categories[idx] for idx in range(len(probs)) 
                                   if probs[idx] > self.threshold]
                else:
                    # Hard labels: threshold probabilities
                    class_names = [self.categories[idx] for idx in range(len(probs)) 
                                   if probs[idx] > self.threshold]
                
                # Apply top-k filtering if requested
                if self.top_k and len(class_names) > self.top_k:
                    # Get top-k by probability
                    top_k_indices = np.argsort(probs)[-self.top_k:]
                    class_names = [self.categories[idx] for idx in top_k_indices if probs[idx] > self.threshold]
                
                # Ensure at least one label (take highest probability)
                if not class_names:
                    max_idx = np.argmax(probs)
                    class_names = [self.categories[max_idx]]
                
                new_file_info['class_names'] = class_names
                
                # Set primary class as highest probability
                primary_idx = np.argmax(probs)
                new_file_info['primary_class'] = self.categories[primary_idx]
                new_file_info['primary_confidence'] = float(probs[primary_idx])
                
                new_files.append(new_file_info)
        
        return new_files
    
    def save_pseudo_labels(self, new_files):
        """Save pseudo-labels to new labels.json file."""
        print(f"Saving pseudo-labels to {self.output_labels_file}")
        
        # Create new labels structure
        new_labels = {
            'dataset': self.original_labels.get('dataset', 'Unknown') + '_PseudoLabeled',
            'categories': self.categories,
            'files': new_files,
            'pseudo_labeled': True,
            'source_model': os.path.basename(self.model_path),
            'threshold': self.threshold,
            'soft_labels': self.use_soft_labels
        }
        
        # Preserve any other metadata from original
        for key in self.original_labels:
            if key not in new_labels and key != 'files':
                new_labels[key] = self.original_labels[key]
        
        # Save to file
        os.makedirs(os.path.dirname(self.output_labels_file) or '.', exist_ok=True)
        with open(self.output_labels_file, 'w') as f:
            json.dump(new_labels, f, indent=2)
        
        print(f"Saved {len(new_files)} pseudo-labeled files")
        
        # Print statistics
        total_labels = sum(len(f['class_names']) for f in new_files)
        avg_labels = total_labels / len(new_files) if new_files else 0
        print(f"Average labels per file: {avg_labels:.2f}")
        
        # Count label distribution
        label_counts = {}
        for file_info in new_files:
            for class_name in file_info['class_names']:
                label_counts[class_name] = label_counts.get(class_name, 0) + 1
        
        print("\nLabel distribution (top 10):")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_labels[:10]:
            print(f"  {class_name}: {count}")
    
    def run(self):
        """Run full pseudo-labeling pipeline."""
        self.load_model()
        self.load_original_labels()
        new_files = self.generate_pseudo_labels()
        self.save_pseudo_labels(new_files)
        
        print("\n✓ Pseudo-labeling complete!")
        print(f"  Model: {self.model_path}")
        print(f"  Data: {self.data_folder}")
        print(f"  Output: {self.output_labels_file}")
        print(f"\nNext steps:")
        print(f"  1. Review the pseudo-labels in: {self.output_labels_file}")
        print(f"  2. Create a new output folder for the next training run")
        print(f"  3. Train a new model on the pseudo-labels:")
        print(f"     python train_models.py {self.data_folder} <new_output_folder> \\")
        print(f"       --model ast --pretrained {self.model_path} --multilabel \\")
        print(f"       [other training options]")
        print(f"\n  Note: The script will automatically use the new pseudo-labels because")
        print(f"        it reads labels.json from the data folder.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels from trained model for self-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate hard pseudo-labels with default threshold (0.5)
  python generate_pseudo_labels.py doc_out_mult/ast_model_best.pt \\
      doc_out_mult/ast_model_config.json \\
      /local/scratch/freangi/doc_data \\
      /local/scratch/freangi/doc_data/labels_pseudo.json
  
  # Generate soft (probabilistic) pseudo-labels
  python generate_pseudo_labels.py model.pt config.json data/ data/labels_pseudo.json --soft-labels
  
  # Use higher threshold to be more conservative
  python generate_pseudo_labels.py model.pt config.json data/ data/labels_pseudo.json --threshold 0.7
  
  # Keep only top-3 predictions per sample
  python generate_pseudo_labels.py model.pt config.json data/ data/labels_pseudo.json --top-k 3

Workflow:
  1. Train initial model:
     python train_models.py data/ output1/ --model ast --multilabel --epochs 25
  
  2. Generate pseudo-labels (this script):
     python generate_pseudo_labels.py output1/ast_model_best.pt output1/ast_model_config.json \\
         data/ data/labels_pseudo.json
  
  3. Replace original labels with pseudo-labels:
     mv data/labels.json data/labels_original.json
     mv data/labels_pseudo.json data/labels.json
  
  4. Train new model on pseudo-labels, initialized from first model:
     python train_models.py data/ output2/ --model ast --multilabel \\
         --pretrained output1/ast_model_best.pt --epochs 25
        """
    )
    
    parser.add_argument('model_path', type=str,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument('model_config', type=str,
                       help="Path to model config JSON file (e.g., ast_model_config.json)")
    parser.add_argument('data_folder', type=str,
                       help="Path to folder containing spectrograms and labels.json")
    parser.add_argument('output_labels_file', type=str,
                       help="Path to save new labels.json with pseudo-labels")
    parser.add_argument('--threshold', type=float, default=0.5,
                       help="Probability threshold for hard labels (default: 0.5)")
    parser.add_argument('--soft-labels', action='store_true',
                       help="Save soft (probabilistic) labels instead of hard thresholded labels")
    parser.add_argument('--top-k', type=int, default=None,
                       help="Only keep top-k predictions per sample (default: None = keep all above threshold)")
    parser.add_argument('--device', type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization to spectrograms (auto-detected from config, but can override)")
    parser.add_argument('--no-baseline-removal', action='store_true',
                       help="Disable baseline removal (default: auto-detect from model config)")
    
    args = parser.parse_args()
    
    labeler = PseudoLabeler(
        model_path=args.model_path,
        model_config=args.model_config,
        data_folder=args.data_folder,
        output_labels_file=args.output_labels_file,
        threshold=args.threshold,
        use_soft_labels=args.soft_labels,
        top_k=args.top_k,
        device=torch.device(args.device) if args.device else None,
        normalize=args.normalize,
        remove_baseline=False if args.no_baseline_removal else None
    )
    
    labeler.run()


if __name__ == '__main__':
    main()
