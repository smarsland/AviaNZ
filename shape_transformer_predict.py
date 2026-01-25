"""
Generate predictions from a trained Shape Transformer model on sequence data.

This script loads a trained Shape Transformer model and generates predictions
in the same format as predict.py for evaluation comparison.
"""

import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shape_transformer_train import ShapeTransformer


class ShapeTransformerPredictor:
    def __init__(self, model_path, model_config, data_folder, output_file, batch_size=32, device=None):
        self.model_path = model_path
        self.model_config = model_config
        self.data_folder = data_folder
        self.output_file = output_file
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.categories = None
        self.sequence_dir = None
    
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        if not os.path.exists(self.model_config):
            raise FileNotFoundError(f"Model config file not found: {self.model_config}")
        
        with open(self.model_config, 'r') as f:
            model_config = json.load(f)
        
        num_classes = model_config['num_classes']
        self.categories = model_config['class_names']
        
        shape_dim = model_config.get('shape_dim', 1024)
        cont_dim = model_config.get('cont_dim', 6)
        d_model = model_config.get('d_model', 96)
        nhead = model_config.get('nhead', 4)
        num_layers = model_config.get('num_layers', 2)
        dropout = model_config.get('dropout', 0.1)
        
        print(f"Number of classes: {num_classes}")
        print(f"Model dimensions: shape_dim={shape_dim}, cont_dim={cont_dim}, d_model={d_model}")
        print(f"Architecture: nhead={nhead}, num_layers={num_layers}, dropout={dropout}")
        
        self.model = ShapeTransformer(
            shape_dim=shape_dim,
            cont_dim=cont_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_data(self):
        """Load sequence data and labels."""
        print(f"Loading data from {self.data_folder}")
        
        self.sequence_dir = os.path.join(self.data_folder, "sequences")
        if not os.path.exists(self.sequence_dir):
            raise FileNotFoundError(f"Sequences directory not found: {self.sequence_dir}")
        
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
        
        self.file_metadata = []
        
        for file_info in files:
            json_path = os.path.join(self.sequence_dir, file_info['filename'].replace('.npy', '.json'))
            if os.path.exists(json_path):
                self.file_metadata.append(file_info)
        
        print(f"Found {len(self.file_metadata)} valid sequence files")
    
    def load_sequence(self, filename, max_length=100):
        """Load a single sequence from JSON file."""
        json_path = os.path.join(self.sequence_dir, filename.replace('.npy', '.json'))
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        shapes = data['shapes']
        seq_len = min(len(shapes), max_length)
        
        shape_dim = 1024
        cont_dim = 6
        
        shape_vectors = torch.zeros(max_length, shape_dim, dtype=torch.float32)
        continuous_features = torch.zeros(max_length, cont_dim, dtype=torch.float32)
        mask = torch.zeros(max_length, dtype=torch.bool)
        
        for i in range(seq_len):
            shape = shapes[i]
            shape_vectors[i] = torch.FloatTensor(shape['shape_vector'])
            continuous_features[i, 0] = shape['time_pos']
            continuous_features[i, 1] = shape['freq_pos']
            continuous_features[i, 2] = shape['duration']
            continuous_features[i, 3] = shape['bandwidth']
            mask[i] = True
        
        return shape_vectors, continuous_features, mask
    
    def predict(self):
        """Generate predictions for all data."""
        print("Generating predictions...")
        
        all_predictions = []
        all_source_files = []
        all_row_ids = []
        
        with torch.no_grad():
            batch_shape_vectors = []
            batch_continuous = []
            batch_masks = []
            batch_metadata = []
            
            for file_info in tqdm(self.file_metadata, desc="Predicting"):
                filename = file_info['filename']
                
                shape_vectors, continuous_features, mask = self.load_sequence(filename)
                
                batch_shape_vectors.append(shape_vectors)
                batch_continuous.append(continuous_features)
                batch_masks.append(mask)
                batch_metadata.append(file_info)
                
                if len(batch_shape_vectors) == self.batch_size:
                    self._process_batch(
                        batch_shape_vectors, batch_continuous, batch_masks, batch_metadata,
                        all_predictions, all_source_files, all_row_ids
                    )
                    batch_shape_vectors = []
                    batch_continuous = []
                    batch_masks = []
                    batch_metadata = []
            
            if len(batch_shape_vectors) > 0:
                self._process_batch(
                    batch_shape_vectors, batch_continuous, batch_masks, batch_metadata,
                    all_predictions, all_source_files, all_row_ids
                )
        
        all_predictions = np.vstack(all_predictions)
        
        print(f"Generated {len(all_predictions)} predictions from {len(self.file_metadata)} files")
        print(f"Predictions shape: {all_predictions.shape}")
        
        self.source_files = all_source_files
        self.row_ids = all_row_ids
        
        return all_predictions
    
    def _process_batch(self, batch_shape_vectors, batch_continuous, batch_masks, batch_metadata,
                       all_predictions, all_source_files, all_row_ids):
        """Process a batch of sequences."""
        shape_vectors = torch.stack(batch_shape_vectors).to(self.device)
        continuous_features = torch.stack(batch_continuous).to(self.device)
        masks = torch.stack(batch_masks).to(self.device)
        
        logits, _ = self.model(shape_vectors, continuous_features, masks)
        probs = torch.sigmoid(logits)
        
        probs_np = probs.cpu().numpy()
        
        for i, file_info in enumerate(batch_metadata):
            all_predictions.append(probs_np[i])
            
            filename = file_info['filename']
            source_file = file_info.get('source_file', filename)
            
            if 'row_id' in file_info:
                row_id = file_info['row_id']
            else:
                row_id = filename
            
            all_source_files.append(source_file)
            all_row_ids.append(row_id)
    
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
        description="Generate predictions from trained Shape Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate predictions on test data
  python shape_transformer_predict.py model.pt model_config.json Sound_Files/NZ_bird_spec predictions.csv
  
  # With custom batch size
  python shape_transformer_predict.py model.pt config.json data/ output.csv --batch-size 64
        """
    )
    
    parser.add_argument('model_path', type=str,
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument('model_config', type=str,
                       help="Path to model config JSON file (e.g., model_config.json)")
    parser.add_argument('data_folder', type=str,
                       help="Path to folder containing sequences/ subdirectory and labels.json")
    parser.add_argument('output_file', type=str,
                       help="Path to output CSV file for predictions")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size for inference (default: 32)")
    parser.add_argument('--device', type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    predictor = ShapeTransformerPredictor(
        model_path=args.model_path,
        model_config=args.model_config,
        data_folder=args.data_folder,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=torch.device(args.device) if args.device else None
    )
    
    predictor.run()


if __name__ == '__main__':
    main()
