"""
Fine-tune BirdClef pretrained models on AviaNZ data.

This script adapts BirdClef's RegNetY-008 model (pretrained on global birds)
to your specific NZ bird dataset via transfer learning.

Strategy:
1. Load BirdClef pretrained weights (backbone features)
2. Replace classifier head with your species count
3. Optionally freeze early layers (faster training, less overfitting)
4. Train for 5-10 epochs on your data

Usage:
    # Basic fine-tuning (all layers trainable)
    python finetune_birdclef.py data/train outputs/birdclef_finetuned
    
    # Freeze backbone, only train classifier head (faster, recommended for small datasets)
    python finetune_birdclef.py data/train outputs/birdclef_finetuned --freeze-backbone
    
    # Partial freeze: only train last N backbone stages + classifier
    python finetune_birdclef.py data/train outputs/birdclef_finetuned --freeze-stages 3
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import timm
from tqdm import tqdm
from pathlib import Path

import config
from data_utils import DataLoader, create_data_loaders
from evaluation_utils import EvaluationManager


class BirdClefFineTuneModel(nn.Module):
    """BirdClef RegNetY-008 adapted for AviaNZ species."""
    
    def __init__(self, num_classes, pretrained_path=None, model_name='regnety_008', freeze_backbone=False, freeze_stages=0):
        super().__init__()
        self.num_classes = num_classes
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=1,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        
        # Get feature dimension
        if 'efficientnet' in model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'regnet' in model_name:
            # RegNetY uses head.fc
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # Create NEW classifier for your species
        self.classifier = nn.Linear(backbone_out, num_classes)
        
        # Load BirdClef pretrained weights if provided (for training)
        if pretrained_path:
            print(f"Loading BirdClef pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Get original number of classes from checkpoint
            orig_num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
            print(f"  Original model: {orig_num_classes} classes (BirdClef global species)")
            print(f"  Target model: {num_classes} classes (your dataset)")
            
            # Load pretrained backbone weights (ignore classifier)
            self.load_pretrained_weights(checkpoint['model_state_dict'], orig_num_classes)
            
            # Freeze strategies
            if freeze_backbone:
                print("  Freezing entire backbone - only training classifier head")
                for param in self.backbone.parameters():
                    param.requires_grad = False
            elif freeze_stages > 0:
                print(f"  Freezing first {freeze_stages} stages of backbone")
                self.freeze_early_stages(freeze_stages)
            else:
                print("  All layers trainable (full fine-tuning)")
            
            # Print trainable parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Total params: {total_params:,}")
            print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def load_pretrained_weights(self, state_dict, orig_num_classes):
        """Load backbone weights from BirdClef checkpoint, skip classifier."""
        # Remove 'backbone.' prefix if present
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                if 'classifier' not in new_key and 'fc' not in new_key:
                    backbone_dict[new_key] = v
        
        # Load into backbone
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_dict, strict=False)
        
        # Filter out expected mismatches (classifier layers)
        missing_keys = [k for k in missing_keys if 'classifier' not in k and 'fc' not in k]
        unexpected_keys = [k for k in unexpected_keys if 'classifier' not in k and 'fc' not in k]
        
        if missing_keys:
            print(f"  Warning: Missing keys in backbone: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")
        
        print("  ✓ Loaded pretrained backbone weights successfully")
    
    def freeze_early_stages(self, num_stages):
        """Freeze first N stages/blocks of the network."""
        # For RegNetY, freeze by stage
        stage_names = ['stem', 's1', 's2', 's3', 's4']
        
        for i, stage_name in enumerate(stage_names[:num_stages]):
            if hasattr(self.backbone, stage_name):
                stage = getattr(self.backbone, stage_name)
                for param in stage.parameters():
                    param.requires_grad = False
                print(f"    Froze stage: {stage_name}")
    
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits


class BirdClefFineTuner:
    """Handles fine-tuning of BirdClef models on AviaNZ data."""
    
    def __init__(self, data_folder, output_folder, pretrained_path, 
                 epochs=10, batch_size=32, lr=1e-4, freeze_backbone=False, 
                 freeze_stages=0, multilabel=False, device=None):
        
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.freeze_stages = freeze_stages
        self.multilabel = multilabel
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"BirdClef Fine-Tuning Setup")
        print(f"  Data: {data_folder}")
        print(f"  Output: {output_folder}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Multi-label: {multilabel}")
    
    def load_data(self):
        """Load data using existing AviaNZ data pipeline."""
        print("\nLoading dataset...")
        
        # Use DataLoader to load and parse the data
        data_loader = DataLoader(self.data_folder, noise_folder=None)
        self.data = data_loader.load_data(self.multilabel, validation_share=0.2)
        
        self.num_classes = self.data['nclasses']
        self.categories = self.data['categories']
        
        print(f"  Classes: {self.num_classes}")
        print(f"  Examples: {self.categories[:5]}...")
        
        # Get dimensions from config
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        
        # Create data loaders
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader, self.val_loader = create_data_loaders(
            self.data, 
            self.batch_size, 
            img_height, 
            img_width, 
            config.DEFAULT_CHANNELS,
            cropping_mode='random',
            noise_ratio=0.0,  # No noise augmentation for fine-tuning
            spec_transform=None,
            num_workers=num_workers,
            width_downsizing=None,
            mixup_alpha=0.0,  # No mixup for fine-tuning
            use_class_balancing=False,
            normalize=False,
            use_sparse_patches=False,
            num_sparse_patches=0,
            use_temporal_roll=True
        )
        
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
    
    def create_model(self):
        """Create model with pretrained BirdClef weights."""
        print("\nCreating model...")
        
        self.model = BirdClefFineTuneModel(
            num_classes=self.num_classes,
            pretrained_path=self.pretrained_path,
            freeze_backbone=self.freeze_backbone,
            freeze_stages=self.freeze_stages
        )
        
        self.model.to(self.device)
        
        # Loss function
        if self.multilabel:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - higher LR for new classifier, lower for pretrained backbone
        backbone_params = [p for n, p in self.model.named_parameters() 
                          if 'classifier' not in n and p.requires_grad]
        classifier_params = [p for n, p in self.model.named_parameters() 
                            if 'classifier' in n and p.requires_grad]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.lr},
            {'params': classifier_params, 'lr': self.lr * 10}  # 10x LR for new head
        ], weight_decay=0.01)
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        print(f"  Optimizer: AdamW")
        print(f"    Backbone LR: {self.lr:.1e}")
        print(f"    Classifier LR: {self.lr * 10:.1e}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Handle target format for loss computation
            if self.multilabel:
                loss = self.criterion(output, target.float())
            else:
                # For single-label, target might be one-hot encoded
                if target.dim() == 2 and target.shape[1] > 1:
                    target_labels = target.argmax(dim=1)
                else:
                    target_labels = target.long()
                loss = self.criterion(output, target_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            if self.multilabel:
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).float().mean().item() * target.size(0)
            else:
                pred = output.argmax(dim=1)
                if target.dim() == 2:
                    target_labels = target.argmax(dim=1)
                else:
                    target_labels = target
                correct += pred.eq(target_labels).sum().item()
            
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Handle target format for loss computation
                if self.multilabel:
                    loss = self.criterion(output, target.float())
                else:
                    if target.dim() == 2 and target.shape[1] > 1:
                        target_labels = target.argmax(dim=1)
                    else:
                        target_labels = target.long()
                    loss = self.criterion(output, target_labels)
                
                total_loss += loss.item()
                
                if self.multilabel:
                    pred = (torch.sigmoid(output) > 0.5).float()
                    correct += (pred == target).float().mean().item() * target.size(0)
                else:
                    pred = output.argmax(dim=1)
                    if target.dim() == 2:
                        target_labels = target.argmax(dim=1)
                    else:
                        target_labels = target
                    correct += pred.eq(target_labels).sum().item()
                
                total += target.size(0)
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        """Main training loop."""
        print("\nStarting fine-tuning...")
        
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('birdclef_finetuned_best.pt')
                print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Save final model
        self.save_model('birdclef_finetuned_final.pt')
        
        # Save training history
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"  Best val accuracy: {best_val_acc:.2f}%")
        print(f"  Models saved to: {self.output_folder}")
    
    def save_model(self, filename):
        """Save model and config."""
        model_path = os.path.join(self.output_folder, filename)
        
        # Save model weights
        torch.save(self.model.state_dict(), model_path)
        
        # Save config for inference
        config_path = model_path.replace('.pt', '_config.json')
        config_dict = {
            'model_type': 'birdclef_finetuned',
            'architecture': 'regnety_008',
            'num_classes': self.num_classes,
            'class_names': self.categories,
            'multilabel': self.multilabel,
            'pretrained_from': self.pretrained_path,
            'freeze_backbone': self.freeze_backbone,
            'freeze_stages': self.freeze_stages,
            'freq_bins': config.DEFAULT_FREQ_BINS,
            'time_bins': config.DEFAULT_TIME_BINS
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BirdClef pretrained model on AviaNZ data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning (all layers trainable, 10 epochs)
  python finetune_birdclef.py data/train outputs/birdclef_ft
  
  # Freeze backbone, train only classifier (fast, good for small datasets)
  python finetune_birdclef.py data/train outputs/birdclef_ft --freeze-backbone --epochs 5
  
  # Partial freeze: train last 2 stages + classifier
  python finetune_birdclef.py data/train outputs/birdclef_ft --freeze-stages 3 --epochs 10
  
  # Multi-label classification
  python finetune_birdclef.py data/train outputs/birdclef_ft --multilabel --epochs 15
  
  # Then generate predictions:
  python predict.py outputs/birdclef_ft/birdclef_finetuned_best.pt \\
                    outputs/birdclef_ft/birdclef_finetuned_best_config.json \\
                    data/test \\
                    birdclef_predictions.csv
  
  # Compare with your other models:
  python compare_models.py --ground-truth data/test/labels.json \\
                           --predictions birdclef_predictions.csv ast_predictions.csv kaytoo_predictions.csv
        """
    )
    
    parser.add_argument('data_folder', type=str,
                       help="Path to training data folder (with labels.json and .npy spectrograms)")
    parser.add_argument('output_folder', type=str,
                       help="Path to output folder for saved models")
    parser.add_argument('--pretrained', default='BirdClefModels/model_fold0.pth',
                       help="Path to BirdClef pretrained checkpoint (default: BirdClefModels/model_fold0.pth)")
    parser.add_argument('--epochs', type=int, default=10,
                       help="Number of training epochs (default: 10, try 5-15)")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-4,
                       help="Learning rate for backbone (default: 1e-4, classifier gets 10x)")
    parser.add_argument('--freeze-backbone', action='store_true',
                       help="Freeze entire backbone, only train classifier (fastest, least overfitting)")
    parser.add_argument('--freeze-stages', type=int, default=0,
                       help="Freeze first N stages of backbone (0-4, default: 0 = all trainable)")
    parser.add_argument('--multilabel', action='store_true',
                       help="Use multi-label classification")
    parser.add_argument('--device', default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.path.exists(args.pretrained):
        print(f"ERROR: Pretrained model not found: {args.pretrained}")
        print("Make sure you have the BirdClef checkpoint in BirdClefModels/")
        return
    
    if not os.path.exists(os.path.join(args.data_folder, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.data_folder}")
        print("Data folder must contain labels.json and .npy spectrogram files")
        return
    
    # Create fine-tuner
    finetuner = BirdClefFineTuner(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        freeze_stages=args.freeze_stages,
        multilabel=args.multilabel,
        device=torch.device(args.device) if args.device else None
    )
    
    # Load data and create model
    finetuner.load_data()
    finetuner.create_model()
    
    # Train
    finetuner.train()


if __name__ == '__main__':
    main()
