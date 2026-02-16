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
            
            # Define dummy classes that might be in checkpoint to avoid unpickling errors
            import sys
            if '__main__' not in sys.modules:
                sys.modules['__main__'] = sys.modules[__name__]
            
            # Create dummy CFG class if needed by checkpoint
            class CFG:
                pass
            
            # Add to current module so unpickler can find it
            sys.modules[__name__].CFG = CFG
            globals()['CFG'] = CFG
            
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Warning: Could not load full checkpoint: {e}")
                print("Attempting to load state_dict only...")
                # Fallback: try loading with torch.load and extracting just state_dict
                checkpoint = {'model_state_dict': torch.load(pretrained_path, map_location='cpu', weights_only=False)}
            
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
                 freeze_stages=0, multilabel=False, device=None,
                 use_class_weights=False, pos_weight_cap=None,
                 normalize=False, mixup_alpha=0.0, noise_ratio=0.0, 
                 noise_folder=None, use_temporal_roll=True, validation_split=0.2,
                 remove_baseline=True):
        
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.freeze_stages = freeze_stages
        self.multilabel = multilabel
        self.use_class_weights = use_class_weights
        self.pos_weight_cap = pos_weight_cap
        self.normalize = normalize
        self.mixup_alpha = mixup_alpha
        self.noise_ratio = noise_ratio
        self.noise_folder = noise_folder
        self.use_temporal_roll = use_temporal_roll
        self.validation_split = validation_split
        self.remove_baseline = remove_baseline
        
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
        if self.multilabel and self.use_class_weights:
            print(f"  Class-weighted BCE: enabled")
        if self.normalize:
            print(f"  Background normalization: enabled")
        if self.mixup_alpha > 0:
            print(f"  Mixup alpha: {self.mixup_alpha}")
        if self.noise_ratio > 0:
            print(f"  Noise augmentation: {self.noise_ratio}")
    
    def load_data(self):
        """Load data using existing AviaNZ data pipeline."""
        print("\nLoading dataset...")
        
        data_loader = DataLoader(self.data_folder, noise_folder=self.noise_folder)
        self.data = data_loader.load_data(self.multilabel, validation_share=self.validation_split)
        
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
            noise_ratio=self.noise_ratio,
            spec_transform=None,
            num_workers=num_workers,
            width_downsizing=None,
            mixup_alpha=self.mixup_alpha,
            use_class_balancing=False,
            normalize=self.normalize,
            use_sparse_patches=False,
            num_sparse_patches=0,
            use_temporal_roll=self.use_temporal_roll,
            remove_baseline=self.remove_baseline
        )
        
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"  Val samples: {len(self.val_loader.dataset)}")
        else:
            print(f"  Val samples: 0 (validation disabled)")
    
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
            pos_weight = None
            if self.use_class_weights:
                pos_weight = self._compute_pos_weight()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

    def _compute_pos_weight(self):
        train_labels = np.array(self.data['train_labels'], dtype=np.float32)
        class_counts = train_labels.sum(axis=0)
        total_samples = len(train_labels)

        pos_counts = class_counts
        neg_counts = total_samples - class_counts
        pos_weight = neg_counts / (pos_counts + 1e-5)

        if self.pos_weight_cap is not None:
            pos_weight = np.clip(pos_weight, 1.0, float(self.pos_weight_cap))

        pos_weight = torch.from_numpy(pos_weight).float().to(self.device)
        print(
            f"  Class weights (pos_weight) - min: {pos_weight.min().item():.2f}, "
            f"max: {pos_weight.max().item():.2f}, mean: {pos_weight.mean().item():.2f}"
        )
        if self.pos_weight_cap is not None:
            capped = (pos_weight == float(self.pos_weight_cap)).sum().item()
            print(f"  Rare classes (capped at {float(self.pos_weight_cap):.0f}): {capped}/{len(pos_weight)}")
        return pos_weight

    def _compute_multilabel_metrics(self, logits, targets, threshold=0.5):
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold)
        targets = targets.to(dtype=torch.bool)

        tp = (preds & targets).sum(dim=0).to(dtype=torch.float32)
        fp = (preds & (~targets)).sum(dim=0).to(dtype=torch.float32)
        fn = ((~preds) & targets).sum(dim=0).to(dtype=torch.float32)

        denom = (2.0 * tp + fp + fn).clamp_min(1e-8)
        f1_per_class = (2.0 * tp) / denom

        support = targets.sum(dim=0)
        valid = support > 0
        macro_f1 = f1_per_class[valid].mean().item() if valid.any() else 0.0

        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()
        micro_denom = (2.0 * tp_micro + fp_micro + fn_micro).clamp_min(1e-8)
        micro_f1 = (2.0 * tp_micro / micro_denom).item()

        bit_acc = (preds == targets).to(dtype=torch.float32).mean().item()
        exact_match = (preds == targets).all(dim=1).to(dtype=torch.float32).mean().item()

        return {
            'bit_acc': bit_acc,
            'exact_match': exact_match,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        metrics_sum = {'bit_acc': 0.0, 'exact_match': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
        
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
                batch_metrics = self._compute_multilabel_metrics(output, target)
                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k] * target.size(0)
            else:
                pred = output.argmax(dim=1)
                if target.dim() == 2:
                    target_labels = target.argmax(dim=1)
                else:
                    target_labels = target
                correct += pred.eq(target_labels).sum().item()
            
            total += target.size(0)
            
            if self.multilabel:
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'macro_f1': metrics_sum['macro_f1'] / max(total, 1),
                    'bit_acc': metrics_sum['bit_acc'] / max(total, 1)
                })
            else:
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        if self.multilabel:
            avg_metrics = {k: metrics_sum[k] / max(total, 1) for k in metrics_sum}
            return total_loss / len(self.train_loader), avg_metrics
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        metrics_sum = {'bit_acc': 0.0, 'exact_match': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
        
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
                    batch_metrics = self._compute_multilabel_metrics(output, target)
                    for k in metrics_sum:
                        metrics_sum[k] += batch_metrics[k] * target.size(0)
                else:
                    pred = output.argmax(dim=1)
                    if target.dim() == 2:
                        target_labels = target.argmax(dim=1)
                    else:
                        target_labels = target
                    correct += pred.eq(target_labels).sum().item()
                
                total += target.size(0)

        if self.multilabel:
            avg_metrics = {k: metrics_sum[k] / max(total, 1) for k in metrics_sum}
            return total_loss / len(self.val_loader), avg_metrics
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        """Main training loop."""
        print("\nStarting fine-tuning...")

        best_val_metric = -1.0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_macro_f1': [],
            'val_macro_f1': [],
            'train_micro_f1': [],
            'val_micro_f1': [],
            'train_exact_match': [],
            'val_exact_match': [],
            'train_bit_acc': [],
            'val_bit_acc': [],
        }
        
        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Only validate if validation set exists
            if self.val_loader is not None:
                val_loss, val_metrics = self.validate()
            else:
                val_loss, val_metrics = None, None
            
            self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss if val_loss is not None else None)

            if self.multilabel:
                history['train_macro_f1'].append(train_metrics['macro_f1'])
                history['val_macro_f1'].append(val_metrics['macro_f1'] if val_metrics else None)
                history['train_micro_f1'].append(train_metrics['micro_f1'])
                history['val_micro_f1'].append(val_metrics['micro_f1'] if val_metrics else None)
                history['train_exact_match'].append(train_metrics['exact_match'])
                history['val_exact_match'].append(val_metrics['exact_match'] if val_metrics else None)
                history['train_bit_acc'].append(train_metrics['bit_acc'])
                history['val_bit_acc'].append(val_metrics['bit_acc'] if val_metrics else None)
                history['train_acc'].append(train_metrics['macro_f1'])
                history['val_acc'].append(val_metrics['macro_f1'] if val_metrics else None)
            else:
                history['train_acc'].append(train_metrics)
                history['val_acc'].append(val_metrics if val_metrics is not None else None)
            
            print(f"Epoch {epoch+1}/{self.epochs}:")
            if self.multilabel:
                print(
                    f"  Train Loss: {train_loss:.4f}, "
                    f"Macro F1: {train_metrics['macro_f1']:.4f}, "
                    f"Micro F1: {train_metrics['micro_f1']:.4f}, "
                    f"Bit Acc: {train_metrics['bit_acc']:.4f}, "
                    f"Exact: {train_metrics['exact_match']:.4f}"
                )
                if val_metrics:
                    print(
                        f"  Val Loss: {val_loss:.4f}, "
                        f"Macro F1: {val_metrics['macro_f1']:.4f}, "
                        f"Micro F1: {val_metrics['micro_f1']:.4f}, "
                        f"Bit Acc: {val_metrics['bit_acc']:.4f}, "
                        f"Exact: {val_metrics['exact_match']:.4f}"
                    )
            else:
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics:.2f}%")
                if val_metrics is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_metrics:.2f}%")
            
            # Determine current metric for model saving
            if val_metrics is not None:
                if self.multilabel:
                    current_metric = val_metrics['macro_f1']
                    metric_name = 'val_macro_f1'
                else:
                    current_metric = val_metrics
                    metric_name = 'val_acc'

                # Save best model based on validation
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    self.save_model('birdclef_finetuned_best.pt')
                    print(f"  ✓ Saved best model ({metric_name}: {current_metric:.4f})")
            else:
                # No validation - save every epoch as "best"
                self.save_model('birdclef_finetuned_best.pt')
                print(f"  ✓ Saved model (no validation)")
        
        # Save final model
        self.save_model('birdclef_finetuned_final.pt')
        
        # Save training history
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Fine-tuning complete!")
        if self.val_loader is not None:
            if self.multilabel:
                print(f"  Best val macro F1: {best_val_metric:.4f}")
            else:
                print(f"  Best val accuracy: {best_val_metric:.2f}%")
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
            'time_bins': config.DEFAULT_TIME_BINS,
            'normalize': self.normalize,
            'remove_baseline': self.remove_baseline
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
  
  # Multi-label with augmentation
  python finetune_birdclef.py data/train outputs/birdclef_ft --multilabel --mixup 0.3 --epochs 15
  
  # For soundscapes: use normalization + noise augmentation
  python finetune_birdclef.py data/train outputs/birdclef_ft --normalize --noise 0.3 --noise-folder noise_data
  
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
    parser.add_argument('--class-weights', action='store_true',
                       help="Use class-weighted BCE (pos_weight) in multilabel mode")
    parser.add_argument('--pos-weight-cap', type=float, default=None,
                       help="Optional cap for multilabel BCE pos_weight (e.g., 20). Only used with --class-weights")
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization to spectrograms (recommended for soundscapes)")
    parser.add_argument('--no-baseline-removal', action='store_true',
                       help="Disable baseline removal (default: enabled). Baseline removal subtracts 10th percentile to fix DC offset differences between datasets")
    parser.add_argument('--mixup', type=float, default=0.0,
                       help="Mixup alpha for data augmentation (default: 0.0 = disabled, try 0.2-0.4)")
    parser.add_argument('--noise', type=float, default=0.0,
                       help="Noise mixing ratio for augmentation (default: 0.0 = disabled, try 0.2-0.5)")
    parser.add_argument('--noise-folder', type=str, default=None,
                       help="Path to noise data folder for augmentation (default: same as data_folder)")
    parser.add_argument('--no-temporal-roll', action='store_true',
                       help="Disable temporal rolling augmentation")
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help="Validation split ratio (default: 0.2 = 20%%, use 0 to disable validation)")
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
        device=torch.device(args.device) if args.device else None,
        use_class_weights=args.class_weights,
        pos_weight_cap=args.pos_weight_cap,
        normalize=args.normalize,
        mixup_alpha=args.mixup,
        noise_ratio=args.noise,
        noise_folder=args.noise_folder,
        use_temporal_roll=not args.no_temporal_roll,
        validation_split=args.validation_split,
        remove_baseline=not args.no_baseline_removal
    )
    
    # Load data and create model
    finetuner.load_data()
    finetuner.create_model()
    
    # Train
    finetuner.train()


if __name__ == '__main__':
    main()
