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
from data_utils import DataLoader, create_data_loaders, SpectrogramDataset
from evaluation_utils import EvaluationManager
from models import GradientReversalLayer, DomainDiscriminator


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
        
        # Store feature dimension for potential domain adaptation
        self.feature_dim = backbone_out
        
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
                 normalize=False, mixup_alpha=0.0, mixup_mode='mixup', noise_ratio=0.0, 
                 noise_folder=None, noise_mode='full', use_temporal_roll=True, validation_split=0.2,
                 remove_baseline=False, test_folder=None, test_folder2=None, background_prob=0.0,
                 use_dann=False, target_folder=None, lambda_domain=0.1):
        
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
        self.mixup_mode = mixup_mode
        self.noise_ratio = noise_ratio
        self.noise_folder = noise_folder
        self.use_dann = use_dann
        self.target_folder = target_folder
        self.lambda_domain = lambda_domain
        self.noise_mode = noise_mode
        self.use_temporal_roll = use_temporal_roll
        self.validation_split = validation_split
        self.remove_baseline = remove_baseline
        self.test_folder = test_folder
        self.test_folder2 = test_folder2
        self.background_prob = background_prob
        
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
            mode_name = {'mixup': 'Mixup', 'cutmix': 'CutMix', 'both': 'Mixup+CutMix'}[self.mixup_mode]
            print(f"  {mode_name} alpha: {self.mixup_alpha}")
        if self.noise_ratio > 0:
            noise_mode_name = {'full': 'full spectrogram', 'background': 'quiet segments', 'both': 'mixed'}
            print(f"  Noise augmentation: expected ratio {self.noise_ratio} (uniformly sampled [0, min({2*self.noise_ratio:.1f}, 1.0)], clipped), mode: {noise_mode_name.get(self.noise_mode, self.noise_mode)}")
        if self.background_prob > 0:
            print(f"  Background replacement: {self.background_prob*100:.1f}% (replaces samples with background, zeros labels)")
        if self.use_dann:
            print(f"  DANN: enabled (lambda={self.lambda_domain})")
            print(f"  Target domain: {self.target_folder}")
            print(f"  IMPORTANT: Watch 'dacc' (domain accuracy) during training:")
            print(f"    - dacc ~100% = discriminator winning, DANN failing (increase lambda)")
            print(f"    - dacc ~50% = DANN working, features domain-invariant")
            print(f"    - If classification collapses, decrease lambda")
    
    def load_data(self):
        """Load data using existing AviaNZ data pipeline."""
        print("\nLoading dataset...")
        
        data_loader = DataLoader(self.data_folder, noise_folder=self.noise_folder)
        self.data = data_loader.load_data(self.multilabel, validation_share=self.validation_split)
        
        self.test_datasets = []
        
        if self.test_folder:
            print(f"  Loading test set 1 from: {self.test_folder}")
            test_loader = DataLoader(self.test_folder, noise_folder=self.noise_folder)
            test_data = test_loader.load_data(self.multilabel, validation_share=0.0)
            
            test_labels = test_data['train_labels']
            if self.data['categories'] != test_data['categories']:
                print(f"  WARNING: Train and test1 categories differ - remapping labels!")
                print(f"    Train: {self.data['categories']}")
                print(f"    Test1: {test_data['categories']}")
                test_labels = self._remap_labels(test_data['train_labels'], test_data['categories'], self.data['categories'])
            
            test_name = f"{Path(self.test_folder).parent.name}/{Path(self.test_folder).name}"
            self.test_datasets.append({
                'name': test_name,
                'path': self.test_folder,
                'filenames': test_data['train_filenames'],
                'labels': test_labels,
                'primary_species': test_data['train_primary_species'],
                'noise_filenames': test_data['train_noise_filenames']
            })
        
        if self.test_folder2:
            print(f"  Loading test set 2 from: {self.test_folder2}")
            test_loader2 = DataLoader(self.test_folder2, noise_folder=self.noise_folder)
            test_data2 = test_loader2.load_data(self.multilabel, validation_share=0.0)
            
            test_labels2 = test_data2['train_labels']
            if self.data['categories'] != test_data2['categories']:
                print(f"  WARNING: Train and test2 categories differ - remapping labels!")
                print(f"    Train: {self.data['categories']}")
                print(f"    Test2: {test_data2['categories']}")
                test_labels2 = self._remap_labels(test_data2['train_labels'], test_data2['categories'], self.data['categories'])
            
            test_name2 = f"{Path(self.test_folder2).parent.name}/{Path(self.test_folder2).name}"
            self.test_datasets.append({
                'name': test_name2,
                'path': self.test_folder2,
                'filenames': test_data2['train_filenames'],
                'labels': test_labels2,
                'primary_species': test_data2['train_primary_species'],
                'noise_filenames': test_data2['train_noise_filenames']
            })
        
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
            remove_baseline=self.remove_baseline,
            mixup_mode=self.mixup_mode,
            noise_mode=self.noise_mode,
            background_prob=self.background_prob
        )
        
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"  Val samples: {len(self.val_loader.dataset)}")
        else:
            print(f"  Val samples: 0 (validation disabled)")
        
        if self.use_dann and self.target_folder:
            print(f"\n  Loading DANN target domain (unlabeled)...")
            target_loader = DataLoader(self.target_folder)
            target_data = target_loader.load_data(use_multilabel=False, validation_share=0.0)
            
            img_height = config.DEFAULT_FREQ_BINS
            img_width = config.DEFAULT_TIME_BINS
            spec_transform = None
            
            target_dataset = SpectrogramDataset(
                target_data['train_filenames'], target_data['train_labels'],
                img_height, img_width, config.DEFAULT_CHANNELS, 'random',
                noise_filenames=target_data['train_noise_filenames'],
                noise_ratio=0.0,
                spec_transform=spec_transform,
                training=True,
                width_downsizing=None,
                normalize=self.normalize,
                use_sparse_patches=False,
                num_sparse_patches=0,
                use_temporal_roll=self.use_temporal_roll,
                remove_baseline=self.remove_baseline,
                noise_mode='full',
                background_prob=0.0
            )
            
            self.target_loader = torch.utils.data.DataLoader(
                target_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
            print(f"  Target domain samples: {len(self.target_loader.dataset)}")
        else:
            self.target_loader = None
    
    def _remap_labels(self, labels, source_categories, target_categories):
        import numpy as np
        labels = np.array(labels)
        category_map = {source_categories.index(cat): target_categories.index(cat) 
                       for cat in source_categories if cat in target_categories}
        
        if labels.ndim == 2:
            remapped = np.zeros_like(labels)
            for src_idx, tgt_idx in category_map.items():
                remapped[:, tgt_idx] = labels[:, src_idx]
            return remapped.tolist()
        else:
            remapped = np.array([category_map.get(int(label), -1) for label in labels])
            if (remapped == -1).any():
                print(f"  ERROR: Some labels couldn't be remapped!")
            return remapped.tolist()
    
    def _compute_accuracy_from_csv(self, csv_path, test_folder):
        import pandas as pd
        labels_path = os.path.join(test_folder, 'labels.json')
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        true_labels = {}
        for item in labels_data['files']:
            true_labels[item['filename']] = item.get('primary_class') or item.get('primary_species')
        
        df = pd.read_csv(csv_path)
        class_columns = [col for col in df.columns if col not in ['row_id', 'File_Path']]
        
        correct = 0
        total = 0
        
        for _, row in df.iterrows():
            filename = row['row_id']
            if filename not in true_labels:
                continue
            
            pred_class = class_columns[row[class_columns].values.argmax()]
            true_class = true_labels[filename]
            
            if pred_class == true_class:
                correct += 1
            total += 1
        
        return 100.0 * correct / total if total > 0 else 0.0
    
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
        
        if self.use_dann:
            print("  Adding DANN domain discriminator...")
            feat_dim = self.model.feature_dim
            
            self.grl = GradientReversalLayer()
            # Simplest possible discriminator - just linear layer
            # Strong discriminator = backbone can't fool it = DANN fails
            self.domain_classifier = nn.Linear(feat_dim, 1)
            self.grl.to(self.device)
            self.domain_classifier.to(self.device)
            self.domain_criterion = nn.BCEWithLogitsLoss()

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
        
        param_groups = [
            {'params': backbone_params, 'lr': self.lr},
            {'params': classifier_params, 'lr': self.lr * 10}
        ]
        
        if self.use_dann:
            # Domain classifier needs LOWER LR than backbone to prevent it from dominating
            # If discriminator is too good, backbone can't fool it
            param_groups.append({'params': self.domain_classifier.parameters(), 'lr': self.lr})
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        
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
        if self.use_dann:
            self.grl.train()
            self.domain_classifier.train()
            
        total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0
        domain_correct = 0
        domain_total = 0

        metrics_sum = {'bit_acc': 0.0, 'exact_match': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
        
        if self.use_dann and self.target_loader:
            # DANN schedule: gradually increase domain adaptation strength
            # Using gentler 0→1 schedule to prevent domain loss from dominating
            p = float(epoch) / float(self.epochs)
            alpha = 1.0 / (1.0 + np.exp(-10 * p))  # Sigmoid: 0 → ~1
            self.grl.lambda_param = alpha * self.lambda_domain
            
            # Debug: print schedule at key epochs
            if epoch == 0 or epoch == self.epochs // 2 or epoch == self.epochs - 1:
                print(f"  [DANN Schedule] Epoch {epoch+1}: alpha={alpha:.3f}, effective_lambda={self.grl.lambda_param:.3f}")
            
            source_iter = iter(self.train_loader)
            target_iter = iter(self.target_loader)
            n_batches = min(len(self.train_loader), len(self.target_loader))
            pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.epochs}")
        else:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx in (pbar if self.use_dann and self.target_loader else enumerate(pbar)):
            if self.use_dann and self.target_loader:
                try:
                    source_data, source_target = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.train_loader)
                    source_data, source_target = next(source_iter)
                
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_target = source_target.to(self.device)
                target_data = target_data.to(self.device)
                
                batch_size_s = source_data.size(0)
                batch_size_t = target_data.size(0)
                
                combined_data = torch.cat([source_data, target_data], dim=0)
                
                self.optimizer.zero_grad()
                
                features = self.model.backbone(combined_data)
                if isinstance(features, dict):
                    features = features['features']
                if len(features.shape) == 4:
                    features = self.model.pooling(features)
                    features = features.view(features.size(0), -1)
                
                class_output = self.model.classifier(features)
                source_class_output = class_output[:batch_size_s]
                
                if self.multilabel:
                    class_loss = self.criterion(source_class_output, source_target.float())
                else:
                    if source_target.dim() == 2 and source_target.shape[1] > 1:
                        is_soft = not torch.all((source_target == 0) | (source_target == 1))
                        if is_soft:
                            log_probs = torch.log_softmax(source_class_output, dim=1)
                            class_loss = -(source_target * log_probs).sum(dim=1).mean()
                        else:
                            target_labels = source_target.argmax(dim=1)
                            class_loss = self.criterion(source_class_output, target_labels)
                    else:
                        target_labels = source_target.long()
                        class_loss = self.criterion(source_class_output, target_labels)
                
                reversed_features = self.grl(features)
                domain_output = self.domain_classifier(reversed_features)
                
                domain_labels_source = torch.zeros(batch_size_s).to(self.device)
                domain_labels_target = torch.ones(batch_size_t).to(self.device)
                domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
                
                domain_loss = self.domain_criterion(domain_output.squeeze(), domain_labels)
                
                # Track domain discriminator accuracy (should approach 50% if DANN working)
                domain_pred = (torch.sigmoid(domain_output.squeeze()) > 0.5).float()
                domain_correct += (domain_pred == domain_labels).sum().item()
                domain_total += domain_labels.size(0)
                domain_acc = 100.0 * domain_correct / domain_total
                
                # GRL handles gradient reversal and scaling internally
                loss = class_loss + domain_loss
                
                loss.backward()
                
                # Gradient clipping prevents instability from domain adaptation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_class_loss += class_loss.item()
                total_domain_loss += domain_loss.item()
                total_loss += loss.item()
                
                if self.multilabel:
                    batch_metrics = self._compute_multilabel_metrics(source_class_output, source_target)
                    for k in metrics_sum:
                        metrics_sum[k] += batch_metrics[k] * source_target.size(0)
                else:
                    pred = source_class_output.argmax(dim=1)
                    if source_target.dim() == 2:
                        target_labels = source_target.argmax(dim=1)
                    else:
                        target_labels = source_target
                    correct += pred.eq(target_labels).sum().item()
                
                total += source_target.size(0)
                
                if self.multilabel:
                    pbar.set_postfix({
                        'cls': f'{class_loss.item():.3f}',
                        'dom': f'{domain_loss.item():.3f}',
                        'f1': f'{metrics_sum["macro_f1"]/max(total,1):.3f}',
                        'dacc': f'{domain_acc:.1f}%',
                        'α': f'{self.grl.lambda_param:.3f}'
                    })
                else:
                    pbar.set_postfix({
                        'cls': f'{class_loss.item():.3f}',
                        'dom': f'{domain_loss.item():.3f}',
                        'acc': f'{100.*correct/total:.1f}%',
                        'dacc': f'{domain_acc:.1f}%',
                        'α': f'{self.grl.lambda_param:.3f}'
                    })
            else:
                batch_idx, (data, target) = batch_idx
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                if self.multilabel:
                    loss = self.criterion(output, target.float())
                else:
                    if target.dim() == 2 and target.shape[1] > 1:
                        is_soft = not torch.all((target == 0) | (target == 1))
                        
                        if is_soft:
                            log_probs = torch.log_softmax(output, dim=1)
                            loss = -(target * log_probs).sum(dim=1).mean()
                        else:
                            target_labels = target.argmax(dim=1)
                            loss = self.criterion(output, target_labels)
                    else:
                        target_labels = target.long()
                        loss = self.criterion(output, target_labels)
                
                loss.backward()
                self.optimizer.step()
                
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
            if self.use_dann and self.target_loader:
                avg_metrics['domain_acc'] = 100.0 * domain_correct / max(domain_total, 1)
            return total_loss / len(self.train_loader), avg_metrics
        
        train_acc = 100. * correct / total
        if self.use_dann and self.target_loader:
            avg_loss = total_loss / n_batches
            return avg_loss, (train_acc, 100.0 * domain_correct / max(domain_total, 1))
        return total_loss / len(self.train_loader), train_acc
    
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
    
    def evaluate_test_set(self, test_data, test_name="Test"):
        from data_utils import SpectrogramDataset
        from torch.utils.data import DataLoader as TorchDataLoader
        
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        
        # Use default spec_transform from config (matches training/validation)
        spec_transform = config.DEFAULT_SPEC_TRANSFORM
        
        test_dataset = SpectrogramDataset(
            test_data['filenames'],
            test_data['labels'],
            img_height,
            img_width,
            config.DEFAULT_CHANNELS,
            cropping_mode='center',
            noise_filenames=None,
            noise_ratio=0.0,
            spec_transform=spec_transform,
            training=False,
            width_downsizing=None,
            normalize=self.normalize,
            use_sparse_patches=False,
            num_sparse_patches=0,
            use_temporal_roll=False,
            remove_baseline=self.remove_baseline
        )
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 2,
            pin_memory=True
        )
        
        self.model.eval()
        correct = 0
        total = 0
        metrics_sum = {'macro_f1': 0.0, 'micro_f1': 0.0, 'macro_precision': 0.0, 'macro_recall': 0.0}
        
        print(f"  DEBUG: Evaluating {len(test_loader.dataset)} test samples")
        print(f"  DEBUG: Model num_classes: {self.model.num_classes}")
        print(f"  DEBUG: Model categories: {self.categories}")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if batch_idx == 0:
                    print(f"  DEBUG: First batch - target shape: {target.shape}, output shape: {output.shape}")
                    print(f"  DEBUG: First sample target: {target[0]}")
                    print(f"  DEBUG: First sample output (top 3): {output[0].topk(3)}")
                
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
            print(f"  {test_name} Macro F1: {avg_metrics['macro_f1']:.4f}")
            print(f"  {test_name} Micro F1: {avg_metrics['micro_f1']:.4f}")
            return avg_metrics['macro_f1']
        else:
            test_acc = 100. * correct / total
            print(f"  {test_name} Accuracy: {test_acc:.2f}%")
            return test_acc
    
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
                metrics_str = (
                    f"  Train Loss: {train_loss:.4f}, "
                    f"Macro F1: {train_metrics['macro_f1']:.4f}, "
                    f"Micro F1: {train_metrics['micro_f1']:.4f}, "
                    f"Bit Acc: {train_metrics['bit_acc']:.4f}, "
                    f"Exact: {train_metrics['exact_match']:.4f}"
                )
                if self.use_dann and 'domain_acc' in train_metrics:
                    metrics_str += f", Domain Acc: {train_metrics['domain_acc']:.1f}%"
                print(metrics_str)
                if val_metrics:
                    print(
                        f"  Val Loss: {val_loss:.4f}, "
                        f"Macro F1: {val_metrics['macro_f1']:.4f}, "
                        f"Micro F1: {val_metrics['micro_f1']:.4f}, "
                        f"Bit Acc: {val_metrics['bit_acc']:.4f}, "
                        f"Exact: {val_metrics['exact_match']:.4f}"
                    )
            else:
                if self.use_dann and isinstance(train_metrics, tuple):
                    train_acc, domain_acc = train_metrics
                    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Domain Acc: {domain_acc:.1f}%")
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
        
        if self.test_datasets:
            print(f"\nEvaluating on test sets using predict.py...")
            
            # Free GPU memory before evaluation
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            best_model_path = os.path.join(self.output_folder, 'birdclef_finetuned_best.pt')
            best_config_path = os.path.join(self.output_folder, 'birdclef_finetuned_best_config.json')
            
            test_results = {}
            for idx, test_data in enumerate(self.test_datasets, 1):
                test_folder = test_data['path']
                test_name = test_data['name']
                output_csv = os.path.join(self.output_folder, f'predictions_{test_name.replace("/", "_")}.csv')
                
                print(f"\nTest Set {idx}: {test_name}")
                
                from predict import ModelPredictor
                predictor = ModelPredictor(
                    best_model_path,
                    best_config_path,
                    test_folder,
                    output_csv,
                    batch_size=self.batch_size,
                    device=self.device
                )
                predictor.run()
                
                accuracy = self._compute_accuracy_from_csv(output_csv, test_folder)
                test_results[test_name] = accuracy
                print(f"  {test_name} Accuracy: {accuracy:.2f}%")
            
            print(f"\n{'='*60}")
            print(f"TEST SET COMPARISON")
            print(f"{'='*60}")
            for name, accuracy in test_results.items():
                print(f"  {name:30s} Accuracy: {accuracy:.2f}%")
            print(f"{'='*60}")
    
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
  
  # Evaluate on separate test set after training (not used for early stopping)
  python finetune_birdclef.py data/train outputs/birdclef_ft --test-folder data/test
  
  # Freeze backbone, train only classifier (fast, good for small datasets)
  python finetune_birdclef.py data/train outputs/birdclef_ft --freeze-backbone --epochs 5
  
  # Partial freeze: train last 2 stages + classifier
  python finetune_birdclef.py data/train outputs/birdclef_ft --freeze-stages 3 --epochs 10
  
  # Multi-label with augmentation
  python finetune_birdclef.py data/train outputs/birdclef_ft --multilabel --mixup 0.3 --epochs 15
  
  # Try CutMix augmentation (paste rectangular regions instead of blending)
  python finetune_birdclef.py data/train outputs/birdclef_ft --mixup 0.3 --mixup-mode cutmix
  
  # Combine both Mixup and CutMix (randomly applies one or the other)
  python finetune_birdclef.py data/train outputs/birdclef_ft --mixup 0.3 --mixup-mode both
  
  # For soundscapes: use normalization + noise augmentation
  python finetune_birdclef.py data/train outputs/birdclef_ft --normalize --noise 0.3 --noise-folder noise_data
  
  # Smart noise extraction: extract only quiet/background segments (no label mixing)
  python finetune_birdclef.py data/train outputs/birdclef_ft --noise 0.3 --noise-folder noise_data --noise-mode background
  
  # Mix of full and background noise extraction
  python finetune_birdclef.py data/train outputs/birdclef_ft --noise 0.3 --noise-folder noise_data --noise-mode both
  
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
    parser.add_argument('--baseline-removal', action='store_true',
                       help="Baseline removal (default: disabled). Baseline removal subtracts 10th percentile to fix DC offset differences between datasets")
    parser.add_argument('--mixup', type=float, default=0.0,
                       help="Mixup alpha for data augmentation (default: 0.0 = disabled, try 0.2-0.4)")
    parser.add_argument('--mixup-mode', type=str, default='mixup', choices=['mixup', 'cutmix', 'both'],
                       help="Augmentation mode when --mixup > 0: 'mixup' (blend entire spectrograms), 'cutmix' (paste rectangular regions), 'both' (randomly apply either). Default: mixup")
    parser.add_argument('--noise', type=float, default=0.0,
                       help="Expected noise mixing ratio for augmentation (uniformly sampled [0, min(2×ratio, 1.0)] so E[noise]≈ratio, clipped to valid range). 0.0=disabled, 0.3=30%% expected noise")
    parser.add_argument('--noise-folder', type=str, default=None,
                       help="Path to noise data folder for augmentation (default: same as data_folder)")
    parser.add_argument('--noise-mode', type=str, default='full', choices=['full', 'background', 'both'],
                       help="Noise extraction mode: 'full' (mix entire spectrogram), 'background' (extract quiet segments only), 'both' (random 50/50). Default: full")
    parser.add_argument('--background-prob', type=float, default=0.0,
                       help="Probability of replacing training sample with its background spectrogram and zeroing labels (teaches model to recognize no-bird-present). 0.0=disabled, 0.1=10%% of samples. Recommended: 0.05-0.15")
    parser.add_argument('--no-temporal-roll', action='store_true',
                       help="Disable temporal rolling augmentation")
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help="Validation split ratio (default: 0.2 = 20%%, use 0 to disable validation)")
    parser.add_argument('--test-folder', type=str, default=None,
                       help="Path to test data folder 1 (with labels.json). Evaluated AFTER training completes (not used for early stopping).")
    parser.add_argument('--test-folder2', type=str, default=None,
                       help="Path to test data folder 2 (with labels.json). Evaluated AFTER training completes (not used for early stopping).")
    parser.add_argument('--device', default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument('--use-dann', action='store_true',
                       help="Enable Domain Adaptive Neural Network (DANN) training for domain adaptation")
    parser.add_argument('--target-folder', type=str, default=None,
                       help="Path to target domain folder for DANN (unlabeled domain for adaptation)")
    parser.add_argument('--lambda-domain', type=float, default=0.3,
                       help="Domain loss weight for DANN (default: 0.3). If dacc stays >95%%, increase lambda. If classification collapses, decrease lambda. Typical working range: 0.2-1.0")
    
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
        mixup_mode=args.mixup_mode,
        noise_ratio=args.noise,
        noise_folder=args.noise_folder,
        noise_mode=args.noise_mode,
        use_temporal_roll=not args.no_temporal_roll,
        validation_split=args.validation_split,
        remove_baseline=args.baseline_removal,
        test_folder=args.test_folder,
        test_folder2=args.test_folder2,
        background_prob=args.background_prob,
        use_dann=args.use_dann,
        target_folder=args.target_folder,
        lambda_domain=args.lambda_domain
    )
    
    # Load data and create model
    finetuner.load_data()
    finetuner.create_model()
    
    # Train
    finetuner.train()


if __name__ == '__main__':
    main()
