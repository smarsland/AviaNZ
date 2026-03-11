"""
Domain-Adversarial Neural Network (DANN) for cross-dataset generalization.

Uses gradient reversal to learn features that are:
- Good for bird species classification
- Unable to distinguish which dataset they came from (domain-invariant)

This is more principled than just merging datasets - it explicitly optimizes
for domain-invariant features while preserving task performance.

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)

Usage:
    # Train on source dataset, adapt to target dataset
    python train_domain_adaptation.py source_data/ target_data/ outputs/dann_model
    
    # With BirdClef pretrained backbone
    python train_domain_adaptation.py source_data/ target_data/ outputs/dann_model \\
        --pretrained BirdClefModels/model_fold0.pth --architecture regnety_008
    
    # Control adversarial strength (0.5-10.0, higher = more domain confusion)
    # For strong domain adaptation, use 1.0-5.0. For lambda << 1.0, adversarial signal is too weak.
    python train_domain_adaptation.py source_data/ target_data/ outputs/dann_model \\
        --lambda-domain 1.0 --epochs 30
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
from torch.utils.data import DataLoader as TorchDataLoader, ConcatDataset
from torch.autograd import Function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from data_utils import DataLoader, SpectrogramDataset


class GradientReversalFunction(Function):
    """Gradient Reversal Layer - multiplies gradient by -lambda during backprop."""
    
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal."""
    
    def __init__(self, lambda_param=1.0):
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)
    
    def set_lambda(self, lambda_param):
        self.lambda_param = lambda_param


class DANNModel(nn.Module):
    """Domain-Adversarial Neural Network for cross-dataset adaptation."""
    
    def __init__(self, num_classes, architecture='resnet18', pretrained_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        
        self.backbone = timm.create_model(
            architecture,
            pretrained=False,
            in_chans=1,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        
        if 'efficientnet' in architecture:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in architecture:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'regnet' in architecture:
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        # Simple classifier like BirdClef - just one linear layer
        self.class_classifier = nn.Linear(backbone_out, num_classes)
        
        self.gradient_reversal = GradientReversalLayer()
        
        # Domain classifier can be deeper since it's a simpler task (binary)
        self.domain_classifier = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        
        if pretrained_path:
            self.load_pretrained_backbone(pretrained_path)
    
    def load_pretrained_backbone(self, pretrained_path):
        print(f"Loading pretrained backbone from {pretrained_path}")
        
        import sys
        if '__main__' not in sys.modules:
            sys.modules['__main__'] = sys.modules[__name__]
        
        class CFG:
            pass
        
        sys.modules[__name__].CFG = CFG
        globals()['CFG'] = CFG
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load full checkpoint: {e}")
            checkpoint = {'model_state_dict': torch.load(pretrained_path, map_location='cpu', weights_only=False)}
        
        backbone_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                if 'classifier' not in new_key and 'fc' not in new_key:
                    backbone_dict[new_key] = v
        
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print("  Loaded pretrained backbone weights")
    
    def forward(self, x, lambda_domain=1.0):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        class_output = self.class_classifier(features)
        
        self.gradient_reversal.set_lambda(lambda_domain)
        reversed_features = self.gradient_reversal(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output, features
    
    def predict(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        return self.class_classifier(features)


class DomainAdaptationTrainer:
    """Trains DANN model for domain adaptation."""
    
    def __init__(self, source_folder, target_folder, output_folder,
                 architecture='resnet18', pretrained_path=None,
                 epochs=30, batch_size=32, lr=1e-4,
                 lambda_domain=1.0, lambda_schedule='fixed',
                 multilabel=False, normalize=False, validation_split=0.2,
                 remove_baseline=False, test_folder=None, test_folder2=None, device=None):
        
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.output_folder = output_folder
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_domain = lambda_domain
        self.lambda_schedule = lambda_schedule
        self.multilabel = multilabel
        self.normalize = normalize
        self.validation_split = validation_split
        self.remove_baseline = remove_baseline
        self.test_folder = test_folder
        self.test_folder2 = test_folder2
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Domain Adaptation Training (DANN)")
        print(f"  Source dataset: {source_folder}")
        print(f"  Target dataset: {target_folder}")
        print(f"  Output: {output_folder}")
        print(f"  Architecture: {architecture}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Lambda (domain): {lambda_domain} ({lambda_schedule} schedule)")
        print(f"  Multi-label: {multilabel}")
    
    def remap_labels(self, labels, source_categories, target_categories):
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
    
    def load_data(self):
        print("\nLoading datasets...")
        
        source_loader = DataLoader(self.source_folder)
        source_data = source_loader.load_data(self.multilabel, validation_share=self.validation_split)
        
        target_loader = DataLoader(self.target_folder)
        target_data = target_loader.load_data(self.multilabel, validation_share=self.validation_split)
        
        if source_data['categories'] != target_data['categories']:
            print(f"  WARNING: Source and target categories differ - remapping labels!")
            print(f"    Source: {source_data['categories']}")
            print(f"    Target: {target_data['categories']}")
            target_data['train_labels'] = self.remap_labels(
                target_data['train_labels'], 
                target_data['categories'], 
                source_data['categories']
            )
            if len(target_data['test_filenames']) > 0:
                target_data['test_labels'] = self.remap_labels(
                    target_data['test_labels'], 
                    target_data['categories'], 
                    source_data['categories']
                )
        
        self.num_classes = source_data['nclasses']
        self.categories = source_data['categories']
        
        print(f"  Classes: {self.num_classes}")
        print(f"  Source samples: {len(source_data['train_filenames'])} train, {len(source_data['test_filenames'])} val")
        print(f"  Target samples: {len(target_data['train_filenames'])} train, {len(target_data['test_filenames'])} val")
        
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        
        source_val_filenames = source_data['test_filenames']
        source_val_labels = source_data['test_labels']
        target_val_filenames = target_data['test_filenames']
        target_val_labels = target_data['test_labels']
        
        self.source_train_dataset = SpectrogramDataset(
            source_data['train_filenames'],
            source_data['train_labels'],
            img_height, img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_filenames=None, noise_ratio=0.0,
            spec_transform=config.DEFAULT_SPEC_TRANSFORM, training=True, width_downsizing=None,
            normalize=self.normalize, use_sparse_patches=False,
            num_sparse_patches=0, use_temporal_roll=True,
            remove_baseline=self.remove_baseline
        )
        
        self.target_train_dataset = SpectrogramDataset(
            target_data['train_filenames'],
            target_data['train_labels'],
            img_height, img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_filenames=None, noise_ratio=0.0,
            spec_transform=config.DEFAULT_SPEC_TRANSFORM, training=True, width_downsizing=None,
            normalize=self.normalize, use_sparse_patches=False,
            num_sparse_patches=0, use_temporal_roll=True,
            remove_baseline=self.remove_baseline
        )
        
        if len(source_val_filenames) > 0:
            self.source_val_dataset = SpectrogramDataset(
                source_val_filenames,
                source_val_labels.tolist() if hasattr(source_val_labels, 'tolist') else source_val_labels,
                img_height, img_width, config.DEFAULT_CHANNELS,
                cropping_mode='center', noise_filenames=None, noise_ratio=0.0,
                spec_transform=config.DEFAULT_SPEC_TRANSFORM, training=False, width_downsizing=None,
                normalize=self.normalize, use_sparse_patches=False,
                num_sparse_patches=0, use_temporal_roll=False,
                remove_baseline=self.remove_baseline
            )
        else:
            self.source_val_dataset = None
        
        if len(target_val_filenames) > 0:
            self.target_val_dataset = SpectrogramDataset(
                target_val_filenames,
                target_val_labels.tolist() if hasattr(target_val_labels, 'tolist') else target_val_labels,
                img_height, img_width, config.DEFAULT_CHANNELS,
                cropping_mode='center', noise_filenames=None, noise_ratio=0.0,
                spec_transform=config.DEFAULT_SPEC_TRANSFORM, training=False, width_downsizing=None,
                normalize=self.normalize, use_sparse_patches=False,
                num_sparse_patches=0, use_temporal_roll=False,
                remove_baseline=self.remove_baseline
            )
        else:
            self.target_val_dataset = None
        
        num_workers = 4 if torch.cuda.is_available() else 2
        
        self.source_train_loader = TorchDataLoader(
            self.source_train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        self.target_train_loader = TorchDataLoader(
            self.target_train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        # For lambda=0 (no domain adaptation), create merged dataset for true equivalence with finetuning
        merged_train_dataset = ConcatDataset([self.source_train_dataset, self.target_train_dataset])
        self.merged_train_loader = TorchDataLoader(
            merged_train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        
        if self.source_val_dataset:
            self.source_val_loader = TorchDataLoader(
                self.source_val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True
            )
        else:
            self.source_val_loader = None
        
        if self.target_val_dataset:
            self.target_val_loader = TorchDataLoader(
                self.target_val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True
            )
        else:
            self.target_val_loader = None
        
        # Create merged validation loader for lambda=0
        if self.source_val_dataset and self.target_val_dataset:
            merged_val_dataset = ConcatDataset([self.source_val_dataset, self.target_val_dataset])
            self.merged_val_loader = TorchDataLoader(
                merged_val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True
            )
        elif self.source_val_dataset:
            self.merged_val_loader = self.source_val_loader
        elif self.target_val_dataset:
            self.merged_val_loader = self.target_val_loader
        else:
            self.merged_val_loader = None
    
    def create_model(self):
        print("\nCreating DANN model...")
        
        self.model = DANNModel(
            num_classes=self.num_classes,
            architecture=self.architecture,
            pretrained_path=self.pretrained_path
        )
        self.model.to(self.device)
        
        if self.multilabel:
            self.class_criterion = nn.BCEWithLogitsLoss()
        else:
            self.class_criterion = nn.CrossEntropyLoss()
        
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Use differential learning rates like BirdClef fine-tuning:
        # - Lower LR for pretrained backbone (needs fine adjustments)
        # - Higher LR for new classifier heads (need fast convergence)
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'class_classifier' in name or 'domain_classifier' in name or 'gradient_reversal' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.lr},
            {'params': classifier_params, 'lr': self.lr * 10}  # 10x LR for new heads
        ], weight_decay=0.01)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Optimizer: AdamW with differential LR")
        print(f"    Backbone LR: {self.lr:.1e}")
        print(f"    Classifier heads LR: {self.lr * 10:.1e}")
    
    def get_lambda_domain(self, epoch):
        if self.lambda_schedule == 'fixed':
            return self.lambda_domain
        elif self.lambda_schedule == 'progressive':
            p = epoch / self.epochs
            return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        else:
            return self.lambda_domain
    
    def train_epoch(self, epoch):
        self.model.train()
        total_class_loss = 0
        total_domain_loss = 0
        total_loss = 0
        correct_class = 0
        correct_domain = 0
        total_samples = 0
        
        lambda_domain = self.get_lambda_domain(epoch)
        
        # When lambda=0, use merged dataset for true equivalence with finetuning
        if lambda_domain == 0.0:
            pbar = tqdm(self.merged_train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to(self.device)
                target = target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with merged data (just like finetuning)
                class_output, domain_output, _ = self.model(data, lambda_domain)
                
                # Compute loss like finetuning
                if self.multilabel:
                    class_loss = self.class_criterion(class_output, target.float())
                else:
                    if target.dim() == 2:
                        target_idx = target.argmax(dim=1)
                    else:
                        target_idx = target.long()
                    class_loss = self.class_criterion(class_output, target_idx)
                
                loss = class_loss
                loss.backward()
                self.optimizer.step()
                
                total_class_loss += class_loss.item()
                total_domain_loss += 0.0
                total_loss += loss.item()
                
                # Track accuracy
                pred_class = class_output.argmax(dim=1)
                if self.multilabel:
                    if target.dim() == 2:
                        labels_primary = target.argmax(dim=1)
                    else:
                        labels_primary = target.long()
                    correct_class += pred_class.eq(labels_primary).sum().item()
                else:
                    if target.dim() == 2:
                        target_idx = target.argmax(dim=1)
                    else:
                        target_idx = target.long()
                    correct_class += pred_class.eq(target_idx).sum().item()
                
                total_samples += target.size(0)
                
                pbar.set_postfix({
                    'class_loss': total_class_loss / (batch_idx + 1),
                    'class_acc': 100. * correct_class / total_samples
                })
            
            train_class_acc = 100. * correct_class / total_samples
            avg_class_loss = total_class_loss / len(self.merged_train_loader)
            avg_domain_loss = 0.0
            train_domain_acc = 0.0
            
            return avg_class_loss, avg_domain_loss, train_class_acc, train_domain_acc
        
        # Standard DANN training with separate source/target loaders
        source_iter = iter(self.source_train_loader)
        target_iter = iter(self.target_train_loader)
        
        num_batches = max(len(self.source_train_loader), len(self.target_train_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx in pbar:
            try:
                source_data, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_train_loader)
                source_data, source_labels = next(source_iter)
            
            try:
                target_data, target_labels = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_train_loader)
                target_data, target_labels = next(target_iter)
            
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)
            
            batch_size_s = source_data.size(0)
            batch_size_t = target_data.size(0)
            
            # CRITICAL: Concatenate source and target into SINGLE batch
            # This makes batch norm see the same mixed distribution as BirdClef training
            combined_data = torch.cat([source_data, target_data], dim=0)
            combined_labels = torch.cat([source_labels, target_labels], dim=0)
            domain_labels = torch.cat([
                torch.zeros(batch_size_s, dtype=torch.long, device=self.device),
                torch.ones(batch_size_t, dtype=torch.long, device=self.device)
            ], dim=0)
            
            # When lambda=0, shuffle the combined batch to match random finetuning behavior
            if lambda_domain == 0.0:
                perm = torch.randperm(combined_data.size(0), device=self.device)
                combined_data = combined_data[perm]
                combined_labels = combined_labels[perm]
                domain_labels = domain_labels[perm]
            
            self.optimizer.zero_grad()
            
            # Single forward pass on combined batch (batch norm sees mixed distribution!)
            class_output, domain_output, _ = self.model(combined_data, lambda_domain)
            
            # Split outputs for separate source/target classification loss
            # Note: after shuffling with lambda=0, this split is meaningless, but we keep the structure
            class_output_s = class_output[:batch_size_s]
            class_output_t = class_output[batch_size_s:]
            source_labels_split = combined_labels[:batch_size_s]
            target_labels_split = combined_labels[batch_size_s:]
            
            # CRITICAL: Train classification on BOTH source and target (like BirdClef merged training)
            # Domain adaptation should learn from all available labeled data
            if lambda_domain == 0.0:
                # When lambda=0, behave exactly like merged training (no equal weighting, no domain loss)
                if self.multilabel:
                    class_loss = self.class_criterion(class_output, combined_labels.float())
                else:
                    if combined_labels.dim() == 2:
                        combined_labels_idx = combined_labels.argmax(dim=1)
                    else:
                        combined_labels_idx = combined_labels.long()
                    class_loss = self.class_criterion(class_output, combined_labels_idx)
                domain_loss = torch.tensor(0.0, device=self.device)
                loss = class_loss
            else:
                # Standard DANN with equal dataset weighting
                if self.multilabel:
                    class_loss_s = self.class_criterion(class_output_s, source_labels_split.float())
                    class_loss_t = self.class_criterion(class_output_t, target_labels_split.float())
                    class_loss = (class_loss_s + class_loss_t) / 2.0
                else:
                    if source_labels_split.dim() == 2:
                        source_labels_idx = source_labels_split.argmax(dim=1)
                    else:
                        source_labels_idx = source_labels_split.long()
                    if target_labels_split.dim() == 2:
                        target_labels_idx = target_labels_split.argmax(dim=1)
                    else:
                        target_labels_idx = target_labels_split.long()
                    class_loss_s = self.class_criterion(class_output_s, source_labels_idx)
                    class_loss_t = self.class_criterion(class_output_t, target_labels_idx)
                    class_loss = (class_loss_s + class_loss_t) / 2.0
                
                # Domain loss on combined output
                domain_loss = self.domain_criterion(domain_output, domain_labels)
                
                # Total loss: classification + domain adversarial loss
                # Lambda scaling is handled by gradient reversal layer (affects feature extractor gradients)
                loss = class_loss + domain_loss
            loss.backward()
            self.optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss
            total_loss += loss.item()
            
            # Track classification accuracy
            if lambda_domain == 0.0:
                # After shuffling, compute accuracy over entire combined batch
                pred_class = class_output.argmax(dim=1)
                if self.multilabel:
                    if combined_labels.dim() == 2:
                        labels_primary = combined_labels.argmax(dim=1)
                    else:
                        labels_primary = combined_labels.long()
                    correct_class += pred_class.eq(labels_primary).sum().item()
                else:
                    if combined_labels.dim() == 2:
                        labels_idx = combined_labels.argmax(dim=1)
                    else:
                        labels_idx = combined_labels.long()
                    correct_class += pred_class.eq(labels_idx).sum().item()
            else:
                # Track classification accuracy on BOTH source and target separately
                pred_class_s = class_output_s.argmax(dim=1)
                pred_class_t = class_output_t.argmax(dim=1)
                
                if self.multilabel:
                    # For multilabel, compare against primary class (argmax of target)
                    if source_labels.dim() == 2:
                        source_labels_primary = source_labels.argmax(dim=1)
                    else:
                        source_labels_primary = source_labels.long()
                    if target_labels_split.dim() == 2:
                        source_labels_primary = source_labels_split.argmax(dim=1)
                    else:
                        source_labels_primary = source_labels_split.long()
                    if target_labels_split.dim() == 2:
                        target_labels_primary = target_labels_split.argmax(dim=1)
                    else:
                        target_labels_primary = target_labels_split.long()
                    correct_class += pred_class_s.eq(source_labels_primary).sum().item()
                    correct_class += pred_class_t.eq(target_labels_primary).sum().item()
                else:
                    correct_class += pred_class_s.eq(source_labels_idx).sum().item()
                    correct_class += pred_class_t.eq(target_labels_idx).sum().item()
            
            pred_domain = domain_output.argmax(dim=1)
            correct_domain += pred_domain.eq(domain_labels).sum().item()
            
            total_samples += batch_size_s + batch_size_t
            
            pbar.set_postfix({
                'cls_loss': total_class_loss / (batch_idx + 1),
                'dom_loss': total_domain_loss / (batch_idx + 1),
                'cls_acc': 100. * correct_class / max(len(self.source_train_dataset) + len(self.target_train_dataset), 1),
                'dom_acc': 100. * correct_domain / total_samples,
                'lambda': lambda_domain
            })
        
        class_acc = 100. * correct_class / (len(self.source_train_dataset) + len(self.target_train_dataset))
        domain_acc = 100. * correct_domain / total_samples
        
        return (total_class_loss / num_batches, 
                total_domain_loss / num_batches,
                class_acc, domain_acc)
    
    def validate(self, val_loader, dataset_name):
        if val_loader is None:
            return 0.0
        
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.predict(data)
                
                # DEBUG: Check if outputs are reasonable
                if total == 0:
                    print(f"  DEBUG [{dataset_name}] First batch output stats: min={output.min().item():.3f}, max={output.max().item():.3f}, mean={output.mean().item():.3f}")
                    print(f"  DEBUG [{dataset_name}] First prediction: {output[0].cpu().numpy()}")
                    print(f"  DEBUG [{dataset_name}] First target: {target[0].cpu().numpy() if target.dim() == 2 else target[0].item()}")
                
                if self.multilabel:
                    # For multilabel, evaluate on primary class (argmax of one-hot)
                    # This allows tracking progress even in multilabel mode
                    pred = output.argmax(dim=1)
                    if target.dim() == 2:
                        target_labels = target.argmax(dim=1)
                    else:
                        target_labels = target.long()
                    correct += pred.eq(target_labels).sum().item()
                    total += target.size(0)
                else:
                    if target.dim() == 2:
                        target_labels = target.argmax(dim=1)
                    else:
                        target_labels = target.long()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target_labels).sum().item()
                    total += target.size(0)
        
        acc = 100. * correct / total if total > 0 else 0.0
        return acc
    
    def train(self):
        print("\nStarting domain adaptation training...")
        
        history = {
            'train_class_loss': [],
            'train_domain_loss': [],
            'train_class_acc': [],
            'train_domain_acc': [],
            'source_val_acc': [],
            'target_val_acc': []
        }
        
        best_target_acc = -1.0
        
        for epoch in range(self.epochs):
            train_class_loss, train_domain_loss, train_class_acc, train_domain_acc = self.train_epoch(epoch)
            
            lambda_domain = self.get_lambda_domain(epoch)
            
            if lambda_domain == 0.0:
                # For lambda=0, validate on merged set like finetuning
                merged_val_acc = self.validate(self.merged_val_loader, "Merged") if self.merged_val_loader else 0.0
                source_val_acc = 0.0
                target_val_acc = 0.0
                
                self.scheduler.step()
                
                history['train_class_loss'].append(train_class_loss)
                history['train_domain_loss'].append(train_domain_loss)
                history['train_class_acc'].append(train_class_acc)
                history['train_domain_acc'].append(train_domain_acc)
                history['source_val_acc'].append(source_val_acc)
                history['target_val_acc'].append(target_val_acc)
                
                print(f"Epoch {epoch+1}/{self.epochs}:")
                print(f"  Train - Class Loss: {train_class_loss:.4f}, Class Acc: {train_class_acc:.2f}%")
                if merged_val_acc > 0:
                    print(f"  Val - Merged Acc: {merged_val_acc:.2f}%")
                
                if merged_val_acc > best_target_acc:
                    best_target_acc = merged_val_acc
                    self.save_model('dann_best.pt')
                    print(f"  ✓ Saved best model (merged val acc: {merged_val_acc:.2f}%)")
            else:
                # Standard DANN validation on separate sets
                source_val_acc = self.validate(self.source_val_loader, "Source")
                target_val_acc = self.validate(self.target_val_loader, "Target")
                
                self.scheduler.step()
                
                history['train_class_loss'].append(train_class_loss)
                history['train_domain_loss'].append(train_domain_loss)
                history['train_class_acc'].append(train_class_acc)
                history['train_domain_acc'].append(train_domain_acc)
                history['source_val_acc'].append(source_val_acc)
                history['target_val_acc'].append(target_val_acc)
                
                print(f"Epoch {epoch+1}/{self.epochs}:")
                print(f"  Train - Class Loss: {train_class_loss:.4f}, Class Acc: {train_class_acc:.2f}%")
                print(f"  Train - Domain Loss: {train_domain_loss:.4f}, Domain Acc: {train_domain_acc:.2f}% (want ~50% = confused)")
                if source_val_acc > 0 or target_val_acc > 0:
                    print(f"  Val - Source Acc: {source_val_acc:.2f}%, Target Acc: {target_val_acc:.2f}%")
                
                if target_val_acc > best_target_acc:
                    best_target_acc = target_val_acc
                    self.save_model('dann_best.pt')
                    print(f"  ✓ Saved best model (target acc: {target_val_acc:.2f}%)")
        
        self.save_model('dann_final.pt')
        
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.plot_training_curves(history)
        
        print(f"\n{'='*60}")
        print(f"DOMAIN ADAPTATION RESULTS")
        print(f"{'='*60}")
        print(f"Best Target Validation Accuracy: {best_target_acc:.2f}%")
        print(f"Final Domain Confusion: {history['train_domain_acc'][-1]:.2f}%")
        print(f"  (50% = perfect confusion, 100% = no adaptation)")
        print(f"{'='*60}")
        print(f"\nModels saved to: {self.output_folder}")
        
        if self.test_folder or self.test_folder2:
            print(f"\nEvaluating on test sets using predict.py...")
            
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            best_model_path = os.path.join(self.output_folder, 'dann_best.pt')
            best_config_path = os.path.join(self.output_folder, 'dann_best_config.json')
            
            test_results = {}
            
            if self.test_folder:
                test_name = f"{Path(self.test_folder).parent.name}/{Path(self.test_folder).name}"
                output_csv = os.path.join(self.output_folder, f'predictions_{test_name.replace("/", "_")}.csv')
                
                print(f"\nTest Set 1: {test_name}")
                
                from predict import ModelPredictor
                predictor = ModelPredictor(
                    best_model_path,
                    best_config_path,
                    self.test_folder,
                    output_csv,
                    batch_size=self.batch_size,
                    device=self.device
                )
                predictor.run()
                
                test_acc = self.compute_accuracy_from_csv(output_csv, self.test_folder)
                test_results[test_name] = test_acc
                print(f"  {test_name} Accuracy: {test_acc:.2f}%")
            
            if self.test_folder2:
                test_name2 = f"{Path(self.test_folder2).parent.name}/{Path(self.test_folder2).name}"
                output_csv2 = os.path.join(self.output_folder, f'predictions_{test_name2.replace("/", "_")}.csv')
                
                print(f"\nTest Set 2: {test_name2}")
                
                from predict import ModelPredictor
                predictor2 = ModelPredictor(
                    best_model_path,
                    best_config_path,
                    self.test_folder2,
                    output_csv2,
                    batch_size=self.batch_size,
                    device=self.device
                )
                predictor2.run()
                
                test_acc2 = self.compute_accuracy_from_csv(output_csv2, self.test_folder2)
                test_results[test_name2] = test_acc2
                print(f"  {test_name2} Accuracy: {test_acc2:.2f}%")
            
            print(f"\n{'='*60}")
            print(f"TEST SET COMPARISON")
            print(f"{'='*60}")
            for name, accuracy in test_results.items():
                print(f"  {name:30s} Accuracy: {accuracy:.2f}%")
            print(f"{'='*60}")
    
    def compute_accuracy_from_csv(self, csv_path, test_folder):
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
    
    def plot_training_curves(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(history['train_class_loss'], label='Class Loss')
        axes[0, 0].plot(history['train_domain_loss'], label='Domain Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(history['train_domain_acc'], label='Domain Acc')
        axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Target (50%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Domain Classifier Accuracy (want ~50%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        if history['source_val_acc'][0] > 0:
            axes[1, 0].plot(history['source_val_acc'], label='Source Val')
            axes[1, 0].plot(history['target_val_acc'], label='Target Val')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy (%)')
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        axes[1, 1].plot(history['train_class_acc'], label='Train Class Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Training Classification Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'), dpi=150)
        plt.close()
    
    def save_model(self, filename):
        model_path = os.path.join(self.output_folder, filename)
        torch.save(self.model.state_dict(), model_path)
        
        config_path = model_path.replace('.pt', '_config.json')
        config_dict = {
            'model_type': 'dann',
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'class_names': self.categories,
            'multilabel': self.multilabel,
            'source_folder': self.source_folder,
            'target_folder': self.target_folder,
            'normalize': self.normalize,
            'remove_baseline': self.remove_baseline,
            'freq_bins': config.DEFAULT_FREQ_BINS,
            'time_bins': config.DEFAULT_TIME_BINS
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Adversarial Neural Network for cross-dataset generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic DANN training
  python train_domain_adaptation.py data/source data/target outputs/dann
  
  # With BirdClef pretrained backbone (recommended)
  python train_domain_adaptation.py data/source data/target outputs/dann \\
      --pretrained BirdClefModels/model_fold0.pth --architecture regnety_008
  
  # Progressive lambda schedule (start weak, increase adaptation)
  python train_domain_adaptation.py data/source data/target outputs/dann \\
      --lambda-domain 1.0 --lambda-schedule progressive --epochs 40
  
  # With normalization (if datasets have different recording equipment)
  python train_domain_adaptation.py data/source data/target outputs/dann \\
      --normalize --epochs 30

How it works:
  1. Trains on SOURCE data with labels (supervised classification)
  2. Also processes TARGET data (unlabeled for domain confusion)
  3. Gradient reversal makes features domain-invariant
  4. Result: classifier works on both datasets

Interpretation:
  - Domain accuracy dropping toward 50% = good (can't tell datasets apart)
  - Target validation accuracy = real measure of generalization
  - Compare to baseline (train on source only)
        """
    )
    
    parser.add_argument('source_folder', type=str,
                       help="Path to source dataset (with labels)")
    parser.add_argument('target_folder', type=str,
                       help="Path to target dataset (with labels for validation)")
    parser.add_argument('output_folder', type=str,
                       help="Path to output folder")
    parser.add_argument('--architecture', default='resnet18',
                       choices=['resnet18', 'resnet34', 'regnety_008', 'efficientnet_b0'],
                       help="Model architecture (default: resnet18)")
    parser.add_argument('--pretrained', default=None,
                       help="Path to pretrained checkpoint (optional)")
    parser.add_argument('--epochs', type=int, default=30,
                       help="Number of epochs (default: 30)")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument('--lambda-domain', type=float, default=1.0,
                       help="Weight for domain adversarial loss (default: 1.0)")
    parser.add_argument('--lambda-schedule', default='fixed',
                       choices=['fixed', 'progressive'],
                       help="Lambda schedule: fixed or progressive (default: fixed)")
    parser.add_argument('--multilabel', action='store_true',
                       help="Use multi-label classification")
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization")
    parser.add_argument('--baseline-removal', action='store_true',
                       help="Enable baseline removal")
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help="Validation split (default: 0.2)")
    parser.add_argument('--test-folder', type=str, default=None,
                       help="Path to test data folder 1 (with labels.json). Evaluated AFTER training completes.")
    parser.add_argument('--test-folder2', type=str, default=None,
                       help="Path to test data folder 2 (with labels.json). Evaluated AFTER training completes.")
    parser.add_argument('--device', default=None,
                       help="Device (cuda/cpu, default: auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.source_folder, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.source_folder}")
        return
    
    if not os.path.exists(os.path.join(args.target_folder, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.target_folder}")
        return
    
    trainer = DomainAdaptationTrainer(
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        output_folder=args.output_folder,
        architecture=args.architecture,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_domain=args.lambda_domain,
        lambda_schedule=args.lambda_schedule,
        multilabel=args.multilabel,
        normalize=args.normalize,
        validation_split=args.validation_split,
        remove_baseline=args.baseline_removal,
        test_folder=args.test_folder,
        test_folder2=args.test_folder2,
        device=torch.device(args.device) if args.device else None
    )
    
    trainer.load_data()
    trainer.create_model()
    trainer.train()


if __name__ == '__main__':
    main()
