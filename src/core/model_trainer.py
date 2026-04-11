
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import math
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from src.data.data_utils import DataLoader, create_data_loaders, SpectrogramDataset
from src.core.models import AST, RegNetModel
from src.evaluation.evaluation_utils import EvaluationManager
from src.core.trainer_config import TrainerConfig, TrainingConfig, ModelConfig
from src.core import config


def compute_multilabel_f1(all_preds, all_targets):
    """Compute macro F1 score for multi-label predictions."""
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    _, _, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    return f1


def compute_multilabel_f1_scores(all_preds, all_targets):
    """Compute macro and micro F1 scores for multi-label predictions."""
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0
    )
    return f1_macro, f1_micro


def compute_multilabel_epoch_metrics(all_preds, all_targets):
    """Compute macro/micro F1, bit accuracy, and exact match for multilabel outputs."""
    if len(all_preds) == 0:
        return {
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'bit_acc': 0.0,
            'exact_match': 0.0,
        }

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    preds = (preds >= 0.5).astype(np.int32)
    targets = (targets >= 0.5).astype(np.int32)

    _, _, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        targets, preds, average='micro', zero_division=0
    )

    preds_bool = preds.astype(bool)
    targets_bool = targets.astype(bool)
    bit_acc = (preds_bool == targets_bool).mean().item() if hasattr((preds_bool == targets_bool).mean(), 'item') else float((preds_bool == targets_bool).mean())
    exact_match = (preds_bool == targets_bool).all(axis=1).mean().item() if hasattr((preds_bool == targets_bool).all(axis=1).mean(), 'item') else float((preds_bool == targets_bool).all(axis=1).mean())

    return {
        'macro_f1': float(f1_macro),
        'micro_f1': float(f1_micro),
        'bit_acc': float(bit_acc),
        'exact_match': float(exact_match),
    }


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard negatives.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples.
        reduction: 'mean' or 'sum' (default: 'mean')
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (N, C) where C is number of classes
            targets: Class indices (N,) or one-hot encoded (N, C)
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        
        # Convert to probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Applies focal loss independently to each label using BCE with logits.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MultilabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (N, C)
            targets: Binary labels (N, C)
        """
        # Support soft targets (e.g., mixup) by using the continuous formulation:
        # p_t = p*target + (1-p)*(1-target)
        # alpha_t = alpha*target + (1-alpha)*(1-target)
        targets = targets.to(dtype=inputs.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        p = torch.sigmoid(inputs)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        focal_term = (1.0 - p_t) ** self.gamma

        alpha = torch.as_tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SmoothBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogits with target smoothing for multi-label classification.
    Moves targets toward 0.5 by epsilon to reduce overconfidence.

    Args:
        epsilon: Smoothing factor in [0, 0.5] (default: 0.05)
        pos_weight: Optional tensor for positive class weighting (class imbalance)
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, epsilon=0.05, pos_weight=None, reduction='mean'):
        super(SmoothBCEWithLogitsLoss, self).__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.epsilon > 0:
            targets_smooth = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        else:
            targets_smooth = targets
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth, pos_weight=self.pos_weight, reduction='none'
        )
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class Trainer:
    """Unified trainer for both AST and RegNetY models."""
    
    def __init__(self, cfg: TrainerConfig):
        """Initialize trainer with structured config."""
        self.cfg = cfg
        self.model_type = getattr(cfg.model, 'model_type', 'ast')
        
        # Unpack commonly used config values
        self.data_folder = cfg.training.data_folder
        self.output_folder = cfg.training.output_folder
        self.max_epochs = cfg.training.max_epochs
        self.batch_size = cfg.training.batch_size
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        self.patience = cfg.training.patience
        self.use_amp = cfg.training.use_amp
        self.seed = cfg.training.seed
        
        self.multilabel = cfg.model.multilabel
        self.dropout = cfg.model.dropout
        self.use_multiscale = cfg.model.use_multiscale
        self.use_sparse_patches = cfg.model.use_sparse_patches
        self.num_sparse_patches = cfg.model.num_sparse_patches
        self.use_reconstruction = cfg.model.use_reconstruction
        self.recon_weight = cfg.model.recon_weight
        self.use_adapters = cfg.model.use_adapters
        self.freeze_layers = cfg.model.freeze_layers
        self.pretrained_path = cfg.model.pretrained_path
        self.freeze_backbone = cfg.model.freeze_backbone
        self.freeze_stages = cfg.model.freeze_stages
        self.model_name = getattr(cfg.model, 'model_name', 'regnety_008')
        
        self.mixup_alpha = cfg.augmentation.mixup_alpha
        self.noise_ratio = cfg.augmentation.noise_ratio
        self.noise_folder = cfg.augmentation.noise_folder
        self.noise_as_samples = cfg.augmentation.noise_as_samples
        self.max_noise_samples = cfg.augmentation.max_noise_samples
        self.use_temporal_roll = cfg.augmentation.use_temporal_roll
        self.normalize = cfg.augmentation.normalize
        self.per_chunk_norm = cfg.augmentation.per_chunk_norm
        
        self.use_focal_loss = cfg.loss.use_focal_loss
        self.use_class_weights = cfg.loss.use_class_weights
        self.pos_weight_cap = cfg.loss.pos_weight_cap
        self.bce_smoothing = cfg.loss.bce_smoothing
        
        self.use_dann = cfg.domain_adaptation.use_dann
        self.target_folder = cfg.domain_adaptation.target_folder
        self.lambda_domain = cfg.domain_adaptation.lambda_domain
        self.use_cleaner = cfg.domain_adaptation.use_cleaner
        
        self.test_folder = cfg.evaluation.test_folder
        self.test_folder2 = cfg.evaluation.test_folder2
        
        self.trial = cfg.trial
        
        # Set random seed
        if self.seed is not None:
            import random
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            print(f"Random seed set to: {self.seed}")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.patience > 0:
            print(f"Early stopping patience: {self.patience} epochs")
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP)")
        
        # Initialize gradient scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Load data
        data_loader = DataLoader(self.data_folder, noise_folder=self.noise_folder)
        self.data = data_loader.load_data(self.multilabel, validation_share=0.2)
        self.num_classes = self.data['nclasses']

        # Optionally include noise spectrograms as explicit all-zero training samples.
        # This is useful for soundscape-style inference even if you evaluate with --birds-only,
        # because it improves calibration and reduces false positives on weak/ambiguous bird activity.
        if self.noise_as_samples and self.data.get('train_noise_filenames'):
            noise_files = list(self.data['train_noise_filenames'])
            if self.max_noise_samples is not None:
                noise_files = noise_files[:int(self.max_noise_samples)]
            if noise_files:
                zeros = np.zeros((len(noise_files), self.num_classes), dtype=np.float32)
                self.data['train_filenames'] = list(self.data['train_filenames']) + noise_files
                self.data['train_labels'] = np.vstack([np.array(self.data['train_labels'], dtype=np.float32), zeros])
                print(f"Added {len(noise_files)} noise samples as all-zero training examples")

        # Load target domain data for DANN
        if self.use_dann:
            if not self.target_folder:
                raise ValueError("use_dann=True requires target_folder")
            print(f"Loading target domain data from {self.target_folder} for DANN...")
            target_loader = DataLoader(self.target_folder, noise_folder=None)
            self.target_data = target_loader.load_data(multilabel, validation_share=0.2)
            print(f"Loaded {len(self.target_data['train_filenames'])} target domain samples")

        # Use dimensions from spectrogram params (single source of truth)
        # Avoid duplicating DEFAULT_FREQ_BINS vs SPECTROGRAM_PARAMS['nfilters']
        self.img_height = cfg.model.freq_bins if cfg.model.freq_bins is not None else config.SPECTROGRAM_PARAMS['nfilters']  # Frequency bins (height)
        self.img_width = cfg.model.time_bins if cfg.model.time_bins is not None else config.DEFAULT_TIME_BINS   # Time bins (width)
        
        # Create data loaders with config defaults
        # Use more workers and prefetch for faster GPU utilization
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader, self.val_loader = create_data_loaders(
            self.data, batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_ratio=self.noise_ratio, 
            spec_transform=None,  # Uses config.DEFAULT_SPEC_TRANSFORM
            num_workers=num_workers, width_downsizing=None, mixup_alpha=mixup_alpha,
            use_class_balancing=False, normalize=self.normalize,
            use_sparse_patches=self.use_sparse_patches, num_sparse_patches=self.num_sparse_patches,
            use_temporal_roll=self.use_temporal_roll
        )
        
        # Create target domain data loader for DANN
        if self.use_dann:
            self.target_train_loader, _ = create_data_loaders(
                self.target_data, batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                cropping_mode='random', noise_ratio=0.0,  # No noise augmentation for target
                spec_transform=None,
                num_workers=num_workers, width_downsizing=None, mixup_alpha=0.0,  # No mixup for target
                use_class_balancing=False, normalize=self.normalize,
                use_sparse_patches=self.use_sparse_patches, num_sparse_patches=self.num_sparse_patches,
                use_temporal_roll=self.use_temporal_roll
            )
            print(f"Created target domain data loader with {len(self.target_train_loader)} batches")
        
        os.makedirs(output_folder, exist_ok=True)
    
    def _compute_class_weights(self):
        """Compute inverse frequency weights for each class (for multilabel BCE loss)."""
        train_labels = np.array(self.data['train_labels'])
        
        class_counts = train_labels.sum(axis=0)
        total_samples = len(train_labels)
        
        pos_counts = class_counts
        neg_counts = total_samples - class_counts
        
        pos_weight = neg_counts / (pos_counts + 1e-5)
        
        pos_weight = np.clip(pos_weight, 1.0, float(self.pos_weight_cap))
        
        pos_weight = torch.from_numpy(pos_weight).float().to(self.device)
        
        print(f"  Class weights - min: {pos_weight.min().item():.2f}, max: {pos_weight.max().item():.2f}, mean: {pos_weight.mean().item():.2f}")
        print(f"  Rare classes (weight={float(self.pos_weight_cap):.0f}): {(pos_weight == float(self.pos_weight_cap)).sum().item()}/{len(pos_weight)}")
        
        return pos_weight
    
    def train(self):
        """Train model (AST or RegNetY)."""
        input_size = (self.img_height, self.img_width)
        print(f"Model input size: {input_size}")
        
        # Create model based on type
        if self.model_type == 'regnet':
            print(f"Creating RegNetY model ({self.model_name})...")
            model = RegNetModel(
                self.num_classes,
                pretrained_path=self.pretrained_path,
                model_name=self.model_name,
                freeze_backbone=self.freeze_backbone,
                freeze_stages=self.freeze_stages
            ).to(self.device)
        else:
            if self.use_multiscale:
                print("Creating Multi-Scale AST model...")
                from models import MultiScaleAST
                model = MultiScaleAST(self.num_classes, self.multilabel, input_size=input_size, dropout=self.dropout, 
                                     use_reconstruction=self.use_reconstruction).to(self.device)
            else:
                print("Creating AST model...")
                model = AST(self.num_classes, self.multilabel, input_size=input_size, dropout=self.dropout, 
                           use_reconstruction=self.use_reconstruction, use_adapters=self.use_adapters,
                           per_chunk_norm=self.per_chunk_norm).to(self.device)
            
            # AST-specific: Interpolate positional embeddings
            if not self.use_multiscale:
                print(f"Interpolating positional embeddings for input size {input_size}...")
                model.interpolate_pos_embed(input_size)
            
            # AST-specific: Load pretrained weights if provided
            if self.pretrained_path:
                print(f"Loading pretrained weights from {self.pretrained_path}")
                self._load_pretrained_weights(model, self.pretrained_path)
            
            # AST-specific: Freeze early transformer layers if requested
            if self.freeze_layers is not None and self.freeze_layers > 0:
                print(f"Freezing first {self.freeze_layers} transformer layers...")
                for i in range(self.freeze_layers):
                    for param in model.ast.encoder.layer[i].parameters():
                        param.requires_grad = False
                
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        # Add DANN components if requested
        if self.use_dann:
            print("  Adding DANN domain discriminator for AST...")
            from models import GradientReversalLayer
            
            # AST uses 768-dim embeddings
            feat_dim = 768
            
            self.grl = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.grl.to(self.device)
            self.domain_classifier.to(self.device)
            self.domain_criterion = nn.BCEWithLogitsLoss()
            print(f"  DANN lambda_domain: {self.lambda_domain}")
        
        # Add spectrogram cleaner if requested
        if self.use_cleaner:
            print("  Adding trainable spectrogram cleaner...")
            from models import SpectrogramCleaner
            
            self.cleaner = SpectrogramCleaner(input_size=input_size).to(self.device)
            
            for param in model.parameters():
                param.requires_grad = False
            
            print(f"  Backbone frozen - training only spectrogram cleaner + classifier")
            
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.cleaner.parameters()) + sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in self.cleaner.parameters())
            print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        # Optimizer and LR schedule
        # Use AdamW (Adam with decoupled weight decay) for better regularization
        param_groups = [{'params': model.parameters(), 'lr': self.learning_rate}]
        if self.use_dann:
            param_groups.append({'params': self.domain_classifier.parameters(), 'lr': self.learning_rate * 0.1})
        if self.use_cleaner:
            param_groups.append({'params': self.cleaner.parameters(), 'lr': self.learning_rate})
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        
        def lr_lambda(epoch):
            if epoch < 5:
                return 1.0
            return 0.85 ** (epoch - 5)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"Using AdamW optimizer with Lambda LR scheduler")
        
        # Use logits from the model and appropriate losses
        # Stronger label smoothing for better generalization
        if self.use_focal_loss:
            if self.multilabel:
                criterion = MultilabelFocalLoss(alpha=0.25, gamma=2.0)
                print("Using Multilabel Focal Loss (alpha=0.25, gamma=2.0)")
            else:
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                print("Using Focal Loss (alpha=0.25, gamma=2.0)")
        elif self.multilabel:
            pos_weight = None
            if self.use_class_weights:
                pos_weight = self._compute_class_weights()
                print(f"Using class-weighted BCE loss")
                print(f"  Weight range: {pos_weight.min().item():.2f} - {pos_weight.max().item():.2f}")
            if self.bce_smoothing and self.bce_smoothing > 0.0:
                criterion = SmoothBCEWithLogitsLoss(epsilon=self.bce_smoothing, pos_weight=pos_weight)
                print(f"Applying BCE target smoothing (epsilon={self.bce_smoothing:.3f})")
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        print(f"Starting training for {self.max_epochs} epochs...")
        
        best_val_acc = 0.0
        best_epoch = -1
        epochs_without_improvement = 0
        
        # Divergence detection
        initial_loss = None
        divergence_threshold = 10.0  # Stop if loss > initial_loss * threshold
        
        # Weight averaging: collect model states from later epochs
        model_states_for_averaging = []
        averaging_start_epoch = max(0, self.max_epochs - 20)  # Average last 20 epochs
        
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            # Check model weights BEFORE starting epoch
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: Model weights are NaN/Inf at START of epoch {epoch+1}!")
                print(f"   ABORTING TRAINING.\n")
                return
            
            # Train
            model.train()
            if self.use_dann:
                self.grl.train()
                self.domain_classifier.train()
            if self.use_cleaner:
                self.cleaner.train()
                
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # For multi-label F1 calculation
            all_train_preds = []
            all_train_targets = []
            
            # DANN: set up alternating batch iteration
            if self.use_dann:
                # Schedule learning rate for domain adaptation
                p = float(epoch) / float(self.max_epochs)
                alpha = 2.0 / (1.0 + np.exp(-5 * p)) - 1.0
                self.grl.lambda_param = alpha * self.lambda_domain
                
                source_iter = iter(self.train_loader)
                target_iter = iter(self.target_train_loader)
                n_batches = min(len(self.train_loader), len(self.target_train_loader))
                
                domain_correct = 0
                domain_total = 0
                total_class_loss = 0.0
                total_domain_loss = 0.0
            
            batch_iterator = range(n_batches) if self.use_dann else enumerate(self.train_loader)
            
            for batch_idx in batch_iterator:
                if self.use_dann:
                    # Get source and target batches
                    try:
                        source_batch = next(source_iter)
                    except StopIteration:
                        source_iter = iter(self.train_loader)
                        source_batch = next(source_iter)
                    
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(self.target_train_loader)
                        target_batch = next(target_iter)
                    
                    # Extract data from batches (handle sparse mode if needed)
                    if self.use_sparse_patches:
                        source_data = source_batch['patches'].to(self.device)
                        source_target = source_batch['label'].to(self.device)
                        source_positions = source_batch['positions'].to(self.device)
                        source_mask = source_batch['mask'].to(self.device)
                        
                        target_data = target_batch['patches'].to(self.device)
                        target_positions = target_batch['positions'].to(self.device)
                        target_mask = target_batch['mask'].to(self.device)
                    else:
                        source_data, source_target = source_batch
                        source_data = source_data.to(self.device, non_blocking=True)
                        source_target = source_target.to(self.device, non_blocking=True)
                        
                        target_data, _ = target_batch
                        target_data = target_data.to(self.device, non_blocking=True)
                    
                    batch_size_s = source_data.size(0)
                    batch_size_t = target_data.size(0)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass for source (for classification)
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        if self.use_sparse_patches:
                            source_output = model(source_data, sparse_mode=True, positions=source_positions, mask=source_mask)
                            source_features = model.get_features(source_data, sparse_mode=True, positions=source_positions, mask=source_mask)
                            target_features = model.get_features(target_data, sparse_mode=True, positions=target_positions, mask=target_mask)
                        else:
                            source_output = model(source_data)
                            source_features = model.get_features(source_data)
                            target_features = model.get_features(target_data)
                        
                        # Classification loss (only on source domain with labels)
                        if self.multilabel:
                            source_output = torch.clamp(source_output, min=-80.0, max=80.0)
                            class_loss = criterion(source_output, source_target)
                        else:
                            if source_target.dim() == 2 and not torch.equal(source_target, source_target.round()):
                                log_probs = F.log_softmax(source_output, dim=1)
                                class_loss = -(source_target * log_probs).sum(dim=1).mean()
                            else:
                                target_idx = source_target.argmax(dim=1)
                                class_loss = criterion(source_output, target_idx)
                        
                        # Domain adaptation loss
                        combined_features = torch.cat([source_features, target_features], dim=0)
                        norm_features = F.normalize(combined_features, p=2, dim=1)
                        reversed_features = self.grl(norm_features)
                        domain_output = self.domain_classifier(reversed_features)
                        
                        domain_labels_source = torch.zeros(batch_size_s).to(self.device)
                        domain_labels_target = torch.ones(batch_size_t).to(self.device)
                        domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
                        
                        domain_loss = self.domain_criterion(domain_output.squeeze(), domain_labels)
                        
                        # Combined loss
                        loss = class_loss + self.grl.lambda_param * domain_loss
                        
                        # Domain accuracy tracking
                        domain_pred = (torch.sigmoid(domain_output.squeeze()) > 0.5).float()
                        domain_correct += (domain_pred == domain_labels).sum().item()
                        domain_total += domain_labels.size(0)
                    
                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    train_loss += loss.item()
                    total_class_loss += class_loss.item()
                    total_domain_loss += domain_loss.item()
                    
                    # Accuracy tracking (only on source labeled data)
                    with torch.no_grad():
                        if self.multilabel:
                            preds = (torch.sigmoid(source_output) > 0.5).float()
                            all_train_preds.append(preds.cpu().numpy())
                            all_train_targets.append(source_target.cpu().numpy())
                        else:
                            pred = source_output.argmax(dim=1)
                            target_idx = source_target.argmax(dim=1) if source_target.dim() == 2 else source_target
                            train_correct += pred.eq(target_idx).sum().item()
                            train_total += source_target.size(0)
                
                else:
                    # Standard training (no DANN)
                    batch_idx, batch = batch_idx
                    
                    # Handle both sparse and standard data formats
                    if self.use_sparse_patches:
                        # Sparse mode: batch is a dict
                        patches = batch['patches'].to(self.device)  # (B, K, 1, 16, 16)
                        positions = batch['positions'].to(self.device)  # (B, K, 2)
                        mask = batch['mask'].to(self.device)  # (B, K)
                        target = batch['label'].to(self.device)  # (B, num_classes)
                        
                        # Apply mixup at embedding level if enabled
                        if self.mixup_alpha > 0:
                            # Generate mixup lambda
                            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                            batch_size = patches.size(0)
                            index = torch.randperm(batch_size).to(self.device)
                            
                            # Mix patches, positions, masks, and targets
                            mixed_patches = lam * patches + (1 - lam) * patches[index]
                            mixed_target = lam * target + (1 - lam) * target[index]
                            
                            # For positions and masks, use the primary sample's (no mixing makes sense here)
                            patches = mixed_patches
                            target = mixed_target
                        
                        optimizer.zero_grad()
                        
                        # Forward with sparse patches
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            if self.use_reconstruction:
                                output, recon = model(patches, sparse_mode=True, positions=positions, mask=mask)
                            else:
                                output = model(patches, sparse_mode=True, positions=positions, mask=mask)
                    else:
                        # Standard mode: batch is (data, target) tuple
                        data, target = batch
                        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                        
                        optimizer.zero_grad()
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            if self.use_cleaner:
                                data = self.cleaner(data)
                            
                            if self.use_reconstruction:
                                output, recon = model(data)
                            else:
                                output = model(data)
                    
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        if self.multilabel:
                            # Clamp logits to prevent numerical overflow in BCE loss
                            # Logits beyond ±80 cause sigmoid(±80) ≈ 0/1 which creates log(0) = -inf
                            output = torch.clamp(output, min=-80.0, max=80.0)
                            loss = criterion(output, target)
                        else:
                            # Detect soft (mixup) labels: if any row has fractional values (not strictly 0/1)
                            if target.dim() == 2 and not torch.equal(target, target.round()):
                                # Proper soft-label cross-entropy (KL to one-hot mixing)
                                log_probs = F.log_softmax(output, dim=1)
                                loss = -(target * log_probs).sum(dim=1).mean()
                            else:
                                # Hard labels path (one-hot vectors)
                                target_idx = target.argmax(dim=1)
                                loss = criterion(output, target_idx)
                        
                        if self.use_reconstruction:
                            target_spec = data.squeeze(1) if data.dim() == 4 else data
                            target_spec = (target_spec - config.AST_MEAN) / config.AST_STD
                            recon_loss = F.mse_loss(recon, target_spec)
                            loss = loss + self.recon_weight * recon_loss
                    
                    # Check for NaN loss BEFORE backward pass
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n❌ CRITICAL: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                        print(f"   DEBUG - Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
                        print(f"   DEBUG - Target stats: min={target.min():.4f}, max={target.max():.4f}, sum={target.sum():.4f}")
                        if torch.isnan(output).any():
                            print(f"   ⚠️  MODEL OUTPUT CONTAINS NaN! Model weights are corrupted.")
                        if torch.isinf(output).any():
                            print(f"   ⚠️  MODEL OUTPUT CONTAINS Inf! Model exploded.")
                        print(f"   Stopping epoch early...")
                        break  # Stop epoch, not just continue
                    
                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        # Check for NaN gradients
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                            print(f"   Stopping epoch early...")
                            self.scaler.update()  # Update scaler to reset state
                            optimizer.zero_grad()  # Clear bad gradients
                            break
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                            print(f"   Stopping epoch early...")
                            optimizer.zero_grad()  # Clear bad gradients  
                            break
                        optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # For metrics, convert soft (mixup) targets back to hard labels
                    # by rounding to nearest integer (0 or 1)
                    target_hard = target.round()
                    
                    if self.multilabel:
                        pred = (torch.sigmoid(output) > 0.5).float()
                        all_train_preds.append(pred.cpu().numpy())
                        all_train_targets.append(target_hard.cpu().numpy())
                    else:
                        pred = output.argmax(dim=1)
                        target_labels = target_hard.argmax(dim=1)
                        train_correct += (pred == target_labels).sum().item()
                        train_total += target_hard.size(0)
            
            # Print progress for DANN training
            if self.use_dann and batch_idx % 10 == 0:
                domain_acc = 100.0 * domain_correct / max(1, domain_total)
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{n_batches}: '
                      f'cls_loss={total_class_loss/(batch_idx+1):.3f}, '
                      f'dom_loss={total_domain_loss/(batch_idx+1):.3f}, '
                      f'domain_acc={domain_acc:.1f}%, alpha={self.grl.lambda_param:.3f}')
            elif not self.use_dann and batch_idx % 10 == 0:
                avg_loss = train_loss / max(1, batch_idx)
                if avg_loss > 100.0:
                    print(f'⚠️  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f} (avg: {avg_loss:.4f} - HIGH!)')
                else:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validate
            model.eval()
            if self.use_cleaner:
                self.cleaner.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # For multi-label F1 calculation
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for batch in self.val_loader:
                    # Handle both sparse and standard data formats
                    if self.use_sparse_patches:
                        patches = batch['patches'].to(self.device)
                        positions = batch['positions'].to(self.device)
                        mask = batch['mask'].to(self.device)
                        target = batch['label'].to(self.device)
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            if self.use_reconstruction:
                                output, _ = model(patches, sparse_mode=True, positions=positions, mask=mask)
                            else:
                                output = model(patches, sparse_mode=True, positions=positions, mask=mask)
                    else:
                        data, target = batch
                        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                        
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            if self.use_cleaner:
                                data = self.cleaner(data)
                            
                            if self.use_reconstruction:
                                output, _ = model(data)
                            else:
                                output = model(data)
                    
                    if self.multilabel:
                        val_loss += criterion(output, target).item()
                        pred = (torch.sigmoid(output) > 0.5).float()
                        all_val_preds.append(pred.cpu().numpy())
                        all_val_targets.append(target.cpu().numpy())
                    else:
                        target_idx = target.argmax(dim=1)
                        val_loss += criterion(output, target_idx).item()
                        pred = output.argmax(dim=1)
                        target_labels = target.argmax(dim=1)
                        val_correct += (pred == target_labels).sum().item()
                        val_total += target.size(0)
            
            # Calculate averages
            train_loss /= max(1, len(self.train_loader))  # Avoid divide by zero if epoch stopped early
            val_loss /= len(self.val_loader)
            
            # Divergence detection
            if initial_loss is None:
                initial_loss = train_loss
            elif train_loss > initial_loss * divergence_threshold:
                print(f"\n❌ TRAINING DIVERGED: Loss increased from {initial_loss:.4f} to {train_loss:.4f}")
                print(f"   This usually means the learning rate is too high.")
                print(f"   Try reducing --lr (current: {self.learning_rate:.2e})")
                print(f"   Recommended: --lr {self.learning_rate/2:.2e} or --lr {self.learning_rate/5:.2e}")
                return
            
            if self.multilabel:
                train_metrics = compute_multilabel_epoch_metrics(all_train_preds, all_train_targets)
                val_metrics = compute_multilabel_epoch_metrics(all_val_preds, all_val_targets)
                train_acc = train_metrics['macro_f1']
                val_acc = val_metrics['macro_f1']
            else:
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # ABORT if model weights contain NaN
            if any(torch.isnan(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: NaN in weights (epoch {epoch+1}). ABORTING.\n")
                return

            scheduler.step()
            
            # Collect model state for weight averaging (after epoch 5 when LR starts decaying)
            if epoch >= averaging_start_epoch:
                model_states_for_averaging.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                self._save_model(model, best=True)
            else:
                epochs_without_improvement += 1
                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    print(f"\n  Early stopping: no improvement for {self.patience} epochs")
                    print(f"  Best {'macro-F1' if self.multilabel else 'val acc'}: {best_val_acc:.4f} at epoch {best_epoch}")
                    break
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s)')
            if self.multilabel:
                print(
                    f"Train Loss: {train_loss:.4f}, "
                    f"Macro F1: {train_metrics['macro_f1']:.4f}, "
                    f"Micro F1: {train_metrics['micro_f1']:.4f}, "
                    f"Bit Acc: {train_metrics['bit_acc']:.4f}, "
                    f"Exact: {train_metrics['exact_match']:.4f}"
                )
                print(
                    f"  Val Loss: {val_loss:.4f}, "
                    f"Macro F1: {val_metrics['macro_f1']:.4f}, "
                    f"Micro F1: {val_metrics['micro_f1']:.4f}, "
                    f"Bit Acc: {val_metrics['bit_acc']:.4f}, "
                    f"Exact: {val_metrics['exact_match']:.4f}"
                )
            else:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print('-' * 60)
            
            if self.trial is not None:
                import optuna
                # Report validation loss for pruning
                self.trial.report(val_loss, epoch)
                # Only check for pruning if not the last epoch
                if epoch < self.max_epochs - 1 and self.trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
        
        # Save final model (only if we actually trained)
        if self.max_epochs > 0:
            self._save_model(model)
            self._save_history(train_losses, val_losses, train_accs, val_accs)
            self._plot_history(train_losses, val_losses, train_accs, val_accs)
        
            # Apply weight averaging (AST paper technique for +1-2% accuracy boost)
            if len(model_states_for_averaging) > 0:
                print(f"\nApplying weight averaging over {len(model_states_for_averaging)} checkpoints...")
                averaged_state = {}
                for key in model_states_for_averaging[0].keys():
                    try:
                        stacked_params = torch.stack([state[key] for state in model_states_for_averaging])
                        if stacked_params.dtype in [torch.float32, torch.float16, torch.float64]:
                            averaged_state[key] = stacked_params.mean(dim=0)
                        else:
                            averaged_state[key] = model_states_for_averaging[-1][key]
                    except (RuntimeError, TypeError):
                        # Skip parameters that can't be stacked (e.g., different shapes across checkpoints)
                        averaged_state[key] = model_states_for_averaging[-1][key]
                model.load_state_dict(averaged_state)
                
                # Evaluate weight-averaged model
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss(label_smoothing=0.1)
                
                # For multi-label F1 calculation
                all_val_preds = []
                all_val_targets = []
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        if self.use_sparse_patches:
                            patches = batch['patches'].to(self.device)
                            positions = batch['positions'].to(self.device)
                            mask = batch['mask'].to(self.device)
                            target = batch['label'].to(self.device)
                            
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                if self.use_reconstruction:
                                    output, _ = model(patches, sparse_mode=True, positions=positions, mask=mask)
                                else:
                                    output = model(patches, sparse_mode=True, positions=positions, mask=mask)
                        else:
                            data, target = batch
                            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                            
                            with torch.amp.autocast('cuda', enabled=self.use_amp):
                                if self.use_reconstruction:
                                    output, _ = model(data)
                                else:
                                    output = model(data)
                        
                        if self.multilabel:
                            val_loss += criterion(output, target).item()
                            pred = (torch.sigmoid(output) > 0.5).float()
                            all_val_preds.append(pred.cpu().numpy())
                            all_val_targets.append(target.cpu().numpy())
                        else:
                            target_idx = target.argmax(dim=1)
                            val_loss += criterion(output, target_idx).item()
                        pred = output.argmax(dim=1)
                        val_correct += (pred == target_idx).sum().item()
                        val_total += target.size(0)
            
            if self.multilabel:
                avg_val_acc = compute_multilabel_f1(all_val_preds, all_val_targets)
                print(f"Weight-averaged model validation macro-F1: {avg_val_acc:.4f}")
            else:
                avg_val_acc = val_correct / val_total
                print(f"Weight-averaged model validation accuracy: {avg_val_acc:.4f}")
            
            # Save weight-averaged model if it's better
            if avg_val_acc > best_val_acc:
                if self.multilabel:
                    print(f"Weight averaging improved macro-F1: {best_val_acc:.4f} -> {avg_val_acc:.4f}")
                else:
                    print(f"Weight averaging improved accuracy: {best_val_acc:.4f} -> {avg_val_acc:.4f}")
                best_val_acc = avg_val_acc
                self._save_model(model, best=True)
        # Evaluate on validation set (using best checkpoint)
        best_path = os.path.join(self.output_folder, 'ast_model_best.pt')
        if os.path.exists(best_path):
            state_dict = torch.load(best_path, map_location=self.device)
            model.load_state_dict(state_dict)
        evaluator = EvaluationManager(self.output_folder, self.data['class_names'], self.multilabel)
        evaluator.evaluate_model(model, self.val_loader, 'ast_model', self.data, device=self.device)

        # Evaluate on test sets if provided (for DANN experiments)
        if self.test_folder:
            print(f"\n{'='*60}")
            print(f"Evaluating on test set 1: {self.test_folder}")
            print(f"{'='*60}")
            test_loader1 = DataLoader(self.test_folder, noise_folder=None)
            test_data1 = test_loader1.load_data(self.multilabel, validation_share=0.0)
            
            test_dataset1 = SpectrogramDataset(
                test_data1['train_filenames'], test_data1['train_labels'],
                self.img_height, self.img_width, config.DEFAULT_CHANNELS, 'center',
                noise_filenames=None,
                noise_ratio=0.0,
                spec_transform=None,
                training=False,
                width_downsizing=None,
                normalize=self.normalize,
                use_sparse_patches=self.use_sparse_patches,
                num_sparse_patches=self.num_sparse_patches,
                use_temporal_roll=False
            )
            
            test_loader_obj1 = torch.utils.data.DataLoader(
                test_dataset1,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Evaluate and save predictions
            test_name1 = Path(self.test_folder).parent.name
            evaluator.evaluate_model(model, test_loader_obj1, f'ast_test_{test_name1}', test_data1, device=self.device)
            
            # Save predictions to CSV
            self._save_test_predictions(model, test_loader_obj1, test_data1, test_name1)
        
        if self.test_folder2:
            print(f"\n{'='*60}")
            print(f"Evaluating on test set 2: {self.test_folder2}")
            print(f"{'='*60}")
            test_loader2 = DataLoader(self.test_folder2, noise_folder=None)
            test_data2 = test_loader2.load_data(self.multilabel, validation_share=0.0)
            
            test_dataset2 = SpectrogramDataset(
                test_data2['train_filenames'], test_data2['train_labels'],
                self.img_height, self.img_width, config.DEFAULT_CHANNELS, 'center',
                noise_filenames=None,
                noise_ratio=0.0,
                spec_transform=None,
                training=False,
                width_downsizing=None,
                normalize=self.normalize,
                use_sparse_patches=self.use_sparse_patches,
                num_sparse_patches=self.num_sparse_patches,
                use_temporal_roll=False
            )
            
            test_loader_obj2 = torch.utils.data.DataLoader(
                test_dataset2,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Evaluate and save predictions
            test_name2 = Path(self.test_folder2).parent.name
            evaluator.evaluate_model(model, test_loader_obj2, f'ast_test_{test_name2}', test_data2, device=self.device)
            
            # Save predictions to CSV
            self._save_test_predictions(model, test_loader_obj2, test_data2, test_name2)

        print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Done! Saved to {self.output_folder}")
        
        best_val_loss = min(val_losses)
        
        if self.trial is not None:
            # Return best validation loss for Optuna to minimize
            return best_val_loss
        
        return {
            'model': model,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
    
    def _load_pretrained_weights(self, model, pretrained_path):
        """
        Load pretrained weights into the model, handling different number of classes.
        
        Args:
            model: The model to load weights into
            pretrained_path: Path to the pretrained model weights (.pt file)
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
        
        # Load pretrained state dict
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        model_state = model.state_dict()
        
        # Filter out incompatible keys (mainly the final classification head)
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_state.items():
            if key in model_state:
                # Check if shapes match
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in current model)")
        
        # Load the modified state dict
        model.load_state_dict(model_state)
        
        print(f"Loaded {len(loaded_keys)} layers from pretrained model")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible layers:")
            for key in skipped_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")
        
        # Typically we skip the final classification layer (mlp_head)
        # The backbone (transformer) weights are transferred
        print("Transfer learning: Using pretrained backbone, training new classification head")

    
    def _save_model(self, model, best=False):
        filename = 'ast_model_best.pt' if best else 'ast_model.pt'
        torch.save(model.state_dict(), os.path.join(self.output_folder, filename))
        
        if self.use_cleaner:
            cleaner_filename = 'cleaner_best.pt' if best else 'cleaner.pt'
            torch.save(self.cleaner.state_dict(), os.path.join(self.output_folder, cleaner_filename))
        
        # Always save configuration for model deployment
        model_config = config.get_model_config()
        
        # Override with actual dimensions used during training
        model_config['freq_bins'] = self.img_height
        model_config['time_bins'] = self.img_width
        
        # Add model-specific information
        model_config['model_type'] = 'MultiScaleAST' if self.use_multiscale else 'AST'
        model_config['num_classes'] = model.num_classes
        model_config['multilabel'] = model.multilabel
        model_config['class_names'] = self.data['class_names']
        model_config['use_reconstruction'] = self.use_reconstruction
        model_config['use_sparse_patches'] = self.use_sparse_patches
        model_config['num_sparse_patches'] = self.num_sparse_patches
        model_config['use_cleaner'] = self.use_cleaner
        
        # Save to JSON
        config_path = os.path.join(self.output_folder, 'ast_model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        if best:
            print(f"Saved best model and configuration")
            print(f"Classes ({model.num_classes}): {', '.join(self.data['class_names'])}")
    
    def _save_history(self, train_losses, val_losses, train_accs, val_accs):
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accs,
            'val_accuracy': val_accs
        }
        with open(os.path.join(self.output_folder, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def _plot_history(self, train_losses, val_losses, train_accs, val_accs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc') 
        ax2.set_title('Multi-Label Accuracy' if self.multilabel else 'Accuracy')
        ax2.legend()
        
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'))
        plt.close()
    
    def _save_test_predictions(self, model, test_loader, test_data, test_name):
        """Save test predictions to CSV file for accuracy computation."""
        import csv
        
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            for batch in test_loader:
                if self.use_sparse_patches:
                    patches = batch['patches'].to(self.device)
                    positions = batch['positions'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    filenames = batch['filename']
                    
                    output = model(patches, sparse_mode=True, positions=positions, mask=mask)
                else:
                    data, target = batch
                    data = data.to(self.device)
                    filenames = [test_data['train_filenames'][i] for i in range(len(data))]
                    
                    output = model(data)
                
                if self.multilabel:
                    preds = (torch.sigmoid(output) > 0.5).cpu().numpy()
                else:
                    preds = output.argmax(dim=1).cpu().numpy()
                
                # Map predictions to class names
                for i, filename in enumerate(filenames):
                    if self.multilabel:
                        pred_classes = [test_data['class_names'][j] for j in range(len(preds[i])) if preds[i][j]]
                        predictions[filename] = ','.join(pred_classes) if pred_classes else 'Empty'
                    else:
                        predictions[filename] = test_data['class_names'][preds[i]]
        
        # Save to CSV
        csv_path = os.path.join(self.output_folder, f'predictions_{test_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'predicted_class'])
            for filename, pred_class in predictions.items():
                writer.writerow([filename, pred_class])
        
        print(f"Saved test predictions to {csv_path}")

