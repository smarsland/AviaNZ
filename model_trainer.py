
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
from sklearn.metrics import precision_recall_fscore_support
from data_utils import DataLoader, create_data_loaders, compute_confusion_weights
from models import AST, CNNModel
from evaluation_utils import EvaluationManager
import config


def compute_multilabel_f1(all_preds, all_targets):
    """Compute macro F1 score for multi-label predictions."""
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    _, _, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    return f1


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
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Class-balancing alpha per element: alpha for positives, (1-alpha) for negatives
        alpha_t = torch.where(targets == 1, torch.as_tensor(self.alpha, device=inputs.device), torch.as_tensor(1.0 - self.alpha, device=inputs.device))
        
        # Focal loss
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
            targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )


class ASTTrainer:
    """Simple AST trainer."""
    
    def __init__(self, data_folder, output_folder, max_epochs, batch_size, 
                 multilabel, learning_rate, mixup_alpha=0.3, 
                 pretrained_path=None, use_class_balancing=False, 
                 scheduler_type='lambda', weight_decay=0.0, 
                 noise_ratio=0.0, noise_folder=None, freq_bins=None, time_bins=None,
                 use_confusion_sampling=False, confusion_eval_freq=None, 
                 confusion_boost_factor=None, confusion_top_k=None, use_focal_loss=False,
                 use_multiscale=False, use_class_weights=False, freeze_layers=None,
                 use_reconstruction=False, recon_weight=0.1, normalize=False,
                 use_sparse_patches=False, num_sparse_patches=20, dropout=0.2,
                 bce_smoothing=0.0, trial=None, use_amp=True):
        
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.multilabel = multilabel
        self.learning_rate = learning_rate
        self.mixup_alpha = mixup_alpha
        self.pretrained_path = pretrained_path
        self.use_class_balancing = use_class_balancing
        self.scheduler_type = scheduler_type
        self.weight_decay = weight_decay
        self.noise_ratio = noise_ratio
        self.noise_folder = noise_folder
        self.use_confusion_sampling = use_confusion_sampling
        self.confusion_eval_freq = confusion_eval_freq if confusion_eval_freq is not None else config.DEFAULT_CONFUSION_EVAL_FREQUENCY
        self.confusion_boost_factor = confusion_boost_factor if confusion_boost_factor is not None else config.DEFAULT_CONFUSION_BOOST_FACTOR
        self.confusion_top_k = confusion_top_k
        self.use_focal_loss = use_focal_loss
        self.use_multiscale = use_multiscale
        self.use_class_weights = use_class_weights
        self.freeze_layers = freeze_layers
        self.use_reconstruction = use_reconstruction
        self.recon_weight = recon_weight
        self.normalize = normalize
        self.use_sparse_patches = use_sparse_patches
        self.num_sparse_patches = num_sparse_patches
        self.dropout = dropout
        self.bce_smoothing = bce_smoothing
        self.trial = trial
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) for faster training")
        
        # Initialize gradient scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Load data
        data_loader = DataLoader(data_folder, noise_folder=noise_folder)
        self.data = data_loader.load_data(multilabel, validation_share=0.2)
        self.num_classes = self.data['nclasses']

        # Use dimensions from spectrogram params (single source of truth)
        # Avoid duplicating DEFAULT_FREQ_BINS vs SPECTROGRAM_PARAMS['nfilters']
        self.img_height = freq_bins if freq_bins is not None else config.SPECTROGRAM_PARAMS['nfilters']  # Frequency bins (height)
        self.img_width = time_bins if time_bins is not None else config.DEFAULT_TIME_BINS   # Time bins (width)
        
        # Create data loaders with config defaults
        # Use more workers and prefetch for faster GPU utilization
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader, self.val_loader = create_data_loaders(
            self.data, batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_ratio=self.noise_ratio, 
            spec_transform=None,  # Uses config.DEFAULT_SPEC_TRANSFORM
            num_workers=num_workers, width_downsizing=None, mixup_alpha=mixup_alpha,
            use_class_balancing=self.use_class_balancing, normalize=self.normalize,
            use_sparse_patches=self.use_sparse_patches, num_sparse_patches=self.num_sparse_patches
        )
        
        if self.use_confusion_sampling:
            from torch.utils.data import DataLoader as TorchDataLoader
            from data_utils import SpectrogramDataset, sparse_collate_fn
            
            # Warning if both balancing strategies are enabled
            if self.use_class_balancing:
                print("\n⚠️  WARNING: Both --balance and --confusion-sampling are enabled!")
                print("   These strategies can interact unpredictably and cause weight explosion.")
                print("   Consider using only one, or reduce --confusion-boost (currently {:.1f})".format(self.confusion_boost_factor))
                # Automatically reduce boost factor to be safer
                if self.confusion_boost_factor > 2.0:
                    original_boost = self.confusion_boost_factor
                    self.confusion_boost_factor = min(2.0, self.confusion_boost_factor / 2.0)
                    print(f"   Auto-reducing boost factor from {original_boost:.1f} to {self.confusion_boost_factor:.1f} for stability\n")
            
            eval_dataset = SpectrogramDataset(
                self.data['train_filenames'], self.data['train_labels'], 
                self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                cropping_mode='center', noise_ratio=0.0, spec_transform=None,
                width_downsizing=None, normalize=self.normalize,
                use_sparse_patches=self.use_sparse_patches,
                num_sparse_patches=self.num_sparse_patches
            )
            num_workers = 4 if torch.cuda.is_available() else 2
            eval_collate = sparse_collate_fn if self.use_sparse_patches else None
            self.eval_train_loader = TorchDataLoader(
                eval_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                collate_fn=eval_collate
            )
            self.current_sample_weights = None
        
        os.makedirs(output_folder, exist_ok=True)
    
    def _compute_class_weights(self):
        """Compute inverse frequency weights for each class (for multilabel BCE loss)."""
        train_labels = np.array(self.data['train_labels'])
        
        class_counts = train_labels.sum(axis=0)
        total_samples = len(train_labels)
        
        pos_counts = class_counts
        neg_counts = total_samples - class_counts
        
        pos_weight = neg_counts / (pos_counts + 1e-5)
        
        pos_weight = torch.from_numpy(pos_weight).float().to(self.device)
        
        return pos_weight
    
    def train(self):
        """Train AST model."""
        if self.use_multiscale:
            print("Creating Multi-Scale AST model...")
        else:
            print("Creating AST model...")
        # Use dimensions we set in __init__
        input_size = (self.img_height, self.img_width)
        print(f"Model input size: {input_size}")
        
        if self.use_multiscale:
            from models import MultiScaleAST
            model = MultiScaleAST(self.num_classes, self.multilabel, input_size=input_size, dropout=self.dropout, use_reconstruction=self.use_reconstruction).to(self.device)
        else:
            model = AST(self.num_classes, self.multilabel, input_size=input_size, dropout=self.dropout, use_reconstruction=self.use_reconstruction).to(self.device)
        
        # Interpolate positional embeddings to match input size
        # AudioSet pretrained uses 128x1024, we may use different dimensions (e.g., 128x512 for ESC-50)
        if not self.use_multiscale:
            print(f"Interpolating positional embeddings for input size {input_size}...")
            model.interpolate_pos_embed(input_size)
        
        # Load pretrained weights if provided
        if self.pretrained_path:
            print(f"Loading pretrained weights from {self.pretrained_path}")
            self._load_pretrained_weights(model, self.pretrained_path)
        
        # Freeze early transformer layers if requested
        if self.freeze_layers is not None and self.freeze_layers > 0:
            print(f"Freezing first {self.freeze_layers} transformer layers...")
            for i in range(self.freeze_layers):
                for param in model.ast.encoder.layer[i].parameters():
                    param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        # Optimizer and LR schedule
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
            print(f"Using Cosine Annealing LR scheduler")
        elif self.scheduler_type == 'cosine_warmup':
            def lr_lambda(epoch):
                warmup_epochs = 5
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                t = (epoch - warmup_epochs) / max(1, (self.max_epochs - warmup_epochs))
                return 0.5 * (1 + math.cos(math.pi * t))
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            print("Using Cosine with Warmup LR scheduler")
        else:
            def lr_lambda(epoch):
                if epoch < 5:
                    return 1.0
                return 0.85 ** (epoch - 5)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"Using Lambda LR scheduler (ESC-50 style)")
        
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
        train_primary_accs = []  # Primary-class accuracy for multi-label
        val_primary_accs = []    # Primary-class accuracy for multi-label
        
        print(f"Starting training for {self.max_epochs} epochs...")
        
        best_val_acc = 0.0
        best_epoch = -1
        
        # Weight averaging: collect model states from later epochs
        model_states_for_averaging = []
        averaging_start_epoch = max(0, self.max_epochs - 20)  # Average last 20 epochs
        
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            # Check model weights BEFORE starting epoch (confusion sampling might have corrupted them)
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: Model weights are NaN/Inf at START of epoch {epoch+1}!")
                print(f"   This happened AFTER confusion sampling updated weights at end of epoch {epoch}.")
                print(f"   ABORTING TRAINING.\n")
                return
            
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_primary_correct = 0
            train_primary_total = 0
            
            # For multi-label F1 calculation
            all_train_preds = []
            all_train_targets = []
            
            for batch_idx, batch in enumerate(self.train_loader):
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
                optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    # Check for NaN gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                        print(f"   Stopping epoch early...")
                        break
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                        print(f"   Stopping epoch early...")
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
                    # Primary-class accuracy: highest prob class vs primary label
                    primary_pred = output.argmax(dim=1)
                    primary_target = target_hard.argmax(dim=1)
                    train_primary_correct += (primary_pred == primary_target).sum().item()
                    train_primary_total += target_hard.size(0)
                else:
                    pred = output.argmax(dim=1)
                    target_labels = target_hard.argmax(dim=1)
                    train_correct += (pred == target_labels).sum().item()
                    train_total += target_hard.size(0)
                
                if batch_idx % 10 == 0:
                    avg_loss = train_loss / max(1, batch_idx)
                    if avg_loss > 100.0:
                        print(f'⚠️  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f} (avg: {avg_loss:.4f} - HIGH!)')
                    else:
                        print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_primary_correct = 0
            val_primary_total = 0
            
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
                            if self.use_reconstruction:
                                output, _ = model(data)
                            else:
                                output = model(data)
                    
                    if self.multilabel:
                        val_loss += criterion(output, target).item()
                        pred = (torch.sigmoid(output) > 0.5).float()
                        all_val_preds.append(pred.cpu().numpy())
                        all_val_targets.append(target.cpu().numpy())
                        # Primary-class accuracy
                        primary_pred = output.argmax(dim=1)
                        primary_target = target.argmax(dim=1)
                        val_primary_correct += (primary_pred == primary_target).sum().item()
                        val_primary_total += target.size(0)
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
            
            if self.multilabel:
                # Compute macro F1 for multi-label (handle empty lists if epoch stopped early)
                if len(all_train_preds) > 0:
                    train_acc = compute_multilabel_f1(all_train_preds, all_train_targets)
                else:
                    train_acc = 0.0
                if len(all_val_preds) > 0:
                    val_acc = compute_multilabel_f1(all_val_preds, all_val_targets)
                else:
                    val_acc = 0.0
            else:
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if self.multilabel:
                train_primary_acc = train_primary_correct / train_primary_total
                val_primary_acc = val_primary_correct / val_primary_total
                train_primary_accs.append(train_primary_acc)
                val_primary_accs.append(val_primary_acc)
            else:
                train_primary_accs.append(train_acc)
                val_primary_accs.append(val_acc)
            
            # ABORT if model weights contain NaN
            if any(torch.isnan(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: NaN in weights (epoch {epoch+1}). ABORTING.\n")
                return

            scheduler.step()
            
            if self.use_confusion_sampling and (epoch + 1) % self.confusion_eval_freq == 0:
                print(f"\nEvaluating confusion matrix to update sampling weights...")
                sample_weights, class_error_rates = compute_confusion_weights(
                    model, self.eval_train_loader, 
                    np.array(self.data['train_labels']),
                    self.num_classes, self.device,
                    boost_factor=self.confusion_boost_factor,
                    top_k=self.confusion_top_k
                )
                
                top_confused = np.argsort(class_error_rates)[-5:][::-1]
                print(f"Top 5 confused classes (error rates):")
                for idx in top_confused:
                    if idx < len(self.data['class_names']):
                        print(f"  {self.data['class_names'][idx]}: {class_error_rates[idx]:.3f}")
                
                from torch.utils.data import WeightedRandomSampler
                from data_utils import MixupCollate
                
                new_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                
                train_collate_fn = MixupCollate(self.mixup_alpha) if self.mixup_alpha > 0 else None
                if self.use_sparse_patches:
                    from data_utils import sparse_collate_fn
                    train_collate_fn = sparse_collate_fn
                
                from torch.utils.data import DataLoader as TorchDataLoader
                from data_utils import SpectrogramDataset
                
                train_dataset = SpectrogramDataset(
                    self.data['train_filenames'], self.data['train_labels'],
                    self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                    cropping_mode='random', noise_ratio=self.noise_ratio,
                    spec_transform=None, width_downsizing=None, normalize=self.normalize,
                    use_sparse_patches=self.use_sparse_patches,
                    num_sparse_patches=self.num_sparse_patches
                )
                
                self.train_loader = TorchDataLoader(
                    train_dataset, batch_size=self.batch_size,
                    shuffle=False, sampler=new_sampler,
                    num_workers=2, pin_memory=True,
                    collate_fn=train_collate_fn
                )
                
                print(f"Updated training sampler with confusion-based weights\n")
            
            # Collect model state for weight averaging (after epoch 5 when LR starts decaying)
            if epoch >= averaging_start_epoch:
                model_states_for_averaging.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self._save_model(model, best=True)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s)')
            if self.multilabel:
                print(f'Train Loss: {train_loss:.4f}, Train Macro-F1: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Macro-F1: {val_acc:.4f}')
                print(f'Val Primary-Class Acc: {val_primary_acc:.4f}')
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
        
        # Save final model
        self._save_model(model)
        self._save_history(train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs)
        self._plot_history(train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs)
        
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
        
        # Save to JSON
        config_path = os.path.join(self.output_folder, 'ast_model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        if best:
            print(f"Saved best model and configuration")
            print(f"Classes ({model.num_classes}): {', '.join(self.data['class_names'])}")
    
    def _save_history(self, train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs):
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accs,
            'val_accuracy': val_accs,
            'train_primary_accuracy': train_primary_accs,
            'val_primary_accuracy': val_primary_accs
        }
        with open(os.path.join(self.output_folder, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def _plot_history(self, train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs):
        if self.multilabel:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc') 
        ax2.set_title('Multi-Label Accuracy' if self.multilabel else 'Accuracy')
        ax2.legend()
        
        if self.multilabel:
            ax3.plot(train_primary_accs, label='Train Primary Acc')
            ax3.plot(val_primary_accs, label='Val Primary Acc')
            ax3.set_title('Primary-Class Accuracy')
            ax3.legend()
        
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'))
        plt.close()

class CNNTrainer:
    """Simple CNN trainer."""
    
    def __init__(self, data_folder, output_folder, max_epochs, batch_size, 
                 multilabel, learning_rate, mixup_alpha=0.3, 
                 pretrained_path=None, use_class_balancing=False, weight_decay=0.0,
                 noise_ratio=0.0, noise_folder=None, freq_bins=None, time_bins=None,
                 use_confusion_sampling=False, confusion_eval_freq=None,
                 confusion_boost_factor=None, confusion_top_k=None, use_focal_loss=False, normalize=False, use_amp=True):
        
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.multilabel = multilabel
        self.learning_rate = learning_rate
        self.mixup_alpha = mixup_alpha
        self.pretrained_path = pretrained_path
        self.use_class_balancing = use_class_balancing
        self.weight_decay = weight_decay
        self.noise_ratio = noise_ratio
        self.noise_folder = noise_folder
        self.use_confusion_sampling = use_confusion_sampling
        self.confusion_eval_freq = confusion_eval_freq if confusion_eval_freq is not None else config.DEFAULT_CONFUSION_EVAL_FREQUENCY
        self.confusion_boost_factor = confusion_boost_factor if confusion_boost_factor is not None else config.DEFAULT_CONFUSION_BOOST_FACTOR
        self.confusion_top_k = confusion_top_k
        self.use_focal_loss = use_focal_loss
        self.confusion_boost_factor = confusion_boost_factor if confusion_boost_factor is not None else config.DEFAULT_CONFUSION_BOOST_FACTOR
        self.confusion_top_k = confusion_top_k
        self.normalize = normalize
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) for faster training")
        
        # Initialize gradient scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Load data
        data_loader = DataLoader(data_folder, noise_folder=noise_folder)
        self.data = data_loader.load_data(multilabel, validation_share=0.2)
        self.num_classes = self.data['nclasses']

        # Use dimensions from spectrogram params (single source of truth)
        self.img_height = freq_bins if freq_bins is not None else config.SPECTROGRAM_PARAMS['nfilters']  # Frequency bins (height)
        self.img_width = time_bins if time_bins is not None else config.DEFAULT_TIME_BINS   # Time bins (width)
        
        # Create data loaders with config defaults
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader, self.val_loader = create_data_loaders(
            self.data, batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_ratio=self.noise_ratio, 
            spec_transform=None,  # Uses config.DEFAULT_SPEC_TRANSFORM
            num_workers=num_workers, width_downsizing=None, mixup_alpha=mixup_alpha,
            use_class_balancing=self.use_class_balancing, normalize=self.normalize,
            use_sparse_patches=False, num_sparse_patches=20
        )
        
        if self.use_confusion_sampling:
            from torch.utils.data import DataLoader as TorchDataLoader
            from data_utils import SpectrogramDataset
            
            # Warning if both balancing strategies are enabled
            if self.use_class_balancing:
                print("\n⚠️  WARNING: Both --balance and --confusion-sampling are enabled!")
                print("   These strategies can interact unpredictably and cause weight explosion.")
                print("   Consider using only one, or reduce --confusion-boost (currently {:.1f})".format(self.confusion_boost_factor))
                # Automatically reduce boost factor to be safer
                if self.confusion_boost_factor > 2.0:
                    original_boost = self.confusion_boost_factor
                    self.confusion_boost_factor = min(2.0, self.confusion_boost_factor / 2.0)
                    print(f"   Auto-reducing boost factor from {original_boost:.1f} to {self.confusion_boost_factor:.1f} for stability\n")
            
            eval_dataset = SpectrogramDataset(
                self.data['train_filenames'], self.data['train_labels'],
                self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                cropping_mode='center', noise_ratio=0.0, spec_transform=None,
                width_downsizing=None, normalize=self.normalize
            )
            num_workers = 4 if torch.cuda.is_available() else 2
            self.eval_train_loader = TorchDataLoader(
                eval_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            self.current_sample_weights = None
        
        os.makedirs(output_folder, exist_ok=True)
    
    def train(self):
        """Train CNN model."""
        print("Creating CNN model...")
        print(f"Model input size: ({self.img_height}, {self.img_width})")
        
        model = CNNModel(self.img_height, self.img_width, self.num_classes).to(self.device)
        
        # Load pretrained weights if provided
        if self.pretrained_path:
            print(f"Loading pretrained weights from {self.pretrained_path}")
            self._load_pretrained_weights(model, self.pretrained_path)

        # Optimizer and LR schedule
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Simple step decay scheduler
        def lr_lambda(epoch):
            if epoch < 10:
                return 1.0
            return 0.9 ** (epoch - 10)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # Loss function
        if self.use_focal_loss:
            if self.multilabel:
                criterion = MultilabelFocalLoss(alpha=0.25, gamma=2.0)
                print("Using Multilabel Focal Loss (alpha=0.25, gamma=2.0)")
            else:
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
                print("Using Focal Loss (alpha=0.25, gamma=2.0)")
        else:
            if self.multilabel:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_primary_accs = []  # Primary-class accuracy for multi-label
        val_primary_accs = []    # Primary-class accuracy for multi-label
        
        print(f"Starting training for {self.max_epochs} epochs...")
        
        best_val_acc = 0.0
        best_epoch = -1
        
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_primary_correct = 0
            train_primary_total = 0
            
            # For multi-label F1 calculation
            all_train_preds = []
            all_train_targets = []
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = model(data)
                    
                    if self.multilabel:
                        # Clamp logits to prevent numerical overflow in BCE loss
                        output = torch.clamp(output, min=-80.0, max=80.0)
                        loss = criterion(output, target)
                    else:
                        # Detect soft (mixup) labels
                        if target.dim() == 2 and not torch.equal(target, target.round()):
                            # Soft-label cross-entropy with logits
                            log_probs = F.log_softmax(output, dim=1)
                            loss = -(target * log_probs).sum(dim=1).mean()
                        else:
                            # Hard labels - use CrossEntropyLoss with label smoothing
                            target_idx = target.argmax(dim=1)
                            loss = criterion(output, target_idx)
                
                # Check for NaN loss BEFORE backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ CRITICAL: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}")
                    print(f"   Model weights corrupted. Stopping epoch early...")
                    break
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                        print(f"   Stopping epoch early...")
                        break
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n❌ CRITICAL: NaN/Inf gradients at epoch {epoch+1}, batch {batch_idx}")
                        print(f"   Stopping epoch early...")
                        break
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                
                # For metrics, convert soft (mixup) targets back to hard labels
                # by rounding to nearest integer (0 or 1)
                target_hard = target.round()
                
                if self.multilabel:
                    pred = (torch.sigmoid(output) > 0.5).float()
                    all_train_preds.append(pred.cpu().numpy())
                    all_train_targets.append(target_hard.cpu().numpy())
                    # Primary-class accuracy: highest prob class vs primary label
                    primary_pred = output.argmax(dim=1)
                    primary_target = target_hard.argmax(dim=1)
                    train_primary_correct += (primary_pred == primary_target).sum().item()
                    train_primary_total += target_hard.size(0)
                else:
                    pred = output.argmax(dim=1)
                    target_labels = target_hard.argmax(dim=1)
                    train_correct += (pred == target_labels).sum().item()
                    train_total += target_hard.size(0)
                
                if batch_idx % 10 == 0:
                    avg_loss = train_loss / max(1, batch_idx)
                    if avg_loss > 100.0:
                        print(f'⚠️  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f} (avg: {avg_loss:.4f} - HIGH!)')
                    else:
                        print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_primary_correct = 0
            val_primary_total = 0
            
            # For multi-label F1 calculation
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        output = model(data)
                    
                    if self.multilabel:
                        val_loss += criterion(output, target).item()
                        pred = (torch.sigmoid(output) > 0.5).float()
                        all_val_preds.append(pred.cpu().numpy())
                        all_val_targets.append(target.cpu().numpy())
                        # Primary-class accuracy
                        primary_pred = output.argmax(dim=1)
                        primary_target = target.argmax(dim=1)
                        val_primary_correct += (primary_pred == primary_target).sum().item()
                        val_primary_total += target.size(0)
                    else:
                        target_idx = target.argmax(dim=1)
                        val_loss += criterion(output, target_idx).item()
                        pred = output.argmax(dim=1)
                        target_labels = target.argmax(dim=1)
                        val_correct += (pred == target_labels).sum().item()
                        val_total += target.size(0)
            
            # Calculate averages
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            
            if self.multilabel:
                # Compute macro F1 for multi-label
                all_train_preds = np.vstack(all_train_preds)
                all_train_targets = np.vstack(all_train_targets)
                _, _, train_f1, _ = precision_recall_fscore_support(
                    all_train_targets, all_train_preds, average='macro', zero_division=0
                )
                train_acc = train_f1
                
                all_val_preds = np.vstack(all_val_preds)
                all_val_targets = np.vstack(all_val_targets)
                _, _, val_f1, _ = precision_recall_fscore_support(
                    all_val_targets, all_val_preds, average='macro', zero_division=0
                )
                val_acc = val_f1
            else:
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            if self.multilabel:
                train_primary_acc = train_primary_correct / train_primary_total
                val_primary_acc = val_primary_correct / val_primary_total
                train_primary_accs.append(train_primary_acc)
                val_primary_accs.append(val_primary_acc)
            else:
                train_primary_accs.append(train_acc)
                val_primary_accs.append(val_acc)
            
            # ABORT if model weights contain NaN
            if any(torch.isnan(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: NaN in weights (epoch {epoch+1}). ABORTING.\n")
                return

            scheduler.step()
            
            if self.use_confusion_sampling and (epoch + 1) % self.confusion_eval_freq == 0:
                print(f"\nEvaluating confusion matrix to update sampling weights...")
                sample_weights, class_error_rates = compute_confusion_weights(
                    model, self.eval_train_loader,
                    np.array(self.data['train_labels']),
                    self.num_classes, self.device,
                    boost_factor=self.confusion_boost_factor,
                    top_k=self.confusion_top_k
                )
                
                top_confused = np.argsort(class_error_rates)[-5:][::-1]
                print(f"Top 5 confused classes (error rates):")
                for idx in top_confused:
                    if idx < len(self.data['class_names']):
                        print(f"  {self.data['class_names'][idx]}: {class_error_rates[idx]:.3f}")
                
                from torch.utils.data import WeightedRandomSampler
                from data_utils import MixupCollate
                
                new_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                
                train_collate_fn = MixupCollate(self.mixup_alpha) if self.mixup_alpha > 0 else None
                
                from torch.utils.data import DataLoader as TorchDataLoader
                from data_utils import SpectrogramDataset
                
                train_dataset = SpectrogramDataset(
                    self.data['train_filenames'], self.data['train_labels'],
                    self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                    cropping_mode='random', noise_ratio=self.noise_ratio,
                    spec_transform=None, width_downsizing=None, normalize=self.normalize
                )
                
                num_workers = 4 if torch.cuda.is_available() else 2
                self.train_loader = TorchDataLoader(
                    train_dataset, batch_size=self.batch_size,
                    shuffle=False, sampler=new_sampler,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=train_collate_fn
                )
                
                print(f"Updated training sampler with confusion-based weights\n")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self._save_model(model, best=True)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s)')
            if self.multilabel:
                print(f'Train Loss: {train_loss:.4f}, Train Macro-F1: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Macro-F1: {val_acc:.4f}')
                print(f'Val Primary-Class Acc: {val_primary_acc:.4f}')
            else:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print('-' * 60)
        
        # Save final model
        self._save_model(model)
        self._save_history(train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs)
        self._plot_history(train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs)
        
        # Evaluate on validation set (using best checkpoint)
        best_path = os.path.join(self.output_folder, 'cnn_model_best.pt')
        if os.path.exists(best_path):
            state_dict = torch.load(best_path, map_location=self.device)
            model.load_state_dict(state_dict)
        evaluator = EvaluationManager(self.output_folder, self.data['class_names'], self.multilabel)
        evaluator.evaluate_model(model, self.val_loader, 'cnn_model', self.data, device=self.device)

        print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Done! Saved to {self.output_folder}")
        return {
            'model': model,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }
    
    def _load_pretrained_weights(self, model, pretrained_path):
        """Load pretrained weights into the model."""
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
        
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        model_state = model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        model.load_state_dict(model_state)
        
        print(f"Loaded {len(loaded_keys)} layers from pretrained model")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} incompatible layers")
    
    def _save_model(self, model, best=False):
        filename = 'cnn_model_best.pt' if best else 'cnn_model.pt'
        torch.save(model.state_dict(), os.path.join(self.output_folder, filename))
        
        # Save configuration for model deployment
        if best:
            model_config = config.get_model_config()
            
            # Add model-specific information
            model_config['model_type'] = 'CNN'
            model_config['num_classes'] = self.num_classes
            model_config['multilabel'] = self.multilabel
            model_config['class_names'] = self.data['class_names']
            model_config['image_height'] = self.img_height
            model_config['image_width'] = self.img_width
            
            # Save to JSON
            config_path = os.path.join(self.output_folder, 'cnn_model_config.json')
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            print(f"Saved model configuration to cnn_model_config.json")
            print(f"Classes ({self.num_classes}): {', '.join(self.data['class_names'])}")
    
    def _save_history(self, train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs):
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accs,
            'val_accuracy': val_accs,
            'train_primary_accuracy': train_primary_accs,
            'val_primary_accuracy': val_primary_accs
        }
        with open(os.path.join(self.output_folder, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def _plot_history(self, train_losses, val_losses, train_accs, val_accs, train_primary_accs, val_primary_accs):
        if self.multilabel:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc') 
        ax2.set_title('Multi-Label Accuracy' if self.multilabel else 'Accuracy')
        ax2.legend()
        
        if self.multilabel:
            ax3.plot(train_primary_accs, label='Train Primary Acc')
            ax3.plot(val_primary_accs, label='Val Primary Acc')
            ax3.set_title('Primary-Class Accuracy')
            ax3.legend()
        
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'))
        plt.close()


class PixelPredictionTrainer:
    
    def __init__(self, data_folder, output_folder, max_epochs, batch_size, 
                 learning_rate, model_type='cnn', pretrained_path=None,
                 freq_bins=None, time_bins=None, normalize=False, use_amp=True):
        
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.pretrained_path = pretrained_path
        self.normalize = normalize
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP) for faster training")
        
        # Initialize gradient scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.img_height = freq_bins if freq_bins is not None else config.DEFAULT_FREQ_BINS
        self.img_width = time_bins if time_bins is not None else config.DEFAULT_TIME_BINS
        
        self._load_pixel_data()
        
        os.makedirs(output_folder, exist_ok=True)
    
    def _load_pixel_data(self):
        spec_folder = os.path.join(self.data_folder, 'data')
        interest_folder = os.path.join(self.data_folder, 'interest_maps')
        labels_file = os.path.join(self.data_folder, 'labels.json')
        
        if not os.path.exists(spec_folder):
            raise ValueError(f"Spectrogram folder not found: {spec_folder}")
        if not os.path.exists(interest_folder):
            raise ValueError(f"Interest maps folder not found: {interest_folder}")
        if not os.path.exists(labels_file):
            raise ValueError(f"Labels file not found: {labels_file}")
        
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
        
        spec_files = []
        interest_files = []
        
        for entry in labels_data:
            filename = entry['filename']
            spec_path = os.path.join(spec_folder, filename)
            interest_path = os.path.join(interest_folder, filename)
            
            if os.path.exists(spec_path) and os.path.exists(interest_path):
                spec_files.append(spec_path)
                interest_files.append(interest_path)
        
        print(f"Found {len(spec_files)} spectrogram-interest map pairs")
        
        from sklearn.model_selection import train_test_split
        train_spec, val_spec, train_interest, val_interest = train_test_split(
            spec_files, interest_files, test_size=0.2, random_state=42
        )
        
        print(f"Train: {len(train_spec)}, Val: {len(val_spec)}")
        
        from data_utils import InterestPixelDataset
        from torch.utils.data import DataLoader as TorchDataLoader
        
        train_dataset = InterestPixelDataset(
            train_spec, train_interest, self.img_height, self.img_width,
            spec_transform="Log", training=True
        )
        val_dataset = InterestPixelDataset(
            val_spec, val_interest, self.img_height, self.img_width,
            spec_transform="Log", training=False
        )
        
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader = TorchDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.val_loader = TorchDataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    def train(self):
        print(f"Training {self.model_type.upper()} for pixel prediction...")
        print(f"Input size: {self.img_height}x{self.img_width}")
        
        if self.model_type == 'cnn':
            from models import PixelPredictionCNN
            model = PixelPredictionCNN(self.img_height, self.img_width).to(self.device)
        elif self.model_type == 'ast':
            from models import ASTPixelPredictor
            model = ASTPixelPredictor(input_size=(self.img_height, self.img_width)).to(self.device)
            
            if self.pretrained_path:
                print(f"Loading pretrained weights from {self.pretrained_path}")
                self._load_pretrained_ast_weights(model, self.pretrained_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        train_losses = []
        val_losses = []
        train_ious = []
        val_ious = []
        
        best_val_iou = 0
        
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0
            train_iou = 0
            
            for spec, target in self.train_loader:
                spec = spec.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = model(spec)
                    loss = criterion(output, target)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_iou += self._compute_iou(output, target)
            
            train_loss /= len(self.train_loader)
            train_iou /= len(self.train_loader)
            
            model.eval()
            val_loss = 0
            val_iou = 0
            
            with torch.no_grad():
                for spec, target in self.val_loader:
                    spec = spec.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        output = model(spec)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    val_iou += self._compute_iou(output, target)
            
            val_loss /= len(self.val_loader)
            val_iou /= len(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            
            print(f"Epoch {epoch+1}/{self.max_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                self._save_model(model, best=True)
                print(f"  New best model saved (IoU: {best_val_iou:.4f})")
            
            self._save_model(model, best=False)
        
        self._save_history(train_losses, val_losses, train_ious, val_ious)
        self._plot_history(train_losses, val_losses, train_ious, val_ious)
        
        print(f"Training complete! Best Val IoU: {best_val_iou:.4f}")
    
    def _compute_iou(self, output, target):
        pred = torch.sigmoid(output) > 0.5
        target_bool = target > 0.5
        
        intersection = (pred & target_bool).float().sum()
        union = (pred | target_bool).float().sum()
        
        iou = intersection / (union + 1e-6)
        return iou.item()
    
    def _load_pretrained_ast_weights(self, model, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        model_state = model.state_dict()
        pretrained_state = {}
        
        for k, v in checkpoint.items():
            if k.startswith('ast.'):
                if k in model_state and model_state[k].shape == v.shape:
                    pretrained_state[k] = v
        
        model.load_state_dict(pretrained_state, strict=False)
        print(f"Loaded {len(pretrained_state)} AST layers from pretrained model")
    
    def _save_model(self, model, best=False):
        suffix = 'best' if best else 'last'
        filename = f'{self.model_type}_pixel_predictor_{suffix}.pt'
        torch.save(model.state_dict(), os.path.join(self.output_folder, filename))
    
    def _save_history(self, train_losses, val_losses, train_ious, val_ious):
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_iou': train_ious,
            'val_iou': val_ious
        }
        with open(os.path.join(self.output_folder, 'pixel_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def _plot_history(self, train_losses, val_losses, train_ious, val_ious):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        ax2.plot(train_ious, label='Train IoU')
        ax2.plot(val_ious, label='Val IoU')
        ax2.set_title('IoU (Intersection over Union)')
        ax2.legend()
        
        plt.savefig(os.path.join(self.output_folder, 'pixel_training_curves.png'))
        plt.close()
