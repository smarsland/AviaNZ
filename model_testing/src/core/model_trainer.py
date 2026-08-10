
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import math
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from model_testing.src.data.data_utils import DataLoader, create_data_loaders, SpectrogramDataset
from model_testing.src.core.models import AST, RegNetModel
from model_testing.src.evaluation.evaluation_utils import EvaluationManager
from model_testing.src.evaluation.attention_viz import visualize_attention
from model_testing.src.core.trainer_config import TrainerConfig, TrainingConfig, ModelConfig
from model_testing.src.core import config
from model_testing.src.core.utils import pick_free_gpu


def _remap_labels_to_train_space(test_data, train_data):
    """Remap test labels from test category space to training category space.

    When a model is trained on N classes but a test set has M classes (e.g.
    training on a large 133-class dataset and testing on a 25-class matched
    split), the label tensors would have mismatched shapes causing a crash.
    This maps each test-set class to its position in the training vocabulary,
    so the resulting label matrix is (N_samples, N_train_classes).  Classes
    in the test set but not in training are silently ignored; training classes
    absent from the test set retain ground-truth 0 (correct: they are not
    present in those samples).
    """
    test_cats = test_data.get('categories', [])
    train_cats = train_data.get('categories', [])
    if list(test_cats) == list(train_cats):
        return test_data  # vocabularies already match – nothing to do

    n_samples = len(test_data['train_labels'])
    n_train = len(train_cats)
    train_cat_to_idx = {c: i for i, c in enumerate(train_cats)}
    remapped = np.zeros((n_samples, n_train), dtype=np.float32)
    for test_idx, cat in enumerate(test_cats):
        if cat in train_cat_to_idx:
            remapped[:, train_cat_to_idx[cat]] = test_data['train_labels'][:, test_idx]

    result = dict(test_data)
    result['train_labels'] = remapped
    result['categories'] = list(train_cats)
    result['class_names'] = train_data.get('class_names', list(train_cats))
    result['nclasses'] = n_train
    n_test_only = sum(1 for c in test_cats if c not in train_cat_to_idx)
    n_train_only = sum(1 for c in train_cats if c not in set(test_cats))
    if n_test_only or n_train_only:
        print(f"  [label remap] test-only classes ignored: {n_test_only}, "
              f"train-only classes (always 0 in GT): {n_train_only}")
    return result


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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Reference: Ben-Baruch et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    Separates focusing strength for positives and negatives:
      - Negatives: p_m^gamma_neg * log(p_m)  where p_m = (1-sigma(x)+margin).clamp(1)
        The margin shifts easy-negative probability toward zero before applying the
        focal weight, so well-classified negatives are doubly suppressed.
      - Positives: (1-sigma(x))^gamma_pos * log(sigma(x))
        gamma_pos=0 gives standard CE for positives (no down-weighting).

    This prevents background samples (all-zero targets) from dominating training.
    Background samples the model correctly identifies contribute ~zero gradient;
    only wrong predictions (false positives on background) get penalised.

    Args:
        gamma_neg: Focusing power for negatives (default: 4)
        gamma_pos: Focusing power for positives (default: 0 = standard CE)
        margin:    Probability margin to shift/zero easy negatives (default: 0.05)
        pos_weight: Optional [C] tensor of per-class weights for positives
        eps:       Numerical stability constant
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, margin=0.05,
                 pos_weight=None, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.pos_weight = pos_weight
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric margin: shift negative probability down so that
        # very confident negative predictions clip to zero
        if self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1.0)

        # Log probabilities
        los_pos = torch.log(xs_pos.clamp(min=self.eps))
        los_neg = torch.log(xs_neg.clamp(min=self.eps))

        # Base BCE loss (margin applied to negatives)
        loss = y * los_pos + (1.0 - y) * los_neg

        # Per-class positive weighting (handles inter-class imbalance: tui vs kaka etc.)
        if self.pos_weight is not None:
            pw = self.pos_weight.unsqueeze(0)          # [1, C]
            loss = loss * (y * (pw - 1.0) + 1.0)      # pw where y=1, 1 where y=0

        # Asymmetric focusing: down-weight easy negatives, standard for positives.
        # IMPORTANT: Never compute x**0 through autograd — the gradient is
        # gamma * x^(gamma-1) which becomes 0 * 0^(-1) = NaN when x=0.
        # For gamma=0 (no focusing) use a constant instead.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Negative focusing weight: p_m^gamma_neg, where p_m is the SHIFTED probability.
            # After margin shift: xs_neg = (1 - xs_pos + margin).clamp(max=1)
            # so the shifted positive probability = 1 - xs_neg.
            if self.gamma_neg > 0:
                xs_pos_shifted = (1.0 - xs_neg)        # p_m = sigma(x) - margin (clipped)
                neg_focus = (xs_pos_shifted * (1.0 - y)) ** self.gamma_neg
            else:
                neg_focus = (1.0 - y)

            # Positive focusing weight: (1 - sigma(x))^gamma_pos on positive terms only.
            if self.gamma_pos > 0:
                pos_focus = ((1.0 - xs_pos) * y) ** self.gamma_pos
            else:
                pos_focus = y

            loss = (neg_focus + pos_focus) * loss

        loss = -loss

        if self.reduction == 'none':
            return loss          # [B, C]
        elif self.reduction == 'sum':
            return loss.sum()
        else:                    # 'mean'
            return loss.mean()


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
        self.seed = cfg.training.seed
        
        # Model configuration
        self.dropout = cfg.model.dropout
        self.use_reconstruction = cfg.model.use_reconstruction
        self.recon_weight = cfg.model.recon_weight
        self.use_adapters = cfg.model.use_adapters
        self.freeze_layers = cfg.model.freeze_layers
        self.pretrained_path = cfg.model.pretrained_path
        self.freeze_backbone = cfg.model.freeze_backbone
        self.freeze_stages = cfg.model.freeze_stages
        self.use_cnn_adapter = getattr(cfg.model, 'use_cnn_adapter', False)
        self.use_sed_head = getattr(cfg.model, 'use_sed_head', False)
        self.use_gated_head = getattr(cfg.model, 'use_gated_head', False)
        self.model_name = getattr(cfg.model, 'model_name', 'regnety_008')
        self.ast_channel_dir = getattr(cfg.model, 'ast_channel_dir', None)
        self.in_chans = getattr(cfg.model, 'in_chans', 1)
        
        # Augmentation configuration
        self.mixup_alpha = cfg.augmentation.mixup_alpha
        self.noise_ratio = cfg.augmentation.noise_ratio
        self.noise_folder = cfg.augmentation.noise_folder
        self.noise_as_samples = cfg.augmentation.noise_as_samples
        self.max_noise_samples = cfg.augmentation.max_noise_samples
        self.use_temporal_roll = cfg.augmentation.use_temporal_roll
        self.bg_subtract = cfg.augmentation.bg_subtract
        self.median_filter = cfg.augmentation.median_filter
        self.no_background = getattr(cfg.augmentation, 'no_background', False)
        self.use_deltas = getattr(cfg.augmentation, 'use_deltas', False)
        self.per_chunk_norm = cfg.augmentation.per_chunk_norm
        self.spec_transform = cfg.augmentation.spec_transform
        self.mixup_mode = cfg.augmentation.mixup_mode
        self.noise_mode = cfg.augmentation.noise_mode
        self.background_prob = cfg.augmentation.background_prob
        
        # Loss configuration
        self.use_class_weights = cfg.loss.use_class_weights
        self.pos_weight_cap = cfg.loss.pos_weight_cap
        self.bce_smoothing = cfg.loss.bce_smoothing
        self.use_asl = cfg.loss.use_asl
        self.rebalance_background = getattr(cfg.loss, 'rebalance_background', True)
        self.background_weight = 1.0  # computed after data load
        self.gate_loss_weight = getattr(cfg.loss, 'gate_loss_weight', 1.0)
        self.asl_gamma_neg = cfg.loss.asl_gamma_neg
        self.asl_gamma_pos = cfg.loss.asl_gamma_pos
        self.asl_margin = cfg.loss.asl_margin
        self.kbird_prior = getattr(cfg.loss, 'kbird_prior', 0.0)
        
        # Domain adaptation configuration
        self.use_dann = cfg.domain_adaptation.use_dann
        self.target_folder = cfg.domain_adaptation.target_folder
        self.lambda_domain = cfg.domain_adaptation.lambda_domain
        
        # Evaluation configuration
        self.test_folder = cfg.evaluation.test_folder
        self.visualize_attention = cfg.evaluation.visualize_attention
        self.viz_samples = cfg.evaluation.viz_samples
        self.eval_only = getattr(cfg.evaluation, 'eval_only', False)
        self.checkpoint_path = getattr(cfg.evaluation, 'checkpoint', None)
        
        # Setup device FIRST — before any torch.cuda call (including seeding).
        # In Exclusive Process mode, cudaErrorDevicesUnavailable is process-wide — once any
        # device fails, the entire CUDA runtime is broken for this process.  We therefore
        # probe GPUs via nvidia-smi (no CUDA context) BEFORE touching torch.cuda, then pin
        # CUDA_VISIBLE_DEVICES to the first free GPU so PyTorch only ever sees one device.
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            chosen = self._pick_free_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(chosen)

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

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

        self.device = torch.device('cuda:0')
        test_tensor = torch.zeros(1, device=self.device)
        del test_tensor
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda:0 (GPU: {gpu_name}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
        
        if self.patience > 0:
            print(f"Early stopping patience: {self.patience} epochs")
        
        # Load data (always multilabel).
        # When max_samples_per_class is set we must subsample the *full* dataset
        # before the train/val split so that the requested N is what actually ends
        # up in the training set rather than N*0.8 (the post-split training slice).
        data_loader = DataLoader(self.data_folder, noise_folder=self.noise_folder)
        max_per_class = cfg.training.max_samples_per_class
        if max_per_class is not None:
            full = data_loader.load_data(use_multilabel=True, validation_share=0.0)
            self.num_classes = full['nclasses']
            # Subsample on the full dataset before splitting
            full = self._subsample_per_class(full, max_per_class)
            split = data_loader.split_data(
                full['train_filenames'],
                np.array(full['train_labels'], dtype=np.float32),
                full.get('train_noise_filenames') or [],
                0.2,
            )
            self.data = {
                'train_filenames':       split[0],
                'train_labels':          split[1],
                'test_filenames':        split[2],
                'test_labels':           split[3],
                'train_noise_filenames': split[4],
                'test_noise_filenames':  split[5],
                'categories':            full['categories'],
                'class_names':           full['class_names'],
                'nclasses':              full['nclasses'],
            }
        else:
            self.data = data_loader.load_data(use_multilabel=True, validation_share=0.2)
            self.num_classes = self.data['nclasses']

        # Spot-check a sample of training spectrograms to catch bad data early.
        _sample_files = self.data['train_filenames'][:20]
        _bad_files = []
        for _f in _sample_files:
            try:
                _sg = np.load(_f, allow_pickle=True)
                if not np.isfinite(_sg).all():
                    _bad_files.append((_f, 'NaN/Inf'))
                elif _sg.max() < 1e-6:
                    _bad_files.append((_f, f'all-near-zero max={_sg.max():.2e}'))
                elif _sg.min() < -100:
                    _bad_files.append((_f, f'suspicious negative min={_sg.min():.2e} '
                                      f'(already log-transformed?)'))
            except Exception as e:
                _bad_files.append((_f, str(e)))
        if _bad_files:
            print(f"\n  ⚠️  WARNING: {len(_bad_files)}/20 sampled training files look suspicious:")
            for _f, _reason in _bad_files[:5]:
                print(f"    {_reason}  →  {_f}")
            print("  This may cause NaN predictions or training instability.\n")
        else:
            _sg_sample = np.load(_sample_files[0], allow_pickle=True)
            print(f"  Training data spot-check OK (sample range: "
                  f"[{_sg_sample.min():.3f}, {_sg_sample.max():.3f}])")

        # Compute background rebalancing weight: equalise total gradient contribution
        # of background (all-zero) vs labelled samples.
        if self.rebalance_background:
            train_labels = np.array(self.data['train_labels'], dtype=np.float32)
            n_bg = int((train_labels.sum(axis=1) == 0).sum())
            n_lab = len(train_labels) - n_bg
            if n_bg > 0 and n_lab > 0:
                self.background_weight = float(n_lab) / float(n_bg)
                print(f"Background rebalancing: {n_lab} labelled, {n_bg} background, "
                      f"bg_weight={self.background_weight:.3f}")
            else:
                self.background_weight = 1.0

        # Drop all-zero (background) training samples when --no-background is set.
        if self.no_background:
            train_labels = np.array(self.data['train_labels'], dtype=np.float32)
            keep = train_labels.sum(axis=1) > 0
            self.data['train_filenames'] = [f for f, k in zip(self.data['train_filenames'], keep) if k]
            self.data['train_labels'] = train_labels[keep]
            print(f"--no-background: kept {keep.sum()} / {len(keep)} training samples (dropped {(~keep).sum()} background)")

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
            self.target_data = target_loader.load_data(use_multilabel=True, validation_share=0.2)
            print(f"Loaded {len(self.target_data['train_filenames'])} target domain samples")

        # Use dimensions from spectrogram params (single source of truth)
        # Avoid duplicating DEFAULT_FREQ_BINS vs SPECTROGRAM_PARAMS['nfilters']
        self.img_height = cfg.model.freq_bins if cfg.model.freq_bins is not None else config.SPECTROGRAM_PARAMS['nfilters']  # Frequency bins (height)
        self.img_width = cfg.model.time_bins if cfg.model.time_bins is not None else config.DEFAULT_TIME_BINS   # Time bins (width)
        
        # Create data loaders with config defaults
        # Use more workers and prefetch for faster GPU utilization
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader, self.val_loader = create_data_loaders(
            self.data, self.batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
            cropping_mode='random', noise_ratio=self.noise_ratio,
            spec_transform=self.spec_transform,
            num_workers=num_workers, width_downsizing=None, mixup_alpha=self.mixup_alpha,
            use_class_balancing=False, bg_subtract=self.bg_subtract,
            median_filter=self.median_filter,
            use_temporal_roll=self.use_temporal_roll,
            mixup_mode=self.mixup_mode,
            noise_mode=self.noise_mode,
            background_prob=self.background_prob,
            ast_channel_dir=self.ast_channel_dir,
            use_deltas=self.use_deltas,
        )
        
        # Create target domain data loader for DANN
        if self.use_dann:
            self.target_train_loader, _ = create_data_loaders(
                self.target_data, self.batch_size, self.img_height, self.img_width, config.DEFAULT_CHANNELS,
                cropping_mode='random', noise_ratio=0.0,  # No noise augmentation for target
                spec_transform=self.spec_transform,
                num_workers=num_workers, width_downsizing=None, mixup_alpha=0.0,  # No mixup for target
                use_class_balancing=False, bg_subtract=self.bg_subtract,
                median_filter=self.median_filter,
                use_temporal_roll=self.use_temporal_roll,
                mixup_mode='mixup',
                noise_mode='full',
                background_prob=0.0
            )
            print(f"Created target domain data loader with {len(self.target_train_loader)} batches")
        
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _background_weighted_loss(self, criterion, output, target):
        """
        Compute loss with background-sample rebalancing.

        criterion must return per-element loss [B, C] (reduction='none').
        Background samples (all-zero target rows) are scaled by self.background_weight
        so their total gradient contribution equals that of labelled samples.
        """
        per_element = criterion(output, target)          # [B, C]
        per_sample = per_element.mean(dim=1)             # [B]
        if self.background_weight != 1.0:
            is_bg = (target.sum(dim=1) == 0).float()     # [B]
            weights = 1.0 + is_bg * (self.background_weight - 1.0)
            return (per_sample * weights).mean()
        return per_sample.mean()

    def _subsample_per_class(self, data, max_per_class):
        """Randomly subsample training data so each class has at most max_per_class examples.

        Background (all-zero) samples are left untouched.  For multi-label samples the
        union of per-class selections is kept, so a multi-label sample survives if it is
        selected for *any* of its active classes.
        """
        import random as _random
        train_filenames = list(data['train_filenames'])
        train_labels = np.array(data['train_labels'], dtype=np.float32)
        n_samples, n_classes = train_labels.shape

        rng = _random.Random(self.seed if self.seed is not None else 42)

        is_labeled = train_labels.sum(axis=1) > 0
        bg_indices = [i for i in range(n_samples) if not is_labeled[i]]
        labeled_indices = [i for i in range(n_samples) if is_labeled[i]]

        # Group labeled samples by class
        class_to_indices = {}
        for idx in labeled_indices:
            for c in range(n_classes):
                if train_labels[idx, c] > 0:
                    if c not in class_to_indices:
                        class_to_indices[c] = []
                    class_to_indices[c].append(idx)

        kept_labeled = set()
        for c, indices in class_to_indices.items():
            if len(indices) > max_per_class:
                indices = rng.sample(indices, max_per_class)
            kept_labeled.update(indices)

        kept_labeled = sorted(kept_labeled)
        all_kept = sorted(bg_indices + kept_labeled)

        data = dict(data)
        data['train_filenames'] = [train_filenames[i] for i in all_kept]
        data['train_labels'] = train_labels[all_kept]

        class_names = data.get('class_names', [])
        new_labels = data['train_labels']
        print(f"--max-samples-per-class {max_per_class}: {n_samples} → {len(all_kept)} training samples "
              f"({len(bg_indices)} background kept)")
        for c in range(n_classes):
            cnt = int(new_labels[:, c].sum())
            name = class_names[c] if c < len(class_names) else str(c)
            print(f"    {name}: {cnt}")
        return data

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
        # --eval-only: skip training entirely, reload saved model, run test evaluation
        if self.eval_only:
            return self._eval_only()

        # CRITICAL: Verify CUDA is actually available before starting
        if not torch.cuda.is_available():
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            raise RuntimeError(
                f"CUDA not available! Cannot train on GPU.\n"
                f"  CUDA_VISIBLE_DEVICES = {cuda_visible}\n"
                f"  torch.cuda.is_available() = False\n"
                f"  Possible causes:\n"
                f"    - CUDA_VISIBLE_DEVICES points to invalid/busy GPU\n"
                f"    - All GPUs are already in use\n"
                f"    - CUDA drivers not loaded\n"
                f"  Solution: Check GPU availability with 'nvidia-smi' and adjust CUDA_VISIBLE_DEVICES"
            )
        
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
                freeze_stages=self.freeze_stages,
                use_cnn_adapter=self.use_cnn_adapter,
                use_sed_head=self.use_sed_head,
                use_gated_head=self.use_gated_head,
                in_chans=self.in_chans,
            ).to(self.device)
        else:
            print("Creating AST model (multilabel)...")
            model = AST(self.num_classes, input_size=input_size, dropout=self.dropout, 
                       use_reconstruction=self.use_reconstruction, use_adapters=self.use_adapters,
                       per_chunk_norm=self.per_chunk_norm,
                       use_cnn_adapter=self.use_cnn_adapter,
                       use_sed_head=self.use_sed_head).to(self.device)
            
            # AST-specific: Interpolate positional embeddings
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

        # Resume from a prior checkpoint (full state_dict load, overrides any pretrained init)
        resume_checkpoint = getattr(self.cfg.training, 'resume_checkpoint', None)
        if resume_checkpoint:
            state = torch.load(resume_checkpoint, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            print(f"Resumed from checkpoint: {resume_checkpoint}")

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
        
        # Optimizer and LR schedule
        # Use AdamW (Adam with decoupled weight decay) for better regularization
        # Differential LR for RegNet: pretrained backbone gets base LR, randomly-init classifier gets 10x
        if self.model_type == 'regnet':
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and 'cnn_adapter' not in n and 'gated_head' not in n and p.requires_grad]
            classifier_params = [p for n, p in model.named_parameters() if ('classifier' in n or 'gated_head' in n) and p.requires_grad]
            adapter_params = [p for n, p in model.named_parameters() if 'cnn_adapter' in n and p.requires_grad]
            param_groups = [
                {'params': backbone_params, 'lr': self.learning_rate},
                {'params': classifier_params, 'lr': self.learning_rate * 10}
            ]
            if adapter_params:
                param_groups.append({'params': adapter_params, 'lr': self.learning_rate * 10})
            print(f"  Backbone LR: {self.learning_rate:.1e}, Classifier LR: {self.learning_rate * 10:.1e}")
        else:
            adapter_params = [p for n, p in model.named_parameters() if 'cnn_adapter' in n]
            other_params = [p for n, p in model.named_parameters() if 'cnn_adapter' not in n]
            if adapter_params:
                param_groups = [
                    {'params': other_params, 'lr': self.learning_rate},
                    {'params': adapter_params, 'lr': self.learning_rate * 10}
                ]
            else:
                param_groups = [{'params': model.parameters(), 'lr': self.learning_rate}]
        if self.use_dann:
            param_groups.append({'params': self.domain_classifier.parameters(), 'lr': self.learning_rate * 0.1})
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7)
        print(f"Using AdamW optimizer with CosineAnnealingLR scheduler")

        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        
        # Use BCE-based loss (always multilabel)
        pos_weight = None
        if self.use_class_weights:
            pos_weight = self._compute_class_weights()
            print(f"Using class-weighted loss")
            print(f"  Weight range: {pos_weight.min().item():.2f} - {pos_weight.max().item():.2f}")
        if self.kbird_prior > 0:
            # Max-k normalisation: compute standard sigmoid probabilities, then if
            # their sum exceeds k scale all of them down proportionally so the total
            # equals k.  This encodes the prior that at most ~k species are active
            # per segment without forcing class competition (unlike k*softmax which
            # uses a softmax and makes classes compete for probability mass).
            # pos_weight is not used in this mode.
            _k = self.kbird_prior
            def criterion(out, tgt, _k=_k):
                # Disable autocast: F.binary_cross_entropy is forbidden inside
                # an autocast context regardless of input dtype.
                with torch.amp.autocast('cuda', enabled=False):
                    p = torch.sigmoid(out.float())                     # [B, C]
                    total = p.sum(dim=1, keepdim=True).clamp(min=1e-7) # [B, 1]
                    scale = (total / _k).clamp(min=1.0)  # >=1 when over-predicting
                    # nan_to_num before clamp: torch.clamp passes NaN through unchanged
                    # which would trigger the CUDA-side assertion in binary_cross_entropy.
                    p_norm = (p / scale).nan_to_num(nan=0.5).clamp(1e-7, 1.0 - 1e-7)
                    return F.binary_cross_entropy(p_norm, tgt.float(), reduction='none')
            print(f"Using k-bird prior (max-{self.kbird_prior:.1f} normalisation)")
        elif self.use_asl:
            criterion = AsymmetricLoss(
                gamma_neg=self.asl_gamma_neg,
                gamma_pos=self.asl_gamma_pos,
                margin=self.asl_margin,
                pos_weight=pos_weight,
                reduction='none',
            )
            print(f"Using Asymmetric Loss (gamma_neg={self.asl_gamma_neg}, gamma_pos={self.asl_gamma_pos}, margin={self.asl_margin})")
        elif self.bce_smoothing and self.bce_smoothing > 0.0:
            criterion = SmoothBCEWithLogitsLoss(epsilon=self.bce_smoothing, pos_weight=pos_weight, reduction='none')
            print(f"Applying BCE target smoothing (epsilon={self.bce_smoothing:.3f})")
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        
        # Training
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        print(f"Starting training for {self.max_epochs} epochs...")
        
        best_val_acc = -1.0   # -1 so even 0.0 F1 at epoch 1 triggers a save
        best_epoch = -1
        epochs_without_improvement = 0
        
        # Divergence detection
        initial_loss = None
        divergence_threshold = 10.0  # Stop if loss > initial_loss * threshold
        
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
                    
                    # Extract data from batches
                    source_data, source_target = source_batch
                    source_data = source_data.to(self.device, non_blocking=True)
                    source_target = source_target.to(self.device, non_blocking=True)
                    
                    target_data, _ = target_batch
                    target_data = target_data.to(self.device, non_blocking=True)
                    
                    batch_size_s = source_data.size(0)
                    batch_size_t = target_data.size(0)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass for source (for classification)
                    source_output = model(source_data)
                    source_features = model.get_features(source_data)
                    target_features = model.get_features(target_data)
                    
                    # Classification loss (only on source domain with labels - always multilabel)
                    source_output = torch.nan_to_num(source_output, nan=0.0, posinf=80.0, neginf=-80.0)
                    class_loss = self._background_weighted_loss(criterion, source_output, source_target)
                    
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
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    total_class_loss += class_loss.item()
                    total_domain_loss += domain_loss.item()
                    
                    # Accuracy tracking (only on source labeled data - always multilabel)
                    with torch.no_grad():
                        preds = (self._get_probs(source_output) > 0.5).float()
                        all_train_preds.append(preds.cpu().numpy())
                        all_train_targets.append(source_target.cpu().numpy())
                
                else:
                    # Standard training (no DANN)
                    batch_idx, batch = batch_idx
                    
                    # Standard data format
                    data, target = batch
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        if self.use_reconstruction:
                            output, recon = model(data)
                        elif self.use_sed_head:
                            output, frame_logits = model(data)
                        elif self.use_gated_head:
                            output, gate_logit = model(data)
                        else:
                            output = model(data)

                            if not torch.isfinite(output).all():
                                print(f"\nFUCKED AT epoch={epoch+1}, batch={batch_idx}")
                                print("input finite:", torch.isfinite(data).all().item())
                                print("output NaN:", torch.isnan(output).sum().item())
                                print("output Inf:", torch.isinf(output).sum().item())

                                for name, p in model.named_parameters():
                                    if not torch.isfinite(p).all():
                                        print("BAD WEIGHT:", name)
                                        break

                                for name, p in model.named_parameters():
                                    if p.grad is not None and not torch.isfinite(p.grad).all():
                                        print("BAD GRADIENT:", name)
                                        break

                                raise RuntimeError("NON-FINITE MODEL OUTPUT")

                        # output = torch.nan_to_num(output, nan=0.0, posinf=80.0, neginf=-80.0)
                        loss = self._background_weighted_loss(criterion, output, target)

                        if self.use_reconstruction:
                            target_spec = data.squeeze(1) if data.dim() == 4 else data
                            target_spec = (target_spec - config.AST_MEAN) / config.AST_STD
                            recon_loss = F.mse_loss(recon, target_spec)
                            loss = loss + self.recon_weight * recon_loss

                        if self.use_sed_head:
                            # Two-way loss: clip-level (attention-weighted) + segment mean.
                            # Mirrors Kaytoo's BCEFocal2WayLoss which supervises both the
                            # final attention-pooled logit and the raw per-frame mean logit,
                            # giving the attention weights a gradient signal to localise birds.
                            frame_logits = torch.nan_to_num(frame_logits, nan=0.0, posinf=80.0, neginf=-80.0)
                            aux_loss = self._background_weighted_loss(criterion, frame_logits, target)
                            loss = loss + aux_loss

                        if self.use_gated_head:
                            is_bird = (target.sum(dim=1) > 0).float()
                            gate_loss = F.binary_cross_entropy_with_logits(gate_logit, is_bird)
                            loss = loss + self.gate_loss_weight * gate_loss
                    
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
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()

                    # For metrics (always multilabel)
                    target_hard = target.round()
                    pred = (self._get_probs(output) > 0.5).float()
                    all_train_preds.append(pred.cpu().numpy())
                    all_train_targets.append(target_hard.cpu().numpy())
            
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
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # For multi-label F1 calculation
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for batch in self.val_loader:
                    # Standard data format
                    data, target = batch
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    if self.use_reconstruction:
                        output, _ = model(data)
                    else:
                        output = model(data)
                    
                    # Always multilabel
                    val_loss += criterion(output, target).mean().item()
                    pred = (self._get_probs(output) > 0.5).float()
                    all_val_preds.append(pred.cpu().numpy())
                    all_val_targets.append(target.cpu().numpy())
            
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
            
            # Calculate metrics (always multilabel)
            train_metrics = compute_multilabel_epoch_metrics(all_train_preds, all_train_targets)
            val_metrics = compute_multilabel_epoch_metrics(all_val_preds, all_val_targets)
            train_acc = train_metrics['macro_f1']

            # Early stopping uses macro F1 on LABELLED val samples only.
            # Background samples (all-zero targets) in the val set penalise false
            # positives and bias selection toward over-conservative models — exactly
            # the opposite of what acc_labelled measures.  Filter them out here.
            all_val_preds_arr  = np.vstack(all_val_preds)
            all_val_targets_arr = np.vstack(all_val_targets)
            labelled_mask = all_val_targets_arr.sum(axis=1) > 0
            if labelled_mask.sum() > 0:
                val_metrics_labelled = compute_multilabel_epoch_metrics(
                    [all_val_preds_arr[labelled_mask]],
                    [all_val_targets_arr[labelled_mask]]
                )
                val_acc = val_metrics_labelled['macro_f1']
            else:
                val_acc = val_metrics['macro_f1']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # ABORT if model weights contain NaN
            if any(torch.isnan(p).any() for p in model.parameters()):
                print(f"\n❌ FATAL: NaN in weights (epoch {epoch+1}). ABORTING.\n")
                return

            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                self._save_model(model, best=True)
            else:
                epochs_without_improvement += 1
                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    print(f"\n  Early stopping: no improvement for {self.patience} epochs")
                    print(f"  Best macro-F1: {best_val_acc:.4f} at epoch {best_epoch}")
                    break
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s)')
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
            print('-' * 60)
        
        # Save final model (only if we actually trained)
        if self.max_epochs > 0:
            self._save_model(model)
            self._save_history(train_losses, val_losses, train_accs, val_accs)
            self._plot_history(train_losses, val_losses, train_accs, val_accs)
        # Evaluate on validation set (using best checkpoint)
        best_path = os.path.join(self.output_folder, f'{self.model_type}_model_best.pt')
        print(f"\nDEBUG: Loading best model from: {best_path}")
        print(f"DEBUG: File exists: {os.path.exists(best_path)}")
        if os.path.exists(best_path):
            import hashlib
            with open(best_path, 'rb') as f:
                model_hash = hashlib.md5(f.read()).hexdigest()[:16]
            print(f"DEBUG: Model checkpoint MD5: {model_hash}")
            state_dict = torch.load(best_path, map_location=self.device)
            model.load_state_dict(state_dict)
        evaluator = EvaluationManager(self.output_folder, self.data['class_names'], is_multilabel=True)
        evaluator.evaluate_model(model, self.val_loader, f'{self.model_type}_model', device=self.device)

        # Save raw validation predictions so per-class thresholds can be tuned
        # over ALL training classes (not just the 9 matched test-set species).
        val_data_for_pred = {
            'train_filenames': self.data['test_filenames'],
            'train_labels':    np.array(self.data['test_labels'], dtype=np.float32),
            'class_names':     self.data['class_names'],
        }
        self._save_test_predictions(model, self.val_loader, val_data_for_pred, 'val')

        print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"\n{'='*60}")
        print(f"✓ TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Model saved to: {self.output_folder}")
        
        best_val_loss = min(val_losses)
        
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

    def _pick_free_gpu(self):
        """Return the index of the GPU with the lowest memory usage."""
        return pick_free_gpu()

    def _get_probs(self, output):
        """Convert raw logits to probabilities using the configured activation.

        When kbird_prior > 0, sigmoid probabilities are normalised so their sum
        never exceeds k, encoding the prior that at most ~k species are active
        per segment.  Otherwise falls back to standard per-class sigmoid.
        """
        if self.kbird_prior > 0:
            p = torch.sigmoid(output)
            total = p.sum(dim=1, keepdim=True).clamp(min=1e-7)
            scale = (total / self.kbird_prior).clamp(min=1.0)
            return (p / scale).clamp(0.0, 1.0)
        return torch.sigmoid(output)

    def _eval_only(self):
        """Load saved model from output_folder and run test evaluation only."""
        input_size = (self.img_height, self.img_width)

        # self.data and self.num_classes are already set by __init__; don't reload.
        print(f"--eval-only: {self.num_classes} classes from {self.data_folder}")

        # Build model skeleton (same arch, no pretrained BirdCLEF weights)
        if self.model_type == 'regnet':
            model = RegNetModel(
                self.num_classes,
                pretrained_path=None,
                model_name=self.model_name,
                use_sed_head=self.use_sed_head,
                use_gated_head=self.use_gated_head,
                in_chans=self.in_chans,
            ).to(self.device)
        else:
            model = AST(self.num_classes, input_size=input_size, dropout=self.dropout,
                       use_reconstruction=self.use_reconstruction, use_adapters=self.use_adapters,
                       per_chunk_norm=self.per_chunk_norm,
                       use_cnn_adapter=self.use_cnn_adapter,
                       use_sed_head=self.use_sed_head).to(self.device)
            model.interpolate_pos_embed(input_size)

        # Load the saved weights: explicit --checkpoint overrides folder lookup
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"--checkpoint: file not found: {ckpt_path}")
        else:
            best_path  = os.path.join(self.output_folder, f'{self.model_type}_model_best.pt')
            final_path = os.path.join(self.output_folder, f'{self.model_type}_model.pt')

            if os.path.exists(best_path) and os.path.exists(final_path):
                # Verify the best checkpoint was saved with the same class count as the
                # current model.  If the dataset was rebuilt since the last training run
                # the best checkpoint may have a stale output-layer shape, which would
                # cause load_state_dict to succeed (if num_classes happens to match the
                # stale count) but produce garbage/NaN predictions at inference time.
                sd_best = torch.load(best_path, map_location='cpu', weights_only=True)
                classifier_key = next((k for k in sd_best if 'classifier' in k and 'weight' in k), None)
                if classifier_key is not None:
                    ckpt_classes = sd_best[classifier_key].shape[0]
                    if ckpt_classes != self.num_classes:
                        print(f"  WARNING: best checkpoint has {ckpt_classes} output classes "
                              f"but current model has {self.num_classes}. "
                              f"Using final checkpoint instead.")
                        ckpt_path = final_path
                    else:
                        ckpt_path = best_path
                else:
                    ckpt_path = best_path
            elif os.path.exists(best_path):
                ckpt_path = best_path
            elif os.path.exists(final_path):
                ckpt_path = final_path
            else:
                raise FileNotFoundError(
                    f"--eval-only: no saved model found in {self.output_folder}\n"
                    f"  Looked for: {best_path}\n"
                    f"              {final_path}"
                )

        print(f"--eval-only: loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if not self.test_folder:
            print("WARNING: --eval-only with no --test-folder specified; nothing to evaluate.")
            return {}

        eval_out = self.output_folder
        os.makedirs(eval_out, exist_ok=True)
        evaluator = EvaluationManager(eval_out, self.data['class_names'], is_multilabel=True)

        print(f"\n{'='*60}")
        print(f"Evaluating on: {self.test_folder}")
        print(f"{'='*60}")
        test_loader1 = DataLoader(self.test_folder, noise_folder=None)
        test_data1_orig = test_loader1.load_data(use_multilabel=True, validation_share=0.0)
        test_data1 = _remap_labels_to_train_space(test_data1_orig, self.data)
        test_dataset1 = SpectrogramDataset(
            test_data1['train_filenames'], test_data1['train_labels'],
            self.img_height, self.img_width, config.DEFAULT_CHANNELS, 'center',
            noise_filenames=None, noise_ratio=0.0, spec_transform=self.spec_transform,
            training=False, width_downsizing=None, bg_subtract=self.bg_subtract,
            median_filter=self.median_filter, use_temporal_roll=False,
            noise_mode='full', background_prob=0.0,
            ast_channel_dir=self.ast_channel_dir, use_deltas=self.use_deltas,
        )
        test_loader_obj1 = torch.utils.data.DataLoader(
            test_dataset1, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=torch.cuda.is_available()
        )
        test_name = Path(self.test_folder).parent.parent.name + "__" + Path(self.test_folder).parent.name
        print(f"  samples={len(test_dataset1)}")
        evaluator.evaluate_model(model, test_loader_obj1, f'{self.model_type}_test_{test_name}', device=self.device)
        self._save_predictions_to(model, test_loader_obj1, test_data1, test_name, eval_out, orig_test_data=test_data1_orig)

        # Save raw validation predictions for threshold tuning over all classes.
        print(f"\n{'='*60}")
        print(f"Saving validation predictions (all {self.num_classes} classes)")
        print(f"{'='*60}")
        val_data_for_pred = {
            'train_filenames': self.data['test_filenames'],
            'train_labels':    np.array(self.data['test_labels'], dtype=np.float32),
            'class_names':     self.data['class_names'],
        }
        val_dataset = SpectrogramDataset(
            val_data_for_pred['train_filenames'], val_data_for_pred['train_labels'],
            self.img_height, self.img_width, config.DEFAULT_CHANNELS, 'center',
            noise_filenames=None, noise_ratio=0.0, spec_transform=self.spec_transform,
            training=False, width_downsizing=None, bg_subtract=self.bg_subtract,
            median_filter=self.median_filter, use_temporal_roll=False,
            noise_mode='full', background_prob=0.0,
            ast_channel_dir=self.ast_channel_dir, use_deltas=self.use_deltas,
        )
        val_loader_obj = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=torch.cuda.is_available()
        )
        self._save_test_predictions(model, val_loader_obj, val_data_for_pred, 'val', out_dir=eval_out)

        # Write a compact summary JSON for downstream scripts and rerun tracking.
        result_data = {
            'name': Path(self.output_folder).name,
            'type': self.model_type,
            'seed': getattr(self, 'seed', None),
            'output_folder': str(self.output_folder),
            'test_name': test_name,
            'status': 'completed',
        }

        # Best-effort accuracy from the report written above.
        report_file = Path(eval_out) / f'{self.model_type}_test_{test_name}_multilabel_report.json'
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            if 'exact_match_accuracy' in report:
                result_data['exact_match_accuracy'] = report['exact_match_accuracy'] * 100

        result_json_path = Path(eval_out) / f'result_{test_name}.json'
        with open(result_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"Saved summary JSON to {result_json_path}")

        print(f"\n--eval-only: results saved to {eval_out}")
        return {}

    def _dataset_spectrogram_config(self):
        """Return the spectrogram settings recorded in the training dataset's
        labels.json, as a dict ready to merge into the saved model config.

        The dataset builders write SpectrogramProcessor.get_metadata() into
        labels.json under 'spectrogram_config'.  Returns {} for older datasets
        that predate that, in which case config.py's defaults stand and the
        caller is warned.
        """
        labels_path = os.path.join(self.data_folder, 'labels.json')
        try:
            with open(labels_path) as f:
                meta = json.load(f).get('spectrogram_config')
        except (OSError, ValueError) as e:
            print(f"WARNING: could not read {labels_path} ({e}); model config will "
                  f"use config.py spectrogram defaults, which may not match the data.")
            return {}

        if not meta:
            print(f"WARNING: {labels_path} has no 'spectrogram_config' (dataset built "
                  f"before that was recorded). Falling back to config.py defaults "
                  f"(sgType={config.SPECTROGRAM_PARAMS['sgType']}) — verify this matches "
                  f"how the data was actually built before deploying the model.")
            return {}

        out = {}
        for key in ('sample_rate', 'window_seconds', 'hop_seconds',
                    'freq_bins', 'spectrogram_params'):
            if key in meta:
                out[key] = meta[key]
        print(f"Model config: spectrogram settings taken from the dataset "
              f"(sgType={out.get('spectrogram_params', {}).get('sgType', '?')}, "
              f"sgScale={out.get('spectrogram_params', {}).get('sgScale', '?')})")
        return out

    def _save_model(self, model, best=False):
        filename = f'{self.model_type}_model_best.pt' if best else f'{self.model_type}_model.pt'
        torch.save(model.state_dict(), os.path.join(self.output_folder, filename))
        
        # Always save configuration for model deployment
        model_config = config.get_model_config()

        # Override the spectrogram settings with the ones the DATA was actually
        # built with. config.get_model_config() returns config.py's static
        # defaults (notably sgType='Reassigned'), which silently disagree with
        # any dataset built with a different --spec-type. Inference reads this
        # file to reproduce the training spectrogram, so a stale sgType here
        # makes the deployed model score real calls against the wrong transform.
        model_config.update(self._dataset_spectrogram_config())

        # Override with actual dimensions used during training
        model_config['freq_bins'] = self.img_height
        model_config['time_bins'] = self.img_width
        
        # Add model-specific information
        if self.model_type == 'regnet':
            model_config['model_type'] = 'RegNet'
            model_config['model_name'] = self.model_name
        else:
            model_config['model_type'] = 'AST'
        model_config['num_classes'] = model.num_classes
        model_config['multilabel'] = True
        model_config['class_names'] = self.data['class_names']
        model_config['use_reconstruction'] = self.use_reconstruction
        model_config['use_cnn_adapter'] = self.use_cnn_adapter
        model_config['use_gated_head'] = self.use_gated_head
        
        # Save ALL augmentation/normalization parameters for inference consistency
        model_config['spec_transform'] = self.spec_transform
        model_config['bg_subtract'] = self.bg_subtract
        model_config['median_filter'] = self.median_filter
        
        # Save to JSON
        config_path = os.path.join(self.output_folder, f'{self.model_type}_model_config.json')
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
        ax2.set_title('Multi-Label Macro F1')
        ax2.legend()
        
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'))
        plt.close()
    
    def _save_predictions_to(self, model, test_loader, test_data, test_name, out_dir, orig_test_data=None):
        """Like _save_test_predictions but writes to an explicit directory."""
        self._save_test_predictions(model, test_loader, test_data, test_name, out_dir=out_dir, orig_test_data=orig_test_data)

    def _save_test_predictions(self, model, test_loader, test_data, test_name, out_dir=None, orig_test_data=None):
        """Save test predictions (per-class probabilities + ground truth) to CSV.

        Prediction columns use the model's training class vocabulary.
        Ground-truth (true_*) columns use the ORIGINAL test-set class vocabulary
        (from orig_test_data when provided, otherwise falls back to training vocab).
        This ensures that when a model is evaluated on a test set with a different
        vocabulary, all test-set classes appear as true_ columns and the F1/accuracy
        metrics in analyze_all_results.py are computed over the full test vocabulary.
        """
        import csv

        model.eval()
        all_probs = []
        all_labels = []
        all_filenames = []

        sample_idx = 0
        filenames_list = test_data['train_filenames']

        _nan_warned = False
        with torch.no_grad():
            for batch in test_loader:
                data, targets = batch
                batch_size = data.size(0)
                data = data.to(self.device)

                # Sanity-check the INPUT before the forward pass.
                if not _nan_warned and (torch.isnan(data).any() or torch.isinf(data).any()):
                    print(f"  ⚠️  NaN/Inf in INPUT spectrogram tensor! "
                          f"nan={torch.isnan(data).sum().item()} "
                          f"inf={torch.isinf(data).sum().item()} "
                          f"shape={tuple(data.shape)} "
                          f"min={data[torch.isfinite(data)].min().item():.3f} "
                          f"max={data[torch.isfinite(data)].max().item():.3f}")

                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]

                # Sanity-check the raw logits before applying _get_probs.
                if not _nan_warned and torch.isnan(output).any():
                    _nan_warned = True
                    n_nan = torch.isnan(output).sum().item()
                    finite = output[torch.isfinite(output)]
                    print(f"  ⚠️  NaN in model OUTPUT logits! "
                          f"nan_count={n_nan}/{output.numel()} "
                          f"finite_range=[{finite.min().item():.3f}, {finite.max().item():.3f}]")
                    print(f"     Input range: min={data.min().item():.3f} max={data.max().item():.3f}")
                    # Check per-layer BN running stats for obvious corruption
                    for name, mod in model.named_modules():
                        if hasattr(mod, 'running_var'):
                            bad = (mod.running_var < 0).sum().item()
                            zero = (mod.running_var == 0).sum().item()
                            if bad or zero:
                                print(f"     BN layer '{name}': "
                                      f"{bad} negative running_var, {zero} zero running_var")

                probs = self._get_probs(output).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(targets.numpy())

                batch_filenames = filenames_list[sample_idx: sample_idx + batch_size]
                all_filenames.extend(batch_filenames)
                sample_idx += batch_size

        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        pred_class_names = test_data['class_names']  # model's training vocabulary

        # Determine ground-truth vocabulary and labels for true_ columns.
        # Use orig_test_data (test-set vocabulary) when available so that
        # test classes absent from the model's training vocab still appear.
        if orig_test_data is not None and list(orig_test_data.get('class_names', [])) != list(pred_class_names):
            true_class_names = orig_test_data['class_names']
            true_labels_arr  = orig_test_data['train_labels']  # (N, n_test_classes)
            n_test_only  = sum(1 for c in true_class_names if c not in set(pred_class_names))
            n_train_only = sum(1 for c in pred_class_names  if c not in set(true_class_names))
            print(f"  [predictions CSV] test-only classes in true_ cols: {n_test_only}, "
                  f"train-only classes (pred cols only, no true_): {n_train_only}")
        else:
            true_class_names = pred_class_names
            true_labels_arr  = all_labels

        csv_path = os.path.join(out_dir or self.output_folder, f'predictions_{test_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename'] + pred_class_names + [f'true_{c}' for c in true_class_names])
            for filename, row_probs, row_labels in zip(all_filenames, all_probs, true_labels_arr):
                writer.writerow(
                    [filename]
                    + [f"{p:.6f}" for p in row_probs]
                    + [int(l) for l in row_labels]
                )

        print(f"Saved {len(all_filenames)} predictions to {csv_path}")

