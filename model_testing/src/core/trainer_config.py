"""
Configuration classes for model training.
Replaces the insane 30+ parameter constructor.
"""

from dataclasses import dataclass
from typing import Optional
from . import config


@dataclass
class TrainingConfig:
    """Core training parameters."""
    data_folder: str
    output_folder: str
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = config.DEFAULT_WEIGHT_DECAY
    patience: int = 0
    seed: Optional[int] = None
    resume_checkpoint: Optional[str] = None
    max_samples_per_class: Optional[int] = None  # Subsample training data per class (scaling experiments)


@dataclass
class ModelConfig:
    """Model architecture and optimization. Always uses multilabel classification."""
    model_type: str = 'ast'  # 'ast' or 'regnet'
    model_name: Optional[str] = None  # e.g., 'regnety_008' for regnet
    freq_bins: Optional[int] = None
    time_bins: Optional[int] = None
    dropout: float = config.DEFAULT_DROPOUT
    use_reconstruction: bool = False
    recon_weight: float = 0.1
    use_adapters: bool = False
    freeze_layers: Optional[int] = None
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False
    freeze_stages: int = 0
    use_cnn_adapter: bool = False
    use_sed_head: bool = False
    use_gated_head: bool = False
    ast_channel_dir: Optional[str] = None   # Pre-computed AST attention maps directory
    in_chans: int = 1                       # Input channels (1 = spec only, 2 = spec + AST attn)


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    mixup_alpha: float = config.DEFAULT_MIXUP_ALPHA
    noise_ratio: float = config.DEFAULT_NOISE_RATIO
    noise_folder: Optional[str] = None
    noise_as_samples: bool = False
    max_noise_samples: Optional[int] = None
    use_temporal_roll: bool = config.DEFAULT_TEMPORAL_ROLL
    bg_subtract: bool = False  # Background subtraction normalization (independent)
    apply_reverb: bool = False  # Apply reverberation to loud noises (independent)
    median_filter: bool = False  # Temporal median filtering (independent)
    no_background: bool = False  # Drop all-zero (background) training samples
    use_deltas: bool = False  # Add delta + delta-delta channels (3-ch input; encodes rate-of-change)
    per_chunk_norm: bool = False
    spec_transform: str = 'Log'
    mixup_mode: str = 'mixup'
    noise_mode: str = 'full'
    validation_split: float = 0.2
    background_prob: float = 0.0


@dataclass
class LossConfig:
    """Loss function configuration."""
    use_class_weights: bool = False
    pos_weight_cap: float = 20.0
    bce_smoothing: float = 0.0
    use_asl: bool = False
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_margin: float = 0.05
    rebalance_background: bool = True  # Down-weight background samples so they equal labelled contribution
    gate_loss_weight: float = 1.0
    kbird_prior: float = 0.0   # When > 0, normalise sigmoid probs so sum <= k (soft species-count constraint)


@dataclass
class DomainAdaptationConfig:
    """Domain adaptation (DANN) settings."""
    use_dann: bool = False
    target_folder: Optional[str] = None
    lambda_domain: float = 0.3


@dataclass
class EvaluationConfig:
    """Test/evaluation settings."""
    test_folder: Optional[str] = None
    visualize_attention: bool = False
    viz_samples: int = 3
    eval_only: bool = False
    checkpoint: Optional[str] = None


@dataclass
class TrainerConfig:
    """Complete trainer configuration. Groups all related settings."""
    training: TrainingConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    loss: LossConfig
    domain_adaptation: DomainAdaptationConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_args(cls, args):
        """Create from argparse Namespace (for backward compatibility)."""
        return cls(
            training=TrainingConfig(
                data_folder=args.data_folder,
                output_folder=args.output_folder,
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=getattr(args, 'weight_decay', config.DEFAULT_WEIGHT_DECAY),
                patience=args.patience,
                seed=getattr(args, 'seed', None),
                resume_checkpoint=getattr(args, 'resume_checkpoint', None),
                max_samples_per_class=getattr(args, 'max_samples_per_class', None),
            ),
            model=ModelConfig(
                model_type=getattr(args, 'model_type', 'ast'),
                model_name=getattr(args, 'model_name', None),
                freq_bins=getattr(args, 'freq_bins', None),
                time_bins=getattr(args, 'time_bins', None),
                dropout=getattr(args, 'dropout', config.DEFAULT_DROPOUT),
                use_reconstruction=getattr(args, 'reconstruct', False),
                recon_weight=getattr(args, 'recon_weight', 0.1),
                use_adapters=getattr(args, 'adapters', False),
                freeze_layers=getattr(args, 'freeze_layers', None),
                pretrained_path=getattr(args, 'pretrained', None),
                freeze_backbone=getattr(args, 'freeze_backbone', False),
                freeze_stages=getattr(args, 'freeze_stages', 0),
                use_cnn_adapter=getattr(args, 'cnn_adapter', False),
                use_sed_head=getattr(args, 'sed_head', False),
                use_gated_head=getattr(args, 'gated_head', False),
                ast_channel_dir=getattr(args, 'ast_channel_dir', None),
                in_chans=3 if getattr(args, 'use_deltas', False) else (2 if getattr(args, 'ast_channel_dir', None) else 1),
            ),
            augmentation=AugmentationConfig(
                mixup_alpha=args.mixup,
                noise_ratio=getattr(args, 'noise', config.DEFAULT_NOISE_RATIO),
                noise_folder=getattr(args, 'noise_folder', None),
                noise_as_samples=getattr(args, 'noise_as_samples', False),
                max_noise_samples=getattr(args, 'max_noise_samples', None),
                use_temporal_roll=getattr(args, 'temporal_roll', config.DEFAULT_TEMPORAL_ROLL),
                bg_subtract=getattr(args, 'bg_subtract', False),
                apply_reverb=getattr(args, 'apply_reverb', False),
                median_filter=getattr(args, 'median_filter', False),
                no_background=getattr(args, 'no_background', False),
                use_deltas=getattr(args, 'use_deltas', False),
                per_chunk_norm=getattr(args, 'per_chunk_norm', False),
                spec_transform=getattr(args, 'spec_transform', 'Log'),
                mixup_mode=getattr(args, 'mixup_mode', 'mixup'),
                noise_mode=getattr(args, 'noise_mode', 'full'),
                validation_split=getattr(args, 'validation_split', 0.2),
                background_prob=getattr(args, 'background_prob', 0.0)
            ),
            loss=LossConfig(
                use_class_weights=getattr(args, 'class_weights', False),
                pos_weight_cap=getattr(args, 'pos_weight_cap', 20.0),
                bce_smoothing=getattr(args, 'bce_smoothing', config.DEFAULT_BCE_SMOOTHING),
                use_asl=getattr(args, 'use_asl', False),
                asl_gamma_neg=getattr(args, 'asl_gamma_neg', 4.0),
                asl_gamma_pos=getattr(args, 'asl_gamma_pos', 0.0),
                asl_margin=getattr(args, 'asl_margin', 0.05),
                rebalance_background=getattr(args, 'rebalance_background', True),
                gate_loss_weight=getattr(args, 'gate_loss_weight', 1.0),
                kbird_prior=getattr(args, 'kbird_prior', 0.0),
            ),
            domain_adaptation=DomainAdaptationConfig(
                use_dann=getattr(args, 'use_dann', False),
                target_folder=getattr(args, 'target_folder', None),
                lambda_domain=getattr(args, 'lambda_domain', 0.3)
            ),
            evaluation=EvaluationConfig(
                test_folder=getattr(args, 'test_folder', None),
                visualize_attention=getattr(args, 'visualize_attention', False),
                viz_samples=getattr(args, 'viz_samples', 3),
                eval_only=getattr(args, 'eval_only', False),
                checkpoint=getattr(args, 'checkpoint', None),
            )
        )
