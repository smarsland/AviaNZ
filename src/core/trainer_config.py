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
    use_amp: bool = True
    seed: Optional[int] = None


@dataclass
class ModelConfig:
    """Model architecture and optimization."""
    multilabel: bool
    model_type: str = 'ast'  # 'ast' or 'regnet'
    model_name: Optional[str] = None  # e.g., 'regnety_008' for regnet
    freq_bins: Optional[int] = None
    time_bins: Optional[int] = None
    dropout: float = config.DEFAULT_DROPOUT
    use_multiscale: bool = False
    use_sparse_patches: bool = False
    num_sparse_patches: int = 20
    use_reconstruction: bool = False
    recon_weight: float = 0.1
    use_adapters: bool = False
    freeze_layers: Optional[int] = None
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False
    freeze_stages: int = 0


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""
    mixup_alpha: float = config.DEFAULT_MIXUP_ALPHA
    noise_ratio: float = config.DEFAULT_NOISE_RATIO
    noise_folder: Optional[str] = None
    noise_as_samples: bool = False
    max_noise_samples: Optional[int] = None
    use_temporal_roll: bool = config.DEFAULT_TEMPORAL_ROLL
    normalize: bool = False
    per_chunk_norm: bool = False
    # BirdClef-specific
    normalize_median_filter: bool = True
    median_only: bool = False
    spec_transform: str = 'Log'
    mixup_mode: str = 'mixup'
    noise_mode: str = 'full'
    validation_split: float = 0.2
    remove_baseline: bool = False
    background_prob: float = 0.0


@dataclass
class LossConfig:
    """Loss function configuration."""
    use_focal_loss: bool = False
    use_class_weights: bool = False
    pos_weight_cap: float = 20.0
    bce_smoothing: float = config.DEFAULT_BCE_SMOOTHING


@dataclass
class DomainAdaptationConfig:
    """Domain adaptation (DANN) settings."""
    use_dann: bool = False
    target_folder: Optional[str] = None
    lambda_domain: float = 0.3
    use_cleaner: bool = False


@dataclass
class EvaluationConfig:
    """Test/evaluation settings."""
    test_folder: Optional[str] = None
    test_folder2: Optional[str] = None


@dataclass
class TrainerConfig:
    """Complete trainer configuration. Groups all related settings."""
    training: TrainingConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    loss: LossConfig
    domain_adaptation: DomainAdaptationConfig
    evaluation: EvaluationConfig
    trial: Optional[object] = None  # For Optuna
    
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
                use_amp=True,
                seed=getattr(args, 'seed', None)
            ),
            model=ModelConfig(
                multilabel=args.multilabel,
                model_type=getattr(args, 'model_type', 'ast'),
                model_name=getattr(args, 'model_name', None),
                freq_bins=getattr(args, 'freq_bins', None),
                time_bins=getattr(args, 'time_bins', None),
                dropout=getattr(args, 'dropout', config.DEFAULT_DROPOUT),
                use_multiscale=getattr(args, 'multiscale', False),
                use_sparse_patches=getattr(args, 'sparse_patches', False),
                num_sparse_patches=getattr(args, 'num_sparse_patches', 20),
                use_reconstruction=getattr(args, 'reconstruct', False),
                recon_weight=getattr(args, 'recon_weight', 0.1),
                use_adapters=getattr(args, 'adapters', False),
                freeze_layers=getattr(args, 'freeze_layers', None),
                pretrained_path=getattr(args, 'pretrained', None),
                freeze_backbone=getattr(args, 'freeze_backbone', False),
                freeze_stages=getattr(args, 'freeze_stages', 0)
            ),
            augmentation=AugmentationConfig(
                mixup_alpha=args.mixup,
                noise_ratio=getattr(args, 'noise', config.DEFAULT_NOISE_RATIO),
                noise_folder=getattr(args, 'noise_folder', None),
                noise_as_samples=getattr(args, 'noise_as_samples', False),
                max_noise_samples=getattr(args, 'max_noise_samples', None),
                use_temporal_roll=getattr(args, 'temporal_roll', config.DEFAULT_TEMPORAL_ROLL),
                normalize=getattr(args, 'normalize', False),
                per_chunk_norm=getattr(args, 'per_chunk_norm', False),
                normalize_median_filter=getattr(args, 'median_filter', False) or getattr(args, 'normalize', False),
                spec_transform=getattr(args, 'spec_transform', 'Log')
            ),
            loss=LossConfig(
                use_focal_loss=getattr(args, 'focal_loss', False),
                use_class_weights=getattr(args, 'class_weights', False),
                pos_weight_cap=getattr(args, 'pos_weight_cap', 20.0),
                bce_smoothing=getattr(args, 'bce_smoothing', config.DEFAULT_BCE_SMOOTHING)
            ),
            domain_adaptation=DomainAdaptationConfig(
                use_dann=getattr(args, 'use_dann', False),
                target_folder=getattr(args, 'target_folder', None),
                lambda_domain=getattr(args, 'lambda_domain', 0.3),
                use_cleaner=getattr(args, 'use_cleaner', False)
            ),
            evaluation=EvaluationConfig(
                test_folder=getattr(args, 'test_folder', None),
                test_folder2=getattr(args, 'test_folder2', None)
            ),
            trial=None
        )
