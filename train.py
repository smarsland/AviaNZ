"""
Train bird audio classification models.
Supports both AST (Audio Spectrogram Transformer) and RegNetY models.
Always uses multilabel classification.
"""
import argparse
import os
from src.core.model_trainer import Trainer
from src.core.trainer_config import TrainerConfig
from src.core import config


def main():
    parser = argparse.ArgumentParser(
        description="Train bird audio classification models (AST or RegNetY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train AST model
  python train.py data/train_spec outputs/run1
  
  # Train RegNetY model (BirdClef fine-tuning)
  python train.py data/train_spec outputs/run2 --model-type regnet --pretrained BirdClefModels/model_fold0.pth
  
  # With early stopping and test evaluation
  python train.py data/train_spec outputs/run3 --patience 15 --test-folder data/test_spec
  
  # With data augmentation
  python train.py data/train_spec outputs/run4 --mixup 0.25 --noise 0.3
  
  # Freeze backbone for fast fine-tuning
  python train.py data/train_spec outputs/run5 --model-type regnet --pretrained BirdClefModels/model_fold0.pth --freeze-backbone
        """
    )
    
    # Required arguments
    parser.add_argument('data_folder', type=str, help="Path to training data folder")
    parser.add_argument('output_folder', type=str, help="Path to output folder")
    
    # Model selection
    parser.add_argument('--model-type', type=str, default='ast', choices=['ast', 'regnet'],
                       help="Model architecture (default: ast)")
    parser.add_argument('--model-name', type=str, default=None,
                       help="Specific model variant (e.g., regnety_008, regnety_016). Default: regnety_008 for regnet")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS,
                       help=f"Number of epochs (default: {config.DEFAULT_EPOCHS})")
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE,
                       help=f"Batch size (default: {config.DEFAULT_BATCH_SIZE})")
    parser.add_argument('--lr', type=float, default=None,
                       help=f"Learning rate (default: auto based on model type)")
    parser.add_argument('--patience', type=int, default=0,
                       help="Early stopping patience in epochs (default: 0 = disabled, try 15)")
    parser.add_argument('--seed', type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Data augmentation
    parser.add_argument('--mixup', type=float, default=config.DEFAULT_MIXUP_ALPHA,
                       help=f"Mixup alpha (default: {config.DEFAULT_MIXUP_ALPHA}, 0 = disabled)")
    parser.add_argument('--noise', type=float, default=config.DEFAULT_NOISE_RATIO,
                       help=f"Noise mixing ratio (default: {config.DEFAULT_NOISE_RATIO}, 0 = disabled)")
    parser.add_argument('--noise-folder', type=str, default=None,
                       help="Path to noise data folder (default: same as data_folder)")
    
    # Preprocessing
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization (recommended for soundscapes)")
    parser.add_argument('--median-filter', action='store_true',
                       help="Apply median filter during normalization (default: enabled if --normalize)")
    parser.add_argument('--spec-transform', type=str, default='Log', choices=['Log', 'PCEN', 'Box-Cox', 'None'],
                       help="Spectrogram transformation (default: Log)")
    
    # Transfer learning
    parser.add_argument('--pretrained', type=str, default=None,
                       help="Path to pretrained model weights (.pt or .pth file)")
    parser.add_argument('--freeze-backbone', action='store_true',
                       help="Freeze backbone, only train classifier (RegNet only)")
    parser.add_argument('--freeze-stages', type=int, default=0,
                       help="Freeze first N stages of backbone (RegNet only, 0-4)")
    
    # Evaluation
    parser.add_argument('--test-folder', type=str, default=None,
                       help="Path to test data folder (evaluated after training)")
    parser.add_argument('--test-folder2', type=str, default=None,
                       help="Path to second test data folder (evaluated after training)")
    
    # Domain adaptation (rarely used)
    parser.add_argument('--use-dann', action='store_true',
                       help="Enable DANN domain adaptation (AST only)")
    parser.add_argument('--target-folder', type=str, default=None,
                       help="Target domain folder for DANN (unlabeled)")
    parser.add_argument('--lambda-domain', type=float, default=0.3,
                       help="Domain loss weight for DANN (default: 0.3)")
    
    args = parser.parse_args()
    
    if args.use_dann and not args.target_folder:
        print("ERROR: Must specify --target-folder when using --use-dann")
        return
    
    if args.use_dann and args.model_type != 'ast':
        print("ERROR: DANN is only supported for AST models")
        return
    
    # Set default learning rate based on model type
    if args.lr is None:
        args.lr = config.DEFAULT_LEARNING_RATE if args.model_type == 'ast' else 1e-4
    
    # Set default model name
    if args.model_name is None:
        args.model_name = 'regnety_008' if args.model_type == 'regnet' else None
    
    print(f"\nTraining {args.model_type.upper()} model...")
    print(f"  Multilabel classification: ALWAYS ENABLED")
    if args.mixup > 0:
        print(f"  Mixup augmentation: α={args.mixup}")
    if args.noise > 0:
        print(f"  Noise augmentation: {args.noise*100:.0f}%")
    
    # Forcefully set multilabel=True
    args.multilabel = True
    
    # Create config from args
    cfg = TrainerConfig.from_args(args)
    
    # Set model type and name in config
    cfg.model.model_type = args.model_type
    if args.model_name:
        cfg.model.model_name = args.model_name
    
    # Run training with unified trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
