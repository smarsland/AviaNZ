"""
Train bird audio classification models.
Supports both AST (Audio Spectrogram Transformer) and RegNetY models.
Always uses multilabel classification.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_testing.src.core.model_trainer import Trainer
from model_testing.src.core.trainer_config import TrainerConfig
from model_testing.src.core import config


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
    parser.add_argument('--bg-subtract', action='store_true',
                       help="Apply background subtraction normalization (works independently)")
    parser.add_argument('--apply-reverb', action='store_true',
                       help="Apply reverberation to loud noises")
    parser.add_argument('--reverb-prob', type=float, default=1.0,
                       help="Probability of applying reverb to a given training sample (default: 1.0)")
    parser.add_argument('--reverb-decay-range', type=float, nargs=2, default=(0.3, 1.2), metavar=('MIN', 'MAX'),
                       help="Range to sample reverb decay/gain from per sample (default: 0.3 1.2)")
    parser.add_argument('--reverb-delay-range', type=int, nargs=2, default=(2, 40), metavar=('MIN', 'MAX'),
                       help="Range (in spectrogram frames) to sample the mean echo delay from (default: 2 40)")
    parser.add_argument('--reverb-threshold', type=float, default=2.5,
                       help="Z-score above which a time-frequency cell counts as \"loud\" and reverberates (default: 2.5)")
    parser.add_argument('--median-filter', action='store_true',
                       help="Apply temporal median filtering (works independently)")
    parser.add_argument('--deltas', action='store_true', dest='use_deltas',
                       help="Add delta and delta-delta channels to the spectrogram (3-ch input). "
                            "Encodes rate-of-change rather than absolute magnitude, improving "
                            "robustness to recording-level and microphone-response differences.")
    parser.add_argument('--no-background', action='store_true', dest='no_background',
                       help="Ignore all-zero (background/no-bird) training samples")
    parser.add_argument('--use-asl', action='store_true', dest='use_asl',
                       help="Use Asymmetric Loss (ASL) instead of BCE — clips easy-negative gradients, "
                            "reducing gradient suppression from all-background samples")
    parser.add_argument('--class-weights', action='store_true', dest='class_weights',
                       help="Weight per-class loss by inverse frequency to prevent rare species from "
                            "being overwhelmed by the majority-class gradient")
    parser.add_argument('--spec-transform', type=str, default='Log', choices=['Log', 'PCEN', 'Box-Cox', 'LogMinMax', 'None'],
                       help="Spectrogram transformation (default: Log). LogMinMax = log then per-clip min-max to [0,1] (Kaytoo-style)")
    
    # Transfer learning
    parser.add_argument('--pretrained', type=str, default=None,
                       help="Path to pretrained model weights (.pt or .pth file)")
    parser.add_argument('--freeze-backbone', action='store_true',
                       help="Freeze backbone, only train classifier (RegNet only)")
    parser.add_argument('--freeze-stages', type=int, default=0,
                       help="Freeze first N stages of backbone (RegNet only, 0-4)")
    parser.add_argument('--freeze-layers', type=int, default=None,
                       help="Freeze first N transformer encoder layers (AST only)")
    parser.add_argument('--cnn-adapter', action='store_true',
                       help="Prepend two trainable CNN layers to the backbone (trained at 10x LR)")
    parser.add_argument('--sed-head', action='store_true', dest='sed_head',
                       help="Replace global avg pool with per-class temporal attention head (SED-style)")
    parser.add_argument('--gated-head', action='store_true', dest='gated_head',
                       help="Two-stage bird-presence gate + species classifier: gate predicts is-any-bird, species head runs on top")
    parser.add_argument('--per-chunk-norm', action='store_true', dest='per_chunk_norm',
                       help="Per-clip min-max normalization (replaces global AudioSet stats). AST only.")

    # Data scaling
    parser.add_argument('--max-samples-per-class', type=int, default=None, dest='max_samples_per_class',
                       help="Randomly subsample training data to at most N samples per class "
                            "(applied before training; useful for scaling experiments)")

    # Multi-label output constraint
    parser.add_argument('--kbird-prior', type=float, default=0.0, dest='kbird_prior',
                       help="When > 0, normalise sigmoid probabilities so their sum never exceeds k, "
                            "encoding the prior that at most ~k species call simultaneously (try 4.0). "
                            "Unlike softmax this does not cause class competition. 0 = standard sigmoid.")

    # Pre-computed AST attention channel
    parser.add_argument('--ast-channel-dir', type=str, default=None, dest='ast_channel_dir',
                       help="Directory of pre-computed AST attention maps (see "
                            "scripts/precompute_ast_attention.py). When set, RegNet is trained "
                            "with 2-channel input: (spectrogram, AST attention map).")
    
    # Evaluation
    parser.add_argument('--test-folder', type=str, default=None,
                       help="Path to test data folder for --eval-only mode")
    parser.add_argument('--eval-only', action='store_true', dest='eval_only',
                       help="Skip training; load saved model from output_folder and evaluate "
                            "on --test-folder / --test-folder2.")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help="Path to a .pt model file to load for --eval-only, overriding "
                            "the default <output_folder>/<model>_model.pt lookup.")
    
    # Visualization
    parser.add_argument('--visualize-attention', action='store_true',
                       help="Generate attention heatmaps for test samples (requires --test-folder)")
    parser.add_argument('--viz-samples', type=int, default=3,
                       help="Number of test samples to visualize (default: 3)")
    
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
    
    if args.visualize_attention and not args.test_folder:
        print("ERROR: Must specify --test-folder when using --visualize-attention")
        return
    
    # Set default learning rate based on model type
    if args.lr is None:
        args.lr = config.DEFAULT_LEARNING_RATE if args.model_type == 'ast' else 1e-4
    
    # Set default model name
    if args.model_name is None:
        args.model_name = 'regnety_008' if args.model_type == 'regnet' else None

    # Auto-load BirdClef pretrained weights for regnet if not specified
    if args.model_type == 'regnet' and args.pretrained is None:
        default_pretrained = 'BirdClefModels/model_fold0.pth'
        if os.path.exists(default_pretrained):
            args.pretrained = default_pretrained
            print(f"  Auto-loading pretrained weights: {default_pretrained}")
    
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
