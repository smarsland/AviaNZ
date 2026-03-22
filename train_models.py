
import argparse
import os
import json
from model_trainer import ASTTrainer, CNNTrainer, PixelPredictionTrainer, KaytooTrainer
import config


def warn_if_multilabel_dataset(data_folder, multilabel_flag, mode):
    if mode != 'classification' or multilabel_flag:
        return

    labels_path = os.path.join(data_folder, 'labels.json')
    if not os.path.exists(labels_path):
        return

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    max_labels = 0
    multi_label_samples = 0
    for file_info in labels.get('files', []):
        class_names = file_info.get('class_names', [])
        class_names = [c for c in class_names if c and c != 'Empty Sample']
        if len(class_names) > 1:
            multi_label_samples += 1
        if len(class_names) > max_labels:
            max_labels = len(class_names)

    if multi_label_samples > 0:
        print("\n⚠️  WARNING: Dataset appears to be multi-label, but --multilabel was NOT set.")
        print(f"   Found {multi_label_samples} samples with >1 label (max labels in a sample: {max_labels}).")
        print("   This will train a single-label (softmax) classifier and can score ~0 in compare_models.py")
        print("   if you evaluate with per-class thresholding. Consider adding --multilabel.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Train models on spectrogram datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train AST model on ESC-50
  python train_models.py "Sound Files/ESC_spec" "Sound Files/ESC_out" --model ast
  
  # Train CNN model on DOC dataset
  python train_models.py "Sound Files/DOC_spec" "outputs/doc_run1" --model cnn --epochs 100 --batch_size 32
  
  # Transfer learning: use ESC-trained model as pretrained weights for DOC
  python train_models.py "Sound Files/NZbird_spec" "Sound Files/Bird_out" --model ast --pretrained "Sound Files/ESC_out/ast_model_best.pt"
  
  # Multi-label training with CNN
  python train_models.py "Sound Files/NZbird_spec" "outputs/test" --model cnn --multilabel --lr 1e-3
  
  # Pixel prediction pre-training with CNN (interesting pixel maps)
  python train_models.py "Sound Files/DOC_spec" "outputs/pixel_pretrain" --mode pixel --model cnn --epochs 50
  
  # Pixel prediction pre-training with AST (interesting pixel maps)
  python train_models.py "Sound Files/DOC_spec" "outputs/ast_pixel_pretrain" --mode pixel --model ast --epochs 30
  
  # Train with background normalization to enhance bird calls
  python train_models.py "Sound Files/NZbird_spec" "outputs/normalized_run" --model ast --normalize
        """
    )
    
    parser.add_argument('data_folder', type=str, help="Path to spectrogram data folder (INPUT)")
    parser.add_argument('output_folder', type=str, help="Path to outputs folder (OUTPUT)")
    parser.add_argument('--mode', type=str, default='classification', choices=['classification', 'pixel'],
                       help="Training mode: 'classification' for standard class prediction, 'pixel' for interesting pixel prediction (pre-training). Default: classification")
    parser.add_argument('--model', type=str, default='ast', choices=['ast', 'cnn', 'kaytoo'],
                       help="Model architecture to use: 'ast' (Audio Spectrogram Transformer), 'cnn' (Convolutional Neural Network), or 'kaytoo' (EfficientNet + attention pooling). Default: ast")
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS, 
                       help=f"Number of epochs (default: {config.DEFAULT_EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, 
                       help=f"Batch size (default: {config.DEFAULT_BATCH_SIZE})")
    parser.add_argument('--multilabel', action='store_true', help="Use multi-label classification (classification mode only)")
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LEARNING_RATE, 
                       help=f"Learning rate (default: {config.DEFAULT_LEARNING_RATE:.2e} from Optuna tuning)")
    parser.add_argument('--mixup', type=float, default=config.DEFAULT_MIXUP_ALPHA,
                       help=f"Mixup alpha parameter (0 = no mixup, default: {config.DEFAULT_MIXUP_ALPHA}). Applied at spectrogram level for efficiency, classification mode only)")
    parser.add_argument('--pretrained', type=str, default=None,
                       help="Path to pretrained model weights (.pt file) for transfer learning")
    parser.add_argument('--weight-decay', type=float, default=config.DEFAULT_WEIGHT_DECAY,
                       help=f"Weight decay (L2 regularization) for Adam optimizer (default: {config.DEFAULT_WEIGHT_DECAY:.1e}, classification mode only)")
    parser.add_argument('--noise', type=float, default=config.DEFAULT_NOISE_RATIO,
                       help=f"Expected noise mixing ratio for AUGMENTATION: uniformly samples from [0, 2\u00d7ratio] so E[noise]=ratio. Creates variable SNR conditions during training (default: {config.DEFAULT_NOISE_RATIO}, 0.0=no mixing, 0.3=30%% expected noise, classification mode only)")
    parser.add_argument('--noise-folder', type=str, default=None,
                       help="Path to noise data folder. Used for augmentation mixing via --noise (default: same as data_folder, classification mode only)")

    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization to spectrograms during training/validation (recommended for soundscapes; must match inference)")

    parser.add_argument('--noise-as-samples', action='store_true',
                       help="Include noise spectrograms as additional all-zero-label training samples (improves empty/background rejection)")
    parser.add_argument('--max-noise-samples', type=int, default=None,
                       help="Maximum number of noise spectrograms to add as all-zero samples (default: use all available)")
    parser.add_argument('--pos-weight-cap', type=float, default=20.0,
                       help="Cap for multilabel BCE pos_weight when using --class-weights (default: 20.0). Increase if dataset has many empty/background samples")
    parser.add_argument('--freq-bins', type=int, default=config.DEFAULT_FREQ_BINS,
                       help=f"Number of frequency bins in spectrograms (default: {config.DEFAULT_FREQ_BINS})")
    parser.add_argument('--time-bins', type=int, default=config.DEFAULT_TIME_BINS,
                       help=f"Number of time bins in spectrograms (default: {config.DEFAULT_TIME_BINS} = ~10 seconds at 10ms hop, matches AudioSet pretraining)")
    parser.add_argument('--focal-loss', action='store_true',
                       help="Use Focal Loss instead of standard BCE/CrossEntropy loss (recommended for highly imbalanced datasets, classification mode only). Focal Loss down-weights easy examples to focus on hard negatives.")
    parser.add_argument('--multiscale', action='store_true',
                       help="Use Multi-Scale CNN frontend instead of standard AST patch embeddings (AST only, classification mode only). Extracts features at 3 different scales for better fine-grained discrimination.")
    parser.add_argument('--class-weights', action='store_true',
                       help="Use class-weighted BCE loss (multilabel only, classification mode only). Weights loss by inverse class frequency - rare classes get higher gradient. Better than --balance for imbalanced datasets.")
    parser.add_argument('--freeze-layers', type=int, default=None,
                       help="Freeze first N transformer encoder layers (AST only). Only trains last layers + classifier. Use 6-8 to keep AudioSet low-level features but learn bird-specific high-level patterns.")
    parser.add_argument('--reconstruct', action='store_true',
                       help="Enable auxiliary reconstruction loss: model also learns to reconstruct input spectrogram from patch embeddings. Acts as regularizer to preserve discriminative features.")
    parser.add_argument('--recon-weight', type=float, default=0.1,
                       help="Weight for reconstruction loss in combined objective (default: 0.1). Total loss = classification_loss + recon_weight * reconstruction_loss")
    parser.add_argument('--sparse-patches', action='store_true',
                       help="⚡ Use sparse patch extraction: only process patches with signal content (much faster, AST only). Extracts top-K patches by signal density instead of processing all ~1000 patches.")
    parser.add_argument('--num-sparse-patches', type=int, default=20,
                       help="Number of patches to extract in sparse mode (default: 20). Standard AST uses ~1000 patches, sparse mode uses only K patches with highest signal content.")
    parser.add_argument('--dropout', type=float, default=config.DEFAULT_DROPOUT,
                       help=f"Dropout rate for AST (default: {config.DEFAULT_DROPOUT}). Increase to 0.4-0.6 to reduce overfitting.")
    parser.add_argument('--bce-smoothing', type=float, default=config.DEFAULT_BCE_SMOOTHING,
                       help=f"Target smoothing epsilon for multilabel BCE (default: {config.DEFAULT_BCE_SMOOTHING}). Reduces overconfidence.")
    parser.add_argument('--temporal-roll', action='store_true', default=True,
                       help="Enable temporal rolling augmentation: randomly shifts start position in tiled/repeated signals (default: True, prevents position bias). Note: Time axis uses tiling not zero-padding.")
    parser.add_argument('--no-temporal-roll', action='store_false', dest='temporal_roll',
                       help="Disable temporal rolling augmentation")
    
    parser.add_argument('--adapters', action='store_true',
                       help="[AST] Use lightweight adapter layers for fine-tuning without forgetting AudioSet features (like LoRA). Adds trainable bottleneck layers while keeping backbone frozen.")
    parser.add_argument('--per-chunk-norm', action='store_true',
                       help="[AST] Use per-chunk min-max normalization (like Kaytoo) instead of global AudioSet mean/std. Handles varying recording levels better. Splits spectrogram into chunks and normalizes each independently.")
    
    # DANN (Domain Adaptation) parameters
    parser.add_argument('--use-dann', action='store_true',
                       help="Enable Domain Adaptive Neural Network (DANN) training for domain adaptation")
    parser.add_argument('--target-folder', type=str, default=None,
                       help="Path to target domain folder for domain adaptation (unlabeled domain)")
    parser.add_argument('--lambda-domain', type=float, default=0.3,
                       help="Domain loss weight (default: 0.3). If dacc stays >95%%, increase lambda.")
    parser.add_argument('--test-folder', type=str, default=None,
                       help="Path to test data folder (with labels.json). Evaluated AFTER training completes.")
    parser.add_argument('--test-folder2', type=str, default=None,
                       help="Path to test data folder 2 (with labels.json). Evaluated AFTER training completes.")
    
    parser.add_argument('--use-cleaner', action='store_true',
                       help="Use trainable spectrogram cleaner network for domain adaptation. Keeps backbone frozen and learns preprocessing transform.")
    
    args = parser.parse_args()
    
    if args.use_dann and not args.target_folder:
        print("ERROR: Must specify --target-folder when using --use-dann")
        return

    warn_if_multilabel_dataset(args.data_folder, args.multilabel, args.mode)
    
    if args.mode == 'pixel':
        print(f"Training model for interesting pixel prediction (pre-training mode)...")
        print(f"Using {args.model.upper()} model")
        trainer = PixelPredictionTrainer(
            data_folder=args.data_folder,
            output_folder=args.output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_type=args.model,
            pretrained_path=args.pretrained,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_temporal_roll=args.temporal_roll
        )
    elif args.model == 'ast':
        print(f"Training Audio Spectrogram Transformer (AST) model...")
        if args.multiscale:
            print("  Using Multi-Scale CNN frontend")
        if args.sparse_patches:
            print(f"  ⚡ Using sparse patch extraction: {args.num_sparse_patches} patches")
        trainer = ASTTrainer(
            data_folder=args.data_folder,
            output_folder=args.output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            multilabel=args.multilabel,
            learning_rate=args.lr,
            mixup_alpha=args.mixup,
            pretrained_path=args.pretrained,
            weight_decay=args.weight_decay,
            noise_ratio=args.noise,
            noise_folder=args.noise_folder,
            normalize=args.normalize,
            noise_as_samples=args.noise_as_samples,
            max_noise_samples=args.max_noise_samples,
            pos_weight_cap=args.pos_weight_cap,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_focal_loss=args.focal_loss,
            use_multiscale=args.multiscale,
            use_class_weights=args.class_weights,
            freeze_layers=args.freeze_layers,
            use_reconstruction=args.reconstruct,
            recon_weight=args.recon_weight,
            use_sparse_patches=args.sparse_patches,
            num_sparse_patches=args.num_sparse_patches,
            dropout=args.dropout,
            bce_smoothing=args.bce_smoothing,
            use_temporal_roll=args.temporal_roll,
            use_adapters=args.adapters,
            per_chunk_norm=args.per_chunk_norm,
            use_amp=False,
            use_dann=args.use_dann,
            target_folder=args.target_folder,
            lambda_domain=args.lambda_domain,
            test_folder=args.test_folder,
            test_folder2=args.test_folder2,
            use_cleaner=args.use_cleaner
        )
    elif args.model == 'cnn':
        print(f"Training Convolutional Neural Network (CNN) model...")
        # Use CNN-specific LR if user didn't override
        cnn_lr = args.lr if args.lr != config.DEFAULT_LEARNING_RATE else config.DEFAULT_CNN_LEARNING_RATE
        if cnn_lr != args.lr:
            print(f"Using CNN-optimized learning rate: {cnn_lr:.1e} (override with --lr if needed)")
        trainer = CNNTrainer(
            data_folder=args.data_folder,
            output_folder=args.output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            multilabel=args.multilabel,
            learning_rate=cnn_lr,
            mixup_alpha=args.mixup,
            pretrained_path=args.pretrained,
            weight_decay=args.weight_decay,
            noise_ratio=args.noise,
            noise_folder=args.noise_folder,
            normalize=args.normalize,
            noise_as_samples=args.noise_as_samples,
            max_noise_samples=args.max_noise_samples,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_focal_loss=args.focal_loss,
            use_temporal_roll=args.temporal_roll
        )
    elif args.model == 'kaytoo':
        print(f"Training Kaytoo model (EfficientNet + attention pooling)...")
        trainer = KaytooTrainer(
            data_folder=args.data_folder,
            output_folder=args.output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            multilabel=args.multilabel,
            learning_rate=args.lr,
            mixup_alpha=args.mixup,
            pretrained_path=args.pretrained,
            weight_decay=args.weight_decay,
            noise_ratio=args.noise,
            noise_folder=args.noise_folder,
            normalize=args.normalize,
            noise_as_samples=args.noise_as_samples,
            max_noise_samples=args.max_noise_samples,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_focal_loss=args.focal_loss,
            use_temporal_roll=args.temporal_roll,
            use_amp=False
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    trainer.train()

if __name__ == "__main__":
    main()

