
import argparse
import os
from model_trainer import ASTTrainer, CNNTrainer, PixelPredictionTrainer
import config

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
    parser.add_argument('--model', type=str, default='ast', choices=['ast', 'cnn'],
                       help="Model architecture to use: 'ast' (Audio Spectrogram Transformer) or 'cnn' (Convolutional Neural Network). Default: ast")
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS, 
                       help=f"Number of epochs (default: {config.DEFAULT_EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, 
                       help=f"Batch size (default: {config.DEFAULT_BATCH_SIZE})")
    parser.add_argument('--multilabel', action='store_true', help="Use multi-label classification (classification mode only)")
    parser.add_argument('--balance', action='store_true', 
                       help="Enable class balancing using WeightedRandomSampler (recommended for imbalanced datasets, classification mode only)")
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LEARNING_RATE, 
                       help=f"Learning rate (default: {config.DEFAULT_LEARNING_RATE})")
    parser.add_argument('--mixup', type=float, default=0.5,
                       help="Mixup alpha parameter (0 = no mixup, 0.5 for ESC-50 per paper, default: 0.5, classification mode only)")
    parser.add_argument('--pretrained', type=str, default=None,
                       help="Path to pretrained model weights (.pt file) for transfer learning")
    parser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'cosine', 'cosine_warmup'],
                       help="Learning rate scheduler: 'lambda' (ESC-50 style), 'cosine' (annealing), or 'cosine_warmup' (5-epoch warmup + cosine, classification mode only)")
    parser.add_argument('--weight-decay', type=float, default=0.0,
                       help="Weight decay (L2 regularization) for Adam optimizer (default: 0.0, try 0.01 or 0.001 for regularization, classification mode only)")
    parser.add_argument('--noise', type=float, default=config.DEFAULT_NOISE_RATIO,
                       help=f"Noise mixing ratio during training (default: {config.DEFAULT_NOISE_RATIO}, 0.0 = no noise, 0.5 = 50%% noise, classification mode only)")
    parser.add_argument('--noise-folder', type=str, default=None,
                       help="Path to noise data folder if different from data_folder (overrides default noise location, classification mode only)")
    parser.add_argument('--freq-bins', type=int, default=config.DEFAULT_FREQ_BINS,
                       help=f"Number of frequency bins in spectrograms (default: {config.DEFAULT_FREQ_BINS})")
    parser.add_argument('--time-bins', type=int, default=config.DEFAULT_TIME_BINS,
                       help=f"Number of time bins in spectrograms (default: {config.DEFAULT_TIME_BINS})")
    parser.add_argument('--confusion-sampling', action='store_true',
                       help="Enable confusion-based curriculum learning: dynamically upweight classes the model struggles with during training (recommended for imbalanced datasets with many classes, classification mode only)")
    parser.add_argument('--confusion-eval-freq', type=int, default=config.DEFAULT_CONFUSION_EVAL_FREQUENCY,
                       help=f"How often (in epochs) to re-evaluate confusion and update sample weights (default: {config.DEFAULT_CONFUSION_EVAL_FREQUENCY}, classification mode only)")
    parser.add_argument('--confusion-boost', type=float, default=config.DEFAULT_CONFUSION_BOOST_FACTOR,
                       help=f"How much to boost confused classes (default: {config.DEFAULT_CONFUSION_BOOST_FACTOR}, e.g., 3.0 = triple weight for worst performers, classification mode only)")
    parser.add_argument('--confusion-top-k', type=int, default=config.DEFAULT_CONFUSION_TOP_K,
                       help=f"Only boost top-k most confused classes (default: {config.DEFAULT_CONFUSION_TOP_K}, None = boost all proportionally to error rate, classification mode only)")
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
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization to spectrograms during training. This normalizes out background noise per frequency band to enhance bird call signals.")
    parser.add_argument('--sparse-patches', action='store_true',
                       help="⚡ Use sparse patch extraction: only process patches with signal content (much faster, AST only). Extracts top-K patches by signal density instead of processing all ~1000 patches.")
    parser.add_argument('--num-sparse-patches', type=int, default=20,
                       help="Number of patches to extract in sparse mode (default: 20). Standard AST uses ~1000 patches, sparse mode uses only K patches with highest signal content.")
    parser.add_argument('--dropout', type=float, default=0.2,
                       help="Dropout rate for AST (default: 0.2). Increase to 0.4-0.6 to reduce overfitting.")
    parser.add_argument('--bce-smoothing', type=float, default=0.0,
                       help="Target smoothing epsilon for multilabel BCE (default: 0.0). Try 0.05 to reduce overconfidence.")
    
    args = parser.parse_args()
    
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
            normalize=args.normalize
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
            use_class_balancing=args.balance,
            scheduler_type=args.scheduler,
            weight_decay=args.weight_decay,
            noise_ratio=args.noise,
            noise_folder=args.noise_folder,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_confusion_sampling=args.confusion_sampling,
            confusion_eval_freq=args.confusion_eval_freq,
            confusion_boost_factor=args.confusion_boost,
            confusion_top_k=args.confusion_top_k,
            use_focal_loss=args.focal_loss,
            use_multiscale=args.multiscale,
            use_class_weights=args.class_weights,
            freeze_layers=args.freeze_layers,
            use_reconstruction=args.reconstruct,
            recon_weight=args.recon_weight,
            normalize=args.normalize,
            use_sparse_patches=args.sparse_patches,
            num_sparse_patches=args.num_sparse_patches,
            dropout=args.dropout,
            bce_smoothing=args.bce_smoothing,
            use_amp=False
        )
    elif args.model == 'cnn':
        print(f"Training Convolutional Neural Network (CNN) model...")
        trainer = CNNTrainer(
            data_folder=args.data_folder,
            output_folder=args.output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            multilabel=args.multilabel,
            learning_rate=args.lr,
            mixup_alpha=args.mixup,
            pretrained_path=args.pretrained,
            use_class_balancing=args.balance,
            weight_decay=args.weight_decay,
            noise_ratio=args.noise,
            noise_folder=args.noise_folder,
            freq_bins=args.freq_bins,
            time_bins=args.time_bins,
            use_confusion_sampling=args.confusion_sampling,
            confusion_eval_freq=args.confusion_eval_freq,
            confusion_boost_factor=args.confusion_boost,
            confusion_top_k=args.confusion_top_k,
            use_focal_loss=args.focal_loss,
            normalize=args.normalize
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    trainer.train()

if __name__ == "__main__":
    main()

