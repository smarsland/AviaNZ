import argparse
import json
import os
import sys
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from model_trainer import ASTTrainer
import config

def objective(trial, data_folder, output_base, fixed_args):
    mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.5)
    dropout = trial.suggest_float('dropout', 0.15, 0.35)
    weight_decay = trial.suggest_float('weight_decay', 5e-6, 5e-5, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    
    normalize = trial.suggest_categorical('normalize', [True, False])
    use_class_weights = trial.suggest_categorical('use_class_weights', [True, False])
    scheduler_type = trial.suggest_categorical('scheduler_type', ['lambda', 'cosine', 'cosine_warmup'])
    
    if fixed_args.get('search_advanced_options', False):
        use_class_balancing = trial.suggest_categorical('use_class_balancing', [True, False])
        # Confusion sampling doesn't work reliably with sparse patches - disable it
        if fixed_args.get('num_sparse_patches', 0) > 0:
            use_confusion_sampling = False
        else:
            use_confusion_sampling = trial.suggest_categorical('use_confusion_sampling', [True, False])
        use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
        noise_ratio = trial.suggest_float('noise_ratio', 0.0, 0.3)
        bce_smoothing = trial.suggest_float('bce_smoothing', 0.0, 0.1)
        # MultiScaleAST doesn't support sparse patches, so disable it when using sparse mode
        if fixed_args.get('num_sparse_patches', 0) > 0:
            use_multiscale = False
        else:
            use_multiscale = trial.suggest_categorical('use_multiscale', [True, False])
    else:
        use_class_balancing = fixed_args.get('balance', False)
        use_confusion_sampling = fixed_args.get('confusion_sampling', False)
        use_focal_loss = fixed_args.get('focal_loss', False)
        noise_ratio = fixed_args.get('noise', config.DEFAULT_NOISE_RATIO)
        bce_smoothing = 0.0
        use_multiscale = fixed_args.get('multiscale', False)
    
    num_sparse_patches = fixed_args.get('num_sparse_patches', 50)
    
    trial_output = os.path.join(output_base, f"trial_{trial.number}")
    os.makedirs(trial_output, exist_ok=True)
    
    try:
        trainer = ASTTrainer(
            data_folder=data_folder,
            output_folder=trial_output,
            max_epochs=fixed_args['epochs'],
            batch_size=fixed_args['batch_size'],
            multilabel=fixed_args['multilabel'],
            learning_rate=learning_rate,
            mixup_alpha=mixup_alpha,
            pretrained_path=fixed_args.get('pretrained'),
            use_class_balancing=use_class_balancing,
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            noise_ratio=noise_ratio,
            noise_folder=fixed_args.get('noise_folder'),
            freq_bins=fixed_args.get('freq_bins', config.DEFAULT_FREQ_BINS),
            time_bins=fixed_args.get('time_bins', config.DEFAULT_TIME_BINS),
            use_confusion_sampling=use_confusion_sampling,
            confusion_eval_freq=fixed_args.get('confusion_eval_freq', config.DEFAULT_CONFUSION_EVAL_FREQUENCY),
            confusion_boost_factor=fixed_args.get('confusion_boost', config.DEFAULT_CONFUSION_BOOST_FACTOR),
            confusion_top_k=fixed_args.get('confusion_top_k', config.DEFAULT_CONFUSION_TOP_K),
            use_focal_loss=use_focal_loss,
            use_multiscale=use_multiscale,
            use_class_weights=use_class_weights,
            freeze_layers=fixed_args.get('freeze_layers'),
            use_reconstruction=fixed_args.get('reconstruct', False),
            recon_weight=fixed_args.get('recon_weight', 0.1),
            normalize=normalize,
            use_sparse_patches=(num_sparse_patches > 0),
            num_sparse_patches=num_sparse_patches,
            dropout=dropout,
            bce_smoothing=bce_smoothing,
            trial=trial
        )
        
        best_val_loss = trainer.train()
        
        trial.set_user_attr('best_val_loss', best_val_loss)
        
        return best_val_loss
        
    except optuna.TrialPruned:
        # Re-raise pruning exceptions from intermediate callbacks
        raise
    except Exception as e:
        import traceback
        print(f"\n{'='*80}")
        print(f"Trial {trial.number} FAILED with exception:")
        print(f"{'='*80}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        trial.set_user_attr('error', str(e))
        raise  # Let it fail properly instead of masking as pruned

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for sparse AST training using Optuna (Bayesian optimization).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search (7 dimensions: hyperparameters + normalize + class_weights + scheduler)
  python hyperparameter_search.py /local/scratch/freangi/doc_data_1000 doc_search --n-trials 100 --multilabel --epochs 25
  
  # Advanced search (13 dimensions: adds confusion sampling, focal loss, noise, multiscale, etc.)
  python hyperparameter_search.py /data output_search --n-trials 150 --multilabel --epochs 25 --search-advanced-options
  
  # Quick sparse search (20 patches, ~1 hour per trial)
  python hyperparameter_search.py /data sparse_20 --n-trials 100 --multilabel --num-sparse-patches 20 --epochs 25
  
  # Resume previous study
  python hyperparameter_search.py /data output_search --n-trials 20 --study-name my_search --resume
  
  # Run on multiple GPUs in parallel (shared filesystem required):
  CUDA_VISIBLE_DEVICES=0 python hyperparameter_search.py /data output --n-trials 50 --study-name shared --multilabel &
  CUDA_VISIBLE_DEVICES=1 python hyperparameter_search.py /data output --n-trials 50 --study-name shared --multilabel &
  CUDA_VISIBLE_DEVICES=2 python hyperparameter_search.py /data output --n-trials 50 --study-name shared --multilabel &
        """
    )
    
    parser.add_argument('data_folder', type=str, help="Path to spectrogram data folder")
    parser.add_argument('output_base', type=str, help="Base path for search outputs (each trial gets subfolder)")
    parser.add_argument('--n-trials', type=int, default=50, help="Number of trials to run (default: 50)")
    parser.add_argument('--study-name', type=str, default='ast_sparse_search', help="Name of the optuna study (default: ast_sparse_search)")
    parser.add_argument('--storage', type=str, default=None, help="Optuna storage URL for distributed search (e.g., sqlite:///search.db or postgresql://...). If None, uses SQLite in storage_dir (or output_base if storage_dir not set).")
    parser.add_argument('--storage-dir', type=str, default=None, help="Directory for Optuna database (shared across machines). If None, uses output_base. Use this to store database in home directory while keeping trial outputs in /local/scratch.")
    parser.add_argument('--resume', action='store_true', help="Resume previous study with same name")
    parser.add_argument('--gpu', type=int, default=None, help="GPU device ID to use (overrides CUDA_VISIBLE_DEVICES)")
    
    parser.add_argument('--search-advanced-options', action='store_true', help="Also search advanced/experimental options (class balancing, confusion sampling, focal loss, noise, bce smoothing, multiscale). Basic important options (normalize, class_weights, scheduler) are always searched.")
    parser.add_argument('--epochs', type=int, default=25, help="Max epochs per trial (default: 25, use 20-30 for sparse models)")
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE, help=f"Batch size (default: {config.DEFAULT_BATCH_SIZE})")
    parser.add_argument('--multilabel', action='store_true', help="Use multi-label classification")
    parser.add_argument('--normalize', action='store_true', help="Apply background normalization")
    parser.add_argument('--pretrained', type=str, default=None, help="Path to pretrained model weights")
    parser.add_argument('--balance', action='store_true', help="Enable class balancing")
    parser.add_argument('--scheduler', type=str, default='lambda', choices=['lambda', 'cosine', 'cosine_warmup'])
    parser.add_argument('--noise', type=float, default=config.DEFAULT_NOISE_RATIO)
    parser.add_argument('--noise-folder', type=str, default=None)
    parser.add_argument('--freq-bins', type=int, default=config.DEFAULT_FREQ_BINS)
    parser.add_argument('--time-bins', type=int, default=config.DEFAULT_TIME_BINS)
    parser.add_argument('--confusion-sampling', action='store_true')
    parser.add_argument('--confusion-eval-freq', type=int, default=config.DEFAULT_CONFUSION_EVAL_FREQUENCY)
    parser.add_argument('--confusion-boost', type=float, default=config.DEFAULT_CONFUSION_BOOST_FACTOR)
    parser.add_argument('--confusion-top-k', type=int, default=config.DEFAULT_CONFUSION_TOP_K)
    parser.add_argument('--focal-loss', action='store_true')
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--class-weights', action='store_true')
    parser.add_argument('--freeze-layers', type=int, default=None)
    parser.add_argument('--reconstruct', action='store_true')
    parser.add_argument('--recon-weight', type=float, default=0.1)
    parser.add_argument('--num-sparse-patches', type=int, default=50, help="Number of sparse patches (fixed, not searched)")
    
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    os.makedirs(args.output_base, exist_ok=True)
    
    fixed_args = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'multilabel': args.multilabel,
        'normalize': args.normalize,
        'pretrained': args.pretrained,
        'balance': args.balance,
        'scheduler': args.scheduler,
        'noise': args.noise,
        'noise_folder': args.noise_folder,
        'freq_bins': args.freq_bins,
        'time_bins': args.time_bins,
        'confusion_sampling': args.confusion_sampling,
        'confusion_eval_freq': args.confusion_eval_freq,
        'confusion_boost': args.confusion_boost,
        'confusion_top_k': args.confusion_top_k,
        'focal_loss': args.focal_loss,
        'multiscale': args.multiscale,
        'class_weights': args.class_weights,
        'freeze_layers': args.freeze_layers,
        'reconstruct': args.reconstruct,
        'recon_weight': args.recon_weight,
        'num_sparse_patches': args.num_sparse_patches,
        'search_advanced_options': args.search_advanced_options,
    }
    
    if args.storage:
        storage = args.storage
    else:
        storage_dir = args.storage_dir if args.storage_dir else args.output_base
        os.makedirs(storage_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(storage_dir, 'optuna_study.db')}"
    
    load_if_exists = True
    
    sampler = TPESampler(seed=42)
    pruner = None
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    print(f"Starting hyperparameter search: {args.n_trials} trials")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Output base (trial runs): {args.output_base}")
    if args.storage_dir:
        print(f"Storage dir (database): {args.storage_dir}")
    print(f"Fixed parameters: {json.dumps(fixed_args, indent=2)}")
    
    study.optimize(
        lambda trial: objective(trial, args.data_folder, args.output_base, fixed_args),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("Search completed!")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    results_file = os.path.join(args.output_base, 'search_results.json')
    with open(results_file, 'w') as f:
        results = {
            'best_trial': study.best_trial.number,
            'best_value': study.best_trial.value,
            'best_params': study.best_trial.params,
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in study.trials
            ]
        }
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    best_trial_output = os.path.join(args.output_base, f"trial_{study.best_trial.number}")
    print(f"Best model saved in: {best_trial_output}")

if __name__ == "__main__":
    main()
