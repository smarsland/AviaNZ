"""
Pseudo-label training pipeline.

Four phases:
  1. Train on source dataset (e.g., doc) for --epochs with early stopping
  2. Fine-tune on --pseudo-pct fraction of target dataset (real labels), same schedule
  3. Generate pseudo labels for the full target training set using the phase-2 model
  4. Fine-tune on the pseudo-labeled full target set, same schedule

Usage:
  python pseudo_train.py SOURCE_TRAIN TARGET_TRAIN OUTPUT_DIR \\
      --test-folder AVIANZ_TEST --test-folder2 DOC_TEST \\
      --model-type regnet --bg-subtract --median-filter \\
      --pseudo-pct 0.25 --epochs 100 --patience 15 --mixup 0.25
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path

from src.core.model_trainer import Trainer
from src.core.trainer_config import (
    TrainerConfig, TrainingConfig, ModelConfig, AugmentationConfig,
    LossConfig, DomainAdaptationConfig, EvaluationConfig,
)
from src.core import config
from src.data.data_utils import SpectrogramDataset


def build_cfg(args, data_folder, output_folder, resume_checkpoint=None,
              test_folder=None, test_folder2=None):
    lr = args.lr
    if lr is None:
        lr = config.DEFAULT_LEARNING_RATE if args.model_type == 'ast' else 1e-4

    model_name = args.model_name
    if model_name is None:
        model_name = 'regnety_008' if args.model_type == 'regnet' else None

    # Only use BirdClef pretrained when NOT resuming (phase 1 only)
    pretrained = None
    if resume_checkpoint is None:
        pretrained = getattr(args, 'pretrained', None)
        if args.model_type == 'regnet' and pretrained is None:
            default = 'BirdClefModels/model_fold0.pth'
            if os.path.exists(default):
                pretrained = default

    return TrainerConfig(
        training=TrainingConfig(
            data_folder=data_folder,
            output_folder=output_folder,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=lr,
            patience=args.patience,
            seed=getattr(args, 'seed', None),
            resume_checkpoint=resume_checkpoint,
        ),
        model=ModelConfig(
            model_type=args.model_type,
            model_name=model_name,
            pretrained_path=pretrained,
            freeze_backbone=getattr(args, 'freeze_backbone', False),
            freeze_stages=getattr(args, 'freeze_stages', 0),
            freeze_layers=getattr(args, 'freeze_layers', None),
            use_cnn_adapter=getattr(args, 'cnn_adapter', False),
            use_sed_head=getattr(args, 'sed_head', False),
        ),
        augmentation=AugmentationConfig(
            mixup_alpha=args.mixup,
            noise_ratio=getattr(args, 'noise', config.DEFAULT_NOISE_RATIO),
            noise_folder=getattr(args, 'noise_folder', None),
            bg_subtract=getattr(args, 'bg_subtract', False),
            median_filter=getattr(args, 'median_filter', False),
            per_chunk_norm=getattr(args, 'per_chunk_norm', False),
            spec_transform=getattr(args, 'spec_transform', 'Log'),
        ),
        loss=LossConfig(),
        domain_adaptation=DomainAdaptationConfig(),
        evaluation=EvaluationConfig(
            test_folder=test_folder,
            test_folder2=test_folder2,
            visualize_attention=getattr(args, 'visualize_attention', False),
            viz_samples=getattr(args, 'viz_samples', 10),
        ),
    )


def make_subset_dir(target_folder, pct, seed, out_dir):
    """
    Create a labels.json in out_dir containing pct fraction of target_folder's files.
    Filenames are stored as absolute paths so DataLoader can locate them regardless
    of the out_dir location (os.path.join(data_dir, abs_path) == abs_path in Python).
    """
    labels_file = os.path.join(target_folder, 'labels.json')
    with open(labels_file) as f:
        label_data = json.load(f)

    categories = label_data.get('categories') or label_data.get('species_list')
    data_dir = os.path.join(target_folder, 'data')

    valid_entries = []
    for file_info in label_data['files']:
        fpath = os.path.join(data_dir, file_info['filename'])
        if os.path.exists(fpath):
            valid_entries.append({
                'filename': fpath,
                'class_names': file_info.get('class_names', []),
            })

    rng = np.random.RandomState(seed if seed is not None else 42)
    n = max(1, round(len(valid_entries) * pct))
    chosen = rng.choice(len(valid_entries), size=n, replace=False)
    subset = [valid_entries[i] for i in sorted(chosen)]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'labels.json'), 'w') as f:
        json.dump({'categories': categories, 'files': subset}, f, indent=2)

    print(f"Subset: selected {n}/{len(valid_entries)} files ({100*pct:.0f}%) → {out_dir}")


def generate_pseudo_labels(model_path, target_folder, pseudo_dir, model_type, args, device):
    """
    Run model inference on all files in target_folder and write pseudo labels.json.
    Threshold for prediction: sigmoid(logit) > 0.5.
    """
    labels_file = os.path.join(target_folder, 'labels.json')
    with open(labels_file) as f:
        label_data = json.load(f)

    categories = label_data.get('categories') or label_data.get('species_list')
    num_classes = len(categories)
    data_dir = os.path.join(target_folder, 'data')

    all_entries = []
    for file_info in label_data['files']:
        fpath = os.path.join(data_dir, file_info['filename'])
        if os.path.exists(fpath):
            all_entries.append(fpath)

    dummy_labels = np.zeros((len(all_entries), num_classes), dtype=np.float32)
    img_height = config.SPECTROGRAM_PARAMS['nfilters']
    img_width = config.DEFAULT_TIME_BINS

    dataset = SpectrogramDataset(
        all_entries, dummy_labels,
        img_height, img_width, config.DEFAULT_CHANNELS, 'center',
        noise_filenames=None, noise_ratio=0.0,
        spec_transform=getattr(args, 'spec_transform', 'Log'),
        training=False,
        bg_subtract=getattr(args, 'bg_subtract', False),
        median_filter=getattr(args, 'median_filter', False),
        use_temporal_roll=False,
        noise_mode='full',
        background_prob=0.0,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )

    from src.core.models import AST, RegNetModel

    model_name = args.model_name or ('regnety_008' if model_type == 'regnet' else None)

    if model_type == 'regnet':
        model = RegNetModel(
            num_classes,
            pretrained_path=None,
            model_name=model_name,
            use_cnn_adapter=getattr(args, 'cnn_adapter', False),
            use_sed_head=getattr(args, 'sed_head', False),
        ).to(device)
    else:
        input_size = (img_height, img_width)
        model = AST(
            num_classes, input_size=input_size,
            use_cnn_adapter=getattr(args, 'cnn_adapter', False),
            use_sed_head=getattr(args, 'sed_head', False),
            per_chunk_norm=getattr(args, 'per_chunk_norm', False),
        ).to(device)
        model.interpolate_pos_embed(input_size)

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch_data, _ in loader:
            batch_data = batch_data.to(device)
            output = model(batch_data)
            preds = (torch.sigmoid(output) > 0.5).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.vstack(all_preds)  # (N, num_classes)

    pseudo_files = []
    for i, fpath in enumerate(all_entries):
        pred_classes = [categories[j] for j in range(num_classes) if all_preds[i, j]]
        pseudo_files.append({'filename': fpath, 'class_names': pred_classes})

    os.makedirs(pseudo_dir, exist_ok=True)
    with open(os.path.join(pseudo_dir, 'labels.json'), 'w') as f:
        json.dump({'categories': categories, 'files': pseudo_files}, f, indent=2)

    n_labeled = sum(1 for pf in pseudo_files if pf['class_names'])
    print(f"Pseudo labels: {n_labeled}/{len(pseudo_files)} files have ≥1 predicted class → {pseudo_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label training: source train → target subset fine-tune → pseudo label → full-target fine-tune"
    )

    parser.add_argument('source_folder', type=str, help="Source training data folder (e.g., doc train)")
    parser.add_argument('target_folder', type=str, help="Target training data folder (e.g., avianz train)")
    parser.add_argument('output_folder', type=str, help="Root output folder (phases written as subfolders)")

    parser.add_argument('--test-folder', type=str, default=None)
    parser.add_argument('--test-folder2', type=str, default=None)
    parser.add_argument('--pseudo-pct', type=float, default=0.25,
                        help="Fraction of target dataset used in phase 2 with real labels (default: 0.25)")

    parser.add_argument('--model-type', type=str, default='ast', choices=['ast', 'regnet'])
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mixup', type=float, default=config.DEFAULT_MIXUP_ALPHA)
    parser.add_argument('--noise', type=float, default=config.DEFAULT_NOISE_RATIO)
    parser.add_argument('--noise-folder', type=str, default=None)
    parser.add_argument('--bg-subtract', action='store_true')
    parser.add_argument('--median-filter', action='store_true')
    parser.add_argument('--spec-transform', type=str, default='Log',
                        choices=['Log', 'PCEN', 'Box-Cox', 'None'])
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--freeze-stages', type=int, default=0)
    parser.add_argument('--freeze-layers', type=int, default=None)
    parser.add_argument('--cnn-adapter', action='store_true')
    parser.add_argument('--sed-head', action='store_true')
    parser.add_argument('--per-chunk-norm', action='store_true', dest='per_chunk_norm')
    parser.add_argument('--visualize-attention', action='store_true')
    parser.add_argument('--viz-samples', type=int, default=10)

    args = parser.parse_args()
    args.multilabel = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    phase1_dir = os.path.join(args.output_folder, 'phase1_source')
    phase2_dir = os.path.join(args.output_folder, 'phase2_target_subset')
    phase3_dir = os.path.join(args.output_folder, 'phase3_pseudo_target')
    subset_dir = os.path.join(args.output_folder, 'target_subset_labels')
    pseudo_dir = os.path.join(args.output_folder, 'pseudo_labels')

    pct_display = f"{args.pseudo_pct*100:.0f}%"

    # ------------------------------------------------------------------
    # Phase 1: train on source dataset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Train on source  [{args.source_folder}]")
    print(f"{'='*60}\n")

    cfg1 = build_cfg(args, args.source_folder, phase1_dir,
                     test_folder=args.test_folder, test_folder2=args.test_folder2)
    Trainer(cfg1).train()

    phase1_ckpt = os.path.join(phase1_dir, f'{args.model_type}_model_best.pt')

    # ------------------------------------------------------------------
    # Phase 2: fine-tune on pseudo_pct% of target with real labels
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tune on {pct_display} of target  [{args.target_folder}]")
    print(f"{'='*60}\n")

    make_subset_dir(args.target_folder, args.pseudo_pct, args.seed, subset_dir)

    cfg2 = build_cfg(args, subset_dir, phase2_dir,
                     resume_checkpoint=phase1_ckpt,
                     test_folder=args.test_folder, test_folder2=args.test_folder2)
    Trainer(cfg2).train()

    phase2_ckpt = os.path.join(phase2_dir, f'{args.model_type}_model_best.pt')

    # ------------------------------------------------------------------
    # Phase 3: generate pseudo labels for the full target training set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generate pseudo labels for full target  [{args.target_folder}]")
    print(f"{'='*60}\n")

    generate_pseudo_labels(phase2_ckpt, args.target_folder, pseudo_dir,
                           args.model_type, args, device)

    # ------------------------------------------------------------------
    # Phase 4: fine-tune on pseudo-labeled full target
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 4: Fine-tune on pseudo-labeled full target")
    print(f"{'='*60}\n")

    cfg4 = build_cfg(args, pseudo_dir, phase3_dir,
                     resume_checkpoint=phase2_ckpt,
                     test_folder=args.test_folder, test_folder2=args.test_folder2)
    Trainer(cfg4).train()

    print(f"\n{'='*60}")
    print(f"PSEUDO-LABEL TRAINING COMPLETE")
    print(f"  Source:       {args.source_folder}")
    print(f"  Target:       {args.target_folder}")
    print(f"  Subset used:  {pct_display}")
    print(f"  Final model:  {phase3_dir}/{args.model_type}_model_best.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
