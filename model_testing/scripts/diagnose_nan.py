#!/usr/bin/env python3
"""
Diagnose NaN predictions / training failure by comparing DOC-only vs combined datasets.

For each dataset this prints:
  1. Label analysis: class counts, positives per class, bad target values
  2. One-batch forward+backward diagnostics: inputs, targets, logits, loss, grads, weights

Usage (on the server):
    python3 scripts/diagnose_nan.py \
        --doc-data   /local/scratch/freangi/combined_dataset/doc_large \
        --combined-data /local/scratch/freangi/combined_dataset/merged_train \
        --doc-model  /local/scratch/freangi/model_tests/regnet_on_doc_bgsub \
        --combined-model /local/scratch/freangi/model_tests/regnet_combined_bgsubtract_seed0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_testing.src.core.models import RegNetModel
from model_testing.src.data.data_utils import DataLoader as AviaNZDataLoader, SpectrogramDataset


# ─────────────────────────────────────────────────────────────────────────────
# Label analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_labels(data_folder: str, tag: str):
    print(f"\n{'='*70}")
    print(f"  LABEL ANALYSIS: {tag}")
    print(f"  folder: {data_folder}")
    print(f"{'='*70}")

    labels_path = Path(data_folder) / 'labels.json'
    if not labels_path.exists():
        print(f"  ERROR: labels.json not found at {labels_path}")
        return

    with open(labels_path) as f:
        label_data = json.load(f)

    categories = label_data.get('categories', label_data.get('species_list', []))
    files = label_data.get('files', [])
    n_classes = len(categories)
    n_files = len(files)

    print(f"\n  Classes   : {n_classes}")
    print(f"  Files     : {n_files}")
    print(f"  Class names (first 20): {categories[:20]}")

    # Build label matrix
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    positives = np.zeros(n_classes, dtype=np.int64)
    non_binary = []
    nan_targets = 0
    n_background = 0

    for entry in files:
        class_names = entry.get('class_names', [])
        if not class_names:
            n_background += 1
        for cn in class_names:
            if cn in cat_to_idx:
                positives[cat_to_idx[cn]] += 1
            else:
                pass  # class in file but not in categories

    zero_positive = [(categories[i], int(positives[i])) for i in range(n_classes) if positives[i] == 0]
    nonzero = [(categories[i], int(positives[i])) for i in range(n_classes) if positives[i] > 0]

    print(f"\n  Background (no-bird) samples: {n_background} / {n_files}")
    print(f"  Classes with ZERO positives : {len(zero_positive)}")
    if zero_positive:
        print(f"    {[c for c, _ in zero_positive[:30]]}")
    print(f"  Classes with positives      : {len(nonzero)}")

    # Show the distribution
    pos_counts = positives[positives > 0]
    if len(pos_counts) > 0:
        print(f"\n  Positive class sample counts:")
        print(f"    min={pos_counts.min()}, max={pos_counts.max()}, "
              f"mean={pos_counts.mean():.1f}, median={np.median(pos_counts):.1f}")

    # Check for classes referenced in files but missing from categories
    all_mentioned = set()
    for entry in files:
        all_mentioned.update(entry.get('class_names', []))
    orphaned = all_mentioned - set(categories)
    if orphaned:
        print(f"\n  ⚠️  Classes in file entries but NOT in categories list: {sorted(orphaned)}")

    # Spot-check a few spectrogram files
    data_dir = Path(data_folder) / 'data'
    if data_dir.exists():
        npy_files = sorted(data_dir.glob('*.npy'))[:10]
        if npy_files:
            print(f"\n  Spectrogram spot-check (first {len(npy_files)} files):")
            for npy in npy_files:
                sg = np.load(npy, allow_pickle=True)
                print(f"    {npy.name}: shape={sg.shape} dtype={sg.dtype} "
                      f"min={sg.min():.4f} max={sg.max():.4f} "
                      f"nan={np.isnan(sg).sum()} inf={np.isinf(sg).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# One-batch forward/backward diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_stats(t: torch.Tensor, name: str):
    finite = t[torch.isfinite(t)]
    print(f"    {name}:")
    print(f"      shape  = {tuple(t.shape)}")
    print(f"      nan    = {torch.isnan(t).sum().item()}")
    print(f"      inf    = {torch.isinf(t).sum().item()}")
    if finite.numel() > 0:
        print(f"      min    = {finite.min().item():.6f}")
        print(f"      max    = {finite.max().item():.6f}")
        print(f"      mean   = {finite.mean().item():.6f}")
        print(f"      std    = {finite.std().item():.6f}")
    else:
        print("      (all non-finite)")


def batch_diagnostic(data_folder: str, model_folder: str, tag: str,
                     device: str = 'cpu', bg_subtract: bool = True):
    print(f"\n{'='*70}")
    print(f"  BATCH DIAGNOSTIC: {tag}")
    print(f"  data  : {data_folder}")
    print(f"  model : {model_folder}")
    print(f"{'='*70}")

    # ── load data ──────────────────────────────────────────────────────────
    loader = AviaNZDataLoader(data_folder, noise_folder=None)
    try:
        data = loader.load_data(use_multilabel=True, validation_share=0.0)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return

    n_classes = data['nclasses']
    filenames = data['train_filenames']
    labels_arr = np.array(data['train_labels'], dtype=np.float32)
    print(f"\n  Loaded: {len(filenames)} files, {n_classes} classes")

    # Check target matrix
    print(f"\n  Target matrix stats:")
    print(f"    shape  = {labels_arr.shape}")
    print(f"    nan    = {np.isnan(labels_arr).sum()}")
    print(f"    inf    = {np.isinf(labels_arr).sum()}")
    non01 = (~np.isin(labels_arr, [0.0, 1.0]) & np.isfinite(labels_arr)).sum()
    print(f"    non-0/1 finite = {non01}")
    pos_per_class = labels_arr.sum(axis=0)
    print(f"    classes with 0 positives: {(pos_per_class == 0).sum()}")
    print(f"    total positives: {pos_per_class.sum():.0f}")
    print(f"    background samples (all-zero rows): {(labels_arr.sum(axis=1) == 0).sum()}")

    # Build one batch (first 16 samples)
    batch_size = min(16, len(filenames))
    batch_files = filenames[:batch_size]
    batch_labels = labels_arr[:batch_size]

    dataset = SpectrogramDataset(
        batch_files, batch_labels,
        img_height=224, img_width=1024, channels=1,
        cropping_mode='center',
        noise_filenames=None, noise_ratio=0.0,
        spec_transform='Log',
        training=False,
        bg_subtract=bg_subtract,
    )

    try:
        samples = [dataset[i] for i in range(batch_size)]
    except Exception as e:
        print(f"\n  ERROR loading spectrogram batch: {e}")
        return

    x = torch.stack([s[0] for s in samples])   # (B, 1, H, W) or (B, H, W)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    y = torch.tensor(batch_labels, dtype=torch.float32)

    print(f"\n  Input batch:")
    _tensor_stats(x, "x")

    print(f"\n  Target batch:")
    _tensor_stats(y, "y")
    print(f"    positives in batch: {(y > 0).sum().item()}")

    # ── load or build model ─────────────────────────────────────────────────
    model_path = Path(model_folder)
    cfg_path = model_path / 'regnet_model_config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_n_classes = cfg['num_classes']
        model_name = cfg.get('model_name', 'regnety_008')
    else:
        model_n_classes = n_classes
        model_name = 'regnety_008'
        print(f"  WARNING: no model config found, using n_classes={n_classes}")

    print(f"\n  Model: {model_name}, {model_n_classes} classes")

    # Try to load saved checkpoint
    ckpt_paths = [
        model_path / 'regnet_model_best.pt',
        model_path / 'regnet_model.pt',
    ]
    loaded_ckpt = None
    for ckpt in ckpt_paths:
        if ckpt.exists():
            loaded_ckpt = ckpt
            break

    model = RegNetModel(model_n_classes, pretrained_path=None, model_name=model_name)
    if loaded_ckpt:
        sd = torch.load(loaded_ckpt, map_location='cpu', weights_only=True)
        try:
            model.load_state_dict(sd)
            print(f"  Loaded checkpoint: {loaded_ckpt.name}")
        except RuntimeError as e:
            print(f"  WARNING: load_state_dict failed ({e}); using random init")
    else:
        print("  No checkpoint found, using random init")

    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    # ── forward pass (eval mode, no grad) ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits_eval = model(x)
        if isinstance(logits_eval, tuple):
            logits_eval = logits_eval[0]
    print(f"\n  Logits (eval mode, saved weights):")
    _tensor_stats(logits_eval, "logits")

    # ── forward + backward (train mode) ─────────────────────────────────────
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    logits_train = model(x)
    if isinstance(logits_train, tuple):
        logits_train = logits_train[0]

    print(f"\n  Logits (train mode):")
    _tensor_stats(logits_train, "logits")

    # Match target size to logits (in case of class count mismatch)
    if y.shape[1] != logits_train.shape[1]:
        print(f"\n  ⚠️  CLASS MISMATCH: targets have {y.shape[1]} classes, "
              f"model outputs {logits_train.shape[1]} classes!")
        # Truncate/pad targets to match model output
        if y.shape[1] > logits_train.shape[1]:
            y_matched = y[:, :logits_train.shape[1]]
            print(f"     Truncating targets to first {logits_train.shape[1]} classes for loss calc")
        else:
            pad = torch.zeros(y.shape[0], logits_train.shape[1] - y.shape[1], device=device)
            y_matched = torch.cat([y, pad], dim=1)
            print(f"     Padding targets with zeros to {logits_train.shape[1]} classes for loss calc")
    else:
        y_matched = y

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits_train, y_matched)
    print(f"\n  Loss:")
    print(f"    value  = {loss.item():.6f}")
    print(f"    nan    = {torch.isnan(loss).item()}")
    print(f"    inf    = {torch.isinf(loss).item()}")

    if not torch.isnan(loss) and not torch.isinf(loss):
        loss.backward()

        # Gradient stats
        all_grads = []
        nan_grad_layers = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad
                if torch.isnan(g).any() or torch.isinf(g).any():
                    nan_grad_layers.append(name)
                all_grads.append(g.abs().max().item())

        max_grad = max(all_grads) if all_grads else float('nan')
        print(f"\n  Gradients after backward:")
        print(f"    max abs grad = {max_grad:.6f}")
        print(f"    nan/inf grad layers ({len(nan_grad_layers)}): "
              f"{nan_grad_layers[:5]}")

        optimizer.step()

        # Weight stats after step
        all_w = []
        nan_w_layers = []
        for name, param in model.named_parameters():
            w = param.data
            if torch.isnan(w).any() or torch.isinf(w).any():
                nan_w_layers.append(name)
            all_w.append(w.abs().max().item())
        max_w = max(all_w) if all_w else float('nan')
        print(f"\n  Weights after optimizer step:")
        print(f"    max abs weight = {max_w:.6f}")
        print(f"    nan/inf layers ({len(nan_w_layers)}): {nan_w_layers[:5]}")
    else:
        print("  (skipping backward — loss is nan/inf)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--doc-data',       required=True,
                        help='Path to doc_large data folder')
    parser.add_argument('--combined-data',  required=True,
                        help='Path to merged_train data folder')
    parser.add_argument('--doc-model',      required=True,
                        help='Path to regnet_on_doc_bgsub experiment folder')
    parser.add_argument('--combined-model', required=True,
                        help='Path to regnet_combined experiment folder')
    parser.add_argument('--no-bg-subtract', action='store_true',
                        help='Skip background subtraction (to isolate its effect)')
    parser.add_argument('--device', default='cpu',
                        help='torch device (default: cpu; use cuda:0 for GPU)')
    args = parser.parse_args()

    bg = not args.no_bg_subtract

    # Label analysis
    analyse_labels(args.doc_data,      'DOC-only (doc_large)')
    analyse_labels(args.combined_data, 'Combined (merged_train)')

    # Batch diagnostics
    batch_diagnostic(args.doc_data,      args.doc_model,
                     'DOC-only',      device=args.device, bg_subtract=bg)
    batch_diagnostic(args.combined_data, args.combined_model,
                     'Combined',      device=args.device, bg_subtract=bg)


if __name__ == '__main__':
    main()
