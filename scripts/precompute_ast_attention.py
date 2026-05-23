#!/usr/bin/env python
"""Precompute raw AST attention maps for all spectrogram files in a dataset.

The attention map is extracted from the raw pre-trained AST
(MIT/ast-finetuned-audioset-10-10-0.4593) without any task-specific fine-tuning.
It captures which time-frequency regions the AudioSet-pretrained model finds most
salient.  We take the CLS-token attention weights from the last transformer layer,
averaged over all 12 attention heads, reshape from the patch grid to the full
spectrogram size (H x W), and save as float32 .npy files.

Usage:
    PYTHONPATH="$PWD" python scripts/precompute_ast_attention.py \\
        /data/avianz_split/train /data/doc_split/train /data/avianz_split/test /data/doc_split/test \\
        --out-dir /data/ast_attn_cache_boxcox \\
        --spec-transform Box-Cox

All output maps are stored flat in --out-dir/<basename>.npy.  Because training and
test basenames are expected to be globally unique, a single directory works for all
data splits.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ASTModel

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.core import config


# Target spectrogram dimensions (must match training pipeline)
H = config.DEFAULT_FREQ_BINS   # 224
W = config.DEFAULT_TIME_BINS   # 1024


# ---------------------------------------------------------------------------
# Positional-embedding interpolation (mirrors AST.interpolate_pos_embed)
# ---------------------------------------------------------------------------

def _interpolate_pos_embed(ast_model: ASTModel, target_size: tuple):
    """Resize positional embeddings to match target_size = (H, W)."""
    pos_embed = ast_model.embeddings.position_embeddings   # (1, N+n_special, C)
    device = pos_embed.device
    dtype  = pos_embed.dtype
    _, N_full, C = pos_embed.shape
    n_special = 2                           # cls + distillation tokens
    num_old_patches = N_full - n_special

    projection = ast_model.embeddings.patch_embeddings.projection
    patch_size = projection.kernel_size     # (ph, pw)
    stride     = projection.stride         # (sh, sw)

    h_new = (target_size[0] - patch_size[0]) // stride[0] + 1
    w_new = (target_size[1] - patch_size[1]) // stride[1] + 1

    # Infer original grid
    if (hasattr(ast_model.config, 'num_mel_bins')
            and hasattr(ast_model.config, 'max_length')):
        h_old = (ast_model.config.num_mel_bins - patch_size[0]) // stride[0] + 1
        w_old = (ast_model.config.max_length   - patch_size[1]) // stride[1] + 1
    elif num_old_patches == 1212:
        h_old, w_old = 12, 101
    else:
        h_old, w_old = None, None
        for h in range(16, 7, -1):
            if num_old_patches % h == 0:
                w = num_old_patches // h
                if 50 <= w <= 200:
                    h_old, w_old = h, w
                    break
        if h_old is None:
            for h in range(1, int(num_old_patches ** 0.5) + 1):
                if num_old_patches % h == 0:
                    h_old, w_old = h, num_old_patches // h
                    break

    if h_old == h_new and w_old == w_new:
        print(f"  Positional embeddings already match {h_old}x{w_old}")
        ast_model._h_patches = h_old
        ast_model._w_patches = w_old
        return

    print(f"  Interpolating positional embeddings {h_old}x{w_old} → {h_new}x{w_new}")
    special_tokens = pos_embed[:, :n_special, :]
    pos_tokens     = pos_embed[:, n_special:, :]

    pos_tokens = pos_tokens.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(h_new, w_new),
                               mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_new * w_new, C)

    new_pos = torch.cat([special_tokens, pos_tokens], dim=1).to(device=device, dtype=dtype)
    ast_model.embeddings.position_embeddings = torch.nn.Parameter(new_pos)

    ast_model._h_patches = h_new
    ast_model._w_patches = w_new


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ast_model(device: torch.device) -> ASTModel:
    print("Loading MIT/ast-finetuned-audioset-10-10-0.4593 ...")
    model = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        attn_implementation="eager",
    )
    model.eval()
    if device.type == 'cuda':
        try:
            model.to(device)
        except Exception as e:
            print(f"  WARNING: Could not move model to {device} ({e})")
            print("  Falling back to CPU — precompute will be slower but correct.")
    else:
        model.to(device)
    _interpolate_pos_embed(model, (H, W))
    actual_device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  AST loaded: {n_params / 1e6:.1f}M parameters  (device: {actual_device})")
    return model


# ---------------------------------------------------------------------------
# Spectrogram transforms (mirrors SpectrogramDataset.apply_spec_transform)
# ---------------------------------------------------------------------------

def _apply_spec_transform(x: np.ndarray, transform: str) -> np.ndarray:
    LOG_OFFSET = 1e-7
    if transform == 'Log':
        return np.log(np.maximum(x, 0.0) + LOG_OFFSET)
    if transform == 'Box-Cox':
        from scipy.stats import boxcox
        flat = np.maximum(x.flatten() + LOG_OFFSET, 1e-10)
        return boxcox(flat, 0.5).reshape(x.shape)
    if transform == 'PCEN':
        from scipy import signal
        gain, bias, power, t, eps = 0.8, 10, 0.25, 0.060, 1e-6
        hop = int(0.010 * 16000)
        s = 1 - np.exp(-hop / (t * 16000))
        M = signal.lfilter([s], [1, s - 1], x, axis=1)
        smooth = (eps + M) ** (-gain)
        return (x * smooth + bias) ** power - bias ** power
    return x   # 'None' or unknown


def _pad_to_hw(x: np.ndarray) -> np.ndarray:
    """Zero-pad (and/or trim) a 2-D spectrogram to exactly (H, W)."""
    h, w = x.shape
    if h < H:
        x = np.concatenate([x, np.zeros((H - h, w), dtype=x.dtype)], axis=0)
    if w < W:
        x = np.concatenate([x, np.zeros((H, W - w), dtype=x.dtype)], axis=1)
    return x[:H, :W]


# ---------------------------------------------------------------------------
# Attention map extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_attention_map(
    model: ASTModel,
    spec_2d: np.ndarray,
    transform: str,
) -> np.ndarray:
    """Return a (H, W) float32 attention map in [0, 1]."""
    device = next(model.parameters()).device
    x = _apply_spec_transform(spec_2d, transform)
    x = (x - config.AST_MEAN) / config.AST_STD
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)  # (1, H, W)

    outputs = model(x_t, output_attentions=True)

    # Last transformer layer: (1, n_heads, N, N)  N = n_special + n_patches
    last_attn = outputs.attentions[-1].squeeze(0)   # (n_heads, N, N)

    # Average over heads; take CLS (index 0) attention to patch tokens (index 2:)
    cls_attn = last_attn.mean(dim=0)[0, 2:]         # (n_patches,)

    # Normalise to [0, 1]
    cls_attn = cls_attn - cls_attn.min()
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)

    # Reshape to patch grid
    h_p = getattr(model, '_h_patches', 21)
    w_p = getattr(model, '_w_patches', 101)
    if cls_attn.shape[0] != h_p * w_p:
        # Fallback: largest factor pair
        total = cls_attn.shape[0]
        h_p = int(total ** 0.5)
        while h_p > 1 and total % h_p != 0:
            h_p -= 1
        w_p = total // h_p

    patch_map = cls_attn[:h_p * w_p].reshape(1, 1, h_p, w_p).float()

    # Upsample to (H, W)
    attn_full = F.interpolate(patch_map, size=(H, W),
                              mode='bilinear', align_corners=False)
    return attn_full.squeeze().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def process_directory(
    data_dir: str,
    out_dir: str,
    transform: str,
    device: torch.device,
    model: ASTModel,
    force: bool,
):
    """Process all .npy files under <data_dir>/data/ and save to out_dir."""
    src = Path(data_dir) / 'data'
    if not src.exists():
        print(f"WARNING: {src} does not exist, skipping")
        return

    npy_files = sorted(src.rglob('*.npy'))
    if not npy_files:
        print(f"WARNING: No .npy files found in {src}")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(npy_files)} files  |  source: {src}")
    print(f"Output dir: {out_dir}")

    errors = 0
    iterator = (tqdm(npy_files, desc="Attn maps", unit="file")
                if HAS_TQDM else npy_files)

    for npy_path in iterator:
        out_path = Path(out_dir) / npy_path.name
        if out_path.exists() and not force:
            continue

        try:
            raw = np.load(npy_path).astype(np.float32)
            while raw.ndim > 2:
                raw = raw.squeeze()
            raw = _pad_to_hw(raw)
            attn = compute_attention_map(model, raw, transform)
            np.save(out_path, attn)
        except Exception as exc:
            if not HAS_TQDM:
                print(f"ERROR {npy_path.name}: {exc}")
            errors += 1

    total = len(npy_files)
    print(f"Done.  Errors: {errors}/{total}  |  Saved: {total - errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute raw AST (AudioSet-pretrained) attention maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'data_dirs', nargs='+',
        help="Dataset directories (each must contain a data/ subfolder)"
    )
    parser.add_argument(
        '--out-dir', required=True,
        help="Single output directory; all maps saved flat as <basename>.npy"
    )
    parser.add_argument(
        '--spec-transform', default='Box-Cox',
        choices=['Log', 'PCEN', 'Box-Cox', 'None'],
        help="Spectral transform applied before AST normalisation (default: Box-Cox)"
    )
    parser.add_argument(
        '--force', action='store_true',
        help="Recompute and overwrite existing .npy files"
    )
    args = parser.parse_args()

    device = torch.device('cpu')
    if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        from src.core.utils import pick_free_gpu
        chosen = pick_free_gpu()
        if chosen is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(chosen)
            device = torch.device('cuda:0')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(f"Requested device: {device}")

    model = load_ast_model(device)
    # After potential CPU fallback, derive the real device from the model
    device = next(model.parameters()).device

    for data_dir in args.data_dirs:
        process_directory(data_dir, args.out_dir, args.spec_transform,
                          device, model, args.force)

    print("\nAll done.")


if __name__ == '__main__':
    main()
