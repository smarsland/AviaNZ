import os
import sys

import torch


def load_pretrained_state_dict(pretrained_path, logger=None):
    if not pretrained_path:
        return None

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

    if logger is None:
        logger = print

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    except (RuntimeError, pickle.UnpicklingError, AttributeError, EOFError, ValueError) as exc:
        logger(f"Warning: could not load pretrained checkpoint {pretrained_path}: {exc}")
        return None

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    if not isinstance(checkpoint, dict):
        logger(f"Warning: pretrained checkpoint {pretrained_path} did not contain a state dict")
        return None

    state_dict = checkpoint
    if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']

    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        state_dict = state_dict['model']

    if 'model_state_dict' in state_dict and isinstance(state_dict['model_state_dict'], dict):
        state_dict = state_dict['model_state_dict']

    if not all(isinstance(v, torch.Tensor) for v in state_dict.values() if isinstance(v, torch.Tensor) or True):
        logger(f"Warning: pretrained checkpoint {pretrained_path} did not contain tensor state dict entries")
        return None

    return state_dict
