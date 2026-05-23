"""Shared utilities for device selection and other common tasks."""

import os
import shutil
import subprocess


def pick_free_gpu():
    """Return the index of the GPU with the lowest memory usage, via nvidia-smi.

    Sets CUDA_VISIBLE_DEVICES to the chosen GPU index so the CUDA runtime only
    ever sees one device.  Must be called *before* any torch.cuda API.

    Raises RuntimeError if nvidia-smi cannot be found or fails.
    """
    smi = shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi'
    if not os.path.isfile(smi):
        raise RuntimeError(
            "nvidia-smi not found — cannot auto-select a free GPU. "
            "Set CUDA_VISIBLE_DEVICES manually before running."
        )

    try:
        out = subprocess.check_output(
            [smi, '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed:\n{e.output}") from e

    best_idx, best_mem = None, float('inf')
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 2:
            continue
        try:
            idx, mem = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        if mem < best_mem:
            best_mem, best_idx = mem, idx

    if best_idx is None:
        raise RuntimeError(f"Could not parse nvidia-smi output:\n{out}")

    print(f"Auto-selected GPU {best_idx} ({best_mem} MiB used)")
    return best_idx
