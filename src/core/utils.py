"""Shared utilities for device selection and other common tasks."""

import os
import shutil
import subprocess


def pick_free_gpu():
    """Return the index of a GPU with no active compute process, preferring the
    one with lowest memory usage.  In Exclusive-Process mode a GPU can show
    near-zero memory yet be locked; nvidia-smi --query-compute-apps detects
    that reliably.

    Must be called *before* any torch.cuda API so that CUDA_VISIBLE_DEVICES can
    still redirect the runtime.

    Returns None if nvidia-smi is unavailable or all GPUs are occupied.
    """
    smi = shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi'
    if not os.path.isfile(smi):
        return None

    # --- memory per GPU ---
    mem_by_idx = {}
    try:
        out = subprocess.check_output(
            [smi, '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True,
        )
        for line in out.splitlines():
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    mem_by_idx[int(parts[0].strip())] = int(parts[1].strip())
                except ValueError:
                    pass
    except subprocess.CalledProcessError:
        pass

    if not mem_by_idx:
        return None

    # --- GPUs that already have an active compute process ---
    occupied = set()
    try:
        out = subprocess.check_output(
            [smi, '--query-compute-apps=gpu_index', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True,
        )
        for line in out.splitlines():
            line = line.strip()
            if line:
                try:
                    occupied.add(int(line))
                except ValueError:
                    pass
    except subprocess.CalledProcessError:
        pass  # if this query fails, treat all as potentially free

    # Prefer GPUs with no active process; among ties prefer lowest memory
    free_gpus = {i: m for i, m in mem_by_idx.items() if i not in occupied}
    candidates = free_gpus if free_gpus else mem_by_idx

    best_idx = min(candidates, key=lambda i: candidates[i])
    best_mem = candidates[best_idx]

    status = "no active process" if best_idx not in occupied else "active process (all GPUs occupied)"
    print(f"Auto-selected GPU {best_idx} ({best_mem} MiB used, {status})")
    return best_idx
