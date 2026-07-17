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
    # --query-compute-apps only supports gpu_uuid, not gpu_index, so we
    # cross-reference with the per-GPU UUID list.
    uuid_to_idx = {}
    try:
        out = subprocess.check_output(
            [smi, '--query-gpu=index,gpu_uuid', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True,
        )
        for line in out.splitlines():
            parts = line.strip().split(',')
            if len(parts) == 2:
                try:
                    uuid_to_idx[parts[1].strip()] = int(parts[0].strip())
                except ValueError:
                    pass
    except subprocess.CalledProcessError:
        pass

    occupied = set()
    try:
        out = subprocess.check_output(
            [smi, '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'],
            stderr=subprocess.STDOUT, text=True,
        )
        for line in out.splitlines():
            uuid = line.strip()
            if uuid and uuid in uuid_to_idx:
                occupied.add(uuid_to_idx[uuid])
    except subprocess.CalledProcessError:
        pass

    # Prefer GPUs with no active process; among ties prefer lowest memory
    free_gpus = {i: m for i, m in mem_by_idx.items() if i not in occupied}

    if not free_gpus:
        lines = [f"  GPU {i}: {mem_by_idx[i]} MiB used" for i in sorted(mem_by_idx)]
        raise RuntimeError(
            "All GPUs are occupied (exclusive process mode). "
            "Try a different server.\n" + "\n".join(lines)
        )

    best_idx = min(free_gpus, key=lambda i: free_gpus[i])
    best_mem = free_gpus[best_idx]
    print(f"Auto-selected GPU {best_idx} ({best_mem} MiB used, no active process)")
    return best_idx
