import torch


def get_device():
    """ Select the best available compute device: CUDA > MPS (Apple Silicon) > CPU. """
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
            return torch.device('cuda')
        except Exception as e:
            print(f"CUDA reported available but failed initialisation ({e}), falling back.")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using device: mps (Apple Silicon GPU)")
        return torch.device('mps')

    print("Using device: cpu")
    return torch.device('cpu')
