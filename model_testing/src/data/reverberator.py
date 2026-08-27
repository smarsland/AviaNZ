import numpy as np

def apply_reverb(img, delay_mean=5, delay_std=4, length=30, decay=0.35):
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape

    # Estimate background and loud pixels
    n_bg_pixels = max(1, W // 10)
    sorted_pixels = np.sort(img, axis=1)
    bg_pixels = sorted_pixels[:, :n_bg_pixels]
    mu0 = np.mean(bg_pixels, axis=1, keepdims=True)
    std0 = np.sqrt(np.var(bg_pixels, axis=1, keepdims=True))
    z = (img - mu0) / (std0 + 1e-6)
    mask = z > 4.0

    # Make log-normal delay distribution
    sigma = np.sqrt(np.log(1.0 + (delay_std / delay_mean) ** 2))
    mu = (np.log(delay_mean) - 0.5 * sigma ** 2)

    # Make the main reverb kernel
    delays = np.arange(1, length + 1, dtype=np.float32)
    kernel = (np.exp(-(np.log(delays) - mu) ** 2 / (2.0 * sigma ** 2)) / (delays * sigma * np.sqrt(2.0 * np.pi)))
    kernel /= kernel.sum()
    kernel *= decay
    reverb = np.zeros_like(img, dtype=np.float32)

    # Apply
    for i, delay in enumerate(range(1, length + 1)):
        amount = kernel[i]
        source = img[:, :-delay]
        source_mask = mask[:, :-delay]
        reverb[:, delay:] += (np.where(source_mask, source, 0.0) * amount)

    return img + reverb