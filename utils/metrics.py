"""
metrics.py — Quality and compression metrics: PSNR, SSIM, MSE,
              compression ratio.
"""

import numpy as np
from tqdm import tqdm


def calculate_mse(frame_a, frame_b):
    """
    Mean Squared Error between two frames (uint8 or float).

    Returns
    -------
    float
    """
    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def calculate_psnr(original, compressed, max_pixel=255.0):
    """
    Peak Signal-to-Noise Ratio between a single original and compressed frame.

    Returns
    -------
    float
        PSNR in dB.  Returns float('inf') if frames are identical.
    """
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((max_pixel ** 2) / mse)


def calculate_ssim(original, compressed):
    """
    Structural Similarity Index between two BGR frames.

    Uses scikit-image's implementation for accuracy.

    Returns
    -------
    float
        SSIM in [0, 1].
    """
    from skimage.metrics import structural_similarity as ssim
    # convert to grayscale for SSIM (standard practice)
    import cv2
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_comp = compressed
    if len(compressed.shape) == 3:
        if compressed.dtype != np.uint8:
            compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        gray_comp = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    return float(ssim(gray_orig, gray_comp))


def batch_psnr(original_frames, compressed_frames, desc="Computing PSNR"):
    """
    Compute per-frame PSNR for two lists of frames.

    Returns
    -------
    list[float]
    """
    n = min(len(original_frames), len(compressed_frames))
    values = []
    for i in tqdm(range(n), desc=desc, unit="frame",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        values.append(calculate_psnr(original_frames[i],
                                     compressed_frames[i]))
    return values


def batch_ssim(original_frames, compressed_frames, desc="Computing SSIM"):
    """
    Compute per-frame SSIM for two lists of frames.

    Returns
    -------
    list[float]
    """
    n = min(len(original_frames), len(compressed_frames))
    values = []
    for i in tqdm(range(n), desc=desc, unit="frame",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        values.append(calculate_ssim(original_frames[i],
                                     compressed_frames[i]))
    return values


def batch_mse(frames, desc="Computing inter-frame MSE"):
    """
    Compute MSE between each pair of consecutive frames.

    Parameters
    ----------
    frames : list[np.ndarray]

    Returns
    -------
    list[float]
        Length = len(frames) - 1
    """
    values = []
    for i in tqdm(range(len(frames) - 1), desc=desc, unit="pair",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        values.append(calculate_mse(frames[i], frames[i + 1]))
    return values


def calculate_compression_ratio(original_frames, compressed_representation):
    """
    Theoretical compression ratio.

    Parameters
    ----------
    original_frames : list[np.ndarray]
        The raw frames (used to compute original raw size).
    compressed_representation : dict
        Must contain a 'total_bytes' key with the size of the
        compressed payload.

    Returns
    -------
    float
        ratio = original_size / compressed_size
    """
    original_bytes = sum(f.nbytes for f in original_frames)
    compressed_bytes = compressed_representation.get("total_bytes", 1)
    ratio = original_bytes / compressed_bytes
    print(f"  Compression ratio: {ratio:.2f}x  "
          f"({original_bytes / 1e6:.1f} MB → "
          f"{compressed_bytes / 1e6:.1f} MB)")
    return ratio


def average_metric(values):
    """Average a list of metric values, ignoring infinities."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return 0.0
    return float(np.mean(finite))
