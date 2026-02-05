"""
temporal.py — Temporal compression via PCA (Principal Component Analysis).

Exploits temporal redundancy: consecutive frames share most of their
pixel data (static background, slow motion).  PCA identifies the
principal directions of variation across a Group of Pictures (GOP)
and discards the least significant components.

Memory-efficient: processes one GOP at a time, reading directly from
the video file.  Downscales to 1080p for PCA computation.
"""

import os
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from config import (
    PCA_TARGET_HEIGHT,
    VARIANCE_THRESHOLD_INIT,
    VARIANCE_THRESHOLD_MIN,
    VARIANCE_THRESHOLD_MAX,
    VARIANCE_ADAPT_STEP,
    PCA_PSNR_LOW,
    PCA_PSNR_HIGH,
    choose_gop_size,
)
from utils.metrics import calculate_psnr


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _downscale(frame, target_h):
    """Downscale a frame to *target_h* keeping aspect ratio."""
    h, w = frame.shape[:2]
    if h <= target_h:
        return frame
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h),
                      interpolation=cv2.INTER_AREA)


def _frames_to_matrix(frames):
    """Stack frames into a 2-D matrix for PCA."""
    shape = frames[0].shape
    vecs = [f.astype(np.float32).ravel() for f in frames]
    return np.vstack(vecs), shape


def _matrix_to_frames(matrix, shape):
    """Reshape rows back into (H, W, 3) uint8 frames."""
    frames = []
    for i in range(matrix.shape[0]):
        f = matrix[i].reshape(shape)
        f = np.clip(f, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def _select_n_components(explained_variance_ratio, target_variance):
    """Return smallest N capturing ≥ target_variance."""
    cumsum = np.cumsum(explained_variance_ratio)
    n = int(np.searchsorted(cumsum, target_variance) + 1)
    return min(n, len(explained_variance_ratio))


# ─────────────────────────────────────────────────────────────────────
# Core PCA compression for a single GOP
# ─────────────────────────────────────────────────────────────────────

def compress_gop(frames, n_components=None, variance_target=None):
    """
    Compress a Group of Pictures with IncrementalPCA.

    Parameters
    ----------
    frames : list[np.ndarray]
        Frames in this GOP (all same size, uint8 BGR).
    n_components : int or None
    variance_target : float or None

    Returns
    -------
    reconstructed : list[np.ndarray]  uint8
    pca_data : dict
    """
    matrix, shape = _frames_to_matrix(frames)
    n_frames, n_features = matrix.shape

    max_comp = min(n_frames, n_features)

    if n_components is not None:
        n_comp = min(n_components, max_comp)
    else:
        n_comp = min(max_comp, n_frames - 1) if max_comp > 1 else 1

    ipca = IncrementalPCA(n_components=n_comp)

    # IncrementalPCA requires batch_size >= n_components
    batch_size = max(n_comp, min(n_frames, 50))
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        # If the final batch is smaller than n_comp, merge it with the previous batch
        # by adjusting the start index backwards (if possible).
        if end - start < n_comp and start > 0:
            start = max(0, end - n_comp)
        
        # Avoid fitting the exact same data again if it perfectly overlaps
        # (this only happens if n_frames < n_comp which shouldn't happen due to math above,
        # but just to be safe).
        if end - start >= n_comp:
            ipca.partial_fit(matrix[start:end])

    if variance_target is not None and n_components is None:
        n_keep = _select_n_components(
            ipca.explained_variance_ratio_, variance_target)
    else:
        n_keep = n_comp

    weights_full = ipca.transform(matrix)
    weights = weights_full[:, :n_keep]
    components = ipca.components_[:n_keep]
    mean = ipca.mean_

    reconstructed_matrix = weights @ components + mean
    reconstructed = _matrix_to_frames(reconstructed_matrix, shape)

    explained = float(np.sum(ipca.explained_variance_ratio_[:n_keep]))

    pca_data = {
        "weights":        weights,
        "components":     components,
        "mean":           mean,
        "explained_var":  explained,
        "n_components":   n_keep,
        "explained_variance_ratio": ipca.explained_variance_ratio_,
    }
    return reconstructed, pca_data


# ─────────────────────────────────────────────────────────────────────
# Streaming temporal compression
# ─────────────────────────────────────────────────────────────────────

def temporal_compress_streaming(video_path, output_path, fps,
                                variance_target=None, n_components=None,
                                gop_size=None, max_frames=None):
    """
    Stream-based temporal compression: reads one GOP at a time from
    the video file, compresses with PCA, writes to output.

    Parameters
    ----------
    video_path : str
    output_path : str
    fps : float
    variance_target : float or None
    n_components : int or None
    gop_size : int or None
    max_frames : int or None

    Returns
    -------
    total_frames : int
    all_pca_data : list[dict]
    final_variance : float
    """
    if gop_size is None:
        gop_size = choose_gop_size(fps)

    if variance_target is None and n_components is None:
        variance_target = VARIANCE_THRESHOLD_INIT

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    n_gops = int(np.ceil(total / gop_size))

    # determine output dimensions from first frame
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ds = _downscale(first_frame, PCA_TARGET_HEIGHT)
    out_h, out_w = ds.shape[:2]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    current_var = variance_target
    all_pca_data = []
    frame_count = 0

    for g in tqdm(range(n_gops), desc="PCA on GOPs",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        # read GOP frames
        gop_frames = []
        for _ in range(gop_size):
            ret, frame = cap.read()
            if not ret:
                break
            ds_frame = _downscale(frame, PCA_TARGET_HEIGHT)
            gop_frames.append(ds_frame)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        if len(gop_frames) < 2:
            for f in gop_frames:
                writer.write(f)
            continue

        reconstructed, pca_data = compress_gop(
            gop_frames, n_components=n_components,
            variance_target=current_var)
        all_pca_data.append(pca_data)

        for f in reconstructed:
            writer.write(f)

        # adaptive variance tuning
        if n_components is None and current_var is not None:
            gop_psnrs = [calculate_psnr(gop_frames[i], reconstructed[i])
                         for i in range(len(gop_frames))]
            avg_psnr = np.mean([p for p in gop_psnrs if np.isfinite(p)])
            if avg_psnr < PCA_PSNR_LOW:
                current_var = min(VARIANCE_THRESHOLD_MAX,
                                  current_var + VARIANCE_ADAPT_STEP)
            elif avg_psnr > PCA_PSNR_HIGH:
                current_var = max(VARIANCE_THRESHOLD_MIN,
                                  current_var - VARIANCE_ADAPT_STEP)

        # free memory
        del gop_frames, reconstructed

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    writer.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  ✓ Temporal compression complete  "
          f"({frame_count} frames, {len(all_pca_data)} GOPs, "
          f"final var≈{current_var:.2f}) → {output_path}  ({size_mb:.1f} MB)")

    return frame_count, all_pca_data, current_var


def temporal_compressed_size(all_pca_data):
    """Estimate compressed size: weights + components + mean."""
    total = 0
    for pd in all_pca_data:
        total += pd["weights"].nbytes
        total += pd["components"].nbytes
        total += pd["mean"].nbytes
    return {"total_bytes": total}
