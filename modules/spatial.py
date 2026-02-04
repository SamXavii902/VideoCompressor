"""
spatial.py — Spatial compression via K-Means colour quantization.

Exploits spatial redundancy: most of the 16.7 M possible 24-bit colours
in a frame are perceptually indistinguishable.  K-Means replaces each
pixel's colour with the nearest cluster centroid, reducing the colour
palette to K representative colours.

Memory-efficient: streams frames one by one through the quantizer and
writes directly to the output video. Never holds all frames in RAM.
"""

import os
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config import (
    KMEANS_MAX_ITER,
    KMEANS_BATCH_SIZE,
    choose_k,
)
from utils.video_io import sample_frames


def quantize_frame(frame, kmeans_model):
    """
    Quantize a single BGR frame using a pre-fitted K-Means model.

    Parameters
    ----------
    frame : np.ndarray   (H, W, 3) uint8
    kmeans_model : MiniBatchKMeans (fitted)

    Returns
    -------
    quantized : np.ndarray  (H, W, 3) uint8
    labels    : np.ndarray  (H*W,) int32  — index per pixel
    """
    h, w, c = frame.shape
    pixels = frame.reshape(-1, c).astype(np.float32)

    labels = kmeans_model.predict(pixels)
    centres = kmeans_model.cluster_centers_.astype(np.uint8)

    quantized = centres[labels].reshape(h, w, c)
    return quantized, labels


def fit_kmeans_from_video(video_path, k, n_sample_frames=10,
                          pixel_sample_ratio=0.05):
    """
    Fit a MiniBatchKMeans model by sampling pixels directly from the video
    file (never loads all frames at once).

    Parameters
    ----------
    video_path : str
    k : int
    n_sample_frames : int
        Number of frames to sample pixels from.
    pixel_sample_ratio : float
        Fraction of pixels per frame to use.

    Returns
    -------
    MiniBatchKMeans (fitted)
    """
    print(f"\n  Fitting K-Means with K={k} …")

    sampled = sample_frames(video_path, n=n_sample_frames)

    all_pixels = []
    for frame in sampled:
        pixels = frame.reshape(-1, 3).astype(np.float32)
        n_sample = max(1, int(len(pixels) * pixel_sample_ratio))
        indices = np.random.choice(len(pixels), n_sample, replace=False)
        all_pixels.append(pixels[indices])

    all_pixels = np.vstack(all_pixels)
    print(f"  Training on {len(all_pixels):,} sampled pixels …")

    model = MiniBatchKMeans(
        n_clusters=k,
        max_iter=KMEANS_MAX_ITER,
        batch_size=KMEANS_BATCH_SIZE,
        n_init=3,
        random_state=42,
    )
    model.fit(all_pixels)
    print(f"  K-Means converged (inertia={model.inertia_:.0f})\n")
    return model


def spatial_compress_streaming(video_path, output_path, fps,
                               k=None, quality="medium",
                               max_frames=None):
    """
    Stream-based spatial compression: reads one frame at a time from the
    input video, quantizes it, and writes it immediately to the output.

    Never holds more than ~2 frames in RAM.

    Parameters
    ----------
    video_path : str
    output_path : str
    fps : float
    k : int or None
    quality : str
    max_frames : int or None

    Returns
    -------
    k_used : int
    frame_count : int
    kmeans_model : MiniBatchKMeans
    """
    # ── dynamic K selection ──────────────────────────────────────────
    if k is None:
        sampled = sample_frames(video_path, n=5)
        k = choose_k(sampled, quality=quality)
        del sampled

    # ── fit model on sampled pixels ──────────────────────────────────
    kmeans_model = fit_kmeans_from_video(video_path, k)

    # ── stream: read → quantize → write ─────────────────────────────
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    count = 0
    with tqdm(total=total, desc=f"Spatial compress (K={k})",
              unit="frame", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            q_frame, _ = quantize_frame(frame, kmeans_model)
            writer.write(q_frame)
            count += 1
            pbar.update(1)
            if max_frames is not None and count >= max_frames:
                break

    cap.release()
    writer.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Spatial compression complete  "
          f"({count} frames, K={k}) → {output_path}  ({size_mb:.1f} MB)")

    return k, count, kmeans_model


def spatial_compressed_size(frame_count, h, w, kmeans_model):
    """
    Estimate the theoretical compressed size in bytes.

    Storage = palette (K × 3 bytes) + per-pixel index (ceil(log2 K) bits).
    """
    k = kmeans_model.n_clusters
    palette_bytes = k * 3

    bits_per_index = int(np.ceil(np.log2(k)))
    total_pixels = frame_count * h * w
    index_bytes = int(np.ceil(total_pixels * bits_per_index / 8))

    total = palette_bytes + index_bytes
    return {"total_bytes": total, "palette_bytes": palette_bytes,
            "index_bytes": index_bytes, "bits_per_pixel": bits_per_index,
            "k": k}


def spatial_compress(frames, k=None, quality="medium"):
    """
    Legacy in-memory spatial compression solely used by analysis/plots.py.
    """
    if k is None:
        k = choose_k(frames, quality)
        
    pixels = np.vstack([f.reshape(-1, 3).astype(np.float32) for f in frames])
    model = MiniBatchKMeans(
        n_clusters=k, max_iter=KMEANS_MAX_ITER, batch_size=KMEANS_BATCH_SIZE, n_init=3, random_state=42
    )
    model.fit(pixels)
    
    quantized_frames = []
    labels_list = []
    for f in frames:
        q, l = quantize_frame(f, model)
        quantized_frames.append(q)
        labels_list.append(l)
        
    return quantized_frames, labels_list, model, k
