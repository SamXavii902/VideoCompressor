"""
adaptive.py — Hybrid adaptive compression pipeline.

Analyses inter-frame motion to decide the best strategy per GOP:
  • Low  motion (static scene)  → heavy PCA  (aggressive temporal compression)
  • High motion (action scene)  → heavy K-Means (preserve per-frame colour detail)

Memory-efficient: reads one GOP at a time from the video file.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from config import (
    ADAPTIVE_CALIBRATION_SECONDS,
    ADAPTIVE_STATIC_VARIANCE,
    ADAPTIVE_DYNAMIC_VARIANCE,
    auto_calibrate_mse_threshold,
    choose_gop_size,
    choose_k,
    PCA_TARGET_HEIGHT,
    KMEANS_MAX_ITER,
    KMEANS_BATCH_SIZE,
)
from utils.metrics import calculate_mse
from utils.video_io import sample_frames
from modules.spatial import quantize_frame, fit_kmeans_from_video
from modules.temporal import compress_gop, _downscale


# ─────────────────────────────────────────────────────────────────────
# Main adaptive pipeline — streaming
# ─────────────────────────────────────────────────────────────────────

def adaptive_compress_streaming(video_path, output_path, fps,
                                quality="medium", max_frames=None):
    """
    Hybrid adaptive compression — streaming from video file.

    1. Auto-calibrate MSE threshold from first few seconds.
    2. For each GOP: read frames → classify → compress → write.
    3. Static GOPs → heavy PCA; Dynamic GOPs → K-Means.

    Parameters
    ----------
    video_path : str
    output_path : str
    fps : float
    quality : str
    max_frames : int or None

    Returns
    -------
    frame_count : int
    report : dict
    """
    gop_size = choose_gop_size(fps)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    # ── 1.  Determine output dimensions ─────────────────────────────
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ds = _downscale(first_frame, PCA_TARGET_HEIGHT)
    out_h, out_w = ds.shape[:2]

    # ── 2.  Auto-calibrate MSE threshold ─────────────────────────────
    cal_count = max(2, int(fps * ADAPTIVE_CALIBRATION_SECONDS))
    cal_count = min(cal_count, total)
    cal_mses = []
    prev_frame = None

    for i in range(cal_count):
        ret, frame = cap.read()
        if not ret:
            break
        ds_frame = _downscale(frame, PCA_TARGET_HEIGHT)
        if prev_frame is not None:
            cal_mses.append(calculate_mse(prev_frame, ds_frame))
        prev_frame = ds_frame

    if cal_mses:
        mse_threshold = auto_calibrate_mse_threshold(cal_mses)
    else:
        mse_threshold = 100.0

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    del prev_frame

    # ── 3.  Pre-fit K-Means for dynamic GOPs ─────────────────────────
    sampled = sample_frames(video_path, n=5,
                            resize_height=PCA_TARGET_HEIGHT)
    k_value = choose_k(sampled, quality=quality)
    kmeans = fit_kmeans_from_video(video_path, k_value,
                                  n_sample_frames=10,
                                  pixel_sample_ratio=0.05)
    del sampled

    # ── 4.  Stream GOPs ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    n_gops = int(np.ceil(total / gop_size))
    gop_classes = []
    gop_mses = []
    gop_slices = []
    strategies = []
    frame_count = 0

    for g in tqdm(range(n_gops), desc="Adaptive compress",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        # read GOP
        gop_frames = []
        start_frame = frame_count
        for _ in range(gop_size):
            ret, frame = cap.read()
            if not ret:
                break
            ds_frame = _downscale(frame, PCA_TARGET_HEIGHT)
            gop_frames.append(ds_frame)
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        if not gop_frames:
            break

        end_frame = start_frame + len(gop_frames)
        gop_slices.append((start_frame, end_frame))

        # classify GOP
        if len(gop_frames) < 2:
            avg_mse = 0.0
            cls = "static"
        else:
            mses = []
            for i in range(len(gop_frames) - 1):
                mses.append(calculate_mse(gop_frames[i], gop_frames[i + 1]))
            avg_mse = float(np.mean(mses))
            cls = "static" if avg_mse < mse_threshold else "dynamic"

        gop_classes.append(cls)
        gop_mses.append(avg_mse)

        # compress
        if cls == "static" and len(gop_frames) >= 2:
            reconstructed, _ = compress_gop(
                gop_frames, variance_target=ADAPTIVE_STATIC_VARIANCE)
            for f in reconstructed:
                writer.write(f)
            strategies.append("PCA (static)")
        else:
            for frame in gop_frames:
                q_frame, _ = quantize_frame(frame, kmeans)
                writer.write(q_frame)
            strategies.append("K-Means (dynamic)")

        del gop_frames
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    writer.release()

    n_static = gop_classes.count("static")
    n_dynamic = gop_classes.count("dynamic")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  ✓ Adaptive compression complete  "
          f"({frame_count} frames, {n_static} static / {n_dynamic} dynamic)")
    print(f"    → {output_path}  ({size_mb:.1f} MB)")

    report = {
        "gop_classes":   gop_classes,
        "gop_mses":      gop_mses,
        "gop_slices":    gop_slices,
        "mse_threshold": mse_threshold,
        "strategies":    strategies,
        "k_used":        k_value,
        "gop_size":      gop_size,
    }
    return frame_count, report
