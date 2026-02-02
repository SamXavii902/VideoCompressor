"""
config.py — Default parameters and dynamic calibration logic for the
Adaptive Video Compression Suite.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Spatial (K-Means) defaults
# ─────────────────────────────────────────────────────────────────────
K_CANDIDATES = [8, 16, 32, 64, 128]

QUALITY_BIAS = {
    "low":    -1,   # shift index towards fewer colours
    "medium":  0,
    "high":    1,   # shift index towards more colours
}

KMEANS_MAX_ITER = 100
KMEANS_BATCH_SIZE = 4096

# ─────────────────────────────────────────────────────────────────────
# Temporal (PCA) defaults
# ─────────────────────────────────────────────────────────────────────
VARIANCE_THRESHOLD_INIT = 0.95      # starting target for explained variance
VARIANCE_THRESHOLD_MIN  = 0.80
VARIANCE_THRESHOLD_MAX  = 0.99
VARIANCE_ADAPT_STEP     = 0.01      # how much to nudge per iteration

PCA_PSNR_LOW  = 30.0   # if PSNR < this ⇒ increase variance threshold
PCA_PSNR_HIGH = 40.0   # if PSNR > this ⇒ decrease variance threshold
# We lower resolution heavily before PCA because PCA over millions of pixels
# takes excessive memory and time.
PCA_TARGET_HEIGHT = 240  # Changed from 480 to 240 to fix 1 GiB numpy memory errors

# ─────────────────────────────────────────────────────────────────────
# Adaptive / Hybrid defaults
# ─────────────────────────────────────────────────────────────────────
ADAPTIVE_CALIBRATION_SECONDS = 2    # seconds of video used to calibrate MSE

# Variance thresholds used by adaptive mode
ADAPTIVE_STATIC_VARIANCE  = 0.85    # heavy PCA for static GOPs
ADAPTIVE_DYNAMIC_VARIANCE = 0.98    # lighter PCA for dynamic GOPs

# ─────────────────────────────────────────────────────────────────────
# Block Motion Estimation & Residual (Codec) defaults
# ─────────────────────────────────────────────────────────────────────
BME_BLOCK_SIZE     = 16      # 16x16 macroblocks
BME_SEARCH_RANGE   = 16      # +/- 16 pixels search window
BME_METRIC         = "sad"   # Sum of Absolute Differences

# "block_matching" or "optical_flow"
MOTION_METHOD      = "optical_flow"

DCT_BLOCK_SIZE     = 8       # Standard 8x8 DCT blocks
DCT_QUALITY_FACTOR = 50      # JPEG-style quantization quality (1-100)

TARGET_BITRATE_KBPS = None   # For RDO: Target kilobits per second. If set, ignores DCT_QUALITY_FACTOR.

# ─────────────────────────────────────────────────────────────────────
# GPU Backend defaults
# ─────────────────────────────────────────────────────────────────────
USE_GPU        = True        # Try to use CUDA if available
GPU_BATCH_SIZE = 64          # Blocks per batch during motion search

# ─────────────────────────────────────────────────────────────────────
# Analysis / plots
# ─────────────────────────────────────────────────────────────────────
ANALYSIS_K_VALUES = [4, 8, 16, 32, 64, 128]
ANALYSIS_N_VALUES = [2, 5, 10, 20, 50, 100]      # fixed component counts
ANALYSIS_SAMPLE_FRAMES = 5   # frames sampled for per-K visual comparison

PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_DPI   = 150

# ─────────────────────────────────────────────────────────────────────
# Output paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "output"


# ═════════════════════════════════════════════════════════════════════
# Dynamic parameter helpers
# ═════════════════════════════════════════════════════════════════════

def choose_k(frame_sample, quality="medium"):
    """
    Dynamically select K based on colour complexity of sampled frames.

    Parameters
    ----------
    frame_sample : list[np.ndarray]
        A small list of BGR frames sampled from the video.
    quality : str
        One of "low", "medium", "high".

    Returns
    -------
    int
        Chosen K value.
    """
    complexities = []
    for frame in frame_sample:
        # quantise colours to 32-bin cube to measure distinct colour regions
        reduced = (frame // 8).astype(np.uint8)
        unique_colours = len(np.unique(
            reduced.reshape(-1, 3), axis=0
        ))
        complexities.append(unique_colours)

    avg_complexity = np.mean(complexities)

    # map complexity to an index in K_CANDIDATES
    # thresholds derived empirically for typical 1080p/4K footage
    if avg_complexity < 500:
        idx = 0          # very simple scene
    elif avg_complexity < 2000:
        idx = 1
    elif avg_complexity < 5000:
        idx = 2
    elif avg_complexity < 10000:
        idx = 3
    else:
        idx = 4          # very complex scene

    # apply quality bias
    idx = max(0, min(len(K_CANDIDATES) - 1,
                     idx + QUALITY_BIAS.get(quality, 0)))

    chosen_k = K_CANDIDATES[idx]
    print(f"  [config] Avg colour complexity: {avg_complexity:.0f} "
          f"→ K = {chosen_k}  (quality={quality})")
    return chosen_k


def choose_gop_size(fps):
    """Return GOP size equal to 1 second of video (= FPS)."""
    gop = max(1, int(round(fps)))
    print(f"  [config] FPS = {fps:.1f} → GOP size = {gop}")
    return gop


def auto_calibrate_mse_threshold(mse_values):
    """
    Choose an MSE threshold that separates static from dynamic GOPs.

    Uses the median of inter-frame MSE values from the calibration window.
    """
    threshold = float(np.median(mse_values))
    print(f"  [config] Auto-calibrated MSE threshold = {threshold:.4f} "
          f"(from {len(mse_values)} frame pairs)")
    return threshold
