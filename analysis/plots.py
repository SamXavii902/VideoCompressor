"""
plots.py — Quantitative analysis and presentation-ready visualizations.

Generates 10 plots covering PSNR, SSIM, compression ratio, visual
comparison grids, PCA scatter, per-frame timelines, and adaptive
decision maps.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    ANALYSIS_K_VALUES,
    ANALYSIS_N_VALUES,
    ANALYSIS_SAMPLE_FRAMES,
    PLOT_STYLE,
    PLOT_DPI,
    PCA_TARGET_HEIGHT,
)
from utils.metrics import (
    calculate_psnr, calculate_ssim, batch_psnr, batch_ssim, average_metric,
)
from modules.spatial import spatial_compress, spatial_compressed_size
from modules.temporal import compress_gop


def _setup_style():
    """Apply a clean, presentation-ready plot style."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("ggplot")


def _save(fig, path):
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved plot → {path}")


# ─────────────────────────────────────────────────────────────────────
# 1 & 2.  PSNR / SSIM  vs  K  (spatial)
# ─────────────────────────────────────────────────────────────────────

def plot_psnr_ssim_vs_k(frames, output_dir):
    """
    Test multiple K values on sampled frames and plot quality metrics.
    Generates: psnr_vs_k.png, ssim_vs_k.png
    """
    _setup_style()
    # sample a subset for speed
    step = max(1, len(frames) // ANALYSIS_SAMPLE_FRAMES)
    sample = frames[::step][:ANALYSIS_SAMPLE_FRAMES]

    psnr_means = []
    ssim_means = []
    ratios = []

    for k in tqdm(ANALYSIS_K_VALUES, desc="Sweeping K values",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        comp, labels, model, _ = spatial_compress(sample, k=k)
        psnrs = [calculate_psnr(sample[i], comp[i])
                 for i in range(len(sample))]
        ssims = [calculate_ssim(sample[i], comp[i])
                 for i in range(len(sample))]
        psnr_means.append(average_metric(psnrs))
        ssim_means.append(average_metric(ssims))

        # compression ratio
        bits_per_px = int(np.ceil(np.log2(k)))
        ratio = 24.0 / bits_per_px  # 24-bit original → bits_per_px
        ratios.append(ratio)

    # PSNR vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ANALYSIS_K_VALUES, psnr_means, "o-", color="#2196F3",
            linewidth=2, markersize=8)
    ax.set_xlabel("K (Number of Colours)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Spatial Quality: PSNR vs K (Colour Palette Size)",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    for i, k in enumerate(ANALYSIS_K_VALUES):
        ax.annotate(f"{psnr_means[i]:.1f}", (k, psnr_means[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    _save(fig, os.path.join(output_dir, "psnr_vs_k.png"))

    # SSIM vs K
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ANALYSIS_K_VALUES, ssim_means, "s-", color="#4CAF50",
            linewidth=2, markersize=8)
    ax.set_xlabel("K (Number of Colours)", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("Spatial Perceptual Quality: SSIM vs K",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    for i, k in enumerate(ANALYSIS_K_VALUES):
        ax.annotate(f"{ssim_means[i]:.3f}", (k, ssim_means[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    _save(fig, os.path.join(output_dir, "ssim_vs_k.png"))

    return ANALYSIS_K_VALUES, psnr_means, ssim_means, ratios


# ─────────────────────────────────────────────────────────────────────
# 3 & 4.  PSNR / SSIM  vs  N  (temporal)
# ─────────────────────────────────────────────────────────────────────

def plot_psnr_ssim_vs_n(frames, output_dir):
    """
    Test multiple N (PCA component counts) on a single GOP
    and plot quality metrics.
    Generates: psnr_vs_n.png, ssim_vs_n.png
    """
    _setup_style()
    import cv2 as _cv2

    # use the first GOP-worth of frames (downscaled)
    gop = frames[:max(len(frames), 60)]
    ds_gop = []
    for f in gop:
        h, w = f.shape[:2]
        if h > PCA_TARGET_HEIGHT:
            scale = PCA_TARGET_HEIGHT / h
            f = _cv2.resize(f, (int(w * scale), PCA_TARGET_HEIGHT),
                            interpolation=_cv2.INTER_AREA)
        ds_gop.append(f)

    # cap N values to what's feasible
    max_n = len(ds_gop) - 1
    n_values = [n for n in ANALYSIS_N_VALUES if n <= max_n]
    if not n_values:
        n_values = [max_n]

    psnr_means = []
    ssim_means = []

    for n in tqdm(n_values, desc="Sweeping N (PCA components)",
                  bar_format="{l_bar}{bar:30}{r_bar}"):
        recon, _ = compress_gop(ds_gop, n_components=n)
        psnrs = [calculate_psnr(ds_gop[i], recon[i])
                 for i in range(len(ds_gop))]
        ssims = [calculate_ssim(ds_gop[i], recon[i])
                 for i in range(len(ds_gop))]
        psnr_means.append(average_metric(psnrs))
        ssim_means.append(average_metric(ssims))

    # PSNR vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, psnr_means, "^-", color="#FF5722",
            linewidth=2, markersize=8)
    ax.set_xlabel("N (PCA Components Retained)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Temporal Quality: PSNR vs N (Principal Components)",
                 fontsize=14, fontweight="bold")
    for i, n in enumerate(n_values):
        ax.annotate(f"{psnr_means[i]:.1f}", (n, psnr_means[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    _save(fig, os.path.join(output_dir, "psnr_vs_n.png"))

    # SSIM vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, ssim_means, "D-", color="#9C27B0",
            linewidth=2, markersize=8)
    ax.set_xlabel("N (PCA Components Retained)", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("Temporal Perceptual Quality: SSIM vs N",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for i, n in enumerate(n_values):
        ax.annotate(f"{ssim_means[i]:.3f}", (n, ssim_means[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    _save(fig, os.path.join(output_dir, "ssim_vs_n.png"))

    return n_values, psnr_means, ssim_means


# ─────────────────────────────────────────────────────────────────────
# 5 & 6.  Compression Ratio  vs  Quality
# ─────────────────────────────────────────────────────────────────────

def plot_compression_vs_quality(k_data, n_data, output_dir):
    """
    Efficiency frontier plots.
    k_data : (k_values, psnr_means, ssim_means, ratios) from spatial sweep
    n_data : (n_values, psnr_means, ssim_means)          from temporal sweep
    Generates: compression_ratio_vs_quality.png
    """
    _setup_style()
    k_values, k_psnr, k_ssim, k_ratios = k_data
    n_values, n_psnr, n_ssim = n_data

    # temporal "ratios" are approximate: total_frames / n_components
    n_ratios = [len(n_values) / max(n, 1) for n in n_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR
    ax1.plot(k_ratios, k_psnr, "o-", label="Spatial (K-Means)",
             color="#2196F3", linewidth=2, markersize=8)
    ax1.plot(n_ratios, n_psnr, "^-", label="Temporal (PCA)",
             color="#FF5722", linewidth=2, markersize=8)
    ax1.set_xlabel("Compression Ratio", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("Compression Ratio vs PSNR", fontsize=14,
                  fontweight="bold")
    ax1.legend(fontsize=10)

    # SSIM
    ax2.plot(k_ratios, k_ssim, "s-", label="Spatial (K-Means)",
             color="#4CAF50", linewidth=2, markersize=8)
    ax2.plot(n_ratios, n_ssim, "D-", label="Temporal (PCA)",
             color="#9C27B0", linewidth=2, markersize=8)
    ax2.set_xlabel("Compression Ratio", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("Compression Ratio vs SSIM", fontsize=14,
                  fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)

    fig.suptitle("Efficiency Frontier: Quality vs Compression",
                 fontsize=16, fontweight="bold", y=1.02)
    _save(fig, os.path.join(output_dir, "compression_ratio_vs_quality.png"))


# ─────────────────────────────────────────────────────────────────────
# 7.  Visual comparison grid
# ─────────────────────────────────────────────────────────────────────

def plot_visual_comparison(frame, output_dir):
    """
    3×3 grid showing the same frame at different K values.
    Generates: visual_comparison_grid.png
    """
    _setup_style()
    import cv2 as _cv2

    k_values = [4, 8, 16, 32, 64, 128, 256, 512]
    # pick first 8, plus original = 9

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    # original
    axes[0].imshow(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (16.7 M colours)", fontsize=11,
                      fontweight="bold")
    axes[0].axis("off")

    for i, k in enumerate(k_values):
        comp, _, _, _ = spatial_compress([frame], k=k)
        psnr_val = calculate_psnr(frame, comp[0])
        ssim_val = calculate_ssim(frame, comp[0])

        axes[i + 1].imshow(_cv2.cvtColor(comp[0], _cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(
            f"K={k}  |  PSNR={psnr_val:.1f}dB  SSIM={ssim_val:.3f}",
            fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle("Visual Comparison: Effect of K on Spatial Compression",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "visual_comparison_grid.png"))


# ─────────────────────────────────────────────────────────────────────
# 8.  PCA frame scatter (2D)
# ─────────────────────────────────────────────────────────────────────

def plot_pca_scatter(frames, gop_size, output_dir):
    """
    Project every frame into 2-D PCA space and colour by GOP index.
    Generates: pca_frame_scatter.png
    """
    _setup_style()
    import cv2 as _cv2
    from sklearn.decomposition import IncrementalPCA

    # downscale heavily for speed (360p)
    ds = []
    for f in frames:
        ds.append(_cv2.resize(f, (640, 360),
                               interpolation=_cv2.INTER_AREA))
    matrix = np.array([f.ravel().astype(np.float32) for f in ds])

    ipca = IncrementalPCA(n_components=2)
    batch = max(1, min(len(matrix), 50))
    for s in range(0, len(matrix), batch):
        ipca.partial_fit(matrix[s:s + batch])
    coords = ipca.transform(matrix)

    # colour by GOP
    gop_ids = [i // gop_size for i in range(len(frames))]
    n_gops = max(gop_ids) + 1

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=gop_ids, cmap="tab20", s=30, alpha=0.8,
                         edgecolors="white", linewidths=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label="GOP Index")
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title("Frame Distribution in PCA Space (coloured by GOP)",
                 fontsize=14, fontweight="bold")

    # annotate explained variance
    ev = ipca.explained_variance_ratio_
    ax.text(0.02, 0.98,
            f"PC1: {ev[0]*100:.1f}%  PC2: {ev[1]*100:.1f}%  var explained",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    _save(fig, os.path.join(output_dir, "pca_frame_scatter.png"))


# ─────────────────────────────────────────────────────────────────────
# 9.  Per-frame PSNR timeline
# ─────────────────────────────────────────────────────────────────────

def plot_quality_timeline(psnr_values, ssim_values, output_dir, title_suffix=""):
    """
    Line chart of PSNR and SSIM per frame.
    Generates: quality_timeline.png
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # PSNR Plot
    ax1.plot(psnr_values, color="#2196F3", linewidth=1.2, alpha=0.9)
    avg_psnr = average_metric(psnr_values)
    ax1.axhline(avg_psnr, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Avg PSNR: {avg_psnr:.1f} dB")
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title(f"Per-Frame Quality Timeline{title_suffix}",
                 fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    
    # SSIM Plot
    ax2.plot(ssim_values, color="#4CAF50", linewidth=1.2, alpha=0.9)
    avg_ssim = average_metric(ssim_values)
    ax2.axhline(avg_ssim, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Avg SSIM: {avg_ssim:.3f}")
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Perceptual SSIM", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "quality_timeline.png"))


# ─────────────────────────────────────────────────────────────────────
# 10.  Adaptive decision map
# ─────────────────────────────────────────────────────────────────────

def plot_adaptive_decision_map(report, output_dir):
    """
    Colour-coded bar showing which strategy was applied per GOP.
    Generates: adaptive_decision_map.png
    """
    _setup_style()
    gop_classes = report["gop_classes"]
    if not gop_classes:
        return
    gop_mses = report["gop_mses"]
    gop_slices = report["gop_slices"]
    threshold = report["mse_threshold"]

    n = len(gop_classes)
    colours = ["#2196F3" if c == "static" else "#F44336" for c in gop_classes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"height_ratios": [1, 2]})

    # top: strategy bar
    for i in range(n):
        start, end = gop_slices[i]
        ax1.barh(0, end - start, left=start, color=colours[i],
                 edgecolor="white", linewidth=0.5, height=0.6)
    ax1.set_xlim(0, gop_slices[-1][1])
    ax1.set_yticks([])
    ax1.set_title("Adaptive Strategy per GOP", fontsize=14,
                  fontweight="bold")
    # legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor="#2196F3", label="PCA (static)"),
        Patch(facecolor="#F44336", label="K-Means (dynamic)"),
    ], loc="upper right", fontsize=10)

    # bottom: MSE per GOP
    midpoints = [(s + e) / 2 for s, e in gop_slices]
    ax2.bar(midpoints, gop_mses, width=[e - s for s, e in gop_slices],
            color=colours, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
                label=f"MSE threshold = {threshold:.2f}")
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Avg Inter-frame MSE", fontsize=12)
    ax2.set_title("Inter-frame Motion (MSE) per GOP", fontsize=14,
                  fontweight="bold")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "adaptive_decision_map.png"))


# ─────────────────────────────────────────────────────────────────────
# Master analysis runner
# ─────────────────────────────────────────────────────────────────────

def run_full_analysis(frames, compressed_frames, fps, gop_size,
                      adaptive_report, output_dir):
    """
    Run all Phase 5 analysis and save plots.

    Parameters
    ----------
    frames : list[np.ndarray]           original (downscaled) frames
    compressed_frames : list[np.ndarray] adaptive-compressed frames
    fps : float
    gop_size : int
    adaptive_report : dict              from adaptive_compress()
    output_dir : str                    directory for plot PNGs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print("  PHASE 5 — QUANTITATIVE ANALYSIS")
    print("═" * 60)

    # 1 & 2: PSNR / SSIM vs K
    print("\n▸ Spatial sweep (PSNR & SSIM vs K) …")
    k_data = plot_psnr_ssim_vs_k(frames, output_dir)

    # 3 & 4: PSNR / SSIM vs N
    print("\n▸ Temporal sweep (PSNR & SSIM vs N) …")
    n_data = plot_psnr_ssim_vs_n(frames, output_dir)

    # 5 & 6: Compression ratio vs quality
    print("\n▸ Compression ratio vs quality …")
    plot_compression_vs_quality(k_data, n_data, output_dir)

    # 7: Visual comparison grid
    print("\n▸ Visual comparison grid …")
    mid = len(frames) // 2
    plot_visual_comparison(frames[mid], output_dir)

    # 8: PCA scatter
    print("\n▸ PCA frame scatter …")
    plot_pca_scatter(frames, gop_size, output_dir)

    # 9: Quality timeline (PSNR & SSIM)
    print("\n▸ Per-frame Quality timeline (PSNR & SSIM) …")
    n = min(len(frames), len(compressed_frames))
    psnr_vals = []
    ssim_vals = []
    for i in tqdm(range(n), desc="Quality timeline", bar_format="{l_bar}{bar:30}{r_bar}"):
        psnr_vals.append(calculate_psnr(frames[i], compressed_frames[i]))
        ssim_vals.append(calculate_ssim(frames[i], compressed_frames[i]))
        
    plot_quality_timeline(psnr_vals, ssim_vals, output_dir, " (Adaptive)")

    # 10: Adaptive decision map
    print("\n▸ Adaptive decision map …")
    if adaptive_report is not None:
        plot_adaptive_decision_map(adaptive_report, output_dir)

    print("\n  ✓ All analysis plots saved to: " + output_dir)
    print("═" * 60 + "\n")
