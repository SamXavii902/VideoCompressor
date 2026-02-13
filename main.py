"""
main.py — CLI entry point for the Adaptive Video Compression Suite.

Usage
-----
  python main.py <video_path> [options]

Modes
-----
  spatial   — K-Means colour quantization only
  temporal  — PCA temporal compression only
  adaptive  — Hybrid adaptive pipeline
  analysis  — Quantitative plots (Phase 5)
  all       — Run spatial → temporal → adaptive → analysis

Examples
--------
  python main.py VID20260208223216.mp4 --mode all --quality medium
  python main.py VID20260208223216.mp4 --mode spatial --playback
  python main.py VID20260208223216.mp4 --mode analysis
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

# ── ensure project root is on sys.path ───────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from config import DEFAULT_OUTPUT_DIR, PCA_TARGET_HEIGHT
from utils.video_io import (
    get_video_metadata, print_metadata,
    playback_side_by_side,
)
from utils.metrics import (
    calculate_psnr, calculate_ssim, average_metric,
)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║         ADAPTIVE VIDEO COMPRESSION SUITE                    ║
║   Spatial (K-Means)  ·  Temporal (PCA)  ·  Hybrid Adaptive  ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive Video Compression Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Path to the input video file (e.g., input.mp4)")
    
    # ── Modes ──────────────────────────────────────────────────────────
    p.add_argument(
        "--mode", 
        choices=["spatial", "temporal", "adaptive", "analysis", "all", "codec"], 
        default="adaptive",
        help="Compression strategy to use. 'codec' uses the advanced BME+Residual pipeline."
    )
    p.add_argument("--quality", default="medium",
                   choices=["low", "medium", "high"],
                   help="Quality preset — influences dynamic K "
                        "(default: medium)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: ./output)")
    
    # ── Advanced Codec Params ──────────────────────────────────────────
    p.add_argument("--block-size", type=int, default=config.BME_BLOCK_SIZE,
                        help="Macroblock size for Block Motion Estimation (default: 16)")
    p.add_argument("--search-range", type=int, default=config.BME_SEARCH_RANGE,
                        help="Search range +/- pixels for Block Motion Estimation (default: 16)")
    p.add_argument("--dct-quality", type=int, default=config.DCT_QUALITY_FACTOR,
                        help="JPEG-style Quality Factor (1-100) for Residual Quantization (default: 50)")
    p.add_argument("--target-bitrate", type=int, default=None,
                        help="Target bitrate in kbps for RDO. Overrides --dct-quality if set.")
    p.add_argument("--use-neural-sr", action="store_true",
                        help="Compress at ultra-low 180p and use PyTorch Neural Net to upscale during playback")
    p.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration (fallback to CPU)")

    # ── Analysis & Playback ──────────────────────────────────────────
    p.add_argument("--playback", action="store_true",
                   help="Open live side-by-side comparison window")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip Phase 5 analysis plots")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit number of frames (for quick testing)")
    return p.parse_args()


def _ensure_output_dir(base):
    os.makedirs(base, exist_ok=True)
    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return base, plots_dir


def _compute_metrics_streaming(original_path, compressed_path,
                               max_frames=None, label=""):
    """
    Compute per-frame PSNR and SSIM by streaming both videos
    frame-by-frame. Never loads all frames into RAM.

    Returns (psnr_values, ssim_values, avg_psnr, avg_ssim).
    """
    from tqdm import tqdm

    cap_orig = cv2.VideoCapture(original_path)
    cap_comp = cv2.VideoCapture(compressed_path)

    total = int(cap_comp.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    # check if resolutions differ → downscale original to match
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    comp_h = int(cap_comp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    comp_w = int(cap_comp.get(cv2.CAP_PROP_FRAME_WIDTH))
    need_resize = (orig_h != comp_h)

    psnr_vals = []
    ssim_vals = []

    for i in tqdm(range(total), desc=f"{label} metrics",
                  unit="frame", bar_format="{l_bar}{bar:30}{r_bar}"):
        ret1, orig = cap_orig.read()
        ret2, comp = cap_comp.read()
        if not ret1 or not ret2:
            break

        if need_resize:
            orig = cv2.resize(orig, (comp_w, comp_h),
                              interpolation=cv2.INTER_AREA)

        psnr_vals.append(calculate_psnr(orig, comp))
        ssim_vals.append(calculate_ssim(orig, comp))

    cap_orig.release()
    cap_comp.release()

    avg_psnr = average_metric(psnr_vals)
    avg_ssim = average_metric(ssim_vals)
    return psnr_vals, ssim_vals, avg_psnr, avg_ssim


def _playback_from_files(original_path, compressed_path, fps,
                         max_frames=None):
    """
    Live side-by-side playback streaming from two video files.
    Press 'q' to quit.
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_comp = cv2.VideoCapture(compressed_path)

    delay = max(1, int(1000 / fps))

    comp_w = int(cap_comp.get(cv2.CAP_PROP_FRAME_WIDTH))
    comp_h = int(cap_comp.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # scale for display
    max_display_w = 1600
    if 2 * comp_w > max_display_w:
        scale = max_display_w / (2 * comp_w)
    else:
        scale = 1.0

    count = 0
    while True:
        ret1, orig = cap_orig.read()
        ret2, comp = cap_comp.read()
        if not ret1 or not ret2:
            break

        # resize original to match compressed dimensions
        orig = cv2.resize(orig, (comp_w, comp_h),
                          interpolation=cv2.INTER_AREA)

        if scale != 1.0:
            disp_h = int(comp_h * scale)
            disp_w = int(comp_w * scale)
            orig = cv2.resize(orig, (disp_w, disp_h))
            comp = cv2.resize(comp, (disp_w, disp_h))

        cv2.putText(orig, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comp, "COMPRESSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        psnr = calculate_psnr(orig, comp)
        cv2.putText(comp, f"PSNR: {psnr:.1f} dB", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([orig, comp])
        cv2.imshow("Original  vs  Compressed", combined)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cv2.destroyAllWindows()
    cap_orig.release()
    cap_comp.release()


# =====================================================================
#  Mode runners (all streaming — never load all frames into RAM)
# =====================================================================

def run_spatial(video_path, meta, args, output_dir):
    """Phase 2: Spatial compression."""
    from modules.spatial import spatial_compress_streaming, spatial_compressed_size

    print("\n" + "=" * 60)
    print("  PHASE 2 — SPATIAL COMPRESSION (K-Means)")
    print("=" * 60)

    out_path = os.path.join(output_dir, "spatial_compressed.mp4")
    k, fcount, model = spatial_compress_streaming(
        video_path, out_path, meta["fps"],
        quality=args.quality, max_frames=args.max_frames)

    # streaming metrics
    psnr_vals, ssim_vals, avg_psnr, avg_ssim = _compute_metrics_streaming(
        video_path, out_path, max_frames=args.max_frames, label="Spatial")

    cr = spatial_compressed_size(fcount, meta["height"], meta["width"], model)
    orig_bytes = fcount * meta["height"] * meta["width"] * 3
    ratio = orig_bytes / cr["total_bytes"]

    print(f"\n  ┌─── Spatial Results ───────────────────┐")
    print(f"  │  K            : {k}")
    print(f"  │  Frames       : {fcount}")
    print(f"  │  Avg PSNR     : {avg_psnr:.2f} dB")
    print(f"  │  Avg SSIM     : {avg_ssim:.4f}")
    print(f"  │  Bits/pixel   : {cr['bits_per_pixel']}")
    print(f"  │  Compression  : {ratio:.2f}x")
    print(f"  └────────────────────────────────────────┘\n")

    if args.playback:
        _playback_from_files(video_path, out_path, meta["fps"],
                             max_frames=args.max_frames)

    return out_path, psnr_vals


def run_temporal(video_path, meta, args, output_dir):
    """Phase 3: Temporal compression."""
    from modules.temporal import temporal_compress_streaming

    print("\n" + "=" * 60)
    print("  PHASE 3 — TEMPORAL COMPRESSION (PCA)")
    print("=" * 60)

    out_path = os.path.join(output_dir, "temporal_compressed.mp4")
    fcount, pca_data, final_var = temporal_compress_streaming(
        video_path, out_path, meta["fps"],
        max_frames=args.max_frames)

    psnr_vals, ssim_vals, avg_psnr, avg_ssim = _compute_metrics_streaming(
        video_path, out_path, max_frames=args.max_frames, label="Temporal")

    n_comp_avg = np.mean([pd["n_components"] for pd in pca_data]) \
        if pca_data else 0

    print(f"\n  ┌─── Temporal Results ──────────────────┐")
    print(f"  │  Frames         : {fcount}")
    print(f"  │  Avg components : {n_comp_avg:.1f}")
    print(f"  │  Final variance : {final_var:.2%}")
    print(f"  │  Avg PSNR       : {avg_psnr:.2f} dB")
    print(f"  │  Avg SSIM       : {avg_ssim:.4f}")
    print(f"  └────────────────────────────────────────┘\n")

    if args.playback:
        _playback_from_files(video_path, out_path, meta["fps"],
                             max_frames=args.max_frames)

    return out_path, psnr_vals


def run_adaptive(video_path, meta, args, output_dir):
    """Phase 4: Adaptive hybrid pipeline."""
    from modules.adaptive import adaptive_compress_streaming

    print("\n" + "=" * 60)
    print("  PHASE 4 — ADAPTIVE HYBRID COMPRESSION")
    print("=" * 60)

    out_path = os.path.join(output_dir, "adaptive_compressed.mp4")
    fcount, report = adaptive_compress_streaming(
        video_path, out_path, meta["fps"],
        quality=args.quality, max_frames=args.max_frames)

    psnr_vals, ssim_vals, avg_psnr, avg_ssim = _compute_metrics_streaming(
        video_path, out_path, max_frames=args.max_frames, label="Adaptive")

    n_static = report["gop_classes"].count("static")
    n_dynamic = report["gop_classes"].count("dynamic")

    print(f"\n  ┌─── Adaptive Results ─────────────────┐")
    print(f"  │  Frames        : {fcount}")
    print(f"  │  Static GOPs   : {n_static}")
    print(f"  │  Dynamic GOPs  : {n_dynamic}")
    print(f"  │  MSE threshold : {report['mse_threshold']:.4f}")
    print(f"  │  K used        : {report['k_used']}")
    print(f"  │  Avg PSNR      : {avg_psnr:.2f} dB")
    print(f"  │  Avg SSIM      : {avg_ssim:.4f}")
    print(f"  └────────────────────────────────────────┘\n")

    if args.playback:
        _playback_from_files(video_path, out_path, meta["fps"],
                             max_frames=args.max_frames)

    return out_path, report, psnr_vals


def run_codec(video_path, meta, args, output_dir):
    """Phase 6: Advanced Codec (BME + Residual)."""
    from modules.codec import codec_compress_streaming, codec_decompress_streaming

    print("\n" + "=" * 60)
    print("  PHASE 6 — ADVANCED CODEC (BME + RESIDUAL)")
    print("=" * 60)

    out_bin = os.path.join(output_dir, "codec_compressed.bin")
    out_mp4 = os.path.join(output_dir, "codec_compressed.mp4")
    
    print(f"\n[{args.mode.upper()} Mode] Encoding {os.path.basename(video_path)}")
    
    if args.use_neural_sr:
        from modules.neural_sr import train_sr_model_on_video
        import config
        sr_model_path = os.path.join(config.DEFAULT_OUTPUT_DIR, "sr_weights.pt")
        # Pre-train a customized CNN for this specific video based on 50 samples
        train_sr_model_on_video(video_path, sr_model_path, upscale_factor=2, epochs=15, sample_frames=50)

    try:
        bitstream = codec_compress_streaming(
            video_path, out_bin, meta["fps"],
            quality_factor=args.dct_quality,
            target_bitrate_kbps=args.target_bitrate,
            max_frames=args.max_frames,
            use_spatial_iframe=True,
            use_neural_sr=args.use_neural_sr)
    except Exception as e:
        print(f"Error during codec compression: {e}")
        import sys
        sys.exit(1)
    fcount = len(bitstream)
    
    # Decode
    codec_decompress_streaming(out_bin, out_mp4, meta["fps"])

    psnr_vals, ssim_vals, avg_psnr, avg_ssim = _compute_metrics_streaming(
        video_path, out_mp4, max_frames=args.max_frames, label="Codec")

    # Storage calc (Codec internally downscales to 360p)
    calc_w, calc_h = 640, 360
    orig_bytes = fcount * calc_h * calc_w * 3
    bin_size = os.path.getsize(out_bin)
    ratio = orig_bytes / bin_size

    print(f"\n  ┌─── Codec Results ────────────────────┐")
    print(f"  │  Frames        : {fcount}")
    print(f"  │  Block Size    : {args.block_size}")
    print(f"  │  Search Range  : {args.search_range}")
    print(f"  │  DCT Quality   : {args.dct_quality}")
    print(f"  │  Compression   : {ratio:.2f}x")
    print(f"  │  Avg PSNR      : {avg_psnr:.2f} dB")
    print(f"  │  Avg SSIM      : {avg_ssim:.4f}")
    print(f"  └────────────────────────────────────────┘\n")

    if args.playback:
        orig_frames = extract_frames(video_path, max_frames=args.max_frames, resize_height=360)
        comp_frames = extract_frames(out_mp4, max_frames=args.max_frames, resize_height=360)
        playback_side_by_side(list(orig_frames), list(comp_frames), meta["fps"], 
                            psnr_values=psnr_vals, ssim_values=ssim_vals, 
                            window_name="Original vs Codec (360p Downscaled)")

    return out_mp4, psnr_vals


def run_analysis(video_path, meta, report, output_dir, plots_dir,
                 max_frames=None):
    """Phase 5: Quantitative analysis — uses sampled frames for
    efficiency, not full video."""
    from analysis.plots import run_full_analysis
    from utils.video_io import sample_frames

    # sample a small set of frames for analysis
    n_sample = min(60, meta["frame_count"])
    if max_frames is not None:
        n_sample = min(n_sample, max_frames)

    frames = sample_frames(video_path, n=n_sample,
                           resize_height=PCA_TARGET_HEIGHT)

    gop_size = report.get("gop_size", int(round(meta["fps"])))

    # for the compressed frames, read from adaptive output if it exists
    adaptive_path = os.path.join(output_dir, "adaptive_compressed.mp4")
    codec_path = os.path.join(output_dir, "codec_compressed.mp4")
    
    comp_frames = frames # fallback
    if os.path.isfile(adaptive_path):
        comp_frames = sample_frames(adaptive_path, n=n_sample)
    elif os.path.isfile(codec_path):
        comp_frames = sample_frames(codec_path, n=n_sample)

    # resize to match if needed
    target_h, target_w = frames[0].shape[:2]
    comp_frames = [
        cv2.resize(f, (target_w, target_h),
                   interpolation=cv2.INTER_AREA)
        if f.shape[:2] != (target_h, target_w) else f
        for f in comp_frames
    ]
    
    run_full_analysis(frames, comp_frames, meta["fps"], gop_size,
                      report, plots_dir)


# =====================================================================
#  Main
# =====================================================================

def main():
    print(BANNER)
    args = parse_args()

    video_path = os.path.abspath(args.input) # Changed from args.video to args.input
    if not os.path.isfile(video_path):
        print(f"  ✗ File not found: {video_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(video_path), DEFAULT_OUTPUT_DIR)
    output_dir, plots_dir = _ensure_output_dir(output_dir)

    # ── metadata ─────────────────────────────────────────────────────
    meta = get_video_metadata(video_path)
    print_metadata(meta)

    mode = args.mode
    adaptive_report = None

    t_start = time.time()

    # ── spatial ──────────────────────────────────────────────────────
    if mode in ("spatial", "all"):
        run_spatial(video_path, meta, args, output_dir)

    # ── temporal ─────────────────────────────────────────────────────
    if mode in ("temporal", "all"):
        run_temporal(video_path, meta, args, output_dir)

    # ── adaptive ─────────────────────────────────────────────────────
    if mode in ("adaptive", "all"):
        _, adaptive_report, _ = run_adaptive(
            video_path, meta, args, output_dir)

    # ── codec ────────────────────────────────────────────────────────
    if mode == "codec":
        run_codec(video_path, meta, args, output_dir)

    # ── analysis ─────────────────────────────────────────────────────
    if mode in ("analysis", "all") and not args.no_plots:
        if adaptive_report is None:
            adaptive_report = {
                "gop_size": int(round(meta["fps"])),
                "gop_classes": [], "gop_mses": [],
                "gop_slices": [], "mse_threshold": 0,
                "strategies": [], "k_used": 0,
            }
        run_analysis(video_path, meta, adaptive_report,
                     output_dir, plots_dir,
                     max_frames=args.max_frames)

    elapsed = time.time() - t_start
    print(f"\n  ✓ All done!  ({elapsed:.0f}s elapsed)")
    print(f"    Output saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("codec_trace.txt", "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
