"""
video_io.py — Frame extraction, video writing, and metadata utilities.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


def get_video_metadata(path):
    """
    Extract metadata from a video file.

    Returns
    -------
    dict
        Keys: fps, width, height, frame_count, duration_sec, codec
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec       = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    duration    = frame_count / fps if fps > 0 else 0.0
    cap.release()

    meta = {
        "fps":         fps,
        "width":       width,
        "height":      height,
        "frame_count": frame_count,
        "duration_sec": round(duration, 2),
        "codec":       codec,
        "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
    }
    return meta


def print_metadata(meta):
    """Pretty-print video metadata to stdout."""
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║       VIDEO METADATA                 ║")
    print("  ╠══════════════════════════════════════╣")
    print(f"  ║  Resolution : {meta['width']}×{meta['height']}")
    print(f"  ║  FPS        : {meta['fps']:.2f}")
    print(f"  ║  Frames     : {meta['frame_count']}")
    print(f"  ║  Duration   : {meta['duration_sec']}s")
    print(f"  ║  Codec      : {meta['codec']}")
    print(f"  ║  File size  : {meta['file_size_mb']} MB")
    print("  ╚══════════════════════════════════════╝\n")


def extract_frames(path, max_frames=None, resize_height=None):
    """
    Generator that yields BGR frames from a video file.

    Parameters
    ----------
    path : str
        Path to the video file.
    max_frames : int or None
        Stop after this many frames (None = all).
    resize_height : int or None
        If set, resize each frame so its height equals this value
        (width scaled proportionally).

    Yields
    ------
    np.ndarray
        BGR frame with shape (H, W, 3), dtype uint8.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    with tqdm(total=total, desc="Extracting frames", unit="frame",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_height is not None:
                h, w = frame.shape[:2]
                scale = resize_height / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, resize_height),
                                   interpolation=cv2.INTER_AREA)
            yield frame
            count += 1
            pbar.update(1)
            if max_frames is not None and count >= max_frames:
                break

    cap.release()


def extract_all_frames(path, max_frames=None, resize_height=None):
    """
    Load all frames into a list (convenience wrapper around the generator).

    Returns
    -------
    list[np.ndarray]
    """
    return list(extract_frames(path, max_frames=max_frames,
                               resize_height=resize_height))


def sample_frames(path, n=5, resize_height=None):
    """
    Sample *n* evenly-spaced frames from a video.

    Returns
    -------
    list[np.ndarray]
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            if resize_height is not None:
                h, w = frame.shape[:2]
                scale = resize_height / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, resize_height),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
    cap.release()
    return frames


def write_video(frames, path, fps, codec="mp4v"):
    """
    Write a list of BGR frames to an .mp4 file.

    Parameters
    ----------
    frames : list[np.ndarray]
        BGR uint8 frames, all the same shape.
    path : str
        Output file path (should end in .mp4).
    fps : float
        Frames per second.
    codec : str
        FourCC codec string (default 'mp4v').
    """
    if len(frames) == 0:
        raise ValueError("No frames to write")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise IOError(f"Cannot create video writer for: {path}")

    with tqdm(total=len(frames), desc="Writing video", unit="frame",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for frame in frames:
            # ensure uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(frame)
            pbar.update(1)

    writer.release()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✓ Saved {len(frames)} frames → {path}  ({size_mb:.1f} MB)")


def playback_side_by_side(original_frames, compressed_frames, fps,
                          psnr_values=None, ssim_values=None,
                          window_name="Original  vs  Compressed"):
    """
    Live side-by-side playback of original and compressed frames.

    Press 'q' to quit early.
    """
    delay = max(1, int(1000 / fps))

    # determine a sensible display width (fit within ~1600 px wide)
    h, w = original_frames[0].shape[:2]
    max_display_w = 1600
    if 2 * w > max_display_w:
        scale = max_display_w / (2 * w)
    else:
        scale = 1.0

    total = min(len(original_frames), len(compressed_frames))

    for i in range(total):
        orig = original_frames[i]
        comp = compressed_frames[i]

        # make sure both are uint8
        if comp.dtype != np.uint8:
            comp = np.clip(comp, 0, 255).astype(np.uint8)

        # resize for display
        if scale != 1.0:
            disp_h = int(h * scale)
            disp_w = int(w * scale)
            orig = cv2.resize(orig, (disp_w, disp_h))
            comp = cv2.resize(comp, (disp_w, disp_h))

        # add labels
        cv2.putText(orig, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comp, "COMPRESSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # overlay metrics
        if psnr_values is not None and i < len(psnr_values):
            txt = f"PSNR: {psnr_values[i]:.1f} dB"
            cv2.putText(comp, txt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ssim_values is not None and i < len(ssim_values):
            txt = f"SSIM: {ssim_values[i]:.4f}"
            cv2.putText(comp, txt, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([orig, comp])
        cv2.imshow(window_name, combined)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
