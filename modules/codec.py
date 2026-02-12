"""
codec.py — Full advanced compression pipeline.

Orchestrates Block Motion Estimation, Residual calculation, DCT+Quantization, and RLE.
Encodes video into I-frames (static keyframes) and P-frames (motion vectors + residuals).
"""

import os
import cv2
import numpy as np
import pickle
import zlib
import multiprocessing
from tqdm import tqdm

from config import (
    BME_BLOCK_SIZE, 
    BME_SEARCH_RANGE, 
    DCT_QUALITY_FACTOR,
    MOTION_METHOD,
    choose_gop_size
)

from utils.video_io import get_video_metadata, extract_frames
from modules.spatial import quantize_frame, fit_kmeans_from_video
from modules.motion import estimate_motion_field, motion_compensate, compute_residual, reconstruct_frame
from modules.residual import encode_residual, decode_residual

def encode_p_frame(current_frame, reference_frame, block_size, search_range, qf, target_bytes=None, method="optical_flow"):
    """
    Encode a P-frame.
    Returns the compressed data payload (dict).
    """
    # 1. Motion Estimation
    motion_vectors, _ = estimate_motion_field(current_frame, reference_frame, block_size, search_range, method=method)
    
    # 2. Motion Compensation
    predicted_frame = motion_compensate(reference_frame, motion_vectors, block_size, method=method)
    
    # 3. Residual
    residual = compute_residual(current_frame, predicted_frame)
    
    # 4. Transform & Encode Residual (RDO Loop)
    if target_bytes is not None:
        best_score = -float('inf')
        best_qf = 10
        best_stream, best_meta = None, None
        
        from utils.metrics import calculate_ssim
        
        # Binary-search style heuristic: Test increasing qualities. 
        # Evaluate using Perceptual SSIM and pick the highest visual quality that fits the bandwidth!
        for test_qf in [10, 20, 30, 45, 60, 80, 95]:
            encoded_stream, metadata = encode_residual(residual, quality_factor=test_qf)
            
            # Approximate motion vector size after arbitrary zlib compression (typically ~25%)
            mv_bytes = motion_vectors.nbytes // 4
            total_bytes = len(encoded_stream) + mv_bytes
            
            if total_bytes <= target_bytes:
                # PERCEPTUAL RDO: Evaluate the human-visual SSIM index on this candidate!
                recon_res = decode_residual(encoded_stream, metadata)
                recon_frame = reconstruct_frame(predicted_frame, recon_res)
                score = calculate_ssim(current_frame, recon_frame)
                
                if score > best_score:
                    best_score = score
                    best_qf = test_qf
                    best_stream, best_meta = encoded_stream, metadata
            else:
                # Exceeded budget, break and use the previous best quality
                break
                
        if best_stream is None:
            # Even QF 10 exceeded budget, fallback to rock-bottom QF 10
            best_stream, best_meta = encode_residual(residual, quality_factor=10)
            
        encoded_stream, metadata = best_stream, best_meta
    else:
        # Standard encoding without RDO rate control
        encoded_stream, metadata = encode_residual(residual, quality_factor=qf)
    
    # Pack P-frame data
    return {
        "type": "P",
        "mv": motion_vectors,
        "res_stream": encoded_stream,
        "res_meta": metadata,
        "method": method
    }

def decode_p_frame(pfdata, reference_frame, block_size):
    """
    Decode a P-frame payload back into a frame.
    """
    method = pfdata.get("method", "block_matching")
    predicted_frame = motion_compensate(reference_frame, pfdata["mv"], block_size, method=method)
    residual = decode_residual(pfdata["res_stream"], pfdata["res_meta"])
    reconstructed = reconstruct_frame(predicted_frame, residual)
    return reconstructed

def encode_b_frame(current_frame, ref_forward, ref_backward, block_size, search_range, qf, target_bytes=None, method="optical_flow"):
    """
    Encode a B-frame which bi-directionally predicts from a past AND future reference frame.
    """
    mv_fwd, _ = estimate_motion_field(current_frame, ref_forward, block_size, search_range, method=method)
    pred_fwd = motion_compensate(ref_forward, mv_fwd, block_size, method=method)
    
    mv_bwd, _ = estimate_motion_field(current_frame, ref_backward, block_size, search_range, method=method)
    pred_bwd = motion_compensate(ref_backward, mv_bwd, block_size, method=method)
    
    # Average the forward and backward predictions for maximum structural certainty
    predicted_frame = ((pred_fwd.astype(np.uint16) + pred_bwd.astype(np.uint16)) // 2).astype(np.uint8)
    residual = compute_residual(current_frame, predicted_frame)
    
    if target_bytes is not None:
        best_score = -float('inf')
        best_qf = 10
        best_stream, best_meta = None, None
        
        from utils.metrics import calculate_ssim
        
        for test_qf in [10, 20, 30, 45, 60, 80, 95]:
            encoded_stream, metadata = encode_residual(residual, quality_factor=test_qf)
            total_bytes = len(encoded_stream) + (mv_fwd.nbytes + mv_bwd.nbytes) // 4
            if total_bytes <= target_bytes:
                # PERCEPTUAL RDO: Evaluate visual quality structure!
                recon_res = decode_residual(encoded_stream, metadata)
                recon_frame = reconstruct_frame(predicted_frame, recon_res)
                score = calculate_ssim(current_frame, recon_frame)
                
                if score > best_score:
                    best_score = score
                    best_qf = test_qf
                    best_stream, best_meta = encoded_stream, metadata
            else:
                break
                
        if best_stream is None:
            best_stream, best_meta = encode_residual(residual, quality_factor=10)
        encoded_stream, metadata = best_stream, best_meta
    else:
        encoded_stream, metadata = encode_residual(residual, quality_factor=qf)
        
    return {
        "type": "B",
        "mv_fwd": mv_fwd,
        "mv_bwd": mv_bwd,
        "res_stream": encoded_stream,
        "res_meta": metadata,
        "method": method
    }

def decode_b_frame(bfdata, ref_forward, ref_backward, block_size):
    """
    Decode a B-frame payload.
    """
    method = bfdata.get("method", "block_matching")
    pred_fwd = motion_compensate(ref_forward, bfdata["mv_fwd"], block_size, method=method)
    pred_bwd = motion_compensate(ref_backward, bfdata["mv_bwd"], block_size, method=method)
    
    predicted_frame = ((pred_fwd.astype(np.uint16) + pred_bwd.astype(np.uint16)) // 2).astype(np.uint8)
    residual = decode_residual(bfdata["res_stream"], bfdata["res_meta"])
    return reconstruct_frame(predicted_frame, residual)

def _encode_gop_worker(args):
    """
    Worker function to encode a single GOP (Group of Pictures).
    Runs in a separate process to achieve multi-core parallelization.
    """
    gop_index, frames_chunk, block_size, search_range, qf, kmeans, motion_method, target_bytes_per_frame, pts_offset = args
    
    # Needs to re-import torch/backend inside worker for safe multiprocessing memory space
    from modules.motion import estimate_motion_field, motion_compensate, compute_residual
    from modules.residual import encode_residual
    from modules.spatial import quantize_frame
    import cv2
    from tqdm import tqdm
    
    gop_bitstream = []
    
    reconstructed_frames = {}
    
    # 1. Encode I-Frame (always frame index 0 of the GOP)
    frame_0 = frames_chunk[0]
    if kmeans is not None:
        i_frame, _ = quantize_frame(frame_0, kmeans)
    else:
        i_frame = frame_0
        
    reconstructed_frames[0] = i_frame.copy()
    
    # Rate Distortion Optimization for I-Frames
    if target_bytes_per_frame is not None:
        final_jpeg_bytes = None
        for test_jq in [10, 30, 50, 70, 85, 95]:
            _, test_bytes = cv2.imencode('.jpg', i_frame, [int(cv2.IMWRITE_JPEG_QUALITY), test_jq])
            if len(test_bytes) > target_bytes_per_frame * 2:
                break
            final_jpeg_bytes = test_bytes
            
        if final_jpeg_bytes is None:
            _, final_jpeg_bytes = cv2.imencode('.jpg', i_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        jpeg_bytes = final_jpeg_bytes
    else:
        _, jpeg_bytes = cv2.imencode('.jpg', i_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    gop_bitstream.append({
        "type": "I",
        "pts": pts_offset,
        "data": jpeg_bytes
    })

    # 2. Build Out-of-Order Execution schedule for Bi-Directional B-Frames
    encode_order = []
    i = 2
    while i < len(frames_chunk):
        encode_order.append(('P', i, i-2))        # P-frame predicts from previous anchor
        encode_order.append(('B', i-1, i-2, i))   # B-frame predicts from both anchors
        i += 2
        
    # Handle the trailing odd frame if any
    if len(frames_chunk) % 2 == 0:
        last_idx = len(frames_chunk) - 1
        encode_order.append(('P', last_idx, last_idx - 1))
        
    # 3. Execute Out-of-Order encoding
    for task in encode_order:
        frame_type = task[0]
        
        if frame_type == 'P':
            f_idx, ref_idx = task[1], task[2]
            payload = encode_p_frame(frames_chunk[f_idx], reconstructed_frames[ref_idx], block_size, search_range, qf, target_bytes=target_bytes_per_frame, method=motion_method)
            reconstructed_frames[f_idx] = decode_p_frame(payload, reconstructed_frames[ref_idx], block_size)
            payload["pts"] = f_idx + pts_offset
            gop_bitstream.append(payload)
            
        elif frame_type == 'B':
            f_idx, ref0_idx, ref1_idx = task[1], task[2], task[3]
            payload = encode_b_frame(frames_chunk[f_idx], reconstructed_frames[ref0_idx], reconstructed_frames[ref1_idx], block_size, search_range, qf, target_bytes=target_bytes_per_frame, method=motion_method)
            
            # B-frames are never used as references for standard GOPs, so we don't strictly need to reconstruct them into the dictionary,
            # but we explicitly decode it to ensure visually sound playback at the end user end
            reconstructed_frames[f_idx] = decode_b_frame(payload, reconstructed_frames[ref0_idx], reconstructed_frames[ref1_idx], block_size)
            payload["pts"] = f_idx + pts_offset
            gop_bitstream.append(payload)
            
    return gop_index, gop_bitstream

def codec_compress_streaming(video_path, output_bin_path, fps, quality_factor=DCT_QUALITY_FACTOR, target_bitrate_kbps=None, max_frames=None, use_spatial_iframe=True, use_neural_sr=False):
    """
    Stream encode a video into a compressed binary format.
    The first frame of every GOP is an I-frame. Subsequent frames are P-frames.
    
    output_bin_path: The custom binary file format to save the stream locally.
    use_spatial_iframe: If True, I-frames are compressed using K-Means (K=64). 
                        If False, I-frames are saved as raw JPGs to the bitstream.
    use_neural_sr: If True, squashes the entire codec to 180p for massive savings,
                   expecting the decoder to use PyTorch to dynamically upscale.
    """
    gop_size = choose_gop_size(fps)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
        
    # Setup spatial compression for I-frames if needed
    kmeans = None
    if use_spatial_iframe:
        kmeans = fit_kmeans_from_video(video_path, k=64, n_sample_frames=5)
        
    target_bytes_per_frame = None
    if target_bitrate_kbps:
        target_bytes_per_frame = int((target_bitrate_kbps * 1000) / (8 * fps))
        print(f"\n[Codec] Rate Control Enabled: Target ~{target_bitrate_kbps} kbps ({target_bytes_per_frame} bytes/frame budget)")
    
    print(f"\n[Codec] Starting Advanced Compression (GOP Size: {gop_size}, QF: {quality_factor})")
    print(f"  [Parallel] Chunking video and starting Multiprocessing Pool...")

    # Build GOP chunks in memory (since 360p frames fit easily in RAM)
    # E.g. 1800 frames / 60 gop_size = 30 arrays of 60 frames
    gop_args = []
    current_chunk = []
    gop_idx = 0
    read_count = 0
    
    width, height = (640, 360)
    if use_neural_sr:
        width, height = (320, 180)
        print("  [Codec] Neural SR enabled! Squashing internal resolution to 180p (4x less pixels!)")
        
    while read_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        current_chunk.append(frame)
        read_count += 1
        
        if len(current_chunk) == gop_size or read_count == total_frames:
            gop_args.append((
                gop_idx, current_chunk, BME_BLOCK_SIZE, 
                BME_SEARCH_RANGE, quality_factor, kmeans, MOTION_METHOD, target_bytes_per_frame,
                # Global PTS offset is the start index of this chunk
                read_count - len(current_chunk)
            ))
            current_chunk = []
            gop_idx += 1
            
    cap.release()
    
    # Execute Multiprocessing Pool
    # We use 8 processes (typically 1 per physical core, avoiding thread hyper-thrashing)
    num_workers = min(8, multiprocessing.cpu_count(), len(gop_args))
    
    bitstream = [{
        "type": "HEADER",
        "use_sr": use_neural_sr,
        "width": width,
        "height": height
    }]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered allows us to yield results as soon as any worker finishes
        # tqdm updates perfectly with this pattern
        results = []
        with tqdm(total=total_frames, desc="Total Video Progress", position=0, leave=True) as pbar:
            for gop_idx_result, gop_stream in pool.imap_unordered(_encode_gop_worker, gop_args):
                results.append((gop_idx_result, gop_stream))
                pbar.update(len(gop_stream))
                
        # Must sort results by gop_index to ensure chronological playback!
        results.sort(key=lambda x: x[0])
        
        for _, gop_stream in results:
            bitstream.extend(gop_stream)
    
    # Save bitstream
    raw_bytes = pickle.dumps(bitstream)
    compressed_bytes = zlib.compress(raw_bytes, level=6)
    
    with open(output_bin_path, 'wb') as f:
        f.write(compressed_bytes)
        
    mb_size = len(compressed_bytes) / (1024*1024)
    print(f"  ✓ Codec encoding complete. Bitstream size: {mb_size:.2f} MB")
    
    return bitstream

def codec_decompress_streaming(input_bin_path, output_mp4_path, fps):
    """
    Decode the custom binary format back into an MP4 video file.
    """
    print(f"\n[Codec] Starting Advanced Decompression")
    
    with open(input_bin_path, 'rb') as f:
        compressed_bytes = f.read()
        
    raw_bytes = zlib.decompress(compressed_bytes)
    bitstream = pickle.loads(raw_bytes)
    
    if not bitstream:
        print("  Error: Empty bitstream")
        return
        
    header = bitstream[0]
    use_sr = False
    
    if header.get("type") == "HEADER":
        use_sr = header.get("use_sr", False)
        # Strip header for decoder loop
        bitstream = bitstream[1:]
        
    if use_sr:
        print("  [Codec] Neural SR Header Detected! Booting PyTorch Upscaler...")
        from modules.neural_sr import init_sr_model, upscale_frame
        init_sr_model("output/sr_weights.pt")
        
    # Output is natively 360p (or whatever the Upscaler / Base res provides natively)
    os.makedirs(os.path.dirname(output_mp4_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # All decodes output 360p standard
    writer = cv2.VideoWriter(output_mp4_path, fourcc, fps, (640, 360))
    
    # Decoder needs a display buffer since frames arrive Out-Of-Order (I P B P B)
    # We yield frames to the CV2 writer strictly in chronological PTS (Presentation Time Stamp) order
    decoder_buffer = {}
    reconstructed_frames = {}
    next_pts_to_write = 0
    
                # Assuming previous P/I frame anchor exists 
                # (For strict GOPs, PTS-1 or PTS-2 might be the anchor. For simplicity, we search backwards for the nearest parsed anchor)
                anchor_pts = max([k for k in reconstructed_frames.keys() if k < pts])
                frame = decode_p_frame(payload, reconstructed_frames[anchor_pts], BME_BLOCK_SIZE)
                reconstructed_frames[pts] = frame.copy()
                decoder_buffer[pts] = frame.copy()
            elif payload["type"] == "B":
                # B-Frames predict from BOTH nearest anchors
                anchor1_pts = max([k for k in reconstructed_frames.keys() if k < pts])
                # The future anchor must have been parsed out-of-order already
                anchor2_list = [k for k in reconstructed_frames.keys() if k > pts]
                anchor2_pts = min(anchor2_list) if anchor2_list else anchor1_pts
                
                frame = decode_b_frame(payload, reconstructed_frames[anchor1_pts], reconstructed_frames[anchor2_pts], BME_BLOCK_SIZE)
                decoder_buffer[pts] = frame.copy()
                
            # Try to drain the display buffer strictly chronologically!
            while next_pts_to_write in decoder_buffer:
                write_frame = decoder_buffer.pop(next_pts_to_write)
                if use_sr:
                    write_frame = upscale_frame(write_frame)
                elif write_frame.shape[:2] != (360, 640):
                    write_frame = cv2.resize(write_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    
                writer.write(write_frame)
                pbar.update(1)
                next_pts_to_write += 1
                
    writer.release()
    print(f"  ✓ Video decompressed to {output_mp4_path}")
