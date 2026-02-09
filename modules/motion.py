"""
motion.py — Block Motion Estimation and Compensation.

Estimates how blocks of pixels move between consecutive frames to reduce
temporal redundancy. Returns motion vectors and predicted frames.
"""

import numpy as np
from tqdm import tqdm

def extract_blocks(frame, block_size):
    """
    Divide frame into non-overlapping macroblocks.
    
    Returns array of shape (n_blocks_h, n_blocks_w, block_size, block_size, 3).
    Ignores right/bottom edges if they do not perfectly divide by block_size.
    """
    h, w, c = frame.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    
    # Crop edges
    cropped = frame[:n_blocks_h * block_size, :n_blocks_w * block_size]
    
    # Reshape and swap axes to get (n_blocks_h, n_blocks_w, block_size, block_size, 3)
    blocks = cropped.reshape(n_blocks_h, block_size, n_blocks_w, block_size, c)
    blocks = blocks.swapaxes(1, 2)
    return blocks

def block_match_sad(current_block, ref_frame, block_ys, block_xs, block_size, search_range):
    """
    Full-search block matching using SSD or SAD.
    Returns the best (dx, dy) motion vector and minimum cost.
    (Currently CPU-bound numpy baseline; GPU acceleration handles this in bulk).
    """
    h, w, _ = ref_frame.shape
    
    # Search window bounds
    min_y = max(0, block_ys - search_range)
    max_y = min(h - block_size, block_ys + search_range)
    min_x = max(0, block_xs - search_range)
    max_x = min(w - block_size, block_xs + search_range)
    
    best_vector = (0, 0)
    min_cost = float('inf')
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            candidate = ref_frame[y:y + block_size, x:x + block_size]
            
            # SAD (Sum of Absolute Differences)
            cost = np.sum(np.abs(current_block.astype(np.int32) - candidate.astype(np.int32)))
            
            if cost < min_cost:
                min_cost = cost
                # Vector is (candidate - original)
                best_vector = (x - block_xs, y - block_ys)
                
            # Early exit perfectly matching blocks
            if cost == 0:
                return best_vector, min_cost
                
    return best_vector, min_cost

def estimate_motion_field(current_frame, ref_frame, block_size, search_range, method="optical_flow"):
    """
    Compute motion vector field for a frame pair.
    If method="optical_flow", uses Farneback pixel optical flow and downsamples to save space.
    If method="block_matching", uses traditional macroblock search.
    """
    if method == "optical_flow":
        import cv2
        from modules.gpu_backend import compute_dense_optical_flow
        
        flow = compute_dense_optical_flow(current_frame, ref_frame)
        h, w = flow.shape[:2]
        
        # Downsample the dense pixel flow to block grid dimensions to save bitstream space
        new_h, new_w = h // block_size, w // block_size
        flow_downsampled = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # We store as float16 to keep bitstream size identical to BME but with sub-pixel floating accuracy
        return flow_downsampled.astype(np.float16), np.zeros((new_h, new_w), dtype=np.float32)

    # Prefer GPU backend if available for Block Matching
    from modules.gpu_backend import USE_GPU
    if USE_GPU:
        from modules.gpu_backend import gpu_motion_estimation
        return gpu_motion_estimation(current_frame, ref_frame, block_size, search_range)
        
    blocks = extract_blocks(current_frame, block_size)
    n_blocks_h, n_blocks_w = blocks.shape[:2]
    
    vectors = np.zeros((n_blocks_h, n_blocks_w, 2), dtype=np.int16)
    costs = np.zeros((n_blocks_h, n_blocks_w), dtype=np.float32)
    
    for i in tqdm(range(n_blocks_h), desc="CPU Motion Search", leave=False):
        for j in range(n_blocks_w):
            current_block = blocks[i, j]
            # Top-left pixel coordinate of this block
            by = i * block_size
            bx = j * block_size
            
            vec, cost = block_match_sad(current_block, ref_frame, by, bx, block_size, search_range)
            vectors[i, j] = vec
            costs[i, j] = cost
            
    return vectors, costs

def motion_compensate(ref_frame, motion_vectors, block_size, method="optical_flow"):
    """
    Reconstruct the predicted frame given the reference frame and motion vectors.
    """
    h, w, c = ref_frame.shape
    
    if method == "optical_flow":
        import cv2
        # Upsample the compressed flow array back to full screen density
        if motion_vectors.shape[:2] != (h, w):
            flow = cv2.resize(motion_vectors.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            flow = motion_vectors.astype(np.float32)
            
        map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
        map_y = np.tile(np.arange(h).reshape(-1, 1), (1, w)).astype(np.float32)
        
        # Optical flow displacement (Flow outputs where pixels moved TO)
        map_x += flow[..., 0]
        map_y += flow[..., 1]
        
        predicted = cv2.remap(ref_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return predicted

    n_blocks_h, n_blocks_w = motion_vectors.shape[:2]
    
    predicted = np.zeros_like(ref_frame)
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            dx, dy = motion_vectors[i, j]
            
            # Destination Top-Left
            dest_y = i * block_size
            dest_x = j * block_size
            
            # Source Top-Left
            src_y = dest_y + dy
            src_x = dest_x + dx
            
            # Ensure safe bounds
            if src_y >= 0 and src_y + block_size <= h and src_x >= 0 and src_x + block_size <= w:
                predicted[dest_y:dest_y+block_size, dest_x:dest_x+block_size] = \
                    ref_frame[src_y:src_y+block_size, src_x:src_x+block_size]
            else:
                # If motion vector is out of bounds (should not happen with good search window),
                # fallback to zero motion
                predicted[dest_y:dest_y+block_size, dest_x:dest_x+block_size] = \
                    ref_frame[dest_y:dest_y+block_size, dest_x:dest_x+block_size]
                    
    # Fill in any cropped margins (right/bottom) statically
    crop_h = n_blocks_h * block_size
    crop_w = n_blocks_w * block_size
    if crop_h < h:
        predicted[crop_h:, :] = ref_frame[crop_h:, :]
    if crop_w < w:
        predicted[:, crop_w:] = ref_frame[:, crop_w:]
        
    return predicted

def compute_residual(current_frame, predicted_frame):
    """
    Calculate the difference between current and predicted.
    Returns int16 array (-255 to 255).
    """
    return current_frame.astype(np.int16) - predicted_frame.astype(np.int16)

def reconstruct_frame(predicted_frame, residual):
    """
    Add residual back to predicted frame.
    Returns uint8 array (0 to 255).
    """
    recon = predicted_frame.astype(np.int16) + residual
    return np.clip(recon, 0, 255).astype(np.uint8)
