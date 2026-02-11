"""
gpu_backend.py — PyTorch hardware acceleration layer.

Provides a unified interface that automatically detects whether an Nvidia CUDA GPU
is available and routes mathematical workloads to it. Automatically falls back to
CPU (while still using PyTorch tensors for vectorized speed) if no GPU is found.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import USE_GPU

_DEVICE_CACHE = None

def get_device():
    """Returns the optimal available PyTorch device (CUDA or CPU)."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    if USE_GPU and torch.cuda.is_available():
        _DEVICE_CACHE = torch.device("cuda")
        print(f"\n  [GPU] PyTorch acceleration enabled: using {torch.cuda.get_device_name(0)}")
    else:
        _DEVICE_CACHE = torch.device("cpu")
        if USE_GPU:
            print("\n  [GPU] CUDA not available — falling back to CPU (PyTorch Tensors).")
        else:
            print("\n  [GPU] GPU acceleration explicitly disabled (using CPU).")
    
    return _DEVICE_CACHE

def to_tensor(ndarray, dtype=torch.float32):
    """Convert numpy array to PyTorch tensor on the correct device."""
    if not isinstance(ndarray, np.ndarray):
        ndarray = np.array(ndarray)
    if not ndarray.flags.c_contiguous:
        ndarray = np.ascontiguousarray(ndarray)
    tensor = torch.from_numpy(ndarray).to(dtype)
    return tensor.to(get_device())

def to_numpy(tensor):
    """Convert PyTorch tensor back to contiguous numpy array."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()

# =====================================================================
# GPU Accelerated Algorithms
# =====================================================================

def gpu_motion_estimation(current_frame, ref_frame, block_size, search_range):
    """
    Massively parallel Block Motion Estimation using PyTorch.
    Computes SAD for all candidate block positions at once.
    
    Args:
        current_frame: np.ndarray (H, W, 3)
        ref_frame: np.ndarray (H, W, 3)
    Returns:
        vectors: np.ndarray (n_blocks_h, n_blocks_w, 2)
        costs: np.ndarray (n_blocks_h, n_blocks_w)
    """
    device = get_device()
    h, w, c = current_frame.shape
    
    # Grid dimensions
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    
    # 1. Prepare tensors: shape (B, C, H, W)
    # Crop edges
    curr_crop = current_frame[:n_blocks_h*block_size, :n_blocks_w*block_size]
    ref_crop = ref_frame[:n_blocks_h*block_size, :n_blocks_w*block_size]
    
    curr_t = to_tensor(curr_crop, torch.float32).permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
    
    # We will pad the reference frame by search_range on all sides,
    # but we must pad with replica mode to avoid false boundary artifacts
    ref_t = to_tensor(ref_crop, torch.float32).permute(2, 0, 1).unsqueeze(0)
    pad = (search_range, search_range, search_range, search_range) # L, R, T, B
    ref_padded = F.pad(ref_t, pad, mode='replicate')
    
    # 2. Extract current blocks using unfold
    # Unfold turns (1, C, H, W) into (1, C*K*K, L)
    # L = total number of blocks
    curr_blocks = F.unfold(curr_t, kernel_size=block_size, stride=block_size)
    curr_blocks = curr_blocks.squeeze(0).transpose(0, 1) # (L, C*K*K)
    n_blocks = curr_blocks.shape[0]
    
    # We will accumulate the best costs and vectors across all candidate searches
    best_costs = torch.full((n_blocks,), float('inf'), device=device)
    best_vecs_dx = torch.zeros(n_blocks, dtype=torch.int16, device=device)
    best_vecs_dy = torch.zeros(n_blocks, dtype=torch.int16, device=device)
    
    # 3. Search window loop
    # We iterate over every possible (dy, dx) shift within the search range.
    # For a +/- 16 search window, this is a 33x33 grid = 1089 iterations
    # Each iteration computes SAD for ALL blocks simultaneously.
    # This is much faster on GPU than looping over blocks!
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            
            # The top-left corner of the corresponding valid reference region
            # since we padded ref by search_range, the "zero" offset is at search_range
            start_y = search_range + dy
            start_x = search_range + dx
            
            # Slice reference area to exactly match current frame crop size
            ref_slice = ref_padded[:, :, start_y:start_y+(n_blocks_h*block_size),
                                         start_x:start_x+(n_blocks_w*block_size)]
                                         
            # Unfold the shifted reference exactly like the current frame
            ref_blocks = F.unfold(ref_slice, kernel_size=block_size, stride=block_size)
            ref_blocks = ref_blocks.squeeze(0).transpose(0, 1) # (L, C*K*K)
            
            # Compute SAD for all blocks at this specific shift
            sad = torch.sum(torch.abs(curr_blocks - ref_blocks), dim=1) # (L,)
            
            # Update best vectors where sad is lower
            better_mask = sad < best_costs
            
            best_costs[better_mask] = sad[better_mask]
            
            # Set the motion vectors where a better block was found
            best_vecs_dx[better_mask] = int(dx)
            best_vecs_dy[better_mask] = int(dy)
            
            # Hard exit perfectly matching blocks early (cost == 0)
            if torch.all(best_costs == 0):
                break
        else:
            continue
        break
        
    # 4. Reshape back to Grid (n_blocks_h, n_blocks_w)
    vectors = torch.stack((best_vecs_dx, best_vecs_dy), dim=1).reshape(n_blocks_h, n_blocks_w, 2)
    costs = best_costs.reshape(n_blocks_h, n_blocks_w)
    return to_numpy(vectors), to_numpy(costs)

def compute_dense_optical_flow(current_frame, ref_frame):
    """
    Computes dense optical flow using OpenCV's Farneback algorithm.
    While this is CPU-bound if cv2.cuda is unavailable, it is highly optimized.
    Returns:
        flow: np.ndarray (H, W, 2) of float32, representing (dx, dy) for every pixel.
    """
    # Convert frames to grayscale for optical flow
    import cv2
    prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def gpu_pca(matrix_np, n_components):
    """
    GPU Accelerated PCA using torch.pca_lowrank().
    Operates fully in GPU memory for massive eigen decomposition matrices.
    """
    device = get_device()
    t = to_tensor(matrix_np, torch.float32)
    
    # Center the data
    mean = torch.mean(t, dim=0, keepdim=True)
    t = t - mean
    
    # U (n_samples, q), S (q,), V (n_features, q)
    # torch.pca_lowrank handles the math efficiently
    U, S, V = torch.pca_lowrank(t, q=n_components, center=False, niter=2)
    
    # weights = U * S
    weights = U * S.unsqueeze(0)
    
    pca_data = {
        "weights": to_numpy(weights),
        "components": to_numpy(V.t()),  # Shape to (n_components, n_features)
        "mean": to_numpy(mean).squeeze(0),
        "n_components": n_components,
    }
    
    return pca_data
