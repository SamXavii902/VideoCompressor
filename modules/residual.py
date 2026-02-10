"""
residual.py — Residual Encoding via DCT, Quantization, and RLE.

Compresses the prediction error (residual) between consecutive frames.
Applies an 8x8 Discrete Cosine Transform (DCT) to compact energy into low frequencies,
divides by a JPEG-style quantization matrix to discard noisy high frequencies,
and encodes the sparse result using Run-Length Encoding (RLE).
"""

import numpy as np
from scipy import fft

# Standard JPEG Luminance Quantization Matrix (Quality = 50)
JPEG_LUMI_Q50 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float32)

def _get_q_matrix(quality_factor):
    """Scale the standard JPEG matrix based on quality_factor (1-100)."""
    q = max(1, min(100, quality_factor))
    if q < 50:
        scale = 5000 / q
    else:
        scale = 200 - 2 * q
        
    scaled_q = np.floor((JPEG_LUMI_Q50 * scale + 50) / 100)
    scaled_q[scaled_q == 0] = 1 # Prevent divide by zero
    return scaled_q

def _zigzag_indices(n):
    """Generate zigzag scan indices for an n x n block."""
    indices = np.empty((n * n, 2), dtype=int)
    index = -1
    for i in range(2 * n - 1):
        if i % 2 == 0:
            # up
            x = min(i, n - 1)
            y = i - x
            while x >= 0 and y < n:
                index += 1
                indices[index] = [y, x]
                x -= 1
                y += 1
        else:
            # down
            y = min(i, n - 1)
            x = i - y
            while y >= 0 and x < n:
                index += 1
                indices[index] = [y, x]
                y -= 1
                x += 1
    return indices

_ZIGZAG_8x8 = _zigzag_indices(8)

def dct_2d_blocks(residual, block_size=8):
    """
    Apply 2D DCT to non-overlapping 8x8 blocks of the residual.
    Expects residual shape (H, W, C).
    """
    h, w, c = residual.shape
    
    # Pad to multiple of block_size if necessary
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        residual = np.pad(residual, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
    ph, pw, _ = residual.shape
    n_h = ph // block_size
    n_w = pw // block_size
    
    # Reshape to (C, n_h, n_w, block_size, block_size)
    blocks = residual.transpose(2, 0, 1).reshape(c, n_h, block_size, n_w, block_size).transpose(0, 1, 3, 2, 4)
    
    # Apply DCT to the last two dimensions (the 8x8 block itself)
    # Using 'ortho' norm preserves scale
    dct_blocks = fft.dctn(blocks, axes=(3, 4), norm='ortho')
    
    return dct_blocks, (h, w) # Return original shape for unpadding later

def idct_2d_blocks(dct_blocks, orig_shape):
    """
    Apply Inverse 2D DCT.
    """
    c, n_h, n_w, block_size, _ = dct_blocks.shape
    
    # Inverse DCT
    blocks = fft.idctn(dct_blocks, axes=(3, 4), norm='ortho')
    
    # Reassemble to (C, H, W)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(c, n_h * block_size, n_w * block_size)
    reconstructed = blocks.transpose(1, 2, 0)
    
    # Unpad
    h, w = orig_shape
    return reconstructed[:h, :w]

def quantize_dct(dct_blocks, quality_factor=50):
    """Divide by Q-matrix and round to integer."""
    q_mat = _get_q_matrix(quality_factor)
    # Add axes to broadcast over (C, n_h, n_w, block_size, block_size)
    q_mat = q_mat[np.newaxis, np.newaxis, np.newaxis, :, :]
    
    # Rounding is critical for compression
    quantized = np.round(dct_blocks / q_mat).astype(np.int16)
    return quantized

def dequantize_dct(quantized, quality_factor=50):
    """Multiply by Q-matrix."""
    q_mat = _get_q_matrix(quality_factor)
    q_mat = q_mat[np.newaxis, np.newaxis, np.newaxis, :, :]
    
    return quantized.astype(np.float32) * q_mat

def rle_encode(quantized_blocks):
    """
    Run-Length Encode the sparse quantized DCT blocks.
    Each 8x8 block is zigzag scanned into a 1D 64-element array first.
    """
    # Flatten to list of 64-element blocks
    c, n_h, n_w, block_size, _ = quantized_blocks.shape
    
    # (C * n_h * n_w, 8, 8)
    flat_blocks = quantized_blocks.reshape(-1, block_size, block_size) 
    
    encoded = []
    
    for block in flat_blocks:
        # Zigzag scan
        vector = block[_ZIGZAG_8x8[:, 0], _ZIGZAG_8x8[:, 1]]
        
        # Trim trailing zeros (very common in DCT)
        non_zero_indices = np.nonzero(vector)[0]
        if len(non_zero_indices) == 0:
            encoded.extend([0, 0]) # EOB (End of Block) marker
            continue
            
        last_non_zero = non_zero_indices[-1]
        trimmed = vector[:last_non_zero + 1]
        
        # Standard RLE on the trimmed vector
        run_len = 0
        for val in trimmed:
            if val == 0:
                run_len += 1
            else:
                encoded.extend([run_len, int(val)])
                run_len = 0
                
        # Append EOB
        encoded.extend([0, 0])
        
    # Convert to efficient numpy uint16 array (run, val, run, val...)
    # We use int16 since val can be negative. Run length fits in 16 bits.
    return np.array(encoded, dtype=np.int16)

def rle_decode(encoded_1d, shape_info):
    """
    Decode RLE array back into quantized DCT blocks.
    """
    c, n_h, n_w, block_size = shape_info
    total_blocks = c * n_h * n_w
    
    decoded_blocks = np.zeros((total_blocks, block_size, block_size), dtype=np.int16)
    
    block_idx = 0
    vector_idx = 0
    
    # Create empty 64-element vector to work with
    vector = np.zeros(64, dtype=np.int16)
    
    encoded = encoded_1d.tolist()
    i = 0
    while i < len(encoded):
        run = encoded[i]
        val = encoded[i+1]
        i += 2
        
        if run == 0 and val == 0: # EOB
            # Inverse zigzag
            decoded_blocks[block_idx][_ZIGZAG_8x8[:, 0], _ZIGZAG_8x8[:, 1]] = vector
            block_idx += 1
            vector.fill(0)
            vector_idx = 0
            if block_idx >= total_blocks:
                break
            continue
            
        vector_idx += run
        vector[vector_idx] = val
        vector_idx += 1
        
    return decoded_blocks.reshape(c, n_h, n_w, block_size, block_size)

# =====================================================================
# Top Level Wrappers
# =====================================================================

def encode_residual(residual, quality_factor=50):
    """
    Compress a residual frame.
    Returns compressed bitstream (np.array) and metadata dictionary.
    """
    from modules.gpu_backend import USE_GPU
    if USE_GPU:
        # TODO: Fast GPU matrix DCT implementation
        pass
        
    dct_blocks, orig_shape = dct_2d_blocks(residual)
    quantized = quantize_dct(dct_blocks, quality_factor)
    
    encoded_stream = rle_encode(quantized)
    
    # Compute size
    total_bytes = encoded_stream.nbytes
    
    c, n_h, n_w, block_size, _ = quantized.shape
    shape_info = (c, n_h, n_w, block_size)
    
    return encoded_stream, {
        "orig_shape": orig_shape,
        "shape_info": shape_info,
        "quality_factor": quality_factor,
        "total_bytes": total_bytes
    }

def decode_residual(encoded_stream, metadata):
    """
    Decompress a residual bitstream back into a residual frame.
    """
    quantized = rle_decode(encoded_stream, metadata["shape_info"])
    dct_blocks = dequantize_dct(quantized, metadata["quality_factor"])
    reconstructed_residual = idct_2d_blocks(dct_blocks, metadata["orig_shape"])
    
    # Ensure it returns as an integer residual map
    return np.round(reconstructed_residual).astype(np.int16)
