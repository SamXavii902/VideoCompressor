import numpy as np
import cv2
import sys
import os

# Ensure modules are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.motion import estimate_motion_field, extract_blocks
from config import BME_BLOCK_SIZE, BME_SEARCH_RANGE

def create_synthetic_frames(block_size, dx, dy):
    """Creates a basic (128x128) reference frame and a shifted current frame."""
    h, w = 128, 128
    
    # Use deterministic random noise to prevent aliasing/periodic matches
    np.random.seed(42)
    ref = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
                
    curr = np.zeros_like(ref)
    
    # Shift reference by (dx, dy) to create current frame
    for y in range(h):
        for x in range(w):
            src_y = y - dy
            src_x = x - dx
            if 0 <= src_y < h and 0 <= src_x < w:
                curr[y, x] = ref[src_y, src_x]
                
    return ref, curr

def run_test():
    block_size = 16
    search_range = 16
    
    print("Testing BME: Synthetic Shift (+5, -3)")
    dx, dy = 5, -3
    ref, curr = create_synthetic_frames(block_size, dx, dy)
    
    print("Extracting blocks...")
    blocks = extract_blocks(curr, block_size)
    print(f"Blocks shape: {blocks.shape}")
    
    print("Estimating motion field...")
    vectors, costs = estimate_motion_field(curr, ref, block_size, search_range)
    
    # Check the motion vector for a central block that wasn't clipped by shifting
    n_h, n_w = vectors.shape[:2]
    mid_y, mid_x = n_h // 2, n_w // 2
    
    pred_dx, pred_dy = vectors[mid_y, mid_x]
    cost = costs[mid_y, mid_x]
    
    print(f"Expected MV: ({dx}, {dy})", flush=True)
    print(f"Predicted MV: ({pred_dx}, {pred_dy}) | SAD Cost: {cost}", flush=True)
    
    with open("utils/test_log.txt", "w") as f:
        f.write(f"Expected MV: ({dx}, {dy})\n")
        f.write(f"Predicted MV: ({pred_dx}, {pred_dy}) | SAD Cost: {cost}\n")
        
    if pred_dx == -dx and pred_dy == -dy:
        print("✅ TEST PASSED: Motion vector perfectly matched (backward estimation).", flush=True)
        return 0
    else:
        print("❌ TEST FAILED: Motion vector mismatch.", flush=True)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(run_test())
    except Exception as e:
        import traceback
        with open("utils/test_trace.txt", "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
