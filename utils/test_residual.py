import numpy as np
import sys
import os

# Ensure modules are discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.residual import encode_residual, decode_residual

def run_test():
    # Create a synthetic residual frame (e.g. 240x320x3)
    # Most of it is 0 (no motion), but we add a few blocks of errors
    h, w, c = 240, 320, 3
    residual = np.zeros((h, w, c), dtype=np.int16)
    
    # Add a block of error (+50 intensity in the red channel)
    residual[10:30, 10:30, 2] = 50
    # Add some random noise to simulate real imperfections
    noise = np.random.randint(-5, 5, size=(h, w, c), dtype=np.int16)
    residual += noise
    
    print(f"Original Residual size: {residual.nbytes / 1024:.2f} KB (Shape: {residual.shape})")
    
    print("\nEncoding Residual (Quality=30)...")
    encoded_stream, metadata = encode_residual(residual, quality_factor=30)
    print(f"Encoded Stream size: {encoded_stream.nbytes / 1024:.2f} KB")
    
    compression_ratio = residual.nbytes / encoded_stream.nbytes
    print(f"Achieved Compression Ratio: {compression_ratio:.2f}x")
    
    print("\nDecoding Residual...")
    decoded_residual = decode_residual(encoded_stream, metadata)
    
    print(f"Decoded Residual shape: {decoded_residual.shape}")
    
    # Calculate Mean Absolute Error to show quantization loss
    mae = np.mean(np.abs(residual - decoded_residual))
    print(f"Mean Absolute Error (Quantization loss): {mae:.2f} pixel intensity units")
    
    if decoded_residual.shape == residual.shape and compression_ratio > 1:
        print("\n✅ TEST PASSED: Residual encoded, decoded, and actually compressed the data.")
        return 0
    else:
        print("\n❌ TEST FAILED: Pipeline broken or failed to compress.")
        return 1

if __name__ == "__main__":
    sys.exit(run_test())
