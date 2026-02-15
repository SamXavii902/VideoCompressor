# VideoCompressor — Advanced AI-Accelerated Video Codec

A high-performance inter-frame video compression suite implemented from scratch in Python, leveraging PyTorch for GPU acceleration and Neural Super-Resolution.

## 🚀 Overview
VideoCompressor is a custom video codec designed to achieve extreme compression ratios while maintaining high visual fidelity. It moves beyond simple frame-by-frame compression by implementing a modern GOP (Group of Pictures) architecture similar to H.264/HEVC, augmented with AI-driven upscaling.

---

## ✨ Key Features
*   **Block Motion Estimation (BME)**: PyTorch-accelerated motion vector calculation using `16x16` macroblocks and a `±16` search window.
*   **Bi-Directional B-Frames**: Sophisticated temporal prediction from both past and future reference frames, reducing residual data.
*   **Neural Super-Resolution (ESPCN)**: A custom Efficient Sub-Pixel Convolutional Neural Network (PyTorch) that compresses video at 180p and reconstructs it back to 360p during playback.
*   **Perceptual Rate-Distortion Optimization (RDO)**: Instead of just targeting PSNR, the codec optimizes for **SSIM (Structural Similarity Index)** to ensure visual quality matches human perception.
*   **Parallel GOP Encoding**: Multiprocessing pipeline that distributes GOP chunks across CPU cores, achieving significant speedups.
*   **Adaptive Hybrid Engine**: Dynamically switches between **Streaming K-Means** (spatial) and **Incremental PCA** (temporal) for non-codec modes.

---

## 📈 Performance Benchmarks
| Mode | Compression Ratio | Perceptual SSIM | PSNR |
| :--- | :--- | :--- | :--- |
| **Extreme (1 Mbps)** | **~213.5x** | 0.703 | 24.4 dB |
| **High Fidelity (100 Mbps)**| **~9.25x** | **0.932** | **33.2 dB** |

---

## 🛠️ Tech Stack
*   **Language**: Python 3.8+
*   **Acceleration**: PyTorch (CUDA/RTX 3050 Optimized)
*   **Computer Vision**: OpenCV, SCIPY (FFT/DCT)
*   **Parallelism**: Multiprocessing Pool

---

## 🏃 Usage
```bash
# Clone the repository
git clone https://github.com/SamXavii902/VideoCompressor.git
cd VideoCompressor

# Install dependencies
pip install -r requirements.txt

# Run the codec with Neural Super-Resolution
python main.py "input_video.mp4" --mode codec --target-bitrate 5000 --use-neural-sr
```

---

## 📄 Documentation
*   [Technical Walkthrough](walkthrough.md): Detailed explanation of BME, Residual encoding, and the custom bitstream.
*   [Task Tracker](task.md): A full log of the 15+ development phases of this project.

Developed by [SamXavii902](https://github.com/SamXavii902).
