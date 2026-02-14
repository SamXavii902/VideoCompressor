"""
neural_sr.py — Lightweight Neural Super-Resolution

Integrates Deep Learning to upscale extremely low-res bitstreams (e.g. 180p)
back to standard definition (e.g. 360p) using an ESPCN architecture.
Includes a quick overfit-training loop that trains the network on the specific
target video before encoding, maximizing texture reconstruction!
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from modules.gpu_backend import get_device

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN)
    Upscales a low-resolution image utilizing sub-pixel convolution (PixelShuffle).
    """
    def __init__(self, upscale_factor=2):
        super(ESPCN, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # Channel multiplier for sub-pixel recombination
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

_SR_MODEL = None

def init_sr_model(model_path=None, upscale_factor=2):
    """Loads the SR model into global context for fast inference during playback."""
    global _SR_MODEL
    _SR_MODEL = ESPCN(upscale_factor=upscale_factor)
    if model_path and os.path.exists(model_path):
        _SR_MODEL.load_state_dict(torch.load(model_path, map_location=get_device()))
    else:
        print(f"  [Neural SR] Warning: Model file {model_path} not found. Output will be randomized noise.")
        
    _SR_MODEL = _SR_MODEL.to(get_device())
    _SR_MODEL.eval()

def upscale_frame(frame_bgr):
    """
    Take a heavily compressed low-res BGR frame (e.g. 180p)
    and pass it through the CNN to yield an upscaled BGR frame.
    """
    global _SR_MODEL
    if _SR_MODEL is None:
        init_sr_model()

    # Convert BGR to RGB tensor [1, 3, H, W] normalized
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(get_device())
    
    with torch.no_grad():
        out = _SR_MODEL(tensor)
        
    # Reconstruct to BGR uint8
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def train_sr_model_on_video(video_path, model_out_path, upscale_factor=2, epochs=25, sample_frames=100):
    """
    Fast-trains the SR model using sample frames from the target video.
    By overfitting the model to the video's specific textures (e.g. "Hitman 3" objects),
    the upscaler performs exceptionally well without needing massive generic pre-training.
    """
    print(f"\n[Neural SR] Training ESPCN on {sample_frames} frames from this video to tailor weights...")
    
    device = get_device()
    model = ESPCN(upscale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 1. Extract sample frames for Training Dataset
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sample_frames)
    
    hr_frames = []
    lr_frames = []
    
    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret: break
        
        # High Res (Ground Truth defaults to 360p for this codec)
        hr = cv2.resize(frame, (640, 360))
        # Low Res (Input defaults to 180p)
        h, w = hr.shape[:2]
        lr = cv2.resize(hr, (w // upscale_factor, h // upscale_factor), interpolation=cv2.INTER_AREA)
        
        hr_frames.append(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
        lr_frames.append(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Convert to Tensors
    lr_tensors = torch.from_numpy(np.array(lr_frames)).permute(0, 3, 1, 2).float() / 255.0
    hr_tensors = torch.from_numpy(np.array(hr_frames)).permute(0, 3, 1, 2).float() / 255.0
    
    # Move dataset to GPU for hyper-fast training
    dataset = torch.utils.data.TensorDataset(lr_tensors, hr_tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    model.train()
    with tqdm(total=epochs, desc="SR Model Training") as pbar:
        for epoch in range(epochs):
            epoch_loss = 0
            for lr_batch, hr_batch in loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                
                optimizer.zero_grad()
                output = model(lr_batch)
                loss = criterion(output, hr_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            pbar.set_postfix({"loss": f"{epoch_loss/len(loader):.4f}"})
            pbar.update(1)
            
    # Save the custom weights
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"  ✓ Saved optimized custom SR weights to {model_out_path}")
