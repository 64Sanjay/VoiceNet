#!/usr/bin/env python3
"""Shared utilities for the demo."""

import os
import numpy as np
import torch


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_time(seconds):
    """Format seconds to readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def normalize_audio(waveform, target_db=-20):
    """Normalize audio to target dB level."""
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        waveform = waveform * (target_rms / rms)
    return waveform


def find_checkpoint(module_path, checkpoint_name="best_model.pt"):
    """Find the best available checkpoint."""
    possible_paths = [
        os.path.join(module_path, "checkpoints", checkpoint_name),
        os.path.join(module_path, "outputs", checkpoint_name),
        os.path.join(module_path, checkpoint_name),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Look for any .pt file
    for root, dirs, files in os.walk(module_path):
        for file in files:
            if file.endswith('.pt'):
                return os.path.join(root, file)
    
    return None