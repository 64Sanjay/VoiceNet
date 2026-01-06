# speaker_identification/utils/audio_utils.py
"""
Audio utility functions.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Union


def load_audio(
    filepath: str,
    target_sr: int = 16000
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (16kHz per paper)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0), target_sr


def save_audio(
    waveform: Union[torch.Tensor, np.ndarray],
    filepath: str,
    sample_rate: int = 16000
):
    """
    Save audio to file.
    
    Args:
        waveform: Audio waveform
        filepath: Output file path
        sample_rate: Sample rate
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    torchaudio.save(filepath, waveform, sample_rate)