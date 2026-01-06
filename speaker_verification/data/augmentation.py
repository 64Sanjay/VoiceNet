# speaker_verification/data/augmentation.py
"""
Data augmentation for CAM++ speaker verification.
Implements speed perturbation, noise addition (MUSAN), and reverb (RIR).
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from typing import Optional, List, Tuple, Union

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class SpeedPerturbation:
    """
    Speed perturbation augmentation.
    
    From the paper:
    "We apply speed perturbation augmentation by randomly sampling 
    a ratio from {0.9, 1.0, 1.1}."
    """
    
    def __init__(self, rates: Tuple[float, ...] = (0.9, 1.0, 1.1)):
        self.rates = rates
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        rate = random.choice(self.rates)
        
        if rate == 1.0:
            return waveform
        
        # Resample to achieve speed change
        original_length = waveform.shape[-1]
        new_length = int(original_length / rate)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            waveform = torch.nn.functional.interpolate(
                waveform, size=new_length, mode='linear', align_corners=False
            )
            waveform = waveform.squeeze()
        else:
            waveform = waveform.unsqueeze(1)
            waveform = torch.nn.functional.interpolate(
                waveform, size=new_length, mode='linear', align_corners=False
            )
            waveform = waveform.squeeze(1)
        
        return waveform


class NoiseAugmentation:
    """
    Add noise from MUSAN dataset.
    
    From the paper:
    "adding noise using the MUSAN dataset"
    """
    
    def __init__(
        self,
        musan_path: str,
        snr_range: Tuple[float, float] = (0, 15),
        sample_rate: int = 16000
    ):
        self.musan_path = Path(musan_path)
        self.snr_range = snr_range
        self.sample_rate = sample_rate
        
        # Load noise file list
        self.noise_files = []
        if self.musan_path.exists():
            for category in ['noise', 'music', 'speech']:
                category_path = self.musan_path / category
                if category_path.exists():
                    self.noise_files.extend(list(category_path.rglob("*.wav")))
        
        if len(self.noise_files) == 0:
            print(f"Warning: No MUSAN files found at {musan_path}")
    
    def _load_noise(self, noise_path: str, target_length: int) -> torch.Tensor:
        """Load and prepare noise file."""
        if HAS_SOUNDFILE:
            noise, sr = sf.read(noise_path, dtype='float32')
            noise = torch.from_numpy(noise)
        elif HAS_LIBROSA:
            noise, sr = librosa.load(noise_path, sr=self.sample_rate, mono=True)
            noise = torch.from_numpy(noise.astype(np.float32))
        else:
            return torch.zeros(target_length)
        
        if noise.dim() > 1:
            noise = noise.mean(dim=-1)
        
        # Repeat or truncate to match target length
        if len(noise) < target_length:
            repeats = target_length // len(noise) + 1
            noise = noise.repeat(repeats)
        
        noise = noise[:target_length]
        return noise
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if len(self.noise_files) == 0:
            return waveform
        
        # Random SNR
        snr_db = random.uniform(*self.snr_range)
        
        # Random noise file
        noise_path = random.choice(self.noise_files)
        noise = self._load_noise(str(noise_path), len(waveform))
        
        # Calculate scaling factor for desired SNR
        signal_power = (waveform ** 2).mean()
        noise_power = (noise ** 2).mean()
        
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (noise_power * snr_linear))
            noise = noise * scale
        
        return waveform + noise


class ReverbAugmentation:
    """
    Add reverberation using RIR dataset.
    
    From the paper:
    "simulating reverberation using the RIR dataset"
    """
    
    def __init__(self, rir_path: str, sample_rate: int = 16000):
        self.rir_path = Path(rir_path)
        self.sample_rate = sample_rate
        
        # Load RIR file list
        self.rir_files = []
        if self.rir_path.exists():
            self.rir_files = list(self.rir_path.rglob("*.wav"))
        
        if len(self.rir_files) == 0:
            print(f"Warning: No RIR files found at {rir_path}")
    
    def _load_rir(self, rir_path: str) -> torch.Tensor:
        """Load RIR file."""
        if HAS_SOUNDFILE:
            rir, sr = sf.read(rir_path, dtype='float32')
            rir = torch.from_numpy(rir)
        elif HAS_LIBROSA:
            rir, sr = librosa.load(rir_path, sr=self.sample_rate, mono=True)
            rir = torch.from_numpy(rir.astype(np.float32))
        else:
            return None
        
        if rir.dim() > 1:
            rir = rir[:, 0]  # Take first channel
        
        # Normalize RIR
        rir = rir / (rir.abs().max() + 1e-8)
        
        return rir
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if len(self.rir_files) == 0:
            return waveform
        
        # Random RIR file
        rir_path = random.choice(self.rir_files)
        rir = self._load_rir(str(rir_path))
        
        if rir is None:
            return waveform
        
        # Convolve with RIR
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        rir = rir.flip(0).unsqueeze(0).unsqueeze(0)
        
        reverbed = torch.nn.functional.conv1d(
            waveform, rir, padding=rir.shape[-1] - 1
        )
        reverbed = reverbed.squeeze()[:len(waveform.squeeze())]
        
        # Normalize to original level
        reverbed = reverbed * (waveform.squeeze().abs().max() / (reverbed.abs().max() + 1e-8))
        
        return reverbed


class SpecAugment(nn.Module):
    """
    SpecAugment for spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 5,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment.
        
        Args:
            spec: Spectrogram of shape (freq, time) or (batch, freq, time)
            
        Returns:
            Augmented spectrogram
        """
        spec = spec.clone()
        
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size, freq_dim, time_dim = spec.shape
        
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, freq_dim - f)
            spec[:, f0:f0+f, :] = 0
        
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_dim))
            t0 = random.randint(0, time_dim - t)
            spec[:, :, t0:t0+t] = 0
        
        if squeeze:
            spec = spec.squeeze(0)
        
        return spec


class SpeakerAugmentor:
    """
    Complete augmentation pipeline for speaker verification.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        speed_perturb: bool = True,
        speed_rates: Tuple[float, ...] = (0.9, 1.0, 1.1),
        noise_aug: bool = True,
        musan_path: str = "./data/musan",
        noise_snr_range: Tuple[float, float] = (0, 15),
        reverb_aug: bool = True,
        rir_path: str = "./data/rir_noises",
        spec_augment: bool = True
    ):
        self.sample_rate = sample_rate
        
        # Waveform augmentations
        self.speed_perturb = SpeedPerturbation(speed_rates) if speed_perturb else None
        self.noise_aug = NoiseAugmentation(musan_path, noise_snr_range, sample_rate) if noise_aug else None
        self.reverb_aug = ReverbAugmentation(rir_path, sample_rate) if reverb_aug else None
        
        # Spectrogram augmentation
        self.spec_augment = SpecAugment() if spec_augment else None
    
    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply waveform-level augmentations."""
        
        # Speed perturbation
        if self.speed_perturb is not None and random.random() < 0.5:
            waveform = self.speed_perturb(waveform, self.sample_rate)
        
        # Noise or reverb (not both)
        if random.random() < 0.5:
            if self.noise_aug is not None and random.random() < 0.5:
                waveform = self.noise_aug(waveform)
            elif self.reverb_aug is not None:
                waveform = self.reverb_aug(waveform)
        
        return waveform
    
    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram-level augmentations."""
        if self.spec_augment is not None:
            spec = self.spec_augment(spec)
        return spec