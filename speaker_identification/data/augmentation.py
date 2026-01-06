# data/augmentation.py
"""
Data augmentation for speaker identification.
Implements Gaussian noise and time-stretch as per the paper.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import random

# Try to import librosa for time stretching
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioAugmentor:
    """
    Audio augmentation class implementing:
    1. Gaussian noise augmentation
    2. Time-stretch augmentation
    
    As per Algorithm 1 in the paper:
    - x_i^(n) ← NoiseAugmentation(x_i)
    - x_i^(t) ← TimeStretch(x_i)
    """
    
    def __init__(
        self,
        noise_snr_db: Tuple[float, float] = (5.0, 20.0),
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        sample_rate: int = 16000
    ):
        """
        Initialize augmentor.
        
        Args:
            noise_snr_db: SNR range in dB for noise augmentation
            time_stretch_range: Range for time stretch factor
            sample_rate: Audio sample rate
        """
        self.noise_snr_db = noise_snr_db
        self.time_stretch_range = time_stretch_range
        self.sample_rate = sample_rate
    
    def add_gaussian_noise(
        self,
        waveform: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Add Gaussian noise to waveform.
        
        Implements NoiseAugmentation(x_i) from Algorithm 1.
        
        Args:
            waveform: Input audio waveform
            snr_db: Signal-to-noise ratio in dB (random if None)
            
        Returns:
            Noise-augmented waveform
        """
        if snr_db is None:
            snr_db = random.uniform(*self.noise_snr_db)
        
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2)
        
        # Avoid division by zero
        if signal_power < 1e-10:
            return waveform
        
        # Calculate required noise power for target SNR
        # SNR_dB = 10 * log10(signal_power / noise_power)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        # Add noise to signal
        noisy_waveform = waveform + noise
        
        return noisy_waveform
    
    def time_stretch(
        self,
        waveform: torch.Tensor,
        stretch_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply time stretching to waveform.
        
        Implements TimeStretch(x_i) from Algorithm 1.
        
        Args:
            waveform: Input audio waveform
            stretch_factor: Time stretch factor (random if None)
            
        Returns:
            Time-stretched waveform
        """
        if stretch_factor is None:
            stretch_factor = random.uniform(*self.time_stretch_range)
        
        original_length = waveform.shape[-1]
        
        # Method 1: Use librosa if available (better quality)
        if HAS_LIBROSA:
            try:
                waveform_np = waveform.numpy()
                
                # Time stretch using librosa
                stretched_np = librosa.effects.time_stretch(
                    waveform_np, 
                    rate=stretch_factor
                )
                
                stretched = torch.from_numpy(stretched_np.astype(np.float32))
                
                # Adjust to original length
                if stretched.shape[-1] < original_length:
                    padding = original_length - stretched.shape[-1]
                    stretched = torch.nn.functional.pad(stretched, (0, padding))
                else:
                    stretched = stretched[:original_length]
                
                return stretched
                
            except Exception:
                pass  # Fall back to simple method
        
        # Method 2: Simple resampling-based time stretch
        # This changes both speed and pitch, but works without extra dependencies
        new_length = int(original_length / stretch_factor)
        
        # Ensure waveform is 3D for interpolation: (batch, channels, length)
        if waveform.dim() == 1:
            waveform_3d = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform_3d = waveform.unsqueeze(0)
        else:
            waveform_3d = waveform
        
        # Resample to new length
        stretched = torch.nn.functional.interpolate(
            waveform_3d,
            size=new_length,
            mode='linear',
            align_corners=False
        )
        
        # Squeeze back to original dimensions
        stretched = stretched.squeeze()
        
        # Adjust to original length (pad or truncate)
        if stretched.shape[-1] < original_length:
            padding = original_length - stretched.shape[-1]
            stretched = torch.nn.functional.pad(stretched, (0, padding))
        else:
            stretched = stretched[..., :original_length]
        
        return stretched
    
    def augment(
        self,
        waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply both augmentations and return augmented versions.
        
        As per Algorithm 1, generates:
        - x_i^(n): Noise-augmented version
        - x_i^(t): Time-stretched version
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Tuple of (noise_augmented, time_stretched) waveforms
        """
        noise_augmented = self.add_gaussian_noise(waveform)
        time_stretched = self.time_stretch(waveform)
        
        return noise_augmented, time_stretched


class SpecAugmentor:
    """
    Spectrogram-level augmentation (optional).
    Can be used in addition to waveform augmentation.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 50,
        n_freq_masks: int = 1,
        n_time_masks: int = 1
    ):
        """
        Initialize spectrogram augmentor.
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def _apply_freq_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        freq_dim = spectrogram.shape[-2]
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, freq_dim - f)
        spectrogram[..., f0:f0+f, :] = 0
        return spectrogram
    
    def _apply_time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        time_dim = spectrogram.shape[-1]
        t = random.randint(0, min(self.time_mask_param, time_dim))
        t0 = random.randint(0, time_dim - t)
        spectrogram[..., t0:t0+t] = 0
        return spectrogram
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram of shape (F, T)
            
        Returns:
            Augmented spectrogram
        """
        augmented = spectrogram.clone()
        
        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            augmented = self._apply_freq_mask(augmented)
        
        # Apply time masks
        for _ in range(self.n_time_masks):
            augmented = self._apply_time_mask(augmented)
        
        return augmented