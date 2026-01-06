"""
Data augmentation for speaker diarization.
Includes noise addition, reverberation, speed perturbation, and more.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple, List, Union
import random
import numpy as np
from pathlib import Path


class AudioAugmentor:
    """
    Comprehensive audio augmentation pipeline for speaker diarization.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        augmentation_prob: float = 0.5,
        noise_path: Optional[str] = None,  # MUSAN noise path
        rir_path: Optional[str] = None,    # Room impulse response path
    ):
        """
        Initialize augmentor.
        
        Args:
            sample_rate: Audio sample rate
            augmentation_prob: Probability of applying augmentation
            noise_path: Path to noise files (MUSAN dataset)
            rir_path: Path to room impulse response files
        """
        self.sample_rate = sample_rate
        self.augmentation_prob = augmentation_prob
        self.noise_path = Path(noise_path) if noise_path else None
        self.rir_path = Path(rir_path) if rir_path else None
        
        # Load noise files if available
        self.noise_files = []
        if self.noise_path and self.noise_path.exists():
            self.noise_files = list(self.noise_path.rglob("*.wav"))
        
        # Load RIR files if available
        self.rir_files = []
        if self.rir_path and self.rir_path.exists():
            self.rir_files = list(self.rir_path.rglob("*.wav"))
        
        # Initialize augmentation transforms
        self.speed_perturb = SpeedPerturbation(sample_rate)
        self.time_masking = T.TimeMasking(time_mask_param=50)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
    
    def __call__(
        self,
        waveform: torch.Tensor,
        augmentations: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Apply augmentations to waveform.
        
        Args:
            waveform: Input waveform [1, T] or [T]
            augmentations: List of augmentations to apply
                          (None = random selection)
            
        Returns:
            Augmented waveform
        """
        if random.random() > self.augmentation_prob:
            return waveform
        
        if augmentations is None:
            augmentations = self._select_random_augmentations()
        
        for aug in augmentations:
            if aug == "noise" and self.noise_files:
                waveform = self.add_noise(waveform)
            elif aug == "reverb" and self.rir_files:
                waveform = self.add_reverb(waveform)
            elif aug == "speed":
                waveform = self.speed_perturb(waveform)
            elif aug == "pitch":
                waveform = self.pitch_shift(waveform)
            elif aug == "volume":
                waveform = self.change_volume(waveform)
            elif aug == "clip":
                waveform = self.random_clip(waveform)
        
        return waveform
    
    def _select_random_augmentations(self) -> List[str]:
        """Randomly select augmentations to apply."""
        available = ["speed", "pitch", "volume"]
        
        if self.noise_files:
            available.append("noise")
        if self.rir_files:
            available.append("reverb")
        
        # Select 1-3 augmentations
        num_augs = random.randint(1, min(3, len(available)))
        return random.sample(available, num_augs)
    
    def add_noise(
        self,
        waveform: torch.Tensor,
        snr_range: Tuple[float, float] = (5.0, 20.0),
    ) -> torch.Tensor:
        """Add noise from MUSAN dataset."""
        if not self.noise_files:
            return waveform
        
        # Load random noise file
        noise_file = random.choice(self.noise_files)
        noise, sr = torchaudio.load(str(noise_file))
        
        # Resample if necessary
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
        
        # Convert to mono
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        
        # Match length
        if noise.shape[-1] < waveform.shape[-1]:
            repeats = waveform.shape[-1] // noise.shape[-1] + 1
            noise = noise.repeat(1, repeats)
        noise = noise[..., :waveform.shape[-1]]
        
        # Random SNR
        snr = random.uniform(*snr_range)
        
        # Mix
        return self._mix_with_snr(waveform, noise, snr)
    
    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add reverberation using room impulse response."""
        if not self.rir_files:
            return waveform
        
        # Load random RIR
        rir_file = random.choice(self.rir_files)
        rir, sr = torchaudio.load(str(rir_file))
        
        # Resample if necessary
        if sr != self.sample_rate:
            rir = torchaudio.functional.resample(rir, sr, self.sample_rate)
        
        # Convert to mono
        if rir.shape[0] > 1:
            rir = rir.mean(dim=0, keepdim=True)
        
        # Normalize RIR
        rir = rir / rir.abs().max()
        
        # Convolve
        reverbed = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            rir.flip(-1).unsqueeze(0),
            padding=rir.shape[-1] - 1,
        ).squeeze(0)
        
        # Trim to original length
        reverbed = reverbed[..., :waveform.shape[-1]]
        
        # Normalize
        reverbed = reverbed / reverbed.abs().max().clamp(min=1e-8)
        
        return reverbed
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        semitones_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> torch.Tensor:
        """Apply pitch shifting."""
        semitones = random.uniform(*semitones_range)
        
        # Use torchaudio pitch shift
        shifted = torchaudio.functional.pitch_shift(
            waveform,
            self.sample_rate,
            semitones,
        )
        
        return shifted
    
    def change_volume(
        self,
        waveform: torch.Tensor,
        gain_range: Tuple[float, float] = (0.5, 1.5),
    ) -> torch.Tensor:
        """Change volume randomly."""
        gain = random.uniform(*gain_range)
        return waveform * gain
    
    def random_clip(
        self,
        waveform: torch.Tensor,
        clip_fraction: float = 0.1,
    ) -> torch.Tensor:
        """Randomly clip a portion of the audio to simulate clipping distortion."""
        if random.random() < clip_fraction:
            threshold = random.uniform(0.5, 0.9)
            waveform = torch.clamp(waveform, -threshold, threshold)
        return waveform
    
    def _mix_with_snr(
        self,
        signal: torch.Tensor,
        noise: torch.Tensor,
        snr_db: float,
    ) -> torch.Tensor:
        """Mix signal with noise at specified SNR."""
        signal_power = (signal ** 2).mean()
        noise_power = (noise ** 2).mean()
        
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
        
        return signal + scale * noise


class SpeedPerturbation(nn.Module):
    """
    Speed perturbation augmentation.
    Commonly used factors: 0.9, 1.0, 1.1
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        speeds: List[float] = [0.9, 0.95, 1.0, 1.05, 1.1],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation."""
        speed = random.choice(self.speeds)
        
        if speed == 1.0:
            return waveform
        
        # Resample to change speed
        new_sr = int(self.sample_rate * speed)
        
        # Resample to new sample rate (changes speed)
        resampled = torchaudio.functional.resample(
            waveform, self.sample_rate, new_sr
        )
        
        # Resample back to original sample rate
        resampled = torchaudio.functional.resample(
            resampled, new_sr, self.sample_rate
        )
        
        return resampled


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.
    Applied to spectrograms/mel-spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram [B, F, T] or [F, T]
            
        Returns:
            Augmented spectrogram
        """
        augmented = spectrogram.clone()
        
        # Apply frequency masks
        for _ in range(self.num_freq_masks):
            augmented = self.freq_masking(augmented)
        
        # Apply time masks
        for _ in range(self.num_time_masks):
            augmented = self.time_masking(augmented)
        
        return augmented


class MixUp(nn.Module):
    """
    MixUp augmentation for speaker diarization.
    Mixes two audio samples with random weights.
    """
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        waveform1: torch.Tensor,
        waveform2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to two samples.
        
        Args:
            waveform1: First waveform
            waveform2: Second waveform
            labels1: First labels
            labels2: Second labels
            
        Returns:
            Mixed waveform and labels
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Ensure same length
        min_len = min(waveform1.shape[-1], waveform2.shape[-1])
        waveform1 = waveform1[..., :min_len]
        waveform2 = waveform2[..., :min_len]
        
        # Mix
        mixed_waveform = lam * waveform1 + (1 - lam) * waveform2
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_waveform, mixed_labels


class RandomCrop(nn.Module):
    """Random crop augmentation for audio."""
    
    def __init__(
        self,
        crop_duration: float,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.crop_samples = int(crop_duration * sample_rate)
        self.sample_rate = sample_rate
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Randomly crop waveform.
        
        Args:
            waveform: Input waveform [C, T]
            
        Returns:
            Cropped waveform and start index
        """
        if waveform.shape[-1] <= self.crop_samples:
            # Pad if too short
            padding = self.crop_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform, 0
        
        # Random start position
        max_start = waveform.shape[-1] - self.crop_samples
        start = random.randint(0, max_start)
        
        return waveform[..., start:start + self.crop_samples], start


def create_augmentation_pipeline(
    sample_rate: int = 16000,
    augmentation_prob: float = 0.5,
    musan_path: Optional[str] = None,
    rir_path: Optional[str] = None,
) -> AudioAugmentor:
    """
    Create standard augmentation pipeline.
    
    Args:
        sample_rate: Audio sample rate
        augmentation_prob: Probability of augmentation
        musan_path: Path to MUSAN dataset
        rir_path: Path to RIR dataset
        
    Returns:
        Configured AudioAugmentor
    """
    return AudioAugmentor(
        sample_rate=sample_rate,
        augmentation_prob=augmentation_prob,
        noise_path=musan_path,
        rir_path=rir_path,
    )