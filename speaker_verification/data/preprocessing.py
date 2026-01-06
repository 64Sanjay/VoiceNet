# speaker_verification/data/preprocessing.py
"""
Audio preprocessing for CAM++ speaker verification.
Extracts 80-dimensional Fbank features as per the paper.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional, Tuple, Union

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class FbankExtractor(nn.Module):
    """
    Extract Fbank features for speaker verification.
    
    From the paper:
    "We use 80-dimensional Fbank features extracted over a 25 ms long window 
    for every 10 ms as input."
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,  # 25ms
        hop_length: int = 160,  # 10ms
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 7600.0
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel filterbank
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect"
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Fbank features from waveform.
        
        Args:
            waveform: Audio waveform of shape (batch, samples) or (samples,)
            
        Returns:
            Fbank features of shape (batch, n_mels, time) or (n_mels, time)
        """
        # Ensure batch dimension
        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        
        # Compute mel spectrogram
        mel_spec = self.mel_scale(waveform)
        
        # Convert to log scale (add small epsilon for stability)
        log_mel = torch.log(mel_spec + 1e-9)
        
        if squeeze:
            log_mel = log_mel.squeeze(0)
        
        return log_mel


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for CAM++.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        f_min: float = 20.0,
        f_max: float = 7600.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Convert ms to samples
        self.win_length = int(win_length_ms * sample_rate / 1000)
        self.hop_length = int(hop_length_ms * sample_rate / 1000)
        
        # Feature extractor
        self.fbank_extractor = FbankExtractor(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform."""
        
        if HAS_SOUNDFILE:
            try:
                waveform, sr = sf.read(audio_path, dtype='float32')
                waveform = torch.from_numpy(waveform)
                
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=-1)
                
                if sr != self.sample_rate:
                    waveform = self._resample(waveform, sr, self.sample_rate)
                
                return waveform, self.sample_rate
            except Exception:
                pass
        
        if HAS_LIBROSA:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return torch.from_numpy(waveform.astype(np.float32)), sr
        
        raise RuntimeError("Could not load audio. Install soundfile or librosa.")
    
    def _resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """Resample audio."""
        if HAS_LIBROSA:
            waveform_np = waveform.numpy()
            resampled = librosa.resample(waveform_np, orig_sr=orig_sr, target_sr=target_sr)
            return torch.from_numpy(resampled.astype(np.float32))
        else:
            # Simple interpolation
            ratio = target_sr / orig_sr
            new_len = int(len(waveform) * ratio)
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            waveform = torch.nn.functional.interpolate(waveform, size=new_len, mode='linear')
            return waveform.squeeze()
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract Fbank features from waveform."""
        return self.fbank_extractor(waveform)
    
    def process(self, audio_path: str) -> torch.Tensor:
        """Complete processing pipeline."""
        waveform, _ = self.load_audio(audio_path)
        features = self.extract_features(waveform)
        return features
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply cepstral mean normalization."""
        mean = features.mean(dim=-1, keepdim=True)
        return features - mean