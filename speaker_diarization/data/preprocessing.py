# """
# Audio preprocessing and feature extraction for speaker diarization.
# Extracts Mel-filterbank, MFCC, and other acoustic features.
# """

# import math
# import torch
# import torch.nn as nn
# import torchaudio
# import torchaudio.transforms as T
# import numpy as np
# from typing import List, Optional, Tuple

# class FeatureExtractor(nn.Module):
#     """
#     Extract acoustic features from audio waveforms.
#     Supports Mel-filterbank, MFCC, and spectrogram features.
#     """
    
#     def __init__(
#         self,
#         sample_rate: int = 16000,
#         n_fft: int = 512,
#         hop_length: int = 160,
#         win_length: int = 400,
#         n_mels: int = 80,
#         n_mfcc: int = 40,
#         fmin: int = 20,
#         fmax: int = 7600,
#         feature_type: str = "mel",  # mel, mfcc, spectrogram
#         normalize: bool = True,
#         deltas: bool = False,
#     ):
#         """
#         Initialize feature extractor.
        
#         Args:
#             sample_rate: Audio sample rate
#             n_fft: FFT size
#             hop_length: Hop length in samples
#             win_length: Window length in samples
#             n_mels: Number of mel filterbanks
#             n_mfcc: Number of MFCC coefficients
#             fmin: Minimum frequency
#             fmax: Maximum frequency
#             feature_type: Type of features to extract
#             normalize: Whether to normalize features
#             deltas: Whether to compute delta and delta-delta
#         """
#         super().__init__()
        
#         self.sample_rate = sample_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.n_mels = n_mels
#         self.n_mfcc = n_mfcc
#         self.feature_type = feature_type
#         self.normalize = normalize
#         self.deltas = deltas
        
#         # Mel spectrogram transform
#         self.mel_transform = T.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             win_length=win_length,
#             n_mels=n_mels,
#             f_min=fmin,
#             f_max=fmax,
#             power=2.0,
#         )
        
#         # MFCC transform
#         self.mfcc_transform = T.MFCC(
#             sample_rate=sample_rate,
#             n_mfcc=n_mfcc,
#             melkwargs={
#                 'n_fft': n_fft,
#                 'hop_length': hop_length,
#                 'win_length': win_length,
#                 'n_mels': n_mels,
#                 'f_min': fmin,
#                 'f_max': fmax,
#             }
#         )
        
#         # Spectrogram transform
#         self.spec_transform = T.Spectrogram(
#             n_fft=n_fft,
#             hop_length=hop_length,
#             win_length=win_length,
#             power=2.0,
#         )
        
#         # Delta computation
#         self.delta_transform = T.ComputeDeltas()
        
#         # Amplitude to dB
#         self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Extract features from waveform.
        
#         Args:
#             waveform: Audio waveform [B, T] or [B, 1, T]
            
#         Returns:
#             Features [B, n_features, T']
#         """
#         # Ensure correct shape
#         if waveform.dim() == 3:
#             waveform = waveform.squeeze(1)
        
#         # Extract features based on type
#         if self.feature_type == "mel":
#             features = self._extract_mel(waveform)
#         elif self.feature_type == "mfcc":
#             features = self._extract_mfcc(waveform)
#         elif self.feature_type == "spectrogram":
#             features = self._extract_spectrogram(waveform)
#         else:
#             raise ValueError(f"Unknown feature type: {self.feature_type}")
        
#         # Add deltas
#         if self.deltas:
#             delta = self.delta_transform(features)
#             delta2 = self.delta_transform(delta)
#             features = torch.cat([features, delta, delta2], dim=1)
        
#         # Normalize
#         if self.normalize:
#             features = self._normalize(features)
        
#         return features
    
#     def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
#         """Extract mel-filterbank features."""
#         mel_spec = self.mel_transform(waveform)
#         mel_spec = self.amplitude_to_db(mel_spec)
#         return mel_spec
    
#     def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
#         """Extract MFCC features."""
#         mfcc = self.mfcc_transform(waveform)
#         return mfcc
    
#     def _extract_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
#         """Extract spectrogram features."""
#         spec = self.spec_transform(waveform)
#         spec = self.amplitude_to_db(spec)
#         return spec
    
#     def _normalize(self, features: torch.Tensor) -> torch.Tensor:
#         """Apply cepstral mean and variance normalization (CMVN)."""
#         mean = features.mean(dim=-1, keepdim=True)
#         std = features.std(dim=-1, keepdim=True)
#         features = (features - mean) / (std + 1e-8)
#         return features
    
#     @property
#     def output_dim(self) -> int:
#         """Get output feature dimension."""
#         if self.feature_type == "mel":
#             base_dim = self.n_mels
#         elif self.feature_type == "mfcc":
#             base_dim = self.n_mfcc
#         else:
#             base_dim = self.n_fft // 2 + 1
        
#         if self.deltas:
#             return base_dim * 3
#         return base_dim


# class SincNetFrontend(nn.Module):
#     """
#     SincNet-based audio frontend for speaker diarization.
#     Learns filterbank directly from raw waveform.
#     Based on: https://arxiv.org/abs/1808.00158
#     """
    
#     def __init__(
#         self,
#         sample_rate: int = 16000,
#         in_channels: int = 1,
#         out_channels: List[int] = [80, 60, 60],
#         kernel_sizes: List[int] = [251, 5, 5],
#         strides: List[int] = [1, 1, 1],
#         padding: str = "same",
#         pool_sizes: List[int] = [3, 3, 3],
#         dropout: float = 0.0,
#     ):
#         """
#         Initialize SincNet frontend.
        
#         Args:
#             sample_rate: Audio sample rate
#             in_channels: Input channels (1 for mono)
#             out_channels: Output channels for each layer
#             kernel_sizes: Kernel sizes for each layer
#             strides: Strides for each layer
#             padding: Padding type
#             pool_sizes: Max pooling sizes
#             dropout: Dropout rate
#         """
#         super().__init__()
        
#         self.sample_rate = sample_rate
        
#         # First layer is SincConv
#         self.sinc_conv = SincConv1d(
#             in_channels=in_channels,
#             out_channels=out_channels[0],
#             kernel_size=kernel_sizes[0],
#             sample_rate=sample_rate,
#         )
        
#         # Build remaining layers
#         layers = []
#         in_ch = out_channels[0]
        
#         for i, (out_ch, k_size, stride, pool_size) in enumerate(
#             zip(out_channels[1:], kernel_sizes[1:], strides[1:], pool_sizes[1:])
#         ):
#             layers.extend([
#                 nn.Conv1d(in_ch, out_ch, k_size, stride=stride, padding=k_size // 2),
#                 nn.BatchNorm1d(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.MaxPool1d(pool_size),
#             ])
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#             in_ch = out_ch
        
#         self.conv_layers = nn.Sequential(*layers)
#         self.out_channels = out_channels[-1]
        
#         # First layer pooling
#         self.first_pool = nn.MaxPool1d(pool_sizes[0])
#         self.first_bn = nn.BatchNorm1d(out_channels[0])
#         self.first_act = nn.LeakyReLU(0.2)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             x: Raw waveform [B, 1, T]
            
#         Returns:
#             Features [B, C, T']
#         """
#         # SincConv layer
#         x = self.sinc_conv(x)
#         x = torch.abs(x)  # Take absolute value
#         x = self.first_pool(x)
#         x = self.first_bn(x)
#         x = self.first_act(x)
        
#         # Remaining conv layers
#         x = self.conv_layers(x)
        
#         return x


# class SincConv1d(nn.Module):
#     """
#     Sinc-based convolution layer.
#     Learns filterbank parameters from raw audio.
#     """
    
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         sample_rate: int = 16000,
#         min_low_hz: float = 50,
#         min_band_hz: float = 50,
#     ):
#         super().__init__()
        
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.sample_rate = sample_rate
#         self.min_low_hz = min_low_hz
#         self.min_band_hz = min_band_hz
        
#         # Initialize filterbanks
#         low_hz = 30
#         high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
#         mel = self._hz_to_mel(torch.tensor([low_hz, high_hz]))
#         mel_points = torch.linspace(mel[0], mel[1], out_channels + 1)
#         hz_points = self._mel_to_hz(mel_points)
        
#         # Filter parameters
#         self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
#         self.band_hz_ = nn.Parameter(
#             (hz_points[1:] - hz_points[:-1]).unsqueeze(1)
#         )
        
#         # Hamming window
#         n_lin = torch.linspace(0, kernel_size / 2 - 1, kernel_size // 2)
#         self.register_buffer(
#             'window_',
#             0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size)
#         )
        
#         # Time points
#         n = (kernel_size - 1) / 2
#         self.register_buffer(
#             'n_',
#             (2 * np.pi * torch.arange(-n, 0) / sample_rate).unsqueeze(0)
#         )
    
#     def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
#         return 2595 * torch.log10(1 + hz / 700)
    
#     def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
#         return 700 * (10 ** (mel / 2595) - 1)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Apply sinc convolution.
        
#         Args:
#             x: Input waveform [B, 1, T]
            
#         Returns:
#             Filtered output [B, out_channels, T']
#         """
#         low = self.min_low_hz + torch.abs(self.low_hz_)
#         high = torch.clamp(
#             low + self.min_band_hz + torch.abs(self.band_hz_),
#             self.min_low_hz,
#             self.sample_rate / 2
#         )
#         band = (high - low)[:, 0]
        
#         # Compute filters
#         f_times_t_low = torch.matmul(low, self.n_)
#         f_times_t_high = torch.matmul(high, self.n_)
        
#         band_pass_left = (
#             (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) /
#             (self.n_ / 2)
#         ) * self.window_
        
#         band_pass_center = 2 * band.unsqueeze(1)
#         band_pass_right = torch.flip(band_pass_left, dims=[1])
        
#         band_pass = torch.cat(
#             [band_pass_left, band_pass_center, band_pass_right],
#             dim=1
#         )
#         band_pass = band_pass / (2 * band.unsqueeze(1))
        
#         # Apply convolution
#         filters = band_pass.unsqueeze(1)
        
#         return nn.functional.conv1d(
#             x, filters,
#             stride=1,
#             padding=self.kernel_size // 2,
#             groups=1,
#         )


# from typing import List  # Add this import at the top


# def extract_features_from_file(
#     audio_path: str,
#     sample_rate: int = 16000,
#     feature_type: str = "mel",
#     n_mels: int = 80,
#     n_mfcc: int = 40,
#     **kwargs,
# ) -> torch.Tensor:
#     """
#     Convenience function to extract features from audio file.
    
#     Args:
#         audio_path: Path to audio file
#         sample_rate: Target sample rate
#         feature_type: Type of features
#         n_mels: Number of mel bands
#         n_mfcc: Number of MFCC coefficients
        
#     Returns:
#         Features tensor
#     """
#     import torchaudio
    
#     # Load audio
#     waveform, sr = torchaudio.load(audio_path)
    
#     # Resample if necessary
#     if sr != sample_rate:
#         waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
#     # Convert to mono
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)
    
#     # Extract features
#     extractor = FeatureExtractor(
#         sample_rate=sample_rate,
#         feature_type=feature_type,
#         n_mels=n_mels,
#         n_mfcc=n_mfcc,
#         **kwargs,
#     )
    
#     with torch.no_grad():
#         features = extractor(waveform.unsqueeze(0))
    
#     return features.squeeze(0)


"""
Audio preprocessing and feature extraction for speaker diarization.
Extracts Mel-filterbank, MFCC, and other acoustic features.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import math


class FeatureExtractor(nn.Module):
    """
    Extract acoustic features from audio waveforms.
    Supports Mel-filterbank, MFCC, and spectrogram features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        n_mfcc: int = 40,
        fmin: int = 20,
        fmax: int = 7600,
        feature_type: str = "mel",  # mel, mfcc, spectrogram
        normalize: bool = True,
        deltas: bool = False,
    ):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length in samples
            win_length: Window length in samples
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            fmin: Minimum frequency
            fmax: Maximum frequency
            feature_type: Type of features to extract
            normalize: Whether to normalize features
            deltas: Whether to compute delta and delta-delta
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.feature_type = feature_type
        self.normalize = normalize
        self.deltas = deltas
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'win_length': win_length,
                'n_mels': n_mels,
                'f_min': fmin,
                'f_max': fmax,
            }
        )
        
        # Spectrogram transform
        self.spec_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0,
        )
        
        # Delta computation
        self.delta_transform = T.ComputeDeltas()
        
        # Amplitude to dB
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from waveform.
        
        Args:
            waveform: Audio waveform [B, T] or [B, 1, T]
            
        Returns:
            Features [B, n_features, T']
        """
        # Ensure correct shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        
        # Extract features based on type
        if self.feature_type == "mel":
            features = self._extract_mel(waveform)
        elif self.feature_type == "mfcc":
            features = self._extract_mfcc(waveform)
        elif self.feature_type == "spectrogram":
            features = self._extract_spectrogram(waveform)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        # Add deltas
        if self.deltas:
            delta = self.delta_transform(features)
            delta2 = self.delta_transform(delta)
            features = torch.cat([features, delta, delta2], dim=1)
        
        # Normalize
        if self.normalize:
            features = self._normalize(features)
        
        return features
    
    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-filterbank features."""
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec
    
    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features."""
        mfcc = self.mfcc_transform(waveform)
        return mfcc
    
    def _extract_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract spectrogram features."""
        spec = self.spec_transform(waveform)
        spec = self.amplitude_to_db(spec)
        return spec
    
    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Apply cepstral mean and variance normalization (CMVN)."""
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True)
        features = (features - mean) / (std + 1e-8)
        return features
    
    @property
    def output_dim(self) -> int:
        """Get output feature dimension."""
        if self.feature_type == "mel":
            base_dim = self.n_mels
        elif self.feature_type == "mfcc":
            base_dim = self.n_mfcc
        else:
            base_dim = self.n_fft // 2 + 1
        
        if self.deltas:
            return base_dim * 3
        return base_dim


class SincNetFrontend(nn.Module):
    """
    SincNet-based audio frontend for speaker diarization.
    Learns filterbank directly from raw waveform.
    Based on: https://arxiv.org/abs/1808.00158
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        in_channels: int = 1,
        out_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        padding: str = "same",
        pool_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize SincNet frontend.
        
        Args:
            sample_rate: Audio sample rate
            in_channels: Input channels (1 for mono)
            out_channels: Output channels for each layer
            kernel_sizes: Kernel sizes for each layer
            strides: Strides for each layer
            padding: Padding type
            pool_sizes: Max pooling sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        # Default values
        if out_channels is None:
            out_channels = [80, 60, 60]
        if kernel_sizes is None:
            kernel_sizes = [251, 5, 5]
        if strides is None:
            strides = [1, 1, 1]
        if pool_sizes is None:
            pool_sizes = [3, 3, 3]
        
        self.sample_rate = sample_rate
        
        # First layer is SincConv
        self.sinc_conv = SincConv1d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            sample_rate=sample_rate,
        )
        
        # Build remaining layers
        layers = []
        in_ch = out_channels[0]
        
        for i, (out_ch, k_size, stride, pool_size) in enumerate(
            zip(out_channels[1:], kernel_sizes[1:], strides[1:], pool_sizes[1:])
        ):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, k_size, stride=stride, padding=k_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.MaxPool1d(pool_size),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.out_channels = out_channels[-1]
        
        # First layer pooling
        self.first_pool = nn.MaxPool1d(pool_sizes[0])
        self.first_bn = nn.BatchNorm1d(out_channels[0])
        self.first_act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Raw waveform [B, 1, T]
            
        Returns:
            Features [B, C, T']
        """
        # SincConv layer
        x = self.sinc_conv(x)
        x = torch.abs(x)  # Take absolute value
        x = self.first_pool(x)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        # Remaining conv layers
        x = self.conv_layers(x)
        
        return x


class SincConv1d(nn.Module):
    """
    Sinc-based convolution layer.
    Learns filterbank parameters from raw audio.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50,
        min_band_hz: float = 50,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filterbanks
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel = self._hz_to_mel(torch.tensor([low_hz, high_hz]))
        mel_points = torch.linspace(mel[0], mel[1], out_channels + 1)
        hz_points = self._mel_to_hz(mel_points)
        
        # Filter parameters
        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter(
            (hz_points[1:] - hz_points[:-1]).unsqueeze(1)
        )
        
        # Hamming window
        n_lin = torch.linspace(0, kernel_size / 2 - 1, kernel_size // 2)
        self.register_buffer(
            'window_',
            0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)
        )
        
        # Time points
        n = (kernel_size - 1) / 2
        self.register_buffer(
            'n_',
            (2 * math.pi * torch.arange(-n, 0) / sample_rate).unsqueeze(0)
        )
    
    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        return 2595 * torch.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinc convolution.
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            Filtered output [B, out_channels, T']
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )
        band = (high - low)[:, 0]
        
        # Compute filters
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) /
            (self.n_ / 2)
        ) * self.window_
        
        band_pass_center = 2 * band.unsqueeze(1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right],
            dim=1
        )
        band_pass = band_pass / (2 * band.unsqueeze(1))
        
        # Apply convolution
        filters = band_pass.unsqueeze(1)
        
        return nn.functional.conv1d(
            x, filters,
            stride=1,
            padding=self.kernel_size // 2,
            groups=1,
        )


def extract_features_from_file(
    audio_path: str,
    sample_rate: int = 16000,
    feature_type: str = "mel",
    n_mels: int = 80,
    n_mfcc: int = 40,
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to extract features from audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        feature_type: Type of features
        n_mels: Number of mel bands
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Features tensor
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Extract features
    extractor = FeatureExtractor(
        sample_rate=sample_rate,
        feature_type=feature_type,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        **kwargs,
    )
    
    with torch.no_grad():
        features = extractor(waveform.unsqueeze(0))
    
    return features.squeeze(0)