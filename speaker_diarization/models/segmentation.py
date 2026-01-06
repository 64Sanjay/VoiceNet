"""
Speaker Segmentation models for speaker diarization.
Provides frame-level speaker activity predictions.

Based on:
- PyanNet: https://arxiv.org/abs/2104.04045
- EEND: https://arxiv.org/abs/1909.06247
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .speaker_encoder import ECAPA_TDNN, AttentiveStatisticsPooling


class SincConvBlock(nn.Module):
    """
    SincNet-style convolutional frontend.
    Learns filterbanks directly from raw waveform.
    """
    
    def __init__(
        self,
        out_channels: int = 80,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        min_low_hz: float = 50,
        min_band_hz: float = 50,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Initialize filterbank parameters
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel_points = torch.linspace(
            self._hz_to_mel(low_hz),
            self._hz_to_mel(high_hz),
            out_channels + 1,
        )
        hz_points = self._mel_to_hz(mel_points)
        
        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))
        
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Hamming window
        n = (kernel_size - 1) / 2
        self.register_buffer(
            'window_',
            torch.hamming_window(kernel_size // 2)
        )
        self.register_buffer(
            'n_',
            (2 * math.pi * torch.arange(-n, 0) / sample_rate).unsqueeze(0)
        )
    
    def _hz_to_mel(self, hz):
        return 2595 * torch.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform [B, 1, T]
        Returns:
            Filtered output [B, out_channels, T']
        """
        # Compute filterbank
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2,
        )
        band = (high - low)[:, 0]
        
        f_low = torch.matmul(low, self.n_)
        f_high = torch.matmul(high, self.n_)
        
        band_pass_left = (
            (torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)
        ) * self.window_
        
        band_pass_center = 2 * band.unsqueeze(1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right],
            dim=1
        )
        band_pass = band_pass / (2 * band.unsqueeze(1) + 1e-8)
        
        filters = band_pass.unsqueeze(1)
        
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class PyanNet(nn.Module):
    """
    PyanNet: End-to-end speaker segmentation model.
    
    Architecture:
    1. SincNet or Mel-spectrogram frontend
    2. LSTM encoder
    3. Feed-forward classifier
    
    Outputs frame-level multi-speaker activity.
    """
    
    def __init__(
        self,
        # Frontend
        sample_rate: int = 16000,
        use_sincnet: bool = True,
        sincnet_channels: int = 80,
        sincnet_kernel_size: int = 251,
        
        # Encoder
        num_lstm_layers: int = 4,
        lstm_hidden_size: int = 128,
        lstm_bidirectional: bool = True,
        
        # Output
        num_speakers: int = 4,
        linear_hidden_size: int = 128,
        
        # Input (if not using SincNet)
        input_dim: int = 80,
    ):
        """
        Initialize PyanNet.
        
        Args:
            sample_rate: Audio sample rate
            use_sincnet: Use SincNet frontend
            sincnet_channels: SincNet output channels
            sincnet_kernel_size: SincNet kernel size
            num_lstm_layers: Number of LSTM layers
            lstm_hidden_size: LSTM hidden size
            lstm_bidirectional: Use bidirectional LSTM
            num_speakers: Maximum number of speakers
            linear_hidden_size: Hidden size for classifier
            input_dim: Input dimension (if not using SincNet)
        """
        super().__init__()
        
        self.use_sincnet = use_sincnet
        self.num_speakers = num_speakers
        
        # Frontend
        if use_sincnet:
            self.sincnet = SincConvBlock(
                out_channels=sincnet_channels,
                kernel_size=sincnet_kernel_size,
                sample_rate=sample_rate,
            )
            encoder_input_dim = sincnet_channels
        else:
            self.sincnet = None
            encoder_input_dim = input_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=0.5 if num_lstm_layers > 1 else 0,
        )
        
        lstm_output_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, linear_hidden_size),
            nn.ReLU(),
            nn.Linear(linear_hidden_size, num_speakers),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
               - If use_sincnet: [B, 1, T] raw waveform
               - Otherwise: [B, C, T] or [B, T, C] features
            return_embeddings: Return LSTM embeddings
            
        Returns:
            outputs: [B, T', num_speakers] frame-level speaker probabilities
        """
        # Frontend
        if self.use_sincnet:
            x = self.sincnet(x)  # [B, C, T]
            x = torch.abs(x)
            x = x.transpose(1, 2)  # [B, T, C]
        else:
            if x.shape[-1] != self.lstm.input_size:
                x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        # LSTM
        x, _ = self.lstm(x)  # [B, T, H]
        
        embeddings = x
        
        # Classifier
        outputs = self.classifier(x)  # [B, T, num_speakers]
        
        if return_embeddings:
            return outputs, embeddings
        return outputs


class EEND(nn.Module):
    """
    End-to-End Neural Diarization (EEND).
    
    Self-attention based model for speaker diarization.
    Can handle overlapping speech.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        num_speakers: int = 4,
        encoder_dim: int = 256,
        num_encoder_layers: int = 4,
        num_attention_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize EEND.
        
        Args:
            input_dim: Input feature dimension
            num_speakers: Maximum number of speakers
            encoder_dim: Transformer encoder dimension
            num_encoder_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            feedforward_dim: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_speakers = num_speakers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(encoder_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Output projection
        self.output_proj = nn.Linear(encoder_dim, num_speakers)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, T, C] or [B, C, T]
            src_mask: Optional attention mask
            
        Returns:
            outputs: [B, T, num_speakers] speaker probabilities
        """
        if x.shape[-1] == self.input_dim:
            pass  # [B, T, C]
        else:
            x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # Output projection
        outputs = torch.sigmoid(self.output_proj(x))
        
        return outputs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EENDWithEDA(nn.Module):
    """
    EEND with Encoder-Decoder based Attractor calculation (EEND-EDA).
    
    Uses LSTM decoder to generate speaker attractors,
    which are then used to predict speaker activities.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        max_speakers: int = 8,
        encoder_dim: int = 256,
        num_encoder_layers: int = 4,
        num_attention_heads: int = 4,
        attractor_dim: int = 256,
        attractor_decoder_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize EEND-EDA.
        
        Args:
            input_dim: Input feature dimension
            max_speakers: Maximum number of speakers
            encoder_dim: Encoder dimension
            num_encoder_layers: Number of encoder layers
            num_attention_heads: Number of attention heads
            attractor_dim: Attractor dimension
            attractor_decoder_layers: Attractor decoder layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.max_speakers = max_speakers
        self.input_dim = input_dim
        
        # Encoder (same as EEND)
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.pos_encoder = PositionalEncoding(encoder_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_attention_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Attractor decoder (LSTM)
        self.attractor_decoder = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=attractor_dim,
            num_layers=attractor_decoder_layers,
            batch_first=True,
        )
        
        # Attractor existence detector
        self.attractor_exist = nn.Linear(attractor_dim, 1)
        
        self.encoder_dim = encoder_dim
        self.attractor_dim = attractor_dim
    
    def forward(
        self,
        x: torch.Tensor,
        num_speakers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, T, C] or [B, C, T]
            num_speakers: Number of speakers (if known)
            
        Returns:
            outputs: [B, T, num_speakers] speaker probabilities
            attractor_probs: [B, num_speakers] attractor existence probabilities
        """
        if x.shape[-1] == self.input_dim:
            pass
        else:
            x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        
        # Encode
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)  # [B, T, D]
        
        # Global representation for attractor initialization
        global_repr = encoded.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Generate attractors
        attractors = []
        attractor_probs = []
        
        decoder_input = global_repr
        hidden = None
        
        for _ in range(self.max_speakers):
            output, hidden = self.attractor_decoder(decoder_input, hidden)
            attractor = output.squeeze(1)  # [B, D]
            attractors.append(attractor)
            
            # Existence probability
            exist_prob = torch.sigmoid(self.attractor_exist(attractor))
            attractor_probs.append(exist_prob)
            
            decoder_input = output
        
        attractors = torch.stack(attractors, dim=1)  # [B, max_speakers, D]
        attractor_probs = torch.cat(attractor_probs, dim=1)  # [B, max_speakers]
        
        # Compute speaker activities using dot product
        # [B, T, D] x [B, D, max_speakers] -> [B, T, max_speakers]
        outputs = torch.sigmoid(
            torch.bmm(encoded, attractors.transpose(1, 2))
        )
        
        return outputs, attractor_probs


class SegmentationModel(nn.Module):
    """
    Unified segmentation model interface.
    """
    
    def __init__(
        self,
        model_type: str = "pyannet",  # pyannet, eend, eend_eda
        **kwargs,
    ):
        super().__init__()
        
        if model_type == "pyannet":
            self.model = PyanNet(**kwargs)
        elif model_type == "eend":
            self.model = EEND(**kwargs)
        elif model_type == "eend_eda":
            self.model = EENDWithEDA(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)