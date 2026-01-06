"""
Speaker Encoder models for speaker diarization.
Includes ECAPA-TDNN and X-vector architectures.

Based on:
- ECAPA-TDNN: https://arxiv.org/abs/2005.07143
- X-vectors: https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            Scaled x: [B, C, T]
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Res2Block(nn.Module):
    """
    Res2Net block with SE attention.
    Multi-scale feature extraction.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        
        self.scale = scale
        width = channels // scale
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    width, width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation * (kernel_size - 1) // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(width))
        
        self.se = SEBlock(channels, channels // se_channels)
        self.width = width
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C, T]
        """
        spx = torch.split(x, self.width, dim=1)
        outputs = []
        
        for i in range(self.scale):
            if i == 0:
                sp = spx[i]
            elif i == 1:
                sp = self.convs[i - 1](spx[i])
                sp = F.relu(self.bns[i - 1](sp))
            else:
                sp = sp + spx[i]
                sp = self.convs[i - 1](sp)
                sp = F.relu(self.bns[i - 1](sp))
            outputs.append(sp)
        
        out = torch.cat(outputs, dim=1)
        out = self.se(out)
        
        return out


class SERes2Block(nn.Module):
    """
    SE-Res2Block for ECAPA-TDNN.
    Combines residual connection with Res2Net and SE attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.res2block = Res2Block(
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            scale=scale,
            se_channels=se_channels,
        )
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C', T]
        """
        residual = self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res2block(out)
        out = self.bn2(self.conv2(out))
        
        out = F.relu(out + residual)
        
        return out


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP).
    Computes attention-weighted mean and std.
    """
    
    def __init__(
        self,
        in_channels: int,
        attention_channels: int = 128,
    ):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, 1),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, 2*C]
        """
        # Compute attention weights
        w = self.attention(x)
        
        # Weighted mean
        mean = torch.sum(x * w, dim=-1)
        
        # Weighted std
        var = torch.sum((x ** 2) * w, dim=-1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-9))
        
        # Concatenate
        out = torch.cat([mean, std], dim=-1)
        
        return out


class TemporalAveragePooling(nn.Module):
    """Simple temporal average pooling."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, C]
        """
        return x.mean(dim=-1)


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN Speaker Encoder.
    
    Emphasized Channel Attention, Propagation and Aggregation in TDNN.
    State-of-the-art speaker verification model.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        channels: List[int] = [1024, 1024, 1024, 1024, 3072],
        kernel_sizes: List[int] = [5, 3, 3, 3, 1],
        dilations: List[int] = [1, 2, 3, 4, 1],
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        embedding_dim: int = 192,
        pooling_type: str = "asp",  # asp, tap
    ):
        """
        Initialize ECAPA-TDNN.
        
        Args:
            input_dim: Input feature dimension (e.g., 80 for mel)
            channels: Channel sizes for each layer
            kernel_sizes: Kernel sizes for each layer
            dilations: Dilation factors for each layer
            attention_channels: Channels for attention
            res2net_scale: Scale factor for Res2Net
            se_channels: SE reduction factor
            embedding_dim: Output embedding dimension
            pooling_type: Pooling type (asp or tap)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Input layer
        self.conv1 = nn.Conv1d(input_dim, channels[0], kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        
        # SE-Res2Blocks
        self.blocks = nn.ModuleList()
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2Block(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    scale=res2net_scale,
                    se_channels=se_channels,
                )
            )
        
        # Multi-layer feature aggregation (MFA)
        in_channels_mfa = sum(channels[:-1])
        self.mfa_conv = nn.Conv1d(in_channels_mfa, channels[-1], kernel_sizes[-1])
        self.mfa_bn = nn.BatchNorm1d(channels[-1])
        
        # Pooling
        if pooling_type == "asp":
            self.pooling = AttentiveStatisticsPooling(channels[-1], attention_channels)
            pooling_out_dim = channels[-1] * 2
        else:
            self.pooling = TemporalAveragePooling()
            pooling_out_dim = channels[-1]
        
        # Final embedding layer
        self.bn2 = nn.BatchNorm1d(pooling_out_dim)
        self.fc = nn.Linear(pooling_out_dim, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Extract speaker embeddings.
        
        Args:
            x: Input features [B, C, T] or [B, T, C]
            return_intermediate: Whether to return intermediate features
            
        Returns:
            embeddings: [B, embedding_dim]
            or (embeddings, intermediate_features) if return_intermediate=True
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if x.shape[1] == self.input_dim:
            pass  # Already [B, C, T]
        elif x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # Input layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # SE-Res2Blocks with skip connections
        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        # Multi-layer feature aggregation
        x = torch.cat(features, dim=1)
        x = F.relu(self.mfa_bn(self.mfa_conv(x)))
        
        intermediate = x if return_intermediate else None
        
        # Pooling
        x = self.pooling(x)
        
        # Embedding
        x = self.bn2(x)
        x = self.fc(x)
        x = self.bn3(x)
        
        if return_intermediate:
            return x, intermediate
        return x


class XVector(nn.Module):
    """
    X-vector Speaker Encoder.
    
    Original architecture from Kaldi/SpeechBrain.
    Time-delay neural network with statistics pooling.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        frame_layers: List[int] = [512, 512, 512, 512, 1500],
        segment_layers: List[int] = [512, 512],
        embedding_dim: int = 512,
    ):
        """
        Initialize X-vector encoder.
        
        Args:
            input_dim: Input feature dimension
            frame_layers: Frame-level layer sizes
            segment_layers: Segment-level layer sizes
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Frame-level TDNN layers
        self.frame_layers = nn.ModuleList()
        
        in_dim = input_dim
        contexts = [[-2, -1, 0, 1, 2], [-2, 0, 2], [-3, 0, 3], [0], [0]]
        
        for i, (out_dim, context) in enumerate(zip(frame_layers, contexts)):
            context_size = len(context) if i < 3 else 1
            layer_in_dim = in_dim * context_size if i < 3 else in_dim
            
            self.frame_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim if i == 0 else frame_layers[i - 1],
                        out_dim,
                        kernel_size=len(context) if i < 3 else 1,
                        dilation=max(1, abs(context[0])) if i < 3 else 1,
                        padding=len(context) // 2 if i < 3 else 0,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                )
            )
            in_dim = out_dim
        
        # Statistics pooling
        self.pooling = StatisticsPooling()
        
        # Segment-level layers
        self.segment_layers = nn.ModuleList()
        in_dim = frame_layers[-1] * 2  # Mean + Std
        
        for out_dim in segment_layers:
            self.segment_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                )
            )
            in_dim = out_dim
        
        # Final embedding
        self.embedding = nn.Linear(segment_layers[-1], embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract x-vector embeddings.
        
        Args:
            x: Input features [B, C, T] or [B, T, C]
            
        Returns:
            embeddings: [B, embedding_dim]
        """
        if x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)
        
        # Frame-level layers
        for layer in self.frame_layers:
            x = layer(x)
        
        # Statistics pooling
        x = self.pooling(x)
        
        # Segment-level layers
        for layer in self.segment_layers:
            x = layer(x)
        
        # Embedding
        x = self.embedding(x)
        
        return x


class StatisticsPooling(nn.Module):
    """Statistics pooling: concatenate mean and std."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            out: [B, 2*C]
        """
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        return torch.cat([mean, std], dim=-1)


class SpeakerEncoder(nn.Module):
    """
    Unified speaker encoder interface.
    Wraps different encoder architectures.
    """
    
    def __init__(
        self,
        encoder_type: str = "ecapa_tdnn",
        input_dim: int = 80,
        embedding_dim: int = 192,
        **kwargs,
    ):
        """
        Initialize speaker encoder.
        
        Args:
            encoder_type: Type of encoder (ecapa_tdnn, xvector)
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            **kwargs: Additional encoder-specific arguments
        """
        super().__init__()
        
        if encoder_type == "ecapa_tdnn":
            self.encoder = ECAPA_TDNN(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                **kwargs,
            )
        elif encoder_type == "xvector":
            self.encoder = XVector(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings."""
        return self.encoder(x)
    
    def extract_embeddings(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract normalized speaker embeddings.
        
        Args:
            x: Input features
            normalize: L2 normalize embeddings
            
        Returns:
            embeddings: Normalized embeddings
        """
        embeddings = self.forward(x)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings