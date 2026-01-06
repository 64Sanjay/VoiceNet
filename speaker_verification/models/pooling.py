# speaker_verification/models/pooling.py
"""
Pooling layers for CAM++ including multi-granularity pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalStatisticsPooling(nn.Module):
    """
    Temporal statistics pooling (mean + std).
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Pooled tensor of shape (batch, channels * 2)
        """
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling.
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, in_channels, 1),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Pooled tensor of shape (batch, channels * 2)
        """
        # Attention weights
        alpha = self.attention(x)
        
        # Weighted mean
        mean = (x * alpha).sum(dim=2)
        
        # Weighted std
        var = ((x - mean.unsqueeze(2)) ** 2 * alpha).sum(dim=2)
        std = torch.sqrt(var.clamp(min=1e-9))
        
        return torch.cat([mean, std], dim=1)


class MultiGranularityPooling(nn.Module):
    """
    Multi-granularity pooling for Context-Aware Masking.
    
    From the paper:
    "A global average pooling is used to extract contextual information at global level.
    Simultaneously, a segment average pooling is used to extract contextual information 
    at segment level."
    
    "We segment the frame-level feature X into consecutive fixed-length 100-frame 
    segments and apply segment average pooling to each."
    """
    
    def __init__(self, segment_length: int = 100):
        super().__init__()
        self.segment_length = segment_length
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Tuple of (global_embedding, segment_embeddings)
            - global_embedding: (batch, channels)
            - segment_embeddings: (batch, channels, num_segments)
        """
        batch_size, channels, time_steps = x.shape
        
        # Global average pooling (Equation 4 in paper)
        # eg = (1/T) * Σ X*t
        global_emb = x.mean(dim=2)  # (batch, channels)
        
        # Segment average pooling (Equation 5 in paper)
        # es^k = (1/(s_{k+1} - s_k)) * Σ X*t for sk ≤ t < s_{k+1}
        num_segments = max(1, time_steps // self.segment_length)
        
        # Pad if necessary
        padded_time = num_segments * self.segment_length
        if time_steps < padded_time:
            padding = padded_time - time_steps
            x_padded = F.pad(x, (0, padding))
        else:
            x_padded = x[:, :, :padded_time]
        
        # Reshape and compute segment means
        x_segments = x_padded.view(batch_size, channels, num_segments, self.segment_length)
        segment_emb = x_segments.mean(dim=3)  # (batch, channels, num_segments)
        
        return global_emb, segment_emb
    
    def get_context_embedding(
        self, 
        x: torch.Tensor, 
        time_idx: int
    ) -> torch.Tensor:
        """
        Get context embedding for a specific time index.
        
        Combines global and segment embeddings as per Equation 6:
        e = eg + es^k, where sk ≤ t < s_{k+1}
        """
        global_emb, segment_emb = self.forward(x)
        
        # Determine which segment this time index belongs to
        segment_idx = min(time_idx // self.segment_length, segment_emb.shape[2] - 1)
        
        # Combine global and segment embeddings
        context_emb = global_emb + segment_emb[:, :, segment_idx]
        
        return context_emb