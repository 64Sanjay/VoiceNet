# speaker_verification/models/dtdnn.py
"""
Densely Connected Time Delay Neural Network (D-TDNN) for CAM++.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .pooling import MultiGranularityPooling


class TDNNLayer(nn.Module):
    """
    Basic TDNN layer with 1D convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ContextAwareMasking(nn.Module):
    """
    Context-Aware Masking (CAM) module.
    
    From the paper:
    "A ratio mask M is predicted based on an extracted contextual embedding e,
    and is expected to contain both speaker of interest and noise characteristic."
    
    Equation 6:
    M^k_*t = σ(W2 * δ(W1 * (eg + es^k) + b1) + b2), sk ≤ t < s_{k+1}
    
    Where:
    - σ: Sigmoid function
    - δ: ReLU function
    - eg: global embedding
    - es^k: segment embedding for segment k
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 2,
        segment_length: int = 100
    ):
        super().__init__()
        
        self.channels = channels
        self.segment_length = segment_length
        
        hidden_dim = channels // reduction
        
        # Multi-granularity pooling
        self.pooling = MultiGranularityPooling(segment_length)
        
        # Mask prediction network (Equation 3/6)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply context-aware masking.
        
        Args:
            x: Input features of shape (batch, channels, time)
            
        Returns:
            Masked features of shape (batch, channels, time)
        """
        batch_size, channels, time_steps = x.shape
        
        # Get global and segment embeddings
        global_emb, segment_emb = self.pooling(x)
        
        # Calculate number of segments
        num_segments = segment_emb.shape[2]
        
        # Create mask for each segment
        masks = []
        for k in range(num_segments):
            # Combine global and segment embeddings (Equation 6)
            context = global_emb + segment_emb[:, :, k]  # (batch, channels)
            
            # Predict mask
            mask = self.fc1(context)
            mask = self.relu(mask)
            mask = self.fc2(mask)
            mask = self.sigmoid(mask)  # (batch, channels)
            
            masks.append(mask)
        
        # Stack masks: (batch, channels, num_segments)
        masks = torch.stack(masks, dim=2)
        
        # Expand masks to time dimension
        # Each segment mask covers segment_length frames
        expanded_masks = masks.repeat_interleave(self.segment_length, dim=2)
        
        # Trim or pad to match time dimension
        if expanded_masks.shape[2] > time_steps:
            expanded_masks = expanded_masks[:, :, :time_steps]
        elif expanded_masks.shape[2] < time_steps:
            padding = time_steps - expanded_masks.shape[2]
            # Repeat last mask for remaining frames
            expanded_masks = F.pad(expanded_masks, (0, padding), mode='replicate')
        
        # Apply mask (Equation 7: F̃ = F ⊙ M)
        return x * expanded_masks


class DTDNNLayer(nn.Module):
    """
    D-TDNN layer with bottleneck and dense connection.
    
    From the paper:
    "The basic unit of D-TDNN consists of a feed-forward neural network (FNN) 
    and a TDNN layer."
    """
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int = 32,
        bn_size: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        use_cam: bool = True,
        cam_reduction: int = 2,
        segment_length: int = 100
    ):
        super().__init__()
        
        self.growth_rate = growth_rate
        
        # Bottleneck (FNN)
        hidden_channels = bn_size * growth_rate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        
        # TDNN layer
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.tdnn = nn.Conv1d(
            hidden_channels, growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
            bias=False
        )
        
        # Context-aware masking
        self.cam = ContextAwareMasking(
            growth_rate, cam_reduction, segment_length
        ) if use_cam else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated features from previous layers (batch, channels, time)
            
        Returns:
            Output features (batch, growth_rate, time)
        """
        # Bottleneck
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.fc1(out)
        
        # TDNN
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.tdnn(out)
        
        # Context-aware masking
        if self.cam is not None:
            out = self.cam(out)
        
        return out


class DTDNNBlock(nn.Module):
    """
    D-TDNN block with multiple D-TDNN layers and dense connections.
    
    From the paper (Equation 1):
    "Sl = Hl([S0, S1, ..., S_{l-1}])"
    
    Where Sl is the output of the l-th layer and [S0, S1, ...] denotes concatenation.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int = 32,
        bn_size: int = 4,
        kernel_size: int = 3,
        use_cam: bool = True,
        cam_reduction: int = 2,
        segment_length: int = 100
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Input channels = initial + accumulated growth
            layer_in_channels = in_channels + i * growth_rate
            
            layer = DTDNNLayer(
                in_channels=layer_in_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                kernel_size=kernel_size,
                dilation=1,  # Can vary per layer
                use_cam=use_cam,
                cam_reduction=cam_reduction,
                segment_length=segment_length
            )
            self.layers.append(layer)
        
        # Output channels after all layers
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, channels, time)
            
        Returns:
            Output features with dense connections (batch, out_channels, time)
        """
        features = [x]
        
        for layer in self.layers:
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=1)
            
            # Apply layer
            out = layer(concat_features)
            
            # Add to feature list for next layer
            features.append(out)
        
        # Return concatenation of all features
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between D-TDNN blocks.
    Reduces channels and optionally subsamples in time.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        subsample: int = 1
    ):
        super().__init__()
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        
        self.pool = nn.AvgPool1d(subsample) if subsample > 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        return x