# speaker_verification/models/frontend.py
"""
Front-end Convolution Module (FCM) for CAM++.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock2D(nn.Module):
    """
    2D Residual block for FCM.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int] = (1, 1)
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class FrontEndConvModule(nn.Module):
    """
    Front-end Convolution Module (FCM) for CAM++.
    
    From the paper:
    "The FCM consists of multiple blocks of two-dimensional convolution with 
    residual connections, which encode acoustic features in the time-frequency 
    domain to exploit high-resolution time-frequency details."
    
    "We decide to incorporate 4 residual blocks in the FCM stem. 
    The number of channels is set to 32 for all residual blocks.
    We use a stride of 2 in the frequency dimension in the last three blocks, 
    resulting in an 8x downsampling in the frequency domain."
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 32,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        # First block: no frequency downsampling
        # Last 3 blocks: stride 2 in frequency
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                stride = (1, 1)  # (freq, time)
            else:
                stride = (2, 1)  # Downsample frequency only
            
            blocks.append(ResidualBlock2D(channels, channels, stride))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Calculate output frequency dimension after downsampling
        # 80 -> 40 -> 20 -> 10 (8x downsampling with 3 stride-2 blocks)
        self.freq_downsample = 2 ** (num_blocks - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input Fbank features of shape (batch, freq, time)
            
        Returns:
            Output features of shape (batch, channels * freq_out, time)
        """
        # Add channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, freq, time)
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.blocks(x)
        
        # Flatten channel and frequency dimensions
        batch, channels, freq, time = x.shape
        x = x.view(batch, channels * freq, time)
        
        return x
    
    @property
    def output_channels(self) -> int:
        """Output channels after flattening."""
        return self.channels * (80 // self.freq_downsample)