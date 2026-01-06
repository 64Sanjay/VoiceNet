# # speaker_verification/models/cam_plus_plus.py
# """
# CAM++ Model for Speaker Verification.

# Based on the paper:
# "CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking"
# """

# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Tuple, Optional

# from .frontend import FrontEndConvModule
# from .dtdnn import DTDNNBlock, TransitionLayer, TDNNLayer
# from .pooling import AttentiveStatisticsPooling, TemporalStatisticsPooling


# class CAMPlusPlus(nn.Module):
#     """
#     CAM++ Speaker Embedding Model.
    
#     Architecture from the paper:
#     1. Front-end Convolution Module (FCM) - 2D CNN for time-frequency processing
#     2. Subsampling TDNN layer (1/2 rate)
#     3. D-TDNN backbone with 3 blocks (12, 24, 16 layers)
#     4. Context-Aware Masking in each D-TDNN layer
#     5. Attentive Statistics Pooling
#     6. Speaker embedding layer
#     """
    
#     def __init__(
#         self,
#         n_mels: int = 80,
#         embedding_dim: int = 192,
#         # FCM parameters
#         fcm_channels: int = 32,
#         fcm_num_blocks: int = 4,
#         # D-TDNN parameters
#         dtdnn_blocks: Tuple[int, ...] = (12, 24, 16),
#         growth_rate: int = 32,
#         bn_size: int = 4,
#         init_channels: int = 128,
#         # CAM parameters
#         use_cam: bool = True,
#         cam_reduction: int = 2,
#         segment_length: int = 100,
#         # Subsampling
#         subsample_rate: int = 2
#     ):
#         super().__init__()
        
#         self.n_mels = n_mels
#         self.embedding_dim = embedding_dim
        
#         # Front-end Convolution Module
#         self.fcm = FrontEndConvModule(
#             in_channels=1,
#             channels=fcm_channels,
#             num_blocks=fcm_num_blocks
#         )
        
#         fcm_out_channels = self.fcm.output_channels
        
#         # Initial TDNN with subsampling
#         self.input_tdnn = nn.Sequential(
#             nn.Conv1d(fcm_out_channels, init_channels, 
#                      kernel_size=5, stride=subsample_rate, padding=2, bias=False),
#             nn.BatchNorm1d(init_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # D-TDNN backbone
#         self.dtdnn_blocks = nn.ModuleList()
#         self.transitions = nn.ModuleList()
        
#         in_channels = init_channels
        
#         for i, num_layers in enumerate(dtdnn_blocks):
#             # D-TDNN block
#             block = DTDNNBlock(
#                 in_channels=in_channels,
#                 num_layers=num_layers,
#                 growth_rate=growth_rate,
#                 bn_size=bn_size,
#                 use_cam=use_cam,
#                 cam_reduction=cam_reduction,
#                 segment_length=segment_length
#             )
#             self.dtdnn_blocks.append(block)
            
#             # Transition layer (except for last block)
#             if i < len(dtdnn_blocks) - 1:
#                 out_channels = block.out_channels // 2
#                 transition = TransitionLayer(
#                     block.out_channels, 
#                     out_channels,
#                     subsample=1
#                 )
#                 self.transitions.append(transition)
#                 in_channels = out_channels
#             else:
#                 in_channels = block.out_channels
        
#         # Final batch norm
#         self.final_bn = nn.BatchNorm1d(in_channels)
        
#         # Attentive statistics pooling
#         self.pooling = AttentiveStatisticsPooling(in_channels, hidden_dim=128)
#         pooling_out = in_channels * 2  # Mean + std
        
#         # Embedding layer
#         self.embedding = nn.Linear(pooling_out, embedding_dim)
#         self.embedding_bn = nn.BatchNorm1d(embedding_dim)
    
#     def forward(
#         self,
#         x: torch.Tensor,
#         return_embedding: bool = True
#     ) -> torch.Tensor:
#         """
#         Forward pass.
        
#         Args:
#             x: Input Fbank features of shape (batch, n_mels, time)
#             return_embedding: If True, return L2-normalized embedding
            
#         Returns:
#             Speaker embedding of shape (batch, embedding_dim)
#         """
#         # Front-end convolution
#         x = self.fcm(x)
        
#         # Subsampling TDNN
#         x = self.input_tdnn(x)
        
#         # D-TDNN backbone with transitions
#         for i, block in enumerate(self.dtdnn_blocks):
#             x = block(x)
            
#             if i < len(self.transitions):
#                 x = self.transitions[i](x)
        
#         # Final batch norm
#         x = self.final_bn(x)
        
#         # Pooling
#         x = self.pooling(x)
        
#         # Embedding
#         x = self.embedding(x)
#         x = self.embedding_bn(x)
        
#         if return_embedding:
#             x = F.normalize(x, p=2, dim=1)
        
#         return x
    
#     def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
#         """Extract speaker embedding (inference mode)."""
#         self.eval()
#         with torch.no_grad():
#             return self.forward(x, return_embedding=True)


# class CAMPlusPlusClassifier(nn.Module):
#     """
#     CAM++ with classification head for training.
#     Uses AAM-Softmax loss.
#     """
    
#     def __init__(
#         self,
#         num_classes: int,
#         embedding_dim: int = 192,
#         scale: float = 32.0,
#         margin: float = 0.2,
#         **cam_kwargs
#     ):
#         super().__init__()
        
#         self.encoder = CAMPlusPlus(
#             embedding_dim=embedding_dim,
#             **cam_kwargs
#         )
        
#         self.scale = scale
#         self.margin = margin
        
#         # Classification weight
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
#         nn.init.xavier_uniform_(self.weight)
    
#     def forward(
#         self,
#         x: torch.Tensor,
#         labels: Optional[torch.Tensor] = None
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass with optional AAM-Softmax.
        
#         Args:
#             x: Input Fbank features
#             labels: Speaker labels (required for training)
            
#         Returns:
#             Dict with 'embedding' and optionally 'logits'
#         """
#         # Get embedding (not normalized for AAM-Softmax)
#         embedding = self.encoder(x, return_embedding=False)
        
#         # Normalize embedding and weight
#         embedding_norm = F.normalize(embedding, p=2, dim=1)
#         weight_norm = F.normalize(self.weight, p=2, dim=1)
        
#         # Cosine similarity
#         cosine = F.linear(embedding_norm, weight_norm)
        
#         output = {
#             'embedding': embedding_norm,
#             'cosine': cosine
#         }
        
#         if labels is not None:
#             # AAM-Softmax
#             theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            
#             # Add margin to target class
#             one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
#             target_theta = theta + self.margin * one_hot
            
#             # Convert back to cosine
#             logits = torch.cos(target_theta) * self.scale
            
#             output['logits'] = logits
        
#         return output
    
#     def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
#         """Extract embedding for inference."""
#         return self.encoder.extract_embedding(x)


# speaker_verification/models/cam_plus_plus.py
# Update the CAMPlusPlusClassifier class - replace the forward method

# speaker_verification/models/cam_plus_plus.py
"""
CAM++ Model for Speaker Verification.

Based on the paper:
"CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .frontend import FrontEndConvModule
from .dtdnn import DTDNNBlock, TransitionLayer, TDNNLayer
from .pooling import AttentiveStatisticsPooling, TemporalStatisticsPooling


class CAMPlusPlus(nn.Module):
    """
    CAM++ Speaker Embedding Model.
    
    Architecture from the paper:
    1. Front-end Convolution Module (FCM) - 2D CNN for time-frequency processing
    2. Subsampling TDNN layer (1/2 rate)
    3. D-TDNN backbone with 3 blocks (12, 24, 16 layers)
    4. Context-Aware Masking in each D-TDNN layer
    5. Attentive Statistics Pooling
    6. Speaker embedding layer
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 192,
        # FCM parameters
        fcm_channels: int = 32,
        fcm_num_blocks: int = 4,
        # D-TDNN parameters
        dtdnn_blocks: Tuple[int, ...] = (12, 24, 16),
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        # CAM parameters
        use_cam: bool = True,
        cam_reduction: int = 2,
        segment_length: int = 100,
        # Subsampling
        subsample_rate: int = 2
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        
        # Front-end Convolution Module
        self.fcm = FrontEndConvModule(
            in_channels=1,
            channels=fcm_channels,
            num_blocks=fcm_num_blocks
        )
        
        fcm_out_channels = self.fcm.output_channels
        
        # Initial TDNN with subsampling
        self.input_tdnn = nn.Sequential(
            nn.Conv1d(fcm_out_channels, init_channels, 
                     kernel_size=5, stride=subsample_rate, padding=2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # D-TDNN backbone
        self.dtdnn_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        in_channels = init_channels
        
        for i, num_layers in enumerate(dtdnn_blocks):
            # D-TDNN block
            block = DTDNNBlock(
                in_channels=in_channels,
                num_layers=num_layers,
                growth_rate=growth_rate,
                bn_size=bn_size,
                use_cam=use_cam,
                cam_reduction=cam_reduction,
                segment_length=segment_length
            )
            self.dtdnn_blocks.append(block)
            
            # Transition layer (except for last block)
            if i < len(dtdnn_blocks) - 1:
                out_channels = block.out_channels // 2
                transition = TransitionLayer(
                    block.out_channels, 
                    out_channels,
                    subsample=1
                )
                self.transitions.append(transition)
                in_channels = out_channels
            else:
                in_channels = block.out_channels
        
        # Final batch norm
        self.final_bn = nn.BatchNorm1d(in_channels)
        
        # Attentive statistics pooling
        self.pooling = AttentiveStatisticsPooling(in_channels, hidden_dim=128)
        pooling_out = in_channels * 2  # Mean + std
        
        # Embedding layer
        self.embedding = nn.Linear(pooling_out, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input Fbank features of shape (batch, n_mels, time)
            return_embedding: If True, return L2-normalized embedding
            
        Returns:
            Speaker embedding of shape (batch, embedding_dim)
        """
        # Front-end convolution
        x = self.fcm(x)
        
        # Subsampling TDNN
        x = self.input_tdnn(x)
        
        # D-TDNN backbone with transitions
        for i, block in enumerate(self.dtdnn_blocks):
            x = block(x)
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Final batch norm
        x = self.final_bn(x)
        
        # Pooling
        x = self.pooling(x)
        
        # Embedding
        x = self.embedding(x)
        x = self.embedding_bn(x)
        
        if return_embedding:
            x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_embedding=True)


class CAMPlusPlusClassifier(nn.Module):
    """
    CAM++ with classification head for training.
    Uses AAM-Softmax loss with adjusted parameters for stable training.
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 192,
        scale: float = 30.0,
        margin: float = 0.1,
        easy_margin: bool = True,
        **cam_kwargs
    ):
        super().__init__()
        
        self.encoder = CAMPlusPlus(
            embedding_dim=embedding_dim,
            **cam_kwargs
        )
        
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Classification weight
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute cos/sin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with AAM-Softmax."""
        # Get embedding (not normalized for AAM-Softmax)
        embedding = self.encoder(x, return_embedding=False)
        
        # Normalize embedding and weight
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embedding_norm, weight_norm)
        
        output = {
            'embedding': embedding_norm,
            'cosine': cosine
        }
        
        if labels is not None:
            # AAM-Softmax with numerical stability
            sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
            
            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            phi = cosine * self.cos_m - sine * self.sin_m
            
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            # One-hot encoding
            one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
            
            # Apply margin only to target class
            logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
            
            output['logits'] = logits
        
        return output
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for inference."""
        return self.encoder.extract_embedding(x)