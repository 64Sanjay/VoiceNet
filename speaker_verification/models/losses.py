# speaker_verification/models/losses.py
"""
Loss functions for CAM++ speaker verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AAMSoftmaxLoss(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax) Loss.
    
    From the paper:
    "Angular additive margin softmax (AAM-Softmax) loss is used for all experiments.
    The margin and scaling factors of AAM-Softmax loss are set to 0.2 and 32 respectively."
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.2
    ):
        super().__init__()
        
        self.scale = scale
        self.margin = margin
        
        # Classification weight
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AAM-Softmax loss.
        
        Args:
            embeddings: Speaker embeddings (batch, embedding_dim)
            labels: Speaker labels (batch,)
            
        Returns:
            Tuple of (loss, logits)
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)
        
        # Convert to angle
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add angular margin to target class
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        target_theta = theta + self.margin * one_hot
        
        # Convert back to cosine and scale
        logits = torch.cos(target_theta) * self.scale
        
        # Cross entropy loss
        loss = self.ce_loss(logits, labels)
        
        return loss, logits


class SoftmaxLoss(nn.Module):
    """Simple softmax loss for comparison."""
    
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(embeddings)
        loss = self.ce_loss(logits, labels)
        return loss, logits