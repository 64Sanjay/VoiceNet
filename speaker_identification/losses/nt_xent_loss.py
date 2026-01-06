# speaker_identification/losses/nt_xent_loss.py
"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
Implements Equation 8 from the paper for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for self-supervised learning.
    
    From Equation 8:
    L_NT = 0.5 * [NT-Xent({z_i}, {z_i^(n)}) + NT-Xent({z_i}, {z_i^(t)})]
    
    This loss enforces consistency between original and augmented views.
    Based on SimCLR (Chen et al., 2020).
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter (τ = 0.5 from Table 3)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.
        
        The loss treats each (z_i, z_j) pair as positive and all other
        samples in the batch as negatives.
        
        Args:
            z_i: Embeddings from first view (B, D)
            z_j: Embeddings from second view (B, D)
            
        Returns:
            NT-Xent loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # L2 normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # Concatenate embeddings: [z_i; z_j] of shape (2B, D)
        embeddings = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix: (2B, 2B)
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive pair mask
        # For i in [0, B-1], positive is at index i + B
        # For i in [B, 2B-1], positive is at index i - B
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)
        
        # Create mask to exclude self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute loss using cross entropy
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward_with_negatives(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss with explicit negatives.
        
        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D)
            negatives: Negative embeddings (B, N, D) or (N, D)
            
        Returns:
            NT-Xent loss value
        """
        batch_size = anchor.shape[0]
        
        # L2 normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=-1)
        
        # Compute positive similarity: (B,)
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # Compute negative similarities
        if negatives.dim() == 2:
            # Shared negatives: (B, N)
            neg_sim = torch.matmul(anchor, negatives.t()) / self.temperature
        else:
            # Per-sample negatives: (B, N)
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),
                negatives.transpose(1, 2)
            ).squeeze(1) / self.temperature
        
        # Combine for softmax: (B, 1 + N)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Target is always index 0 (positive)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MultiViewNTXentLoss(nn.Module):
    """
    NT-Xent Loss for multiple augmented views.
    
    Computes the average NT-Xent loss between the original view
    and each augmented view, as per Equation 8.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize multi-view NT-Xent loss.
        
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.nt_xent = NTXentLoss(temperature=temperature)
    
    def forward(
        self,
        z_original: torch.Tensor,
        z_noise_aug: torch.Tensor,
        z_time_aug: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-view NT-Xent loss.
        
        Implements Equation 8:
        L_NT = 0.5 * [NT-Xent({z_i}, {z_i^(n)}) + NT-Xent({z_i}, {z_i^(t)})]
        
        Args:
            z_original: Original embeddings (B, D)
            z_noise_aug: Noise-augmented embeddings (B, D)
            z_time_aug: Time-stretched embeddings (B, D)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # NT-Xent between original and noise-augmented
        loss_noise = self.nt_xent(z_original, z_noise_aug)
        
        # NT-Xent between original and time-stretched
        loss_time = self.nt_xent(z_original, z_time_aug)
        
        # Average (Equation 8)
        total_loss = 0.5 * (loss_noise + loss_time)
        
        loss_dict = {
            'nt_xent_noise': loss_noise.item(),
            'nt_xent_time': loss_time.item(),
            'nt_xent_total': total_loss.item()
        }
        
        return total_loss, loss_dict