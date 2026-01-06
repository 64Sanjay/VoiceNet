# speaker_identification/losses/joint_loss.py
"""
Joint loss combining triplet loss and NT-Xent loss.
Implements Equation 9 from the paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .triplet_loss import OnlineHardTripletLoss
from .nt_xent_loss import MultiViewNTXentLoss


class WSIJointLoss(nn.Module):
    """
    WSI Joint Loss function.
    
    From Equation 9:
    L = L_triplet + λ * L_NT
    
    Where:
    - L_triplet: Online hard triplet loss (Equation 7)
    - L_NT: NT-Xent self-supervised loss (Equation 8)
    - λ: Self-supervised loss weight (1.0 from Table 3)
    """
    
    def __init__(
        self,
        triplet_margin: float = 1.0,
        nt_xent_temperature: float = 0.5,
        lambda_weight: float = 1.0
    ):
        """
        Initialize joint loss.
        
        Args:
            triplet_margin: Margin for triplet loss (m = 1.0 from paper)
            nt_xent_temperature: Temperature for NT-Xent (τ = 0.5 from paper)
            lambda_weight: Weight for self-supervised loss (λ = 1.0 from paper)
        """
        super().__init__()
        
        self.triplet_loss = OnlineHardTripletLoss(margin=triplet_margin)
        self.nt_xent_loss = MultiViewNTXentLoss(temperature=nt_xent_temperature)
        self.lambda_weight = lambda_weight
    
    def forward(
        self,
        z_original: torch.Tensor,
        z_noise_aug: torch.Tensor,
        z_time_aug: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss.
        
        Implements Equation 9:
        L = L_triplet + λ * L_NT
        
        Args:
            z_original: Original embeddings (B, D)
            z_noise_aug: Noise-augmented embeddings (B, D)
            z_time_aug: Time-stretched embeddings (B, D)
            labels: Speaker labels (B,)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute triplet loss on original embeddings (Equation 7)
        loss_triplet = self.triplet_loss(z_original, labels)
        
        # Compute NT-Xent loss (Equation 8)
        loss_nt_xent, nt_xent_dict = self.nt_xent_loss(
            z_original, z_noise_aug, z_time_aug
        )
        
        # Combine losses (Equation 9)
        total_loss = loss_triplet + self.lambda_weight * loss_nt_xent
        
        # Build loss dictionary
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_triplet': loss_triplet.item(),
            'loss_nt_xent': loss_nt_xent.item(),
            **nt_xent_dict
        }
        
        return total_loss, loss_dict


class WSIJointLossWithStats(WSIJointLoss):
    """
    Joint loss with additional training statistics.
    Useful for monitoring training progress.
    """
    
    def forward(
        self,
        z_original: torch.Tensor,
        z_noise_aug: torch.Tensor,
        z_time_aug: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss with statistics.
        
        Args:
            z_original: Original embeddings (B, D)
            z_noise_aug: Noise-augmented embeddings (B, D)
            z_time_aug: Time-stretched embeddings (B, D)
            labels: Speaker labels (B,)
            
        Returns:
            Tuple of (total_loss, loss_dict with stats)
        """
        total_loss, loss_dict = super().forward(
            z_original, z_noise_aug, z_time_aug, labels
        )
        
        # Add embedding statistics
        with torch.no_grad():
            # Embedding norms
            loss_dict['emb_norm_original'] = z_original.norm(dim=1).mean().item()
            loss_dict['emb_norm_noise'] = z_noise_aug.norm(dim=1).mean().item()
            loss_dict['emb_norm_time'] = z_time_aug.norm(dim=1).mean().item()
            
            # Inter-view similarities
            z_orig_norm = torch.nn.functional.normalize(z_original, p=2, dim=1)
            z_noise_norm = torch.nn.functional.normalize(z_noise_aug, p=2, dim=1)
            z_time_norm = torch.nn.functional.normalize(z_time_aug, p=2, dim=1)
            
            loss_dict['sim_orig_noise'] = (z_orig_norm * z_noise_norm).sum(dim=1).mean().item()
            loss_dict['sim_orig_time'] = (z_orig_norm * z_time_norm).sum(dim=1).mean().item()
        
        return total_loss, loss_dict