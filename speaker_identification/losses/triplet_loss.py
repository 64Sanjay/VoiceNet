# speaker_identification/losses/triplet_loss.py
"""
Online Hard Triplet Loss implementation.
Implements Equation 7 and Section 3.2 from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class OnlineHardTripletLoss(nn.Module):
    """
    Online Hard Triplet Loss with online hard mining.
    
    From Equation 7:
    L_triplet = max(0, m + ||z_A - z_P||_2 - ||z_A - z_N||_2)
    
    From Section 3.2 (Online Hard Triplet Mining Strategy):
    - For each anchor, select the hardest positive (most dissimilar)
    - For each anchor, select the hardest negative (most similar)
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Triplet loss margin (m = 1.0 from paper)
        """
        super().__init__()
        self.margin = margin
    
    def _pairwise_distances(
        self,
        embeddings: torch.Tensor,
        squared: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances.
        
        Args:
            embeddings: Embeddings of shape (B, D)
            squared: Whether to return squared distances
            
        Returns:
            Distance matrix of shape (B, B)
        """
        # Compute dot product
        dot_product = torch.matmul(embeddings, embeddings.t())
        
        # Get squared L2 norm
        square_norm = torch.diag(dot_product)
        
        # Compute distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        
        # Ensure non-negative
        distances = torch.clamp(distances, min=0.0)
        
        if not squared:
            # Add small epsilon to avoid gradient issues with sqrt(0)
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)
        
        return distances
    
    def _get_anchor_positive_triplet_mask(
        self,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get mask for valid anchor-positive pairs.
        
        Args:
            labels: Speaker labels of shape (B,)
            
        Returns:
            Boolean mask of shape (B, B)
        """
        # Check if labels[i] == labels[j] and i != j
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        return labels_equal & indices_not_equal
    
    def _get_anchor_negative_triplet_mask(
        self,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get mask for valid anchor-negative pairs.
        
        Args:
            labels: Speaker labels of shape (B,)
            
        Returns:
            Boolean mask of shape (B, B)
        """
        # Check if labels[i] != labels[j]
        return labels.unsqueeze(0) != labels.unsqueeze(1)
    
    def _get_hardest_positive(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get hardest positive (most dissimilar positive) for each anchor.
        
        From Section 3.2:
        "A positive sample is selected... that is most dissimilar 
        (i.e., with the maximum Euclidean distance)"
        
        Args:
            distances: Pairwise distance matrix (B, B)
            labels: Speaker labels (B,)
            
        Returns:
            Hardest positive distances (B,)
        """
        # Get mask for valid positives
        mask = self._get_anchor_positive_triplet_mask(labels)
        
        # Set distances for invalid pairs to 0
        anchor_positive_dist = distances * mask.float()
        
        # Get hardest positive (maximum distance)
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
        
        return hardest_positive_dist
    
    def _get_hardest_negative(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get hardest negative (most similar negative) for each anchor.
        
        From Section 3.2:
        "A negative sample is selected as one... that is most similar 
        (i.e., with the minimum Euclidean distance)"
        
        Args:
            distances: Pairwise distance matrix (B, B)
            labels: Speaker labels (B,)
            
        Returns:
            Hardest negative distances (B,)
        """
        # Get mask for valid negatives
        mask = self._get_anchor_negative_triplet_mask(labels)
        
        # Add maximum value for invalid pairs
        max_dist = distances.max()
        anchor_negative_dist = distances + (1.0 - mask.float()) * max_dist
        
        # Get hardest negative (minimum distance)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        
        return hardest_negative_dist
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute online hard triplet loss.
        
        Implements Equation 7:
        L_triplet = max(0, m + ||z_A - z_P||_2 - ||z_A - z_N||_2)
        
        Args:
            embeddings: Speaker embeddings of shape (B, D)
            labels: Speaker labels of shape (B,)
            
        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        # Get hardest positive and negative for each anchor
        hardest_positive_dist = self._get_hardest_positive(distances, labels)
        hardest_negative_dist = self._get_hardest_negative(distances, labels)
        
        # Compute triplet loss: max(0, margin + d(a,p) - d(a,n))
        triplet_loss = F.relu(
            self.margin + hardest_positive_dist - hardest_negative_dist
        )
        
        # Average over batch
        loss = triplet_loss.mean()
        
        return loss


class BatchHardTripletLoss(OnlineHardTripletLoss):
    """
    Batch Hard Triplet Loss variant.
    More efficient implementation for larger batches.
    """
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Compute batch hard triplet loss with optional statistics.
        
        Args:
            embeddings: Speaker embeddings (B, D)
            labels: Speaker labels (B,)
            return_stats: Whether to return mining statistics
            
        Returns:
            Loss value and optionally mining statistics
        """
        # Get distances
        distances = self._pairwise_distances(embeddings)
        
        # Get hardest positives and negatives
        hardest_positive_dist = self._get_hardest_positive(distances, labels)
        hardest_negative_dist = self._get_hardest_negative(distances, labels)
        
        # Compute loss
        triplet_loss = F.relu(
            self.margin + hardest_positive_dist - hardest_negative_dist
        )
        
        # Count valid triplets (non-zero loss)
        valid_triplets = (triplet_loss > 0).sum()
        num_triplets = triplet_loss.numel()
        
        loss = triplet_loss.mean()
        
        if return_stats:
            stats = {
                'num_valid_triplets': valid_triplets.item(),
                'num_triplets': num_triplets,
                'fraction_valid': (valid_triplets / num_triplets).item(),
                'avg_positive_dist': hardest_positive_dist.mean().item(),
                'avg_negative_dist': hardest_negative_dist.mean().item()
            }
            return loss, stats
        
        return loss