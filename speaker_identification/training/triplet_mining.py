# speaker_identification/training/triplet_mining.py
"""
Online hard triplet mining utilities.
Implements Section 3.2 from the paper.
"""

import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TripletBatch:
    """Container for triplet batch data."""
    anchors: torch.Tensor
    positives: torch.Tensor
    negatives: torch.Tensor
    anchor_labels: torch.Tensor


class OnlineHardTripletMiner:
    """
    Online Hard Triplet Miner.
    
    From Section 3.2:
    "Rather than pre-constructing triplets, triplet selection is performed 
    online within each mini-batch."
    
    For each anchor:
    - Positive: Same speaker, maximum distance (hardest positive)
    - Negative: Different speaker, minimum distance (hardest negative)
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet miner.
        
        Args:
            margin: Triplet loss margin
        """
        self.margin = margin
    
    def _pairwise_distances(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances.
        
        Args:
            embeddings: Embeddings (B, D)
            
        Returns:
            Distance matrix (B, B)
        """
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-16)
        return distances
    
    def mine_hard_triplets(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard triplets from a batch.
        
        Args:
            embeddings: Embeddings (B, D)
            labels: Speaker labels (B,)
            
        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute distance matrix
        distances = self._pairwise_distances(embeddings)
        
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Find positives (same label, different index)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
            if positive_mask.sum() == 0:
                continue  # No valid positive
            
            # Find hardest positive (maximum distance)
            positive_distances = distances[i].clone()
            positive_distances[~positive_mask] = -float('inf')
            hardest_positive_idx = positive_distances.argmax()
            
            # Find negatives (different label)
            negative_mask = labels != anchor_label
            if negative_mask.sum() == 0:
                continue  # No valid negative
            
            # Find hardest negative (minimum distance)
            negative_distances = distances[i].clone()
            negative_distances[~negative_mask] = float('inf')
            hardest_negative_idx = negative_distances.argmin()
            
            anchor_indices.append(i)
            positive_indices.append(hardest_positive_idx.item())
            negative_indices.append(hardest_negative_idx.item())
        
        return (
            torch.tensor(anchor_indices, device=device),
            torch.tensor(positive_indices, device=device),
            torch.tensor(negative_indices, device=device)
        )
    
    def get_triplet_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> TripletBatch:
        """
        Get triplet embeddings using hard mining.
        
        Args:
            embeddings: Embeddings (B, D)
            labels: Speaker labels (B,)
            
        Returns:
            TripletBatch with anchor, positive, negative embeddings
        """
        a_idx, p_idx, n_idx = self.mine_hard_triplets(embeddings, labels)
        
        return TripletBatch(
            anchors=embeddings[a_idx],
            positives=embeddings[p_idx],
            negatives=embeddings[n_idx],
            anchor_labels=labels[a_idx]
        )