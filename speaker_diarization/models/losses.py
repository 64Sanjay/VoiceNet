"""
Loss functions for speaker diarization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from itertools import permutations
import numpy as np


class DiarizationLoss(nn.Module):
    """
    Combined loss for speaker diarization.
    
    Supports:
    - Binary Cross Entropy for speaker detection
    - Permutation Invariant Training (PIT)
    - VAD loss
    - Embedding loss (optional)
    """
    
    def __init__(
        self,
        num_speakers: int = 4,
        use_pit: bool = True,
        pit_weight: float = 1.0,
        vad_weight: float = 0.5,
        embedding_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.use_pit = use_pit
        self.pit_weight = pit_weight
        self.vad_weight = vad_weight
        self.embedding_weight = embedding_weight
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        vad: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        Args:
            outputs: Model outputs dict with 'speakers', 'vad', optionally 'embeddings'
            labels: Ground truth labels [B, T, num_speakers]
            vad: Ground truth VAD [B, T]
            lengths: Valid lengths [B]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Extract predictions
        if isinstance(outputs, dict):
            pred_speakers = outputs.get('speakers', outputs.get('predictions'))
            pred_vad = outputs.get('vad', None)
            embeddings = outputs.get('embeddings', None)
        else:
            pred_speakers = outputs
            pred_vad = None
            embeddings = None
        
        batch_size, time_steps, num_speakers = pred_speakers.shape
        
        # Create mask for valid positions
        if lengths is not None:
            mask = torch.arange(time_steps, device=lengths.device)[None, :] < lengths[:, None]
            mask = mask.float()
        else:
            mask = torch.ones(batch_size, time_steps, device=pred_speakers.device)
        
        # Speaker detection loss
        if self.use_pit:
            speaker_loss = self._pit_loss(pred_speakers, labels, mask)
        else:
            speaker_loss = self._bce_loss(pred_speakers, labels, mask)
        
        # VAD loss
        vad_loss = torch.tensor(0.0, device=pred_speakers.device)
        if pred_vad is not None and vad is not None:
            vad_loss = self._bce_loss_1d(pred_vad, vad, mask)
        elif pred_vad is not None:
            # Derive VAD from labels
            derived_vad = (labels.sum(dim=-1) > 0).float()
            vad_loss = self._bce_loss_1d(pred_vad, derived_vad, mask)
        
        # Embedding loss (optional)
        embedding_loss = torch.tensor(0.0, device=pred_speakers.device)
        if embeddings is not None and self.embedding_weight > 0:
            embedding_loss = self._embedding_loss(embeddings, labels)
        
        # Total loss
        total_loss = (
            self.pit_weight * speaker_loss +
            self.vad_weight * vad_loss +
            self.embedding_weight * embedding_loss
        )
        
        loss_dict = {
            'speaker_loss': speaker_loss.item(),
            'vad_loss': vad_loss.item(),
            'embedding_loss': embedding_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def _bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross entropy loss with masking."""
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1).expand_as(bce)
        bce = bce * mask_expanded
        
        return bce.sum() / mask_expanded.sum()
    
    def _bce_loss_1d(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross entropy for 1D predictions (VAD)."""
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        bce = bce * mask
        
        return bce.sum() / mask.sum()
    
    def _pit_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Permutation Invariant Training loss.
        Finds best permutation of speaker labels.
        """
        batch_size, time_steps, num_speakers = pred.shape
        
        # For efficiency, use Hungarian algorithm approximation for many speakers
        if num_speakers > 4:
            return self._pit_loss_hungarian(pred, target, mask)
        
        # Full permutation search for small number of speakers
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for b in range(batch_size):
            pred_b = pred[b]  # [T, S]
            target_b = target[b]  # [T, S]
            mask_b = mask[b]  # [T]
            
            min_loss = float('inf')
            
            # Try all permutations
            for perm in permutations(range(num_speakers)):
                perm_target = target_b[:, list(perm)]
                loss = self._compute_frame_bce(pred_b, perm_target, mask_b)
                min_loss = min(min_loss, loss)
            
            total_loss = total_loss + min_loss
        
        return total_loss / batch_size
    
    def _pit_loss_hungarian(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate PIT using greedy assignment."""
        batch_size, time_steps, num_speakers = pred.shape
        
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for b in range(batch_size):
            pred_b = pred[b]
            target_b = target[b]
            mask_b = mask[b]
            
            # Compute cost matrix
            cost_matrix = torch.zeros(num_speakers, num_speakers, device=pred.device)
            for i in range(num_speakers):
                for j in range(num_speakers):
                    cost_matrix[i, j] = self._compute_frame_bce(
                        pred_b[:, i:i+1], target_b[:, j:j+1], mask_b
                    )
            
            # Greedy assignment
            used_targets = set()
            for i in range(num_speakers):
                best_j = -1
                best_cost = float('inf')
                for j in range(num_speakers):
                    if j not in used_targets and cost_matrix[i, j].item() < best_cost:
                        best_cost = cost_matrix[i, j].item()
                        best_j = j
                if best_j >= 0:
                    used_targets.add(best_j)
                    total_loss = total_loss + cost_matrix[i, best_j]
        
        return total_loss / (batch_size * num_speakers)
    
    def _compute_frame_bce(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE for a single permutation."""
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        if pred.dim() == 2:
            bce = bce.mean(dim=-1)
        
        return (bce * mask).sum() / mask.sum()
    
    def _embedding_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss for speaker embeddings.
        Same speakers should have similar embeddings.
        """
        # Simplified version - can be enhanced with proper contrastive loss
        return torch.tensor(0.0, device=embeddings.device)


class BCEWithLogitsLossWeighted(nn.Module):
    """
    Weighted Binary Cross Entropy with Logits.
    Handles class imbalance in speaker diarization.
    """
    
    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            input: Predictions (logits) [B, T, S]
            target: Ground truth [B, T, S]
            mask: Valid positions mask [B, T]
            
        Returns:
            loss: Scalar loss
        """
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=input.device)
        else:
            # Auto-compute based on class frequency
            pos_ratio = target.sum() / target.numel()
            pos_weight = (1 - pos_ratio) / (pos_ratio + 1e-8)
            pos_weight = torch.clamp(pos_weight, 0.1, 10.0)
        
        loss = F.binary_cross_entropy_with_logits(
            input, target,
            pos_weight=pos_weight.expand_as(input) if isinstance(pos_weight, torch.Tensor) else None,
            reduction='none',
        )
        
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            return loss.sum() / mask.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PermutationInvariantTrainingLoss(nn.Module):
    """Standalone PIT loss module."""
    
    def __init__(
        self, 
        num_speakers: int = 4,
        loss_fn: str = 'bce',  # 'bce' or 'mse'
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.loss_fn = loss_fn
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
        """
        Compute PIT loss.
        
        Args:
            pred: Predictions [B, T, S]
            target: Ground truth [B, T, S]
            mask: Valid mask [B, T]
            
        Returns:
            loss: Scalar loss
            best_perms: Best permutations for each batch
        """
        if mask is None:
            mask = torch.ones(pred.shape[0], pred.shape[1], device=pred.device)
        
        batch_size, time_steps, num_speakers = pred.shape
        total_loss = torch.tensor(0.0, device=pred.device)
        best_perms = []
        
        for b in range(batch_size):
            min_loss = float('inf')
            best_perm = tuple(range(num_speakers))
            
            for perm in permutations(range(num_speakers)):
                perm_target = target[b, :, list(perm)]
                
                if self.loss_fn == 'bce':
                    eps = 1e-7
                    pred_b = pred[b].clamp(eps, 1 - eps)
                    loss = -(perm_target * torch.log(pred_b) + 
                            (1 - perm_target) * torch.log(1 - pred_b))
                else:
                    loss = (pred[b] - perm_target) ** 2
                
                loss = loss.mean(dim=-1)
                loss = (loss * mask[b]).sum() / mask[b].sum()
                
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_perm = perm
            
            total_loss = total_loss + min_loss
            best_perms.append(best_perm)
        
        return total_loss / batch_size, best_perms


class PITLossEfficient(nn.Module):
    """
    Efficient PIT loss using Hungarian algorithm.
    Better for large number of speakers.
    """
    
    def __init__(self, num_speakers: int = 4):
        super().__init__()
        self.num_speakers = num_speakers
        
        try:
            from scipy.optimize import linear_sum_assignment
            self.linear_sum_assignment = linear_sum_assignment
            self.use_scipy = True
        except ImportError:
            self.use_scipy = False
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute efficient PIT loss."""
        if mask is None:
            mask = torch.ones(pred.shape[0], pred.shape[1], device=pred.device)
        
        batch_size, time_steps, num_speakers = pred.shape
        total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        for b in range(batch_size):
            # Compute pairwise loss matrix
            cost_matrix = np.zeros((num_speakers, num_speakers))
            
            for i in range(num_speakers):
                for j in range(num_speakers):
                    eps = 1e-7
                    p = pred[b, :, i].clamp(eps, 1 - eps)
                    t = target[b, :, j]
                    bce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
                    cost_matrix[i, j] = (bce * mask[b]).sum().item() / mask[b].sum().item()
            
            # Find optimal assignment
            if self.use_scipy:
                row_ind, col_ind = self.linear_sum_assignment(cost_matrix)
                perm = col_ind.tolist()
            else:
                # Greedy fallback
                perm = self._greedy_assignment(cost_matrix)
            
            # Compute loss with optimal permutation
            perm_target = target[b, :, perm]
            eps = 1e-7
            pred_b = pred[b].clamp(eps, 1 - eps)
            loss = -(perm_target * torch.log(pred_b) + 
                    (1 - perm_target) * torch.log(1 - pred_b))
            loss = loss.mean(dim=-1)
            batch_loss = (loss * mask[b]).sum() / mask[b].sum()
            
            total_loss = total_loss + batch_loss
        
        return total_loss / batch_size
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[int]:
        """Greedy assignment when scipy not available."""
        n = cost_matrix.shape[0]
        assignment = [-1] * n
        used_cols = set()
        
        for i in range(n):
            best_j = -1
            best_cost = float('inf')
            for j in range(n):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                assignment[i] = best_j
                used_cols.add(best_j)
        
        return assignment


class DeepClusteringLoss(nn.Module):
    """
    Deep Clustering loss for speaker separation/diarization.
    
    Learns embeddings where same-speaker frames are close together.
    """
    
    def __init__(self, embedding_dim: int = 40):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute deep clustering loss.
        
        Args:
            embeddings: Frame embeddings [B, T, D]
            labels: Speaker labels [B, T, S]
            mask: Valid mask [B, T]
            
        Returns:
            loss: Deep clustering loss
        """
        batch_size, time_steps, embed_dim = embeddings.shape
        _, _, num_speakers = labels.shape
        
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        for b in range(batch_size):
            V = embeddings[b]  # [T, D]
            Y = labels[b]  # [T, S]
            
            if mask is not None:
                valid_idx = mask[b] > 0
                V = V[valid_idx]
                Y = Y[valid_idx]
            
            T = V.shape[0]
            
            # Normalize embeddings
            V = F.normalize(V, p=2, dim=-1)
            
            # Deep clustering objective: ||VV^T - YY^T||_F^2
            VVt = torch.mm(V, V.t())  # [T, T]
            YYt = torch.mm(Y, Y.t())  # [T, T]
            
            loss = torch.norm(VVt - YYt, p='fro') ** 2 / (T * T)
            total_loss = total_loss + loss
        
        return total_loss / batch_size


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for speaker embeddings.
    
    Pulls same-speaker embeddings together,
    pushes different-speaker embeddings apart.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'cosine',  # 'cosine' or 'euclidean'
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Speaker embeddings [B, D]
            labels: Speaker labels [B]
            
        Returns:
            loss: Contrastive loss
        """
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute pairwise distances
        if self.distance == 'cosine':
            sim_matrix = torch.mm(embeddings, embeddings.t())
            dist_matrix = 1 - sim_matrix
        else:
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label matrix
        labels = labels.view(-1, 1)
        same_speaker = (labels == labels.t()).float()
        diff_speaker = 1 - same_speaker
        
        # Mask diagonal
        mask = 1 - torch.eye(batch_size, device=embeddings.device)
        same_speaker = same_speaker * mask
        diff_speaker = diff_speaker * mask
        
        # Contrastive loss
        pos_loss = (dist_matrix * same_speaker).sum() / (same_speaker.sum() + 1e-8)
        neg_loss = (F.relu(self.margin - dist_matrix) * diff_speaker).sum() / (diff_speaker.sum() + 1e-8)
        
        return pos_loss + neg_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces loss for well-classified examples,
    focuses on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            input: Predictions [B, T, S] (probabilities, not logits)
            target: Ground truth [B, T, S]
            mask: Valid mask [B, T]
            
        Returns:
            loss: Focal loss
        """
        eps = 1e-7
        p = input.clamp(eps, 1 - eps)
        
        # Focal loss
        ce_loss = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
        
        p_t = target * p + (1 - target) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * ce_loss
        
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            return loss.sum() / mask.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining multiple objectives.
    Automatically learns task weights.
    """
    
    def __init__(
        self,
        task_names: List[str],
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.task_names = task_names
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Log variance parameterization for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in task_names
            })
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combine multiple task losses.
        
        Args:
            losses: Dictionary of task losses
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses and weights
        """
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        loss_dict = {}
        
        for name in self.task_names:
            if name not in losses:
                continue
            
            task_loss = losses[name]
            
            if self.learnable_weights:
                # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
                precision = torch.exp(-self.log_vars[name])
                weighted_loss = precision * task_loss + self.log_vars[name]
                loss_dict[f'{name}_weight'] = precision.item()
            else:
                weighted_loss = task_loss
            
            total_loss = total_loss + weighted_loss
            loss_dict[name] = task_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict