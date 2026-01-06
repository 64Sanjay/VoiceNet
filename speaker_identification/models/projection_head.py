# speaker_identification/models/projection_head.py
"""
Projection head for mapping pooled embeddings to speaker embeddings.
Implements Equation 3 from the paper.
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Projection head for WSI model.
    
    From Section 2.1:
    "The pooled representation is transformed via a projection head f_proj(·) 
    into a compact speaker embedding: z = f_proj(ē), z ∈ R^256"
    
    Architecture: Two dense layers with ReLU activation.
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # Whisper-tiny encoder output dim
        hidden_dim: int = 512,
        output_dim: int = 256,  # Embedding Dimension: 256 (from Table 3)
        dropout: float = 0.1
    ):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input dimension (from Whisper encoder)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (256 as per paper)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            # First dense layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second dense layer
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.output_dim = output_dim
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project pooled embeddings to speaker embedding space.
        
        Implements Equation 3:
        z = f_proj(ē), z ∈ R^256
        
        Args:
            x: Pooled embeddings of shape (B, input_dim)
            
        Returns:
            Speaker embeddings of shape (B, 256)
        """
        return self.projection(x)


class ProjectionHeadWithNorm(nn.Module):
    """
    Projection head with L2 normalization.
    Useful for cosine similarity-based evaluation.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
        normalize: bool = True
    ):
        super().__init__()
        
        self.projection = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
        self.normalize = normalize
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional L2 normalization."""
        z = self.projection(x)
        
        if self.normalize:
            z = nn.functional.normalize(z, p=2, dim=-1)
        
        return z