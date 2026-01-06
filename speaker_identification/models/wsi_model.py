# speaker_identification/models/wsi_model.py
"""
WSI (Whisper Speaker Identification) model.
Complete model architecture combining Whisper encoder and projection head.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .whisper_encoder import WhisperEncoderWrapper
from .projection_head import ProjectionHead


class WSIModel(nn.Module):
    """
    WSI Model for Speaker Identification.
    
    Architecture from Figure 1 and Section 2.1:
    1. Whisper Encoder: Extracts frame-level embeddings from log-mel spectrogram
    2. Mean Pooling: Aggregates frame-level to utterance-level
    3. Projection Head: Maps to compact speaker embedding space
    
    During training, generates embeddings for:
    - Original audio (z_i)
    - Noise-augmented audio (z_i^(n))
    - Time-stretched audio (z_i^(t))
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize WSI model.
        
        Args:
            whisper_model_name: Pretrained Whisper model name
            embedding_dim: Output embedding dimension (256 per paper)
            projection_hidden_dim: Projection head hidden dimension
            freeze_encoder: Whether to freeze Whisper encoder
            dropout: Dropout probability
        """
        super().__init__()
        
        # Whisper Encoder (Section 2.1, Component 1)
        self.encoder = WhisperEncoderWrapper(
            model_name=whisper_model_name,
            freeze_encoder=freeze_encoder
        )
        
        # Projection Head (Section 2.1, Component 2)
        self.projection_head = ProjectionHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
    
    def encode(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to speaker embedding.
        
        Implements Equation 4:
        z_i = f_proj(pool(f_enc(X_i)))
        
        Args:
            input_features: Log-mel spectrogram of shape (B, 80, T)
            
        Returns:
            Speaker embeddings of shape (B, 256)
        """
        # Get pooled output from encoder (Equation 2)
        pooled = self.encoder.get_pooled_output(input_features)
        
        # Project to embedding space (Equation 3)
        embeddings = self.projection_head(pooled)
        
        return embeddings
    
    def forward(
        self,
        original: torch.Tensor,
        noise_augmented: Optional[torch.Tensor] = None,
        time_stretched: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with multi-view learning.
        
        Implements Equations 4, 5, 6 from Section 2.2:
        - z_i: Embedding of original
        - z_i^(n): Embedding of noise-augmented
        - z_i^(t): Embedding of time-stretched
        
        Args:
            original: Original audio features (B, 80, T)
            noise_augmented: Noise-augmented features (B, 80, T)
            time_stretched: Time-stretched features (B, 80, T)
            
        Returns:
            Dictionary with embeddings:
            {
                'original': z_i,
                'noise_augmented': z_i^(n) (if provided),
                'time_stretched': z_i^(t) (if provided)
            }
        """
        outputs = {}
        
        # Original embeddings (Equation 4)
        outputs['original'] = self.encode(original)
        
        # Noise-augmented embeddings (Equation 5)
        if noise_augmented is not None:
            outputs['noise_augmented'] = self.encode(noise_augmented)
        
        # Time-stretched embeddings (Equation 6)
        if time_stretched is not None:
            outputs['time_stretched'] = self.encode(time_stretched)
        
        return outputs
    
    def get_embedding(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Get speaker embedding for inference.
        
        Args:
            input_features: Log-mel spectrogram of shape (B, 80, T)
            
        Returns:
            L2-normalized speaker embeddings of shape (B, 256)
        """
        embeddings = self.encode(input_features)
        
        # L2 normalize for cosine similarity
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    @torch.no_grad()
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.
        
        Implements Equation 10:
        sim(z_1, z_2) = (z_1 · z_2) / (||z_1|| * ||z_2||)
        
        Args:
            embedding1: First embedding (B, D) or (D,)
            embedding2: Second embedding (B, D) or (D,)
            
        Returns:
            Cosine similarity scores
        """
        # Ensure normalized
        embedding1 = nn.functional.normalize(embedding1, p=2, dim=-1)
        embedding2 = nn.functional.normalize(embedding2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=-1)
        
        return similarity


class WSIModelForVerification(WSIModel):
    """
    WSI model specifically for speaker verification tasks.
    Adds utility methods for verification decisions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = 0.5  # Default threshold
    
    def set_threshold(self, threshold: float):
        """Set decision threshold for verification."""
        self.threshold = threshold
    
    @torch.no_grad()
    def verify(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify if two audio samples belong to the same speaker.
        
        Args:
            features1: Features of first audio
            features2: Features of second audio
            
        Returns:
            Tuple of (decision, similarity_score)
            decision: Boolean tensor (True if same speaker)
        """
        emb1 = self.get_embedding(features1)
        emb2 = self.get_embedding(features2)
        
        similarity = self.compute_similarity(emb1, emb2)
        decision = similarity >= self.threshold
        
        return decision, similarity