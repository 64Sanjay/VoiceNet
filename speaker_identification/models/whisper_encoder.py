# speaker_identification/models/whisper_encoder.py
"""
Whisper encoder wrapper for extracting frame-level embeddings.
Implements Equation 1 and 2 from the paper.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import Optional


class WhisperEncoderWrapper(nn.Module):
    """
    Wrapper around Whisper encoder for speaker embedding extraction.
    
    From Section 2.1:
    "Given an input log-mel spectrogram X ∈ R^(F×T), the encoder extracts 
    frame-level embeddings: E = {e_1, e_2, ..., e_T}, e_t ∈ R^D"
    
    These embeddings are then aggregated via global mean pooling (Equation 2).
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        freeze_encoder: bool = False,
        output_hidden_states: bool = False
    ):
        """
        Initialize Whisper encoder wrapper.
        
        Args:
            model_name: Hugging Face model name (default: whisper-tiny)
            freeze_encoder: Whether to freeze encoder weights
            output_hidden_states: Whether to output all hidden states
        """
        super().__init__()
        
        # Load pretrained Whisper model
        self.whisper = WhisperModel.from_pretrained(model_name)
        
        # We only need the encoder
        self.encoder = self.whisper.encoder
        
        # Get encoder output dimension
        # whisper-tiny: 384, whisper-base: 512, whisper-small: 768
        self.output_dim = self.encoder.config.d_model
        
        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze()
        
        self.output_hidden_states = output_hidden_states
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract frame-level embeddings from input features.
        
        Implements Equation 1:
        E = {e_1, e_2, ..., e_T}, e_t ∈ R^D
        
        Args:
            input_features: Log-mel spectrogram of shape (B, F, T)
                           where B=batch, F=frequency bins (80), T=time frames
            attention_mask: Optional attention mask
            
        Returns:
            Frame-level embeddings of shape (B, T', D)
            where T' is the output sequence length and D is the hidden dimension
        """
        # Whisper encoder expects input of shape (batch, feature_size, sequence_length)
        # which is already our input format (B, 80, T)
        
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
            return_dict=True
        )
        
        # Get the last hidden state: (B, T', D)
        hidden_states = encoder_outputs.last_hidden_state
        
        return hidden_states
    
    def get_pooled_output(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get pooled (mean) output from encoder.
        
        Implements Equation 2:
        ē = (1/T) * Σ e_t
        
        Args:
            input_features: Log-mel spectrogram of shape (B, F, T)
            attention_mask: Optional attention mask
            
        Returns:
            Pooled embeddings of shape (B, D)
        """
        # Get frame-level embeddings
        frame_embeddings = self.forward(input_features, attention_mask)
        
        # Global mean pooling over time dimension
        # Equation 2: ē = (1/T) * Σ e_t
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).expand_as(frame_embeddings)
            summed = (frame_embeddings * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / count
        else:
            # Simple mean pooling
            pooled = torch.mean(frame_embeddings, dim=1)
        
        return pooled