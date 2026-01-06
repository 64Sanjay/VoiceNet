"""Training module for speaker diarization."""

from .trainer import (
    Trainer,
    train_model,
)

__all__ = [
    "Trainer",
    "train_model",
]