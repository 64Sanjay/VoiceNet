# speaker_identification/__init__.py
"""
WSI (Whisper Speaker Identification) Module

A framework for speaker identification using pre-trained Whisper encoder
with joint loss optimization (triplet loss + NT-Xent loss).

Based on the paper: "Whisper Speaker Identification: Leveraging Pre-trained
Multilingual Transformers for Robust Speaker Embeddings"
"""

from .models.wsi_model import WSIModel
from .config.config import WSIConfig, get_default_config
from .data.preprocessing import AudioPreprocessor
from .data.augmentation import AudioAugmentor
from .evaluation.evaluator import WSIEvaluator

__version__ = "1.0.0"
__all__ = [
    "WSIModel",
    "WSIConfig",
    "get_default_config",
    "AudioPreprocessor",
    "AudioAugmentor",
    "WSIEvaluator"
]