# speaker_verification/__init__.py
"""
CAM++ Speaker Verification Module

A fast and efficient speaker verification system using Context-Aware Masking
with Multi-Granularity Pooling.

Based on the paper:
"CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking"
"""

from .models.cam_plus_plus import CAMPlusPlus, CAMPlusPlusClassifier
from .config.config import get_config, get_small_config

__version__ = "1.0.0"
__all__ = [
    "CAMPlusPlus",
    "CAMPlusPlusClassifier", 
    "get_config",
    "get_small_config"
]