#!/usr/bin/env python3
"""Demo package for unified speaker recognition."""

from .verification_tab import create_verification_tab
from .identification_tab import create_identification_tab
from .diarization_tab import create_diarization_tab

__all__ = [
    'create_verification_tab',
    'create_identification_tab', 
    'create_diarization_tab'
]