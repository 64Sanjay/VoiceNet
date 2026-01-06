"""Data module for speaker diarization."""

from .dataset import (
    DiarizationDataset,
    AISHELL4Dataset,
    AMIDataset,
    InferenceDataset,
    DiarizationSample,
    collate_fn,
    create_dataloaders,
)
from .augmentation import (
    AudioAugmentor,
    SpeedPerturbation,
    SpecAugment,
    MixUp,
    RandomCrop,
    create_augmentation_pipeline,
)
from .preprocessing import (
    FeatureExtractor,
    SincNetFrontend,
    SincConv1d,
    extract_features_from_file,
)

__all__ = [
    # Dataset
    "DiarizationDataset",
    "AISHELL4Dataset",
    "AMIDataset",
    "InferenceDataset",
    "DiarizationSample",
    "collate_fn",
    "create_dataloaders",
    # Augmentation
    "AudioAugmentor",
    "SpeedPerturbation",
    "SpecAugment",
    "MixUp",
    "RandomCrop",
    "create_augmentation_pipeline",
    # Preprocessing
    "FeatureExtractor",
    "SincNetFrontend",
    "SincConv1d",
    "extract_features_from_file",
]