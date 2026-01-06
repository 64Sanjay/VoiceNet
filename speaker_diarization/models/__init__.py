"""Models module for speaker diarization."""

from .speaker_encoder import (
    ECAPA_TDNN,
    XVector,
    SpeakerEncoder,
    SEBlock,
    Res2Block,
    SERes2Block,
    AttentiveStatisticsPooling,
    StatisticsPooling,
)
from .segmentation import (
    PyanNet,
    EEND,
    EENDWithEDA,
    SegmentationModel,
    SincConvBlock,
    PositionalEncoding,
)
from .clustering import (
    SpeakerClustering,
    AgglomerativeHierarchicalClustering,
    SpectralClusteringWrapper,
    VBxClustering,
    OnlineClustering,
    PLDAScoring,
    create_clustering,
)
from .losses import (
    DiarizationLoss,
    BCEWithLogitsLossWeighted,
    PermutationInvariantTrainingLoss,
    PITLossEfficient,
    DeepClusteringLoss,
    ContrastiveLoss,
    FocalLoss,
    MultiTaskLoss,
)
from .diarization_model import (
    SpeakerDiarizationModel,
    DiarizationPipeline,
    load_pretrained_model,
)

__all__ = [
    # Speaker encoder
    "ECAPA_TDNN",
    "XVector",
    "SpeakerEncoder",
    "SEBlock",
    "Res2Block",
    "SERes2Block",
    "AttentiveStatisticsPooling",
    "StatisticsPooling",
    # Segmentation
    "PyanNet",
    "EEND",
    "EENDWithEDA",
    "SegmentationModel",
    "SincConvBlock",
    "PositionalEncoding",
    # Clustering
    "SpeakerClustering",
    "AgglomerativeHierarchicalClustering",
    "SpectralClusteringWrapper",
    "VBxClustering",
    "OnlineClustering",
    "PLDAScoring",
    "create_clustering",
    # Losses
    "DiarizationLoss",
    "BCEWithLogitsLossWeighted",
    "PermutationInvariantTrainingLoss",
    "PITLossEfficient",
    "DeepClusteringLoss",
    "ContrastiveLoss",
    "FocalLoss",
    "MultiTaskLoss",
    # Diarization model
    "SpeakerDiarizationModel",
    "DiarizationPipeline",
    "load_pretrained_model",
]