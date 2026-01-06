"""
Speaker Diarization Module

A complete implementation of end-to-end neural speaker diarization.

Features:
- Multiple segmentation models (PyanNet, EEND, EEND-EDA)
- State-of-the-art speaker encoders (ECAPA-TDNN, X-vector)
- Various clustering algorithms (AHC, Spectral, VBx)
- Comprehensive evaluation metrics (DER, JER)
- Data augmentation pipeline
- Training and inference pipelines

Usage:
    from speaker_diarization import SpeakerDiarizationModel, DiarizationPipeline
    
    # Load pretrained model
    model = SpeakerDiarizationModel(num_speakers=4)
    
    # Create pipeline
    pipeline = DiarizationPipeline(model)
    
    # Run diarization
    segments = pipeline(waveform, sample_rate=16000)
    
    # segments: [(start, end, speaker_id), ...]
"""

__version__ = "1.0.0"
__author__ = "Speaker Recognition Team"

from .config import (
    DiarizationConfig,
    AudioConfig,
    EncoderConfig,
    SegmentationConfig,
    ClusteringConfig,
    TrainingConfig,
    DataConfig,
    EvaluationConfig,
    get_aishell4_config,
    get_ami_config,
    get_low_resource_config,
)

from .models import (
    # Main models
    SpeakerDiarizationModel,
    DiarizationPipeline,
    load_pretrained_model,
    
    # Encoder
    ECAPA_TDNN,
    XVector,
    SpeakerEncoder,
    
    # Segmentation
    PyanNet,
    EEND,
    EENDWithEDA,
    
    # Clustering
    AgglomerativeHierarchicalClustering,
    SpectralClusteringWrapper,
    VBxClustering,
    create_clustering,
    
    # Losses
    DiarizationLoss,
    PermutationInvariantTrainingLoss,
)

from .data import (
    DiarizationDataset,
    AISHELL4Dataset,
    AMIDataset,
    InferenceDataset,
    FeatureExtractor,
    create_augmentation_pipeline,
    create_dataloaders,
)

from .evaluation import (
    DiarizationMetrics,
    Evaluator,
    compute_der_simple,
    compute_jer,
)

from .training import (
    Trainer,
    train_model,
)

from .utils import (
    RTTMReader,
    RTTMWriter,
    RTTMSegment,
    AudioProcessor,
    set_seed,
    setup_logging,
)


__all__ = [
    # Version
    "__version__",
    
    # Config
    "DiarizationConfig",
    "AudioConfig",
    "EncoderConfig",
    "SegmentationConfig",
    "ClusteringConfig",
    "TrainingConfig",
    "DataConfig",
    "EvaluationConfig",
    "get_aishell4_config",
    "get_ami_config",
    "get_low_resource_config",
    
    # Models
    "SpeakerDiarizationModel",
    "DiarizationPipeline",
    "load_pretrained_model",
    "ECAPA_TDNN",
    "XVector",
    "SpeakerEncoder",
    "PyanNet",
    "EEND",
    "EENDWithEDA",
    "AgglomerativeHierarchicalClustering",
    "SpectralClusteringWrapper",
    "VBxClustering",
    "create_clustering",
    "DiarizationLoss",
    "PermutationInvariantTrainingLoss",
    
    # Data
    "DiarizationDataset",
    "AISHELL4Dataset",
    "AMIDataset",
    "InferenceDataset",
    "FeatureExtractor",
    "create_augmentation_pipeline",
    "create_dataloaders",
    
    # Evaluation
    "DiarizationMetrics",
    "Evaluator",
    "compute_der_simple",
    "compute_jer",
    
    # Training
    "Trainer",
    "train_model",
    
    # Utils
    "RTTMReader",
    "RTTMWriter",
    "RTTMSegment",
    "AudioProcessor",
    "set_seed",
    "setup_logging",
]