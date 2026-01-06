# speaker_verification/config/config.py
"""
Configuration for CAM++ Speaker Verification System.
Based on the paper: "CAM++: A Fast and Efficient Network for Speaker Verification"
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400  # 25ms at 16kHz
    hop_length: int = 160  # 10ms at 16kHz
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 7600.0
    
    # Training audio length
    min_duration: float = 2.0  # seconds
    max_duration: float = 6.0  # seconds
    training_duration: float = 3.0  # 3s crops during training


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Speed perturbation
    speed_perturb: bool = True
    speed_perturb_rates: Tuple[float, ...] = (0.9, 1.0, 1.1)
    
    # Noise augmentation (MUSAN)
    noise_aug: bool = True
    noise_snr_range: Tuple[float, float] = (0, 15)
    
    # Reverb augmentation (RIR)
    reverb_aug: bool = True
    
    # SpecAugment
    spec_augment: bool = True
    freq_mask_param: int = 10
    time_mask_param: int = 5


@dataclass
class CAMPlusPlusConfig:
    """CAM++ model architecture configuration from the paper."""
    
    # Front-end Convolution Module (FCM)
    fcm_channels: int = 32
    fcm_num_blocks: int = 4
    fcm_freq_stride: int = 2  # Downsampling in frequency
    
    # D-TDNN Backbone
    # Paper: "expand the number of layers per block to 12, 24 and 16"
    dtdnn_blocks: Tuple[int, ...] = (12, 24, 16)
    # Paper: "reducing the original growth rate k from 64 to 32"
    growth_rate: int = 32
    bn_size: int = 4  # Bottleneck size multiplier
    init_channels: int = 128
    
    # Context-Aware Masking
    cam_reduction: int = 2
    segment_length: int = 100  # frames per segment for segment pooling
    
    # Pooling
    embedding_dim: int = 192
    
    # TDNN kernel size
    kernel_size: int = 3
    dilation: Tuple[int, ...] = (1, 2, 3)
    
    # Subsampling
    subsample_rate: int = 2  # 1/2 subsampling before D-TDNN


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Batch and epochs
    batch_size: int = 128
    num_epochs: int = 100
    
    # Optimizer (SGD with cosine annealing from paper)
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    min_lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Warmup
    warmup_epochs: int = 5
    
    # AAM-Softmax loss (from paper)
    aam_margin: float = 0.2
    aam_scale: float = 32
    
    # Gradient clipping
    gradient_clip: float = 5.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    
    # Data loading
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Scoring
    scoring_method: str = "cosine"  # cosine similarity
    
    # Metrics
    compute_eer: bool = True
    compute_mindcf: bool = True
    mindcf_p_target: float = 0.01


@dataclass
class CAMPlusPlusSystemConfig:
    """Complete system configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: CAMPlusPlusConfig = field(default_factory=CAMPlusPlusConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General
    seed: int = 42
    device: str = "cuda"
    
    # Dataset paths
    train_data_path: str = "./data/voxceleb2/train"
    val_data_path: str = "./data/voxceleb1/test"
    musan_path: str = "./data/musan"
    rir_path: str = "./data/rir_noises"


def get_config() -> CAMPlusPlusSystemConfig:
    """Get default configuration."""
    return CAMPlusPlusSystemConfig()


# def get_small_config() -> CAMPlusPlusSystemConfig:
#     """Get smaller configuration for testing."""
#     config = CAMPlusPlusSystemConfig()
#     config.model.dtdnn_blocks = (6, 12, 8)
#     config.model.growth_rate = 16
#     config.training.batch_size = 32
#     config.training.num_epochs = 20
#     return config

# In config/config.py, in the small config section:
def get_small_config():
    config = get_config()
    
    # Smaller model for testing
    config.model.fcm_channels = 16
    config.model.fcm_num_blocks = 2
    config.model.dtdnn_blocks = (6, 12, 8)
    config.model.growth_rate = 16
    config.model.bn_size = 2  # Make sure this is included
    config.model.init_channels = 64
    
    return config