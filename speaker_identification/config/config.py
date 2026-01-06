# speaker_identification/config/config.py
"""
Configuration classes for WSI model training and evaluation.
Based on Table 3 from the paper.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    # Audio parameters
    sample_rate: int = 16000  # Audio Sampling Rate: 16 kHz
    fixed_input_frames: int = 3000  # Fixed Input Frames: 3000 (zero-padded)
    segment_duration: float = 4.0  # Each instance is 4-second audio segment
    
    # Dataset paths
    train_data_path: str = "./data/voxtube/train"
    val_data_path: str = "./data/voxtube/val"
    test_data_path: str = "./data/voxtube/test"
    
    # Data augmentation parameters
    noise_snr_db: Tuple[float, float] = (5.0, 20.0)  # SNR range for Gaussian noise
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)  # Time stretch factor range
    
    # DataLoader parameters
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Backbone Architecture: openai/whisper-tiny
    whisper_model_name: str = "openai/whisper-tiny"
    
    # Embedding Dimension: 256
    embedding_dim: int = 256
    
    # Whisper encoder output dimension (whisper-tiny: 384)
    encoder_output_dim: int = 384
    
    # Projection head hidden dimension
    projection_hidden_dim: int = 512
    
    # Freeze whisper encoder initially
    freeze_encoder: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Triplet Loss Margin: 1.0
    triplet_margin: float = 1.0
    
    # Self-Supervised Loss Weight: 1.0 (lambda in equation 9)
    self_supervised_weight: float = 1.0
    
    # NT-Xent Loss Temperature: 0.5
    nt_xent_temperature: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration parameters from Table 3."""
    
    # Batch Size: 16
    batch_size: int = 16
    
    # Epochs: 3
    epochs: int = 3
    
    # Learning Rate: 1 × 10^-5
    learning_rate: float = 1e-5
    
    # Optimizer: Adam
    optimizer: str = "adam"
    
    # Weight decay
    weight_decay: float = 0.0
    
    # Gradient clipping
    gradient_clip_val: Optional[float] = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 1
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "wsi-speaker-identification"


@dataclass
class WSIConfig:
    """Complete WSI configuration."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device configuration
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.data.sample_rate == 16000, "Sample rate must be 16kHz as per paper"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.model.embedding_dim == 256, "Embedding dimension must be 256 as per paper"


def get_default_config() -> WSIConfig:
    """Get default configuration matching the paper."""
    return WSIConfig()


def get_config_for_dataset(dataset_name: str) -> WSIConfig:
    """Get configuration adjusted for specific dataset."""
    config = WSIConfig()
    
    if dataset_name == "voxtube":
        config.data.train_data_path = "./data/voxtube/train"
        config.data.val_data_path = "./data/voxtube/val"
        config.data.test_data_path = "./data/voxtube/test"
    elif dataset_name == "jvs":
        config.data.train_data_path = "./data/jvs/train"
        config.data.val_data_path = "./data/jvs/val"
        config.data.test_data_path = "./data/jvs/test"
    elif dataset_name == "callhome":
        config.data.train_data_path = "./data/callhome/train"
        config.data.val_data_path = "./data/callhome/val"
        config.data.test_data_path = "./data/callhome/test"
    elif dataset_name == "voxconverse":
        config.data.train_data_path = "./data/voxconverse/train"
        config.data.val_data_path = "./data/voxconverse/val"
        config.data.test_data_path = "./data/voxconverse/test"
    
    return config