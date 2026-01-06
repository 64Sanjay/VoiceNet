"""
General helper utilities for speaker diarization.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get torch device."""
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_gpu_info() -> None:
    """Print GPU information."""
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU Information")
        print("=" * 60)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"  CUDA Version: {torch.version.cuda}")
        try:
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        except Exception:
            pass
        print("=" * 60)
    else:
        print("=" * 60)
        print("WARNING: CUDA not available, using CPU!")
        print("=" * 60)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    output_dir: Optional[Path] = None,  # Add this parameter
    name: str = "speaker_diarization",
) -> logging.Logger:
    """Setup logging configuration."""

    # Handle case where output_dir is passed as first positional argument
    if isinstance(log_level, Path) or (isinstance(log_level, str) and Path(log_level).exists()):
        output_dir = Path(log_level)
        log_level = "INFO"
        log_file = str(output_dir / "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    elif output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "train.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    scheduler: Optional[Any] = None,
    **kwargs,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    path = Path(path)
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def move_to_device(
    data: Any, 
    device: torch.device,
) -> Any:
    """Move data to device recursively."""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = "Meter"):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int = 0) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class MetricTracker:
    """Track multiple metrics during training."""
    
    def __init__(self, *metrics: str):
        self.metrics = {m: AverageMeter(m) for m in metrics}
        self.history: Dict[str, List[float]] = {m: [] for m in metrics}
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                if isinstance(value, tuple):
                    self.metrics[key].update(value[0], value[1])
                else:
                    self.metrics[key].update(value)
    
    def reset(self) -> None:
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
    
    def save_epoch(self) -> None:
        """Save current epoch metrics to history."""
        for name, meter in self.metrics.items():
            self.history[name].append(meter.avg)
    
    def get_average(self, metric: str) -> float:
        """Get average value for a metric."""
        return self.metrics[metric].avg if metric in self.metrics else 0.0
    
    def __str__(self) -> str:
        return " | ".join(str(m) for m in self.metrics.values())