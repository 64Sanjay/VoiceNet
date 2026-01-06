#!/usr/bin/env python3
"""
Training script for speaker diarization.

Usage:
    python train.py --data-dir ./data/aishell4 --output-dir ./outputs
    python train.py --config ./config.yaml --output-dir ./outputs
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from project modules with fallbacks
try:
    from config.config import DiarizationConfig, get_aishell4_config
except ImportError:
    DiarizationConfig = None
    get_aishell4_config = None

try:
    from data.dataset import DiarizationDataset, collate_fn
except ImportError:
    DiarizationDataset = None
    collate_fn = None

try:
    from models.diarization_model import SpeakerDiarizationModel
except ImportError:
    SpeakerDiarizationModel = None

try:
    from models.segmentation import PyanNet, EEND
except ImportError:
    PyanNet = None
    EEND = None

try:
    from models.losses import DiarizationLoss
except ImportError:
    DiarizationLoss = None

try:
    from training.trainer import Trainer, train_model
except ImportError:
    Trainer = None
    train_model = None

try:
    from utils.helpers import set_seed, print_gpu_info, setup_logging, AverageMeter
except ImportError:
    set_seed = None
    print_gpu_info = None
    setup_logging = None
    AverageMeter = None


# ============================================================================
# FALLBACK IMPLEMENTATIONS (if modules not available)
# ============================================================================

def _set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU Information")
        print("=" * 60)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print("=" * 60)
    else:
        print("WARNING: CUDA not available, using CPU!")


def _setup_logging(output_dir: Path = None, log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    handlers = [logging.StreamHandler()]
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / 'train.log'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


# Use imported or fallback functions
if set_seed is None:
    set_seed = _set_seed
if print_gpu_info is None:
    print_gpu_info = _print_gpu_info
if setup_logging is None:
    setup_logging = _setup_logging


# ============================================================================
# SIMPLE MODEL (fallback if project model not available)
# ============================================================================

class SimpleConvBlock(nn.Module):
    """Simple convolutional block."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_ch),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block."""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, key_padding_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleDiarizationModel(nn.Module):
    """Simple diarization model for training."""
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_speakers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_speakers = max_speakers
        self.num_speakers = max_speakers
        self.n_mels = n_mels
        self.sample_rate = 16000
        
        # CNN frontend
        self.frontend = nn.Sequential(
            SimpleConvBlock(n_mels, 128),
            nn.MaxPool1d(2),
            SimpleConvBlock(128, 256),
            SimpleConvBlock(256, hidden_dim),
        )
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 5000, hidden_dim) * 0.02)
        
        # Transformer
        self.transformer = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_speakers),
        )
        
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, lengths=None):
        batch_size, _, orig_time = x.shape
        
        # CNN frontend
        x = self.frontend(x)
        x = x.transpose(1, 2)
        
        # Positional encoding
        seq_len = x.size(1)
        x = x + self.pos_enc[:, :seq_len, :]
        
        # Padding mask
        mask = None
        if lengths is not None:
            lengths_down = lengths // 2
            mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths_down[:, None]
        
        # Transformer
        for layer in self.transformer:
            x = layer(x, mask)
        
        # Output heads
        speakers = torch.sigmoid(self.speaker_head(x))
        vad = torch.sigmoid(self.vad_head(x).squeeze(-1))
        
        # Upsample
        speakers = nn.functional.interpolate(
            speakers.transpose(1, 2), size=orig_time, mode='linear', align_corners=False
        ).transpose(1, 2)
        
        vad = nn.functional.interpolate(
            vad.unsqueeze(1), size=orig_time, mode='linear', align_corners=False
        ).squeeze(1)
        
        return {'speakers': speakers, 'vad': vad}


# ============================================================================
# SIMPLE DATASET (fallback if project dataset not available)
# ============================================================================

class SimpleDiarizationDataset(torch.utils.data.Dataset):
    """Simple dataset for diarization training."""
    
    def __init__(
        self,
        audio_paths: List[str],
        rttm_paths: List[str],
        sample_rate: int = 16000,
        segment_duration: float = 3.0,
        max_speakers: int = 4,
        n_mels: int = 80,
        augment: bool = False,
    ):
        import torchaudio
        
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.max_speakers = max_speakers
        self.n_mels = n_mels
        self.augment = augment
        
        # Caches
        self._rttm_cache = {}
        self._speaker_map_cache = {}
        
        # Index segments
        self.segments = []
        self._index_segments(audio_paths, rttm_paths)
        
        print(f"  Indexed {len(self.segments)} segments from {len(audio_paths)} files")
        
        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
        )
        
        # Augmentation
        if augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=100)
        else:
            self.freq_mask = None
            self.time_mask = None
    
    def _index_segments(self, audio_paths, rttm_paths):
        import torchaudio
        
        segment_step = self.segment_duration / 2
        
        for audio_path, rttm_path in zip(audio_paths, rttm_paths):
            if not Path(audio_path).exists() or not Path(rttm_path).exists():
                continue
            
            try:
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
            except Exception as e:
                print(f"Error reading {audio_path}: {e}")
                continue
            
            rttm_segments = self._parse_rttm(rttm_path)
            file_id = Path(audio_path).stem
            start = 0.0
            
            while start + self.segment_duration <= duration:
                end = start + self.segment_duration
                
                has_speech = any(
                    seg['start'] < end and seg['end'] > start
                    for seg in rttm_segments
                )
                
                if has_speech:
                    self.segments.append({
                        'audio_path': audio_path,
                        'rttm_path': rttm_path,
                        'file_id': file_id,
                        'start': start,
                        'end': end,
                    })
                
                start += segment_step
    
    def _parse_rttm(self, rttm_path: str):
        if rttm_path in self._rttm_cache:
            return self._rttm_cache[rttm_path]
        
        segments = []
        try:
            with open(rttm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == 'SPEAKER':
                        segments.append({
                            'start': float(parts[3]),
                            'end': float(parts[3]) + float(parts[4]),
                            'speaker': parts[7]
                        })
        except Exception as e:
            print(f"Error parsing {rttm_path}: {e}")
        
        self._rttm_cache[rttm_path] = segments
        return segments
    
    def _get_speaker_map(self, rttm_path: str):
        if rttm_path in self._speaker_map_cache:
            return self._speaker_map_cache[rttm_path]
        
        segments = self._parse_rttm(rttm_path)
        speakers = sorted(set(seg['speaker'] for seg in segments))
        speaker_map = {spk: i for i, spk in enumerate(speakers[:self.max_speakers])}
        
        self._speaker_map_cache[rttm_path] = speaker_map
        return speaker_map
    
    def _load_audio(self, path: str, start: float, end: float):
        import torchaudio
        
        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            start_frame = int(start * sr)
            num_frames = int((end - start) * sr)
            
            waveform, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)
            
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform
        except Exception as e:
            print(f"Error loading {path}: {e}")
            num_samples = int(self.segment_duration * self.sample_rate)
            return torch.zeros(1, num_samples)
    
    def _create_labels(self, rttm_path, seg_start, seg_end, num_frames):
        rttm_segments = self._parse_rttm(rttm_path)
        speaker_map = self._get_speaker_map(rttm_path)
        
        labels = torch.zeros(num_frames, self.max_speakers)
        vad = torch.zeros(num_frames)
        
        seg_duration = seg_end - seg_start
        
        for seg in rttm_segments:
            if seg['start'] >= seg_end or seg['end'] <= seg_start:
                continue
            
            if seg['speaker'] not in speaker_map:
                continue
            
            spk_idx = speaker_map[seg['speaker']]
            
            rel_start = max(0, (seg['start'] - seg_start) / seg_duration)
            rel_end = min(1, (seg['end'] - seg_start) / seg_duration)
            
            start_frame = int(rel_start * num_frames)
            end_frame = int(rel_end * num_frames)
            
            labels[start_frame:end_frame, spk_idx] = 1.0
            vad[start_frame:end_frame] = 1.0
        
        return labels, vad
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        seg = self.segments[idx]
        
        waveform = self._load_audio(seg['audio_path'], seg['start'], seg['end'])
        
        # Augmentation
        if self.augment and torch.rand(1).item() < 0.3:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Mel features
        mel = self.mel_transform(waveform)
        mel = (mel + 1e-6).log()
        
        # SpecAugment
        if self.augment and self.freq_mask is not None:
            if torch.rand(1).item() < 0.5:
                mel = self.freq_mask(mel)
            if torch.rand(1).item() < 0.5:
                mel = self.time_mask(mel)
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Labels
        num_frames = mel.shape[-1]
        labels, vad = self._create_labels(seg['rttm_path'], seg['start'], seg['end'], num_frames)
        
        return {
            'features': mel.squeeze(0),
            'labels': labels,
            'vad': vad,
        }


def simple_collate_fn(batch):
    """Simple collate function."""
    max_len = max(item['features'].shape[-1] for item in batch)
    
    features, labels, vad, lengths = [], [], [], []
    
    for item in batch:
        feat = item['features']
        lab = item['labels']
        v = item['vad']
        
        orig_len = feat.shape[-1]
        lengths.append(orig_len)
        
        if feat.shape[-1] < max_len:
            pad = max_len - feat.shape[-1]
            feat = torch.nn.functional.pad(feat, (0, pad))
            lab = torch.nn.functional.pad(lab, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, pad))
        
        features.append(feat)
        labels.append(lab)
        vad.append(v)
    
    return {
        'features': torch.stack(features),
        'labels': torch.stack(labels),
        'vad': torch.stack(vad),
        'lengths': torch.tensor(lengths),
    }


# ============================================================================
# SIMPLE TRAINER (fallback if project trainer not available)
# ============================================================================

class SimpleTrainer:
    """Simple trainer for diarization."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        output_dir: Path,
        device: torch.device,
        logger: logging.Logger,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.device = device
        self.logger = logger
        self.num_epochs = num_epochs
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
        )
        
        # AMP
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Checkpoints
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_loss(self, outputs, labels, vad, lengths):
        """Compute training loss."""
        pred_speakers = outputs['speakers']
        pred_vad = outputs['vad']
        
        batch_size, time_steps, _ = pred_speakers.shape
        
        # Mask
        mask = torch.arange(time_steps, device=lengths.device)[None, :] < lengths[:, None]
        mask = mask.float()
        
        eps = 1e-7
        
        # Speaker loss
        pred_speakers = pred_speakers.clamp(eps, 1 - eps)
        speaker_loss = -(labels * torch.log(pred_speakers) + 
                        (1 - labels) * torch.log(1 - pred_speakers))
        speaker_loss = (speaker_loss.mean(-1) * mask).sum() / mask.sum()
        
        # VAD loss
        pred_vad = pred_vad.clamp(eps, 1 - eps)
        vad_loss = -(vad * torch.log(pred_vad) + 
                    (1 - vad) * torch.log(1 - pred_vad))
        vad_loss = (vad_loss * mask).sum() / mask.sum()
        
        return speaker_loss + 0.5 * vad_loss, speaker_loss, vad_loss
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            vad = batch['vad'].to(self.device, non_blocking=True)
            lengths = batch['lengths'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(features, lengths)
                loss, spk_loss, vad_loss = self._compute_loss(outputs, labels, vad, lengths)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            self.global_step += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Validate."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        from tqdm import tqdm
        for batch in tqdm(self.val_loader, desc="Validating"):
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            vad = batch['vad'].to(self.device, non_blocking=True)
            lengths = batch['lengths'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(features, lengths)
                loss, _, _ = self._compute_loss(outputs, labels, vad, lengths)
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
        self.logger.info(f"Saved: {path}")
    
    def train(self):
        """Full training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                val_loss = val_metrics.get('val_loss', float('inf'))
                self.logger.info(f"  Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    self.logger.info("  ✓ New best model!")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete! Time: {total_time/60:.1f} minutes")
        self.logger.info("=" * 60)
        
        self.save_checkpoint('final_model.pt')


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train speaker diarization model")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/aishell4", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (YAML or JSON)")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers")
    
    # Model
    parser.add_argument("--max-speakers", type=int, default=4, help="Maximum speakers")
    parser.add_argument("--segment-duration", type=float, default=3.0, help="Segment duration (seconds)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bands")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--use-project-modules", action="store_true", help="Use project modules if available")
    
    return parser.parse_args()


def find_files(data_dir: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Find audio and RTTM files."""
    audio_dir = data_dir / "audio"
    rttm_dir = data_dir / "rttm"
    
    train_audio, train_rttm = [], []
    val_audio, val_rttm = [], []
    
    # Load from file lists
    if (data_dir / "train.txt").exists():
        with open(data_dir / "train.txt") as f:
            train_files = [line.strip() for line in f if line.strip()]
        for f in train_files:
            audio_path = audio_dir / f"{f}.wav"
            rttm_path = rttm_dir / f"{f}.rttm"
            if audio_path.exists() and rttm_path.exists():
                train_audio.append(str(audio_path))
                train_rttm.append(str(rttm_path))
    
    if (data_dir / "val.txt").exists():
        with open(data_dir / "val.txt") as f:
            val_files = [line.strip() for line in f if line.strip()]
        for f in val_files:
            audio_path = audio_dir / f"{f}.wav"
            rttm_path = rttm_dir / f"{f}.rttm"
            if audio_path.exists() and rttm_path.exists():
                val_audio.append(str(audio_path))
                val_rttm.append(str(rttm_path))
    
    # Fallback: scan directories
    if not train_audio and audio_dir.exists():
        print("No train.txt found, scanning audio directory...")
        audio_files = sorted(audio_dir.glob("*.wav"))
        
        for audio_path in audio_files:
            rttm_path = rttm_dir / f"{audio_path.stem}.rttm"
            if rttm_path.exists():
                train_audio.append(str(audio_path))
                train_rttm.append(str(rttm_path))
        
        # Split 90/10
        if train_audio:
            split_idx = int(len(train_audio) * 0.9)
            val_audio = train_audio[split_idx:]
            val_rttm = train_rttm[split_idx:]
            train_audio = train_audio[:split_idx]
            train_rttm = train_rttm[:split_idx]
    
    return train_audio, train_rttm, val_audio, val_rttm


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Print GPU info
    print_gpu_info()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"diarization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Find data files
    data_dir = Path(args.data_dir)
    train_audio, train_rttm, val_audio, val_rttm = find_files(data_dir)
    
    logger.info(f"Train files: {len(train_audio)}")
    logger.info(f"Val files: {len(val_audio)}")
    
    if not train_audio:
        logger.error("No training files found!")
        logger.error(f"Expected structure:")
        logger.error(f"  {data_dir}/audio/*.wav")
        logger.error(f"  {data_dir}/rttm/*.rttm")
        logger.error(f"  {data_dir}/train.txt (optional)")
        return
    
    # Save config
    config_dict = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_speakers': args.max_speakers,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'segment_duration': args.segment_duration,
        'n_mels': args.n_mels,
        'seed': args.seed,
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {json.dumps(config_dict, indent=2)}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Try to use project modules, fall back to simple implementations
    use_project = args.use_project_modules and DiarizationDataset is not None
    
    if use_project:
        logger.info("Using project DiarizationDataset")
        try:
            # Create config object if available
            if DiarizationConfig is not None:
                config = get_aishell4_config() if get_aishell4_config else DiarizationConfig()
                config.training.batch_size = args.batch_size
                config.training.num_epochs = args.num_epochs
                config.training.learning_rate = args.learning_rate
                config.segmentation.num_speakers = args.max_speakers
                config.segmentation.transformer_hidden_dim = args.hidden_dim
                
                train_dataset = DiarizationDataset(
                    audio_paths=train_audio,
                    rttm_paths=train_rttm,
                    config=config,
                    augment=True,
                )
            else:
                train_dataset = DiarizationDataset(
                    audio_paths=train_audio,
                    rttm_paths=train_rttm,
                    segment_duration=args.segment_duration,
                    max_speakers=args.max_speakers,
                    augment=True,
                )
        except Exception as e:
            logger.warning(f"Failed to use project dataset: {e}")
            use_project = False
    
    if not use_project:
        logger.info("Using SimpleDiarizationDataset")
        train_dataset = SimpleDiarizationDataset(
            audio_paths=train_audio,
            rttm_paths=train_rttm,
            segment_duration=args.segment_duration,
            max_speakers=args.max_speakers,
            n_mels=args.n_mels,
            augment=True,
        )
    
    val_dataset = None
    if val_audio:
        if use_project:
            try:
                if DiarizationConfig is not None:
                    val_dataset = DiarizationDataset(
                        audio_paths=val_audio,
                        rttm_paths=val_rttm,
                        config=config,
                        augment=False,
                    )
                else:
                    val_dataset = DiarizationDataset(
                        audio_paths=val_audio,
                        rttm_paths=val_rttm,
                        segment_duration=args.segment_duration,
                        max_speakers=args.max_speakers,
                        augment=False,
                    )
            except Exception:
                use_project = False
        
        if not use_project:
            val_dataset = SimpleDiarizationDataset(
                audio_paths=val_audio,
                rttm_paths=val_rttm,
                segment_duration=args.segment_duration,
                max_speakers=args.max_speakers,
                n_mels=args.n_mels,
                augment=False,
            )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        logger.error("No training samples! Check audio and RTTM files.")
        return
    
    # Collate function
    collate = collate_fn if (use_project and collate_fn is not None) else simple_collate_fn
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
    
    # Create model
    logger.info("Creating model...")
    
    use_project_model = args.use_project_modules and SpeakerDiarizationModel is not None
    
    if use_project_model:
        try:
            model = SpeakerDiarizationModel(
                num_speakers=args.max_speakers,
                n_mels=args.n_mels,
            )
            logger.info("Using SpeakerDiarizationModel")
        except Exception as e:
            logger.warning(f"Failed to create project model: {e}")
            use_project_model = False
    
    if not use_project_model:
        model = SimpleDiarizationModel(
            n_mels=args.n_mels,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_speakers=args.max_speakers,
        )
        logger.info("Using SimpleDiarizationModel")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    use_project_trainer = args.use_project_modules and Trainer is not None
    
    if use_project_trainer:
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                output_dir=str(output_dir),
                device=device,
                use_tensorboard=not args.no_tensorboard,
            )
            logger.info("Using project Trainer")
        except Exception as e:
            logger.warning(f"Failed to create project trainer: {e}")
            use_project_trainer = False
    
    if not use_project_trainer:
        trainer = SimpleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            device=device,
            logger=logger,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        logger.info("Using SimpleTrainer")
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from {args.resume}")
    
    # Train!
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()