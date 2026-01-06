# speaker_verification/data/dataset.py
"""
Dataset classes for CAM++ speaker verification.
Supports VoxCeleb and CN-Celeb datasets.
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict

from .preprocessing import AudioPreprocessor
from .augmentation import SpeakerAugmentor


class SpeakerVerificationDataset(Dataset):
    """
    Dataset for speaker verification training.
    
    From the paper:
    "3s-long samples are randomly cropped from each audio to construct 
    the training minibatches."
    """
    
    def __init__(
        self,
        data_path: str,
        preprocessor: AudioPreprocessor,
        augmentor: Optional[SpeakerAugmentor] = None,
        sample_rate: int = 16000,
        duration: float = 3.0,  # 3 seconds as per paper
        min_duration: float = 2.0,
        train: bool = True
    ):
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.sample_rate = sample_rate
        self.duration = duration
        self.min_duration = min_duration
        self.train = train
        
        # Number of samples for training duration
        self.num_samples = int(duration * sample_rate)
        self.min_samples = int(min_duration * sample_rate)
        
        # Load dataset
        self.samples = []
        self.speaker_to_idx = {}
        self.speaker_to_files = defaultdict(list)
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from directory structure."""
        
        # Check for manifest file
        manifest_path = self.data_path / "train_manifest.csv"
        
        if manifest_path.exists():
            self._load_from_manifest(manifest_path)
        else:
            self._load_from_directory()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.speaker_to_idx)} speakers")
    
    def _load_from_manifest(self, manifest_path: Path):
        """Load from CSV manifest."""
        df = pd.read_csv(manifest_path)
        
        for idx, row in df.iterrows():
            speaker_id = str(row['speaker_id'])
            audio_path = row['audio_path']
            
            if speaker_id not in self.speaker_to_idx:
                self.speaker_to_idx[speaker_id] = len(self.speaker_to_idx)
            
            self.samples.append({
                'audio_path': audio_path,
                'speaker_id': speaker_id,
                'speaker_idx': self.speaker_to_idx[speaker_id]
            })
            
            self.speaker_to_files[speaker_id].append(audio_path)
    
    def _load_from_directory(self):
        """Load from directory structure: data_path/speaker_id/*.wav"""
        
        for speaker_dir in sorted(self.data_path.iterdir()):
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                
                if speaker_id not in self.speaker_to_idx:
                    self.speaker_to_idx[speaker_id] = len(self.speaker_to_idx)
                
                for audio_file in speaker_dir.glob("*.wav"):
                    self.samples.append({
                        'audio_path': str(audio_file),
                        'speaker_id': speaker_id,
                        'speaker_idx': self.speaker_to_idx[speaker_id]
                    })
                    self.speaker_to_files[speaker_id].append(str(audio_file))
    
    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        speaker_idx = sample['speaker_idx']
        
        # Load audio
        waveform, _ = self.preprocessor.load_audio(audio_path)
        
        # Apply waveform augmentation (training only)
        if self.train and self.augmentor is not None:
            waveform = self.augmentor.augment_waveform(waveform)
        
        # Crop to fixed duration (BOTH training and validation need fixed length for batching)
        waveform = self._crop_to_fixed_length(waveform)
        
        # Extract Fbank features
        features = self.preprocessor.extract_features(waveform)
        
        # Apply spectrogram augmentation (training only)
        if self.train and self.augmentor is not None:
            features = self.augmentor.augment_spectrogram(features)
        
        # Normalize
        features = self.preprocessor.normalize_features(features)
        
        # Ensure tensor is contiguous and return a fresh copy
        features = features.clone().contiguous()
        
        return features, speaker_idx
    
    def _crop_to_fixed_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Crop or pad waveform to fixed duration.
        
        For training: random crop
        For validation: center crop
        """
        current_length = len(waveform)
        
        if current_length >= self.num_samples:
            if self.train:
                # Random start position for training
                start = random.randint(0, current_length - self.num_samples)
            else:
                # Center crop for validation
                start = (current_length - self.num_samples) // 2
            return waveform[start:start + self.num_samples]
        elif current_length >= self.min_samples:
            # Pad to target length
            padding = self.num_samples - current_length
            return F.pad(waveform, (0, padding))
        else:
            # Too short, repeat and crop
            repeats = self.num_samples // current_length + 1
            waveform = waveform.repeat(repeats)
            if self.train:
                start = random.randint(0, len(waveform) - self.num_samples)
            else:
                start = 0
            return waveform[start:start + self.num_samples]


class VerificationTrialDataset(Dataset):
    """
    Dataset for verification trials (evaluation).
    Returns pairs of embeddings for scoring.
    """
    
    def __init__(
        self,
        trial_file: str,
        data_path: str,
        preprocessor: AudioPreprocessor,
        max_duration: float = 10.0,  # Max duration for evaluation
        sample_rate: int = 16000
    ):
        self.trial_file = trial_file
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Load trials
        self.trials = self._load_trials()
    
    def _load_trials(self) -> List[Dict]:
        """Load verification trials."""
        trials = []
        
        with open(self.trial_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    audio1 = parts[1]
                    audio2 = parts[2]
                    
                    trials.append({
                        'label': label,
                        'audio1': audio1,
                        'audio2': audio2
                    })
        
        return trials
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trial = self.trials[idx]
        
        # Load and process both audio files
        features1 = self._process_audio(trial['audio1'])
        features2 = self._process_audio(trial['audio2'])
        
        return features1, features2, trial['label']
    
    def _process_audio(self, audio_path: str) -> torch.Tensor:
        """Load and process audio file."""
        full_path = self.data_path / audio_path
        
        waveform, _ = self.preprocessor.load_audio(str(full_path))
        
        # Limit to max duration
        if len(waveform) > self.max_samples:
            # Center crop
            start = (len(waveform) - self.max_samples) // 2
            waveform = waveform[start:start + self.max_samples]
        
        features = self.preprocessor.extract_features(waveform)
        features = self.preprocessor.normalize_features(features)
        
        # Ensure contiguous
        features = features.clone().contiguous()
        
        return features


def collate_fn(batch):
    """
    Custom collate function to handle potential edge cases.
    Ensures all tensors are properly cloned and contiguous.
    """
    features = []
    labels = []
    
    for feat, label in batch:
        # Ensure each tensor is a fresh, contiguous copy
        if isinstance(feat, torch.Tensor):
            features.append(feat.clone().contiguous())
        else:
            features.append(torch.tensor(feat).contiguous())
        
        if isinstance(label, torch.Tensor):
            labels.append(label.clone())
        else:
            labels.append(torch.tensor(label))
    
    # Stack into batches
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return features, labels


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader for speaker verification."""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
        collate_fn=collate_fn  # Use custom collate function
    )