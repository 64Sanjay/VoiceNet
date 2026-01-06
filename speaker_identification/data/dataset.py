# speaker_identification/data/dataset.py
"""
Dataset classes for speaker identification training and evaluation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import random
from collections import defaultdict

from .preprocessing import AudioPreprocessor
from .augmentation import AudioAugmentor


class SpeakerDataset(Dataset):
    """
    Dataset for speaker identification with multi-view augmentation.
    
    Supports VoxTube, JVS, CallHome, and VoxConverse datasets.
    Each sample returns: (original_features, noise_aug_features, time_aug_features, speaker_id)
    """
    
    def __init__(
        self,
        data_path: str,
        preprocessor: AudioPreprocessor,
        augmentor: Optional[AudioAugmentor] = None,
        split: str = "train",
        return_augmented: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory or manifest file
            preprocessor: Audio preprocessor instance
            augmentor: Audio augmentor instance (None for eval)
            split: Dataset split ('train', 'val', 'test')
            return_augmented: Whether to return augmented views
        """
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split
        self.return_augmented = return_augmented and (split == "train")
        
        # Load dataset
        self.samples = self._load_samples()
        
        # Create speaker ID mapping
        self.speaker_to_idx = self._create_speaker_mapping()
        self.idx_to_speaker = {v: k for k, v in self.speaker_to_idx.items()}
        
        # Create index for each speaker (for triplet mining)
        self.speaker_to_samples = self._create_speaker_index()
    
    def _load_samples(self) -> List[Dict]:
        """
        Load samples from dataset.
        
        Returns:
            List of sample dictionaries with 'audio_path' and 'speaker_id'
        """
        samples = []
        
        # Check for manifest file
        manifest_path = self.data_path / f"{self.split}_manifest.csv"
        
        if manifest_path.exists():
            # Load from manifest CSV
            df = pd.read_csv(manifest_path)
            for _, row in df.iterrows():
                samples.append({
                    'audio_path': row['audio_path'],
                    'speaker_id': row['speaker_id'],
                    'language': row.get('language', 'unknown')
                })
        else:
            # Load from directory structure: data_path/speaker_id/*.wav
            for speaker_dir in sorted(self.data_path.iterdir()):
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name
                    for audio_file in speaker_dir.glob("*.wav"):
                        samples.append({
                            'audio_path': str(audio_file),
                            'speaker_id': speaker_id,
                            'language': 'unknown'
                        })
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def _create_speaker_mapping(self) -> Dict[str, int]:
        """Create mapping from speaker ID to index."""
        unique_speakers = sorted(set(s['speaker_id'] for s in self.samples))
        return {spk: idx for idx, spk in enumerate(unique_speakers)}
    
    def _create_speaker_index(self) -> Dict[int, List[int]]:
        """Create index mapping speaker ID to sample indices."""
        speaker_index = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            spk_idx = self.speaker_to_idx[sample['speaker_id']]
            speaker_index[spk_idx].append(idx)
        return dict(speaker_index)
    
    @property
    def num_speakers(self) -> int:
        """Return number of unique speakers."""
        return len(self.speaker_to_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(
        self,
        idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If training: (original, noise_aug, time_aug, speaker_idx)
            If eval: (original, speaker_idx)
        """
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        speaker_id = sample['speaker_id']
        speaker_idx = self.speaker_to_idx[speaker_id]
        
        # Load and preprocess audio
        waveform, _ = self.preprocessor.load_audio(audio_path)
        
        if self.return_augmented and self.augmentor is not None:
            # Generate augmented views for training
            # As per Algorithm 1: x_i^(n) and x_i^(t)
            noise_aug, time_aug = self.augmentor.augment(waveform)
            
            # Extract features for all views
            original_features = self.preprocessor.preprocess_waveform(waveform)
            noise_features = self.preprocessor.preprocess_waveform(noise_aug)
            time_features = self.preprocessor.preprocess_waveform(time_aug)
            
            return original_features, noise_features, time_features, speaker_idx
        else:
            # Only return original for evaluation
            features = self.preprocessor.preprocess_waveform(waveform)
            return features, speaker_idx


class SpeakerBatchSampler(Sampler):
    """
    Batch sampler ensuring each batch has multiple samples per speaker.
    This is important for online hard triplet mining.
    """
    
    def __init__(
        self,
        dataset: SpeakerDataset,
        batch_size: int = 16,
        samples_per_speaker: int = 2,
        drop_last: bool = True
    ):
        """
        Initialize batch sampler.
        
        Args:
            dataset: Speaker dataset
            batch_size: Batch size
            samples_per_speaker: Number of samples per speaker in each batch
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_speaker = samples_per_speaker
        self.drop_last = drop_last
        
        # Calculate speakers per batch
        self.speakers_per_batch = batch_size // samples_per_speaker
        
        # Get speakers with enough samples
        self.valid_speakers = [
            spk for spk, indices in dataset.speaker_to_samples.items()
            if len(indices) >= samples_per_speaker
        ]
    
    def __iter__(self):
        """Generate batches."""
        # Shuffle speakers
        speakers = self.valid_speakers.copy()
        random.shuffle(speakers)
        
        batches = []
        current_batch = []
        
        for speaker in speakers:
            # Get samples for this speaker
            speaker_samples = self.dataset.speaker_to_samples[speaker].copy()
            random.shuffle(speaker_samples)
            
            # Take samples_per_speaker samples
            selected = speaker_samples[:self.samples_per_speaker]
            current_batch.extend(selected)
            
            # Check if batch is complete
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch[:self.batch_size])
                current_batch = current_batch[self.batch_size:]
        
        # Handle remaining samples
        if len(current_batch) >= self.batch_size:
            batches.append(current_batch[:self.batch_size])
        elif not self.drop_last and len(current_batch) > 0:
            batches.append(current_batch)
        
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        total_samples = sum(
            min(len(self.dataset.speaker_to_samples[spk]), self.samples_per_speaker)
            for spk in self.valid_speakers
        )
        n_batches = total_samples // self.batch_size
        return n_batches


def create_dataloader(
    dataset: SpeakerDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    use_speaker_sampler: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for speaker dataset.
    
    Args:
        dataset: Speaker dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if use_speaker_sampler)
        use_speaker_sampler: Use SpeakerBatchSampler for triplet mining
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    if use_speaker_sampler:
        sampler = SpeakerBatchSampler(
            dataset,
            batch_size=batch_size,
            samples_per_speaker=2
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )