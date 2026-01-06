"""
Dataset classes for speaker diarization.
Supports AISHELL-4, AMI, VoxConverse, and custom datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
import random
from dataclasses import dataclass

from ..utils.rttm_utils import RTTMReader, RTTMSegment, segments_to_frames
from ..utils.audio_utils import AudioProcessor
from .augmentation import AudioAugmentor, create_augmentation_pipeline
from .preprocessing import FeatureExtractor


@dataclass
class DiarizationSample:
    """Represents a single diarization sample."""
    audio_path: str
    rttm_path: Optional[str]
    duration: float
    file_id: str
    num_speakers: Optional[int] = None


class DiarizationDataset(Dataset):
    """
    Base dataset class for speaker diarization.
    
    Provides frame-level speaker labels for training
    end-to-end neural diarization models.
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        rttm_paths: List[str],
        sample_rate: int = 16000,
        segment_duration: float = 5.0,
        frame_duration: float = 0.016,  # 16ms frames
        max_speakers: int = 4,
        feature_extractor: Optional[FeatureExtractor] = None,
        augmentor: Optional[AudioAugmentor] = None,
        return_waveform: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            audio_paths: List of audio file paths
            rttm_paths: List of RTTM annotation file paths
            sample_rate: Audio sample rate
            segment_duration: Duration of segments in seconds
            frame_duration: Duration of each frame in seconds
            max_speakers: Maximum number of speakers per segment
            feature_extractor: Feature extraction module
            augmentor: Audio augmentation module
            return_waveform: Whether to return raw waveform
        """
        self.audio_paths = audio_paths
        self.rttm_paths = rttm_paths
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.frame_duration = frame_duration
        self.max_speakers = max_speakers
        self.return_waveform = return_waveform
        
        # Audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Feature extractor
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sample_rate=sample_rate,
            feature_type="mel",
            n_mels=80,
        )
        
        # Augmentor
        self.augmentor = augmentor
        
        # Load all RTTM annotations
        self.rttm_reader = RTTMReader()
        self.annotations = {}
        self._load_annotations()
        
        # Create samples
        self.samples = self._create_samples()
    
    def _load_annotations(self):
        """Load all RTTM annotations."""
        for rttm_path in self.rttm_paths:
            segments = self.rttm_reader.read(rttm_path)
            self.annotations.update(segments)
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples with segment information."""
        samples = []
        
        for audio_path in self.audio_paths:
            audio_path = Path(audio_path)
            file_id = audio_path.stem
            
            # Get audio duration
            try:
                duration = self.audio_processor.get_duration(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue
            
            # Get annotations for this file
            file_annotations = self.annotations.get(file_id, [])
            
            if not file_annotations:
                continue
            
            # Create segments
            segment_samples = int(self.segment_duration * self.sample_rate)
            hop_samples = segment_samples // 2  # 50% overlap
            
            num_segments = max(1, int((duration - self.segment_duration) / (hop_samples / self.sample_rate)) + 1)
            
            for i in range(num_segments):
                start_time = i * (hop_samples / self.sample_rate)
                end_time = start_time + self.segment_duration
                
                if end_time > duration:
                    break
                
                # Get speakers active in this segment
                segment_annotations = [
                    ann for ann in file_annotations
                    if ann.start < end_time and ann.end > start_time
                ]
                
                if not segment_annotations:
                    continue
                
                samples.append({
                    'audio_path': str(audio_path),
                    'file_id': file_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'annotations': segment_annotations,
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
                - features: [C, T] mel spectrogram features
                - labels: [T', max_speakers] frame-level speaker labels
                - waveform: [1, S] raw waveform (if return_waveform=True)
        """
        sample = self.samples[idx]
        
        # Load audio segment
        waveform, _ = self.audio_processor.load(
            sample['audio_path'],
            start=sample['start_time'],
            duration=self.segment_duration,
        )
        
        # Apply augmentation
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        
        # Extract features
        features = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)
        
        # Create frame-level labels
        labels = self._create_labels(
            sample['annotations'],
            sample['start_time'],
            sample['end_time'],
            features.shape[-1],
        )
        
        result = {
            'features': features,
            'labels': labels,
            'file_id': sample['file_id'],
            'start_time': sample['start_time'],
        }
        
        if self.return_waveform:
            result['waveform'] = waveform
        
        return result
    
    def _create_labels(
        self,
        annotations: List[RTTMSegment],
        start_time: float,
        end_time: float,
        num_frames: int,
    ) -> torch.Tensor:
        """Create frame-level labels from annotations."""
        # Get unique speakers in this segment
        speakers = sorted(set(ann.speaker_id for ann in annotations))
        speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
        
        # Initialize labels
        labels = torch.zeros(num_frames, self.max_speakers)
        
        # Frame duration in seconds
        frame_dur = self.segment_duration / num_frames
        
        for ann in annotations:
            if ann.speaker_id not in speaker_to_idx:
                continue
            
            spk_idx = speaker_to_idx[ann.speaker_id]
            if spk_idx >= self.max_speakers:
                continue
            
            # Convert to frame indices (relative to segment start)
            rel_start = max(0, ann.start - start_time)
            rel_end = min(self.segment_duration, ann.end - start_time)
            
            start_frame = int(rel_start / frame_dur)
            end_frame = int(rel_end / frame_dur)
            
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            
            labels[start_frame:end_frame, spk_idx] = 1.0
        
        return labels


class AISHELL4Dataset(DiarizationDataset):
    """
    AISHELL-4 dataset for speaker diarization.
    
    AISHELL-4 is a sizable real-recorded Mandarin speech dataset
    collected by 8-channel circular microphone array for speech
    processing in conference scenarios.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",  # train, test
        channel: int = 0,  # Which channel to use (0-7)
        **kwargs,
    ):
        """
        Initialize AISHELL-4 dataset.
        
        Args:
            data_root: Path to AISHELL-4 dataset root
            split: Dataset split (train/test)
            channel: Microphone channel to use
            **kwargs: Additional arguments for parent class
        """
        self.data_root = Path(data_root)
        self.split = split
        self.channel = channel
        
        # Get audio and RTTM paths
        audio_paths, rttm_paths = self._get_file_lists()
        
        super().__init__(
            audio_paths=audio_paths,
            rttm_paths=rttm_paths,
            **kwargs,
        )
    
    def _get_file_lists(self) -> Tuple[List[str], List[str]]:
        """Get lists of audio and RTTM files."""
        split_dir = self.data_root / self.split
        
        audio_paths = []
        rttm_paths = []
        
        # AISHELL-4 structure: {split}/{session_id}/{session_id}.wav
        for session_dir in sorted(split_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            
            # Audio file (use specified channel)
            audio_file = session_dir / f"{session_id}_channel{self.channel}.wav"
            if not audio_file.exists():
                # Try without channel specification
                audio_file = session_dir / f"{session_id}.wav"
            
            if not audio_file.exists():
                continue
            
            # RTTM file
            rttm_file = session_dir / f"{session_id}.rttm"
            if not rttm_file.exists():
                # Try in separate rttm directory
                rttm_file = self.data_root / "rttm" / f"{session_id}.rttm"
            
            if audio_file.exists():
                audio_paths.append(str(audio_file))
                if rttm_file.exists():
                    rttm_paths.append(str(rttm_file))
        
        return audio_paths, rttm_paths


class AMIDataset(DiarizationDataset):
    """
    AMI Meeting Corpus dataset for speaker diarization.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        mic_type: str = "Mix-Headset",  # Mix-Headset, Array1-01, etc.
        **kwargs,
    ):
        """
        Initialize AMI dataset.
        
        Args:
            data_root: Path to AMI corpus root
            split: Dataset split
            mic_type: Microphone type
            **kwargs: Additional arguments
        """
        self.data_root = Path(data_root)
        self.split = split
        self.mic_type = mic_type
        
        audio_paths, rttm_paths = self._get_file_lists()
        
        super().__init__(
            audio_paths=audio_paths,
            rttm_paths=rttm_paths,
            **kwargs,
        )
    
    def _get_file_lists(self) -> Tuple[List[str], List[str]]:
        """Get lists of audio and RTTM files."""
        # Load split file
        split_file = self.data_root / f"{self.split}.txt"
        
        if not split_file.exists():
            # Use all files in directory
            audio_dir = self.data_root / "audio"
            rttm_dir = self.data_root / "rttm"
            
            audio_paths = sorted(audio_dir.glob("*.wav"))
            rttm_paths = sorted(rttm_dir.glob("*.rttm"))
            
            return [str(p) for p in audio_paths], [str(p) for p in rttm_paths]
        
        with open(split_file, 'r') as f:
            meeting_ids = [line.strip() for line in f if line.strip()]
        
        audio_paths = []
        rttm_paths = []
        
        for meeting_id in meeting_ids:
            audio_file = self.data_root / "audio" / f"{meeting_id}.{self.mic_type}.wav"
            rttm_file = self.data_root / "rttm" / f"{meeting_id}.rttm"
            
            if audio_file.exists():
                audio_paths.append(str(audio_file))
            if rttm_file.exists():
                rttm_paths.append(str(rttm_file))
        
        return audio_paths, rttm_paths


class InferenceDataset(Dataset):
    """
    Dataset for inference on new audio files.
    Does not require RTTM annotations.
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        sample_rate: int = 16000,
        segment_duration: float = 5.0,
        segment_step: float = 2.5,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        """
        Initialize inference dataset.
        
        Args:
            audio_paths: List of audio file paths
            sample_rate: Audio sample rate
            segment_duration: Duration of segments
            segment_step: Step between segments
            feature_extractor: Feature extraction module
        """
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_step = segment_step
        
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sample_rate=sample_rate
        )
        
        # Create segments
        self.segments = self._create_segments()
    
    def _create_segments(self) -> List[Dict]:
        """Create segments for all audio files."""
        segments = []
        
        for audio_path in self.audio_paths:
            audio_path = Path(audio_path)
            duration = self.audio_processor.get_duration(audio_path)
            
            start = 0.0
            while start < duration:
                end = min(start + self.segment_duration, duration)
                
                segments.append({
                    'audio_path': str(audio_path),
                    'file_id': audio_path.stem,
                    'start_time': start,
                    'end_time': end,
                    'total_duration': duration,
                })
                
                start += self.segment_step
        
        return segments
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an inference segment."""
        segment = self.segments[idx]
        
        # Load audio
        waveform, _ = self.audio_processor.load(
            segment['audio_path'],
            start=segment['start_time'],
            duration=segment['end_time'] - segment['start_time'],
        )
        
        # Pad if necessary
        expected_samples = int(self.segment_duration * self.sample_rate)
        if waveform.shape[-1] < expected_samples:
            padding = expected_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Extract features
        features = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)
        
        return {
            'features': features,
            'waveform': waveform,
            'file_id': segment['file_id'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'total_duration': segment['total_duration'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples
    """
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    result = {
        'features': features,
        'labels': labels,
        'file_ids': [item['file_id'] for item in batch],
        'start_times': torch.tensor([item['start_time'] for item in batch]),
    }
    
    if 'waveform' in batch[0]:
        result['waveforms'] = torch.stack([item['waveform'] for item in batch])
    
    return result


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Training and validation dataloaders
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
    
    return train_loader, val_loader