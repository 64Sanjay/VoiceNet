# data/preprocessing.py
"""
Audio preprocessing using Whisper feature extractor.
Handles resampling to 16kHz and standardization to 3000 frames.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from transformers import WhisperFeatureExtractor
import warnings

# Try different audio loading backends
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioPreprocessor:
    """
    Preprocessor for audio data following WSI paper specifications.
    
    - Resamples to 16kHz
    - Uses Whisper feature extractor for log-mel spectrogram
    - Standardizes to 3000 frames via zero-padding or truncation
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        fixed_frames: int = 3000,
        whisper_model_name: str = "openai/whisper-tiny"
    ):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate (16kHz as per paper)
            fixed_frames: Fixed number of input frames (3000 as per paper)
            whisper_model_name: Whisper model name for feature extractor
        """
        self.sample_rate = sample_rate
        self.fixed_frames = fixed_frames
        
        # Initialize Whisper feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            whisper_model_name
        )
        
        # Whisper expects specific audio length for 30 seconds
        # 30 seconds * 16000 Hz = 480000 samples
        self.max_audio_samples = 480000
        
        # Check available audio backends
        if not HAS_SOUNDFILE and not HAS_LIBROSA:
            raise ImportError(
                "Either 'soundfile' or 'librosa' is required for audio loading. "
                "Install with: pip install soundfile librosa"
            )
    
    def load_audio(
        self,
        audio_path: str,
        target_sr: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and resample audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (defaults to self.sample_rate)
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        target_sr = target_sr or self.sample_rate
        
        # Try soundfile first (faster)
        if HAS_SOUNDFILE:
            try:
                waveform, orig_sr = sf.read(audio_path, dtype='float32')
                
                # Convert to torch tensor
                waveform = torch.from_numpy(waveform)
                
                # Handle stereo -> mono
                if waveform.ndim > 1:
                    waveform = torch.mean(waveform, dim=-1)
                
                # Resample if necessary
                if orig_sr != target_sr:
                    if HAS_LIBROSA:
                        waveform_np = waveform.numpy()
                        waveform_np = librosa.resample(
                            waveform_np, 
                            orig_sr=orig_sr, 
                            target_sr=target_sr
                        )
                        waveform = torch.from_numpy(waveform_np.astype(np.float32))
                    else:
                        # Simple resampling using interpolation
                        waveform = self._resample_simple(waveform, orig_sr, target_sr)
                
                return waveform, target_sr
                
            except Exception as e:
                warnings.warn(f"soundfile failed, trying librosa: {e}")
        
        # Fallback to librosa
        if HAS_LIBROSA:
            waveform, orig_sr = librosa.load(audio_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform.astype(np.float32))
            return waveform, target_sr
        
        raise RuntimeError(f"Could not load audio file: {audio_path}")
    
    def _resample_simple(
        self, 
        waveform: torch.Tensor, 
        orig_sr: int, 
        target_sr: int
    ) -> torch.Tensor:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return waveform
        
        # Calculate new length
        orig_len = waveform.shape[0]
        new_len = int(orig_len * target_sr / orig_sr)
        
        # Use interpolation
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        waveform = torch.nn.functional.interpolate(
            waveform, 
            size=new_len, 
            mode='linear', 
            align_corners=False
        )
        waveform = waveform.squeeze()
        
        return waveform
    
    def extract_features(
        self,
        waveform: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Extract log-mel spectrogram features using Whisper feature extractor.
        
        Args:
            waveform: Audio waveform tensor or numpy array
            
        Returns:
            Log-mel spectrogram features
        """
        # Convert to numpy if tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Ensure 1D array and float32
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        waveform = waveform.astype(np.float32)
        
        # Use Whisper feature extractor
        # This returns log-mel spectrogram with 80 mel bins
        features = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        return features.input_features.squeeze(0)  # Shape: (80, T)
    
    def pad_or_truncate(
        self,
        features: torch.Tensor,
        target_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Pad or truncate features to fixed number of frames.
        
        As per paper: "Each input is standardized by zero-padding or 
        truncating to 3000 frames"
        
        Args:
            features: Input features of shape (F, T) where F=frequency bins, T=time
            target_frames: Target number of frames (defaults to self.fixed_frames)
            
        Returns:
            Padded/truncated features of shape (F, target_frames)
        """
        target_frames = target_frames or self.fixed_frames
        current_frames = features.shape[-1]
        
        if current_frames == target_frames:
            return features
        elif current_frames < target_frames:
            # Zero-padding
            padding = target_frames - current_frames
            padded = torch.nn.functional.pad(
                features,
                (0, padding),  # Pad only the time dimension
                mode='constant',
                value=0
            )
            return padded
        else:
            # Truncation - take from the beginning
            return features[..., :target_frames]
    
    def preprocess(
        self,
        audio_path: str
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed log-mel spectrogram of shape (80, 3000)
        """
        # Load and resample audio
        waveform, _ = self.load_audio(audio_path)
        
        # Extract log-mel spectrogram
        features = self.extract_features(waveform)
        
        # Pad or truncate to fixed frames
        features = self.pad_or_truncate(features)
        
        return features
    
    def preprocess_waveform(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocess a waveform tensor directly.
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Preprocessed log-mel spectrogram
        """
        # Extract features
        features = self.extract_features(waveform)
        
        # Pad or truncate
        features = self.pad_or_truncate(features)
        
        return features


class BatchPreprocessor:
    """Batch preprocessing for efficient data loading."""
    
    def __init__(self, preprocessor: AudioPreprocessor):
        self.preprocessor = preprocessor
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of (features, speaker_id) tuples
            
        Returns:
            Batched features and speaker IDs
        """
        features = torch.stack([item[0] for item in batch])
        speaker_ids = torch.tensor([item[1] for item in batch])
        
        return features, speaker_ids