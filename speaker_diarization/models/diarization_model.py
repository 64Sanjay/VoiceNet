"""
End-to-end Speaker Diarization Model.

Combines:
- Feature extraction
- Speaker segmentation (frame-level predictions)
- Speaker embedding extraction
- Clustering for speaker assignment

Based on pyannote.audio architecture.
"""

"""
End-to-end Speaker Diarization Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
import numpy as np

# Robust imports
try:
    from .speaker_encoder import ECAPA_TDNN, SpeakerEncoder
    from .segmentation import PyanNet, EEND, EENDWithEDA, SegmentationModel
    from .clustering import (
        AgglomerativeHierarchicalClustering,
        SpectralClusteringWrapper,
        VBxClustering,
        create_clustering,
    )
except ImportError:
    from speaker_encoder import ECAPA_TDNN, SpeakerEncoder
    from segmentation import PyanNet, EEND, EENDWithEDA, SegmentationModel
    from clustering import (
        AgglomerativeHierarchicalClustering,
        SpectralClusteringWrapper,
        VBxClustering,
        create_clustering,
    )

# Feature extractor - make optional
try:
    from ..data.preprocessing import FeatureExtractor
    HAS_FEATURE_EXTRACTOR = True
except ImportError:
    HAS_FEATURE_EXTRACTOR = False
    FeatureExtractor = None


class SpeakerDiarizationModel(nn.Module):
    """
    End-to-end Speaker Diarization Model.
    
    Pipeline:
    1. Extract acoustic features (mel-spectrogram)
    2. Segment audio into speaker regions (frame-level)
    3. Extract speaker embeddings for each segment
    4. Cluster embeddings to assign speaker identities
    """
    
    def __init__(
        self,
        # Segmentation config
        segmentation_model: str = "pyannet",
        num_speakers: int = 4,
        
        # Encoder config
        encoder_type: str = "ecapa_tdnn",
        embedding_dim: int = 192,
        
        # Feature config
        sample_rate: int = 16000,
        n_mels: int = 80,
        
        # Additional config
        use_sincnet: bool = False,
        **kwargs,
    ):
        """
        Initialize diarization model.
        
        Args:
            segmentation_model: Type of segmentation model
            num_speakers: Maximum number of speakers
            encoder_type: Type of speaker encoder
            embedding_dim: Speaker embedding dimension
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            use_sincnet: Use SincNet frontend
        """
        super().__init__()
        
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            feature_type="mel",
        )
        
        # Segmentation model
        if segmentation_model == "pyannet":
            self.segmentation = PyanNet(
                sample_rate=sample_rate,
                use_sincnet=use_sincnet,
                num_speakers=num_speakers,
                input_dim=n_mels,
                **kwargs.get('segmentation_kwargs', {}),
            )
        elif segmentation_model == "eend":
            self.segmentation = EEND(
                input_dim=n_mels,
                num_speakers=num_speakers,
                **kwargs.get('segmentation_kwargs', {}),
            )
        elif segmentation_model == "eend_eda":
            self.segmentation = EENDWithEDA(
                input_dim=n_mels,
                max_speakers=num_speakers,
                **kwargs.get('segmentation_kwargs', {}),
            )
        else:
            raise ValueError(f"Unknown segmentation model: {segmentation_model}")
        
        # Speaker encoder (for refinement/clustering)
        self.speaker_encoder = SpeakerEncoder(
            encoder_type=encoder_type,
            input_dim=n_mels,
            embedding_dim=embedding_dim,
        )
        
        self.segmentation_model_type = segmentation_model
    
    def forward(
        self,
        x: torch.Tensor,
        extract_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
               - Raw waveform: [B, 1, T]
               - Features: [B, C, T] or [B, T, C]
            extract_embeddings: Also extract speaker embeddings
            
        Returns:
            predictions: Frame-level speaker probabilities [B, T', num_speakers]
            embeddings: Speaker embeddings [B, D] (if extract_embeddings=True)
        """
        # Extract features if input is waveform
        if x.dim() == 3 and x.shape[1] == 1:
            # Raw waveform input
            features = self.feature_extractor(x.squeeze(1))
        elif x.dim() == 2:
            # Already features [B, T] - add channel dim
            features = x.unsqueeze(1)
            features = self.feature_extractor(features)
        else:
            features = x
        
        # Segmentation
        if self.segmentation_model_type == "eend_eda":
            predictions, attractor_probs = self.segmentation(features)
        else:
            predictions = self.segmentation(features)
        
        if extract_embeddings:
            embeddings = self.speaker_encoder(features)
            return predictions, embeddings
        
        return predictions
    
    def get_speaker_segments(
        self,
        predictions: torch.Tensor,
        threshold: float = 0.5,
        min_duration: float = 0.1,
        frame_duration: float = 0.016,
    ) -> List[List[Tuple[float, float, int]]]:
        """
        Convert frame-level predictions to speaker segments.
        
        Args:
            predictions: [B, T, num_speakers]
            threshold: Activation threshold
            min_duration: Minimum segment duration
            frame_duration: Frame duration in seconds
            
        Returns:
            List of segments for each batch item
            Each segment is (start_time, end_time, speaker_id)
        """
        batch_size = predictions.shape[0]
        all_segments = []
        
        for b in range(batch_size):
            segments = []
            pred = predictions[b].cpu().numpy()
            binary_pred = pred > threshold
            
            for spk in range(self.num_speakers):
                spk_active = binary_pred[:, spk]
                
                # Find contiguous regions
                in_segment = False
                start_frame = 0
                
                for frame in range(len(spk_active)):
                    if spk_active[frame] and not in_segment:
                        start_frame = frame
                        in_segment = True
                    elif not spk_active[frame] and in_segment:
                        end_frame = frame
                        start_time = start_frame * frame_duration
                        end_time = end_frame * frame_duration
                        
                        if end_time - start_time >= min_duration:
                            segments.append((start_time, end_time, spk))
                        
                        in_segment = False
                
                # Handle last segment
                if in_segment:
                    end_time = len(spk_active) * frame_duration
                    start_time = start_frame * frame_duration
                    if end_time - start_time >= min_duration:
                        segments.append((start_time, end_time, spk))
            
            # Sort by start time
            segments.sort(key=lambda x: x[0])
            all_segments.append(segments)
        
        return all_segments


class DiarizationPipeline:
    """
    Complete speaker diarization pipeline.
    
    Handles:
    - Audio loading and preprocessing
    - Sliding window inference
    - Post-processing and clustering
    - Output generation (RTTM)
    """
    
    def __init__(
        self,
        model: SpeakerDiarizationModel,
        clustering_method: str = "ahc",
        clustering_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        
        # Inference config
        window_duration: float = 5.0,
        window_step: float = 2.5,
        
        # Post-processing
        min_segment_duration: float = 0.1,
        min_silence_duration: float = 0.1,
        merge_threshold: float = 0.3,
    ):
        """
        Initialize pipeline.
        
        Args:
            model: Trained diarization model
            clustering_method: Clustering algorithm
            clustering_threshold: Clustering threshold
            device: Device to use
            window_duration: Sliding window duration
            window_step: Sliding window step
            min_segment_duration: Minimum segment duration
            min_silence_duration: Minimum silence duration
            merge_threshold: Threshold for merging segments
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.window_duration = window_duration
        self.window_step = window_step
        self.min_segment_duration = min_segment_duration
        self.min_silence_duration = min_silence_duration
        self.merge_threshold = merge_threshold
        
        # Initialize clustering
        self.clustering = create_clustering(
            method=clustering_method,
            threshold=clustering_threshold,
        )
    
    @torch.no_grad()
    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """
        Run diarization on audio.
        
        Args:
            waveform: Audio waveform [1, T] or [T]
            sample_rate: Sample rate
            num_speakers: Number of speakers (optional)
            
        Returns:
            List of (start, end, speaker_id) tuples
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if necessary
        if sample_rate != self.model.sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.model.sample_rate
            )
        
        duration = waveform.shape[-1] / self.model.sample_rate
        
        # Sliding window inference
        all_predictions = []
        all_embeddings = []
        window_starts = []
        
        window_samples = int(self.window_duration * self.model.sample_rate)
        step_samples = int(self.window_step * self.model.sample_rate)
        
        start = 0
        while start < waveform.shape[-1]:
            end = min(start + window_samples, waveform.shape[-1])
            window = waveform[:, start:end]
            
            # Pad if necessary
            if window.shape[-1] < window_samples:
                padding = window_samples - window.shape[-1]
                window = F.pad(window, (0, padding))
            
            # Add batch dimension and move to device
            window = window.unsqueeze(0).to(self.device)
            
            # Inference
            predictions, embeddings = self.model(window, extract_embeddings=True)
            
            all_predictions.append(predictions.cpu())
            all_embeddings.append(embeddings.cpu())
            window_starts.append(start / self.model.sample_rate)
            
            start += step_samples
        
        # Aggregate predictions
        frame_predictions = self._aggregate_predictions(
            all_predictions, window_starts, duration
        )
        
        # Get initial segments
        segments = self._predictions_to_segments(frame_predictions)
        
        # Cluster embeddings if needed
        if len(all_embeddings) > 1:
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            cluster_labels = self.clustering.cluster(embeddings, num_speakers)
            
            # Assign cluster labels to segments
            segments = self._assign_cluster_labels(segments, cluster_labels, window_starts)
        
        # Post-processing
        segments = self._merge_segments(segments)
        segments = self._filter_short_segments(segments)
        
        return segments
    
    def _aggregate_predictions(
        self,
        predictions: List[torch.Tensor],
        window_starts: List[float],
        total_duration: float,
    ) -> np.ndarray:
        """Aggregate overlapping window predictions."""
        frame_duration = 0.016  # 16ms frames
        num_frames = int(np.ceil(total_duration / frame_duration))
        num_speakers = predictions[0].shape[-1]
        
        # Accumulator for predictions and counts
        pred_sum = np.zeros((num_frames, num_speakers))
        pred_count = np.zeros(num_frames)
        
        for pred, start in zip(predictions, window_starts):
            pred = pred.squeeze(0).numpy()
            start_frame = int(start / frame_duration)
            
            for i, frame_pred in enumerate(pred):
                frame_idx = start_frame + i
                if frame_idx < num_frames:
                    pred_sum[frame_idx] += frame_pred
                    pred_count[frame_idx] += 1
        
        # Average predictions
        pred_count = np.maximum(pred_count, 1)
        frame_predictions = pred_sum / pred_count[:, np.newaxis]
        
        return frame_predictions
    
    def _predictions_to_segments(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
        frame_duration: float = 0.016,
    ) -> List[Tuple[float, float, int]]:
        """Convert frame predictions to segments."""
        segments = []
        num_speakers = predictions.shape[-1]
        
        binary_pred = predictions > threshold
        
        for spk in range(num_speakers):
            spk_active = binary_pred[:, spk]
            
            in_segment = False
            start_frame = 0
            
            for frame in range(len(spk_active)):
                if spk_active[frame] and not in_segment:
                    start_frame = frame
                    in_segment = True
                elif not spk_active[frame] and in_segment:
                    start_time = start_frame * frame_duration
                    end_time = frame * frame_duration
                    segments.append((start_time, end_time, spk))
                    in_segment = False
            
            if in_segment:
                start_time = start_frame * frame_duration
                end_time = len(spk_active) * frame_duration
                segments.append((start_time, end_time, spk))
        
        return sorted(segments, key=lambda x: x[0])
    
    def _assign_cluster_labels(
        self,
        segments: List[Tuple[float, float, int]],
        cluster_labels: np.ndarray,
        window_starts: List[float],
    ) -> List[Tuple[float, float, int]]:
        """Assign cluster labels to segments based on window embeddings."""
        # Map original speaker indices to cluster labels
        speaker_to_cluster = {}
        
        for seg_start, seg_end, spk_id in segments:
            # Find which window this segment belongs to
            for i, win_start in enumerate(window_starts):
                win_end = win_start + self.window_duration
                if seg_start >= win_start and seg_start < win_end:
                    if spk_id not in speaker_to_cluster:
                        speaker_to_cluster[spk_id] = cluster_labels[i]
                    break
        
        # Apply mapping
        new_segments = []
        for start, end, spk_id in segments:
            cluster_id = speaker_to_cluster.get(spk_id, spk_id)
            new_segments.append((start, end, cluster_id))
        
        return new_segments
    
    def _merge_segments(
        self,
        segments: List[Tuple[float, float, int]],
    ) -> List[Tuple[float, float, str]]:
        """Merge adjacent segments of the same speaker."""
        if not segments:
            return []
        
        # Group by speaker
        speaker_segments = {}
        for start, end, spk in segments:
            if spk not in speaker_segments:
                speaker_segments[spk] = []
            speaker_segments[spk].append((start, end))
        
        # Merge within each speaker
        merged = []
        for spk, segs in speaker_segments.items():
            segs = sorted(segs, key=lambda x: x[0])
            
            current_start, current_end = segs[0]
            
            for start, end in segs[1:]:
                if start - current_end <= self.merge_threshold:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end, f"speaker_{spk}"))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end, f"speaker_{spk}"))
        
        return sorted(merged, key=lambda x: x[0])
    
    def _filter_short_segments(
        self,
        segments: List[Tuple[float, float, str]],
    ) -> List[Tuple[float, float, str]]:
        """Remove segments shorter than minimum duration."""
        return [
            (start, end, spk)
            for start, end, spk in segments
            if end - start >= self.min_segment_duration
        ]


def load_pretrained_model(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> SpeakerDiarizationModel:
    """
    Load pretrained diarization model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to
        **kwargs: Model configuration overrides
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    config.update(kwargs)
    
    model = SpeakerDiarizationModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model