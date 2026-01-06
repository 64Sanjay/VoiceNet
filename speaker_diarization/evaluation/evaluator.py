"""
Evaluation pipeline for speaker diarization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import json
import logging
from collections import defaultdict

# Robust imports
try:
    from .metrics import DiarizationMetrics, compute_der_simple, DERComponents
except ImportError:
    from metrics import DiarizationMetrics, compute_der_simple, DERComponents

try:
    from ..utils.rttm_utils import RTTMReader, RTTMWriter, RTTMSegment
    from ..utils.audio_utils import AudioProcessor
except ImportError:
    try:
        from utils.rttm_utils import RTTMReader, RTTMWriter, RTTMSegment
        from utils.audio_utils import AudioProcessor
    except ImportError:
        RTTMReader = None
        RTTMWriter = None
        RTTMSegment = None
        AudioProcessor = None

try:
    from ..models.diarization_model import SpeakerDiarizationModel, DiarizationPipeline
except ImportError:
    try:
        from models.diarization_model import SpeakerDiarizationModel, DiarizationPipeline
    except ImportError:
        SpeakerDiarizationModel = None
        DiarizationPipeline = None


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation pipeline for speaker diarization.
    
    Handles:
    - Loading test data
    - Running inference
    - Computing metrics
    - Generating reports
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = None,
        collar: float = 0.25,
        skip_overlap: bool = False,
        
        # Inference settings
        window_duration: float = 5.0,
        window_step: float = 2.5,
        
        # Post-processing
        min_segment_duration: float = 0.1,
        merge_threshold: float = 0.3,
        
        # Clustering
        clustering_method: str = "ahc",
        clustering_threshold: float = 0.5,
        
        # Audio settings
        sample_rate: int = 16000,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained diarization model
            device: Device to use
            collar: Forgiveness collar in seconds
            skip_overlap: Skip overlapping speech in evaluation
            window_duration: Sliding window duration
            window_step: Sliding window step
            min_segment_duration: Minimum segment duration
            merge_threshold: Threshold for merging segments
            clustering_method: Clustering algorithm
            clustering_threshold: Clustering threshold
            sample_rate: Audio sample rate
        """
        # Setup device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        # Model
        self.model = model.to(self.device)
        self.model.eval()
        
        # Get sample rate from model if available
        if hasattr(model, 'sample_rate'):
            sample_rate = model.sample_rate
        self.sample_rate = sample_rate
        
        # Metrics
        self.metrics = DiarizationMetrics(
            collar=collar,
            skip_overlap=skip_overlap,
        )
        self.collar = collar
        self.skip_overlap = skip_overlap
        
        # Inference settings
        self.window_duration = window_duration
        self.window_step = window_step
        self.min_segment_duration = min_segment_duration
        self.merge_threshold = merge_threshold
        self.clustering_method = clustering_method
        self.clustering_threshold = clustering_threshold
        
        # Pipeline (if available)
        self.pipeline = None
        if DiarizationPipeline is not None:
            try:
                self.pipeline = DiarizationPipeline(
                    model=model,
                    clustering_method=clustering_method,
                    clustering_threshold=clustering_threshold,
                    device=device,
                    window_duration=window_duration,
                    window_step=window_step,
                    min_segment_duration=min_segment_duration,
                    merge_threshold=merge_threshold,
                )
            except Exception as e:
                logger.warning(f"Could not create DiarizationPipeline: {e}")
        
        # Audio processor
        self.audio_processor = None
        if AudioProcessor is not None:
            self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # RTTM reader
        self.rttm_reader = None
        if RTTMReader is not None:
            self.rttm_reader = RTTMReader()
    
    @torch.no_grad()
    def evaluate_file(
        self,
        audio_path: Union[str, Path],
        rttm_path: Union[str, Path],
        num_speakers: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single audio file.
        
        Args:
            audio_path: Path to audio file
            rttm_path: Path to reference RTTM file
            num_speakers: Number of speakers (optional)
            
        Returns:
            Dictionary of metrics
        """
        audio_path = Path(audio_path)
        rttm_path = Path(rttm_path)
        
        if self.audio_processor is None or self.rttm_reader is None:
            raise RuntimeError("AudioProcessor and RTTMReader required for file evaluation")
        
        # Load audio
        waveform, sr = self.audio_processor.load(audio_path)
        
        # Load reference
        reference_segments = self.rttm_reader.read(rttm_path)
        file_id = audio_path.stem
        
        # Try to find matching key
        if file_id not in reference_segments:
            for key in reference_segments:
                if key in file_id or file_id in key:
                    file_id = key
                    break
        
        ref_segments = reference_segments.get(file_id, [])
        reference = [
            (seg.start, seg.end, seg.speaker_id)
            for seg in ref_segments
        ]
        
        # Run inference
        if self.pipeline is not None:
            hypothesis = self.pipeline(
                waveform,
                sample_rate=sr,
                num_speakers=num_speakers,
            )
        else:
            # Fallback: simple inference
            hypothesis = self._simple_inference(waveform, num_speakers)
        
        # Compute metrics
        metrics = self.metrics(reference, hypothesis)
        
        return metrics
    
    def _simple_inference(
        self,
        waveform: torch.Tensor,
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """
        Simple inference without full pipeline.
        
        Args:
            waveform: Audio waveform
            num_speakers: Number of speakers
            
        Returns:
            List of (start, end, speaker_id) tuples
        """
        # This is a simplified version - full pipeline should be used for production
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        # Get predictions
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            outputs = self.model(waveform.unsqueeze(0))
        
        # Extract predictions
        if isinstance(outputs, dict):
            preds = outputs.get('speakers', outputs.get('predictions'))
        elif isinstance(outputs, tuple):
            preds = outputs[0]
        else:
            preds = outputs
        
        preds = preds.squeeze(0).cpu().numpy()  # [T, S]
        
        # Convert to segments
        segments = self._predictions_to_segments(preds)
        
        return segments
    
    def _predictions_to_segments(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
        frame_duration: float = 0.016,
    ) -> List[Tuple[float, float, str]]:
        """Convert frame-level predictions to segments."""
        segments = []
        num_frames, num_speakers = predictions.shape
        
        binary_pred = predictions > threshold
        
        for spk in range(num_speakers):
            spk_active = binary_pred[:, spk]
            
            in_segment = False
            start_frame = 0
            
            for frame in range(num_frames):
                if spk_active[frame] and not in_segment:
                    start_frame = frame
                    in_segment = True
                elif not spk_active[frame] and in_segment:
                    start_time = start_frame * frame_duration
                    end_time = frame * frame_duration
                    
                    if end_time - start_time >= self.min_segment_duration:
                        segments.append((start_time, end_time, f"speaker_{spk}"))
                    
                    in_segment = False
            
            # Handle last segment
            if in_segment:
                start_time = start_frame * frame_duration
                end_time = num_frames * frame_duration
                
                if end_time - start_time >= self.min_segment_duration:
                    segments.append((start_time, end_time, f"speaker_{spk}"))
        
        # Merge nearby segments
        segments = self._merge_segments(segments)
        
        return sorted(segments, key=lambda x: x[0])
    
    def _merge_segments(
        self,
        segments: List[Tuple[float, float, str]],
    ) -> List[Tuple[float, float, str]]:
        """Merge nearby segments from the same speaker."""
        if not segments:
            return []
        
        # Group by speaker
        by_speaker = defaultdict(list)
        for start, end, spk in segments:
            by_speaker[spk].append((start, end))
        
        merged = []
        for spk, segs in by_speaker.items():
            segs = sorted(segs, key=lambda x: x[0])
            
            current_start, current_end = segs[0]
            
            for start, end in segs[1:]:
                if start - current_end <= self.merge_threshold:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end, spk))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end, spk))
        
        return merged
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        audio_dir: Union[str, Path],
        rttm_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        file_list: Optional[List[str]] = None,
        num_speakers: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset.
        
        Args:
            audio_dir: Directory containing audio files
            rttm_dir: Directory containing RTTM files
            output_dir: Directory to save predictions
            file_list: Optional list of files to evaluate
            num_speakers: Number of speakers (optional)
            save_predictions: Whether to save predictions
            
        Returns:
            Aggregate metrics
        """
        audio_dir = Path(audio_dir)
        rttm_dir = Path(rttm_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files to evaluate
        if file_list:
            audio_files = [audio_dir / f"{f}.wav" for f in file_list]
        else:
            audio_files = sorted(audio_dir.glob("*.wav"))
        
        # Aggregate results
        all_metrics = defaultdict(list)
        all_predictions = {}
        evaluated_files = []
        
        for audio_path in tqdm(audio_files, desc="Evaluating"):
            file_id = audio_path.stem
            rttm_path = rttm_dir / f"{file_id}.rttm"
            
            if not audio_path.exists():
                logger.warning(f"Audio not found: {audio_path}, skipping")
                continue
            
            if not rttm_path.exists():
                logger.warning(f"RTTM not found for {file_id}, skipping")
                continue
            
            try:
                # Evaluate file
                metrics = self.evaluate_file(
                    audio_path,
                    rttm_path,
                    num_speakers=num_speakers,
                )
                
                # Store metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                evaluated_files.append(file_id)
                
                logger.info(f"{file_id}: DER={metrics.get('der', 0):.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {file_id}: {e}")
                continue
        
        if not all_metrics:
            logger.warning("No files were successfully evaluated!")
            return {}
        
        # Compute aggregate metrics
        aggregate = {}
        for key, values in all_metrics.items():
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))
        
        aggregate['num_files'] = len(evaluated_files)
        
        # Save results
        if output_dir:
            # Aggregate results
            results_path = output_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(aggregate, f, indent=2)
            logger.info(f"Saved results to {results_path}")
            
            # Per-file results
            per_file_results = {}
            for i, file_id in enumerate(evaluated_files):
                per_file_results[file_id] = {
                    key: values[i]
                    for key, values in all_metrics.items()
                }
            
            per_file_path = output_dir / "per_file_results.json"
            with open(per_file_path, 'w') as f:
                json.dump(per_file_results, f, indent=2)
        
        return aggregate
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of data (for validation during training).
        
        Args:
            batch: Batch dictionary with 'features' and 'labels'
            
        Returns:
            Batch metrics
        """
        features = batch['features'].to(self.device)
        labels = batch['labels'].to(self.device)
        lengths = batch.get('lengths', None)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            outputs = self.model(features)
        
        # Extract predictions
        if isinstance(outputs, dict):
            predictions = outputs.get('speakers', outputs.get('predictions'))
        elif isinstance(outputs, tuple):
            predictions = outputs[0]
        else:
            predictions = outputs
        
        # Compute frame-level metrics
        pred_binary = (predictions > 0.5).float()
        
        # Create mask if lengths provided
        if lengths is not None:
            lengths = lengths.to(self.device)
            batch_size, time_steps = labels.shape[:2]
            mask = torch.arange(time_steps, device=self.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(labels).float()
        else:
            mask = torch.ones_like(labels)
        
        # Per-frame accuracy
        correct = ((pred_binary == labels).float() * mask).sum()
        total = mask.sum()
        accuracy = (correct / total).item() * 100 if total > 0 else 0
        
        # Per-speaker metrics
        true_positives = ((pred_binary == 1) & (labels == 1)).float() * mask
        false_positives = ((pred_binary == 1) & (labels == 0)).float() * mask
        false_negatives = ((pred_binary == 0) & (labels == 1)).float() * mask
        
        tp = true_positives.sum()
        fp = false_positives.sum()
        fn = false_negatives.sum()
        
        precision = (tp / (tp + fp + 1e-8)).item() * 100
        recall = (tp / (tp + fn + 1e-8)).item() * 100
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def get_summary(self, results: Dict[str, float]) -> str:
        """Get a formatted summary of results."""
        lines = [
            "=" * 60,
            "Evaluation Results",
            "=" * 60,
        ]
        
        if 'der_mean' in results:
            lines.append(f"DER: {results['der_mean']:.2f}% ± {results.get('der_std', 0):.2f}%")
        
        if 'jer_mean' in results:
            lines.append(f"JER: {results['jer_mean']:.2f}% ± {results.get('jer_std', 0):.2f}%")
        
        if 'coverage_mean' in results:
            lines.append(f"Coverage: {results['coverage_mean']:.2f}%")
        
        if 'purity_mean' in results:
            lines.append(f"Purity: {results['purity_mean']:.2f}%")
        
        if 'num_files' in results:
            lines.append(f"Files evaluated: {results['num_files']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class OracleEvaluator:
    """
    Oracle evaluation with ground truth information.
    
    Useful for analyzing model components separately.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Union[str, torch.device] = None,
        sample_rate: int = 16000,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = sample_rate
        self.metrics = DiarizationMetrics()
    
    @torch.no_grad()
    def evaluate_with_oracle_vad(
        self,
        audio_path: str,
        reference: List[Tuple[float, float, str]],
    ) -> Dict[str, float]:
        """
        Evaluate using oracle VAD (ground truth speech regions).
        
        Tests clustering/speaker assignment only.
        """
        # Use reference speech regions as VAD
        speech_regions = [
            (start, end) for start, end, _ in reference
        ]
        
        # TODO: Implement oracle VAD evaluation
        # Extract embeddings only for speech regions
        # Cluster and assign speakers
        
        logger.warning("Oracle VAD evaluation not fully implemented")
        return {'oracle_vad': True}
    
    @torch.no_grad()
    def evaluate_with_oracle_num_speakers(
        self,
        audio_path: str,
        reference: List[Tuple[float, float, str]],
    ) -> Dict[str, float]:
        """
        Evaluate using oracle number of speakers.
        
        Tests everything except speaker count estimation.
        """
        num_speakers = len(set(spk for _, _, spk in reference))
        
        # TODO: Implement with oracle speaker count
        
        logger.warning("Oracle num speakers evaluation not fully implemented")
        return {'oracle_num_speakers': num_speakers}


def evaluate_model(
    model_path: str,
    test_audio_dir: str,
    test_rttm_dir: str,
    output_dir: str,
    device: Union[str, torch.device] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model_path: Path to model checkpoint
        test_audio_dir: Test audio directory
        test_rttm_dir: Test RTTM directory
        output_dir: Output directory
        device: Device to use
        **kwargs: Additional evaluator arguments
        
    Returns:
        Evaluation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to load model
    try:
        from ..models.diarization_model import load_pretrained_model
        model = load_pretrained_model(model_path, device=device)
    except ImportError:
        try:
            from models.diarization_model import load_pretrained_model
            model = load_pretrained_model(model_path, device=device)
        except ImportError:
            # Fallback: load checkpoint directly
            checkpoint = torch.load(model_path, map_location=device)
            logger.warning("Could not use load_pretrained_model, loading state dict only")
            raise RuntimeError("Model loading not fully supported without diarization_model module")
    
    # Create evaluator
    evaluator = Evaluator(model, device=device, **kwargs)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        audio_dir=test_audio_dir,
        rttm_dir=test_rttm_dir,
        output_dir=output_dir,
        save_predictions=True,
    )
    
    # Print summary
    print(evaluator.get_summary(results))
    
    return results