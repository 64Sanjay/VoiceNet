#!/usr/bin/env python3
"""
Main evaluation script for speaker diarization.

Usage:
    python evaluate.py --model-path ./outputs/checkpoints/best_model.pt \
                       --test-audio ./data/aishell4/test/audio \
                       --test-rttm ./data/aishell4/test/rttm \
                       --output-dir ./results
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Robust imports with fallbacks
try:
    from models.diarization_model import SpeakerDiarizationModel, DiarizationPipeline, load_pretrained_model
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    load_pretrained_model = None
    DiarizationPipeline = None

try:
    from evaluation.evaluator import Evaluator
    from evaluation.metrics import DiarizationMetrics, compute_der_simple, DERComponents
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False
    Evaluator = None

try:
    from utils.helpers import setup_logging as _setup_logging, print_gpu_info
    from utils.rttm_utils import RTTMReader, RTTMWriter, RTTMSegment
    from utils.audio_utils import AudioProcessor
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    RTTMReader = None
    AudioProcessor = None


# ============================================================================
# FALLBACK IMPLEMENTATIONS
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging with fallback."""
    if HAS_UTILS:
        try:
            return _setup_logging(log_level=log_level, log_file=str(log_file) if log_file else None)
        except Exception:
            pass
    
    # Fallback
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


class SimpleRTTMReader:
    """Simple RTTM reader fallback."""
    
    def read(self, rttm_path: str) -> Dict[str, List]:
        segments = defaultdict(list)
        
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
                    file_id = parts[1]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    
                    segments[file_id].append({
                        'start': start,
                        'end': start + duration,
                        'speaker': speaker,
                    })
        
        return dict(segments)


class SimpleAudioProcessor:
    """Simple audio processor fallback."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load(self, path: str):
        import torchaudio
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, self.sample_rate
    
    def get_duration(self, path: str) -> float:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate


class SimpleDiarizationMetrics:
    """Simple metrics calculator fallback."""
    
    def __init__(self, collar: float = 0.25, skip_overlap: bool = False):
        self.collar = collar
        self.skip_overlap = skip_overlap
    
    def __call__(
        self,
        reference: List[Tuple[float, float, str]],
        hypothesis: List[Tuple[float, float, str]],
    ) -> Dict[str, float]:
        """Compute simple DER metrics."""
        if not reference:
            return {'der': 0.0, 'false_alarm': 0.0, 'missed': 0.0, 'confusion': 0.0}
        
        # Simple frame-based DER computation
        frame_duration = 0.01  # 10ms frames
        
        # Get total duration
        max_time = max(
            max((seg[1] for seg in reference), default=0),
            max((seg[1] for seg in hypothesis), default=0)
        )
        num_frames = int(max_time / frame_duration) + 1
        
        # Create frame-level labels
        ref_frames = np.zeros((num_frames, 10))  # Max 10 speakers
        hyp_frames = np.zeros((num_frames, 10))
        
        ref_speakers = sorted(set(s[2] for s in reference))
        hyp_speakers = sorted(set(s[2] for s in hypothesis))
        
        ref_spk_map = {s: i for i, s in enumerate(ref_speakers)}
        hyp_spk_map = {s: i for i, s in enumerate(hyp_speakers)}
        
        for start, end, spk in reference:
            start_frame = int((start + self.collar) / frame_duration)
            end_frame = int((end - self.collar) / frame_duration)
            if spk in ref_spk_map and start_frame < end_frame:
                ref_frames[start_frame:end_frame, ref_spk_map[spk]] = 1
        
        for start, end, spk in hypothesis:
            start_frame = int(start / frame_duration)
            end_frame = int(end / frame_duration)
            if spk in hyp_spk_map:
                hyp_frames[start_frame:end_frame, hyp_spk_map[spk]] = 1
        
        # Compute metrics
        ref_speech = ref_frames.sum(axis=1) > 0
        hyp_speech = hyp_frames.sum(axis=1) > 0
        
        total_speech = ref_speech.sum() * frame_duration
        
        # False alarm: hypothesis speech when no reference speech
        false_alarm = ((~ref_speech) & hyp_speech).sum() * frame_duration
        
        # Missed: reference speech when no hypothesis speech
        missed = (ref_speech & (~hyp_speech)).sum() * frame_duration
        
        # Confusion: both speaking but different speakers (simplified)
        both_speech = ref_speech & hyp_speech
        n_ref_spk = ref_frames[both_speech].sum(axis=1)
        n_hyp_spk = hyp_frames[both_speech].sum(axis=1)
        confusion = (np.abs(n_ref_spk - n_hyp_spk) * frame_duration).sum()
        
        if total_speech > 0:
            der = 100 * (false_alarm + missed + confusion) / total_speech
        else:
            der = 0.0
        
        return {
            'der': der,
            'false_alarm': 100 * false_alarm / max(total_speech, 1e-8),
            'missed': 100 * missed / max(total_speech, 1e-8),
            'confusion': 100 * confusion / max(total_speech, 1e-8),
        }


class SimpleEvaluator:
    """Simple evaluator fallback."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        collar: float = 0.25,
        skip_overlap: bool = False,
        sample_rate: int = 16000,
        min_segment_duration: float = 0.1,
        merge_threshold: float = 0.3,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.merge_threshold = merge_threshold
        
        self.metrics = SimpleDiarizationMetrics(collar=collar, skip_overlap=skip_overlap)
        self.audio_processor = SimpleAudioProcessor(sample_rate=sample_rate)
        self.rttm_reader = SimpleRTTMReader()
    
    @torch.no_grad()
    def evaluate_file(
        self,
        audio_path: str,
        rttm_path: str,
        num_speakers: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate a single file."""
        audio_path = Path(audio_path)
        rttm_path = Path(rttm_path)
        
        # Load reference
        ref_segments = self.rttm_reader.read(str(rttm_path))
        file_id = audio_path.stem
        
        # Find matching file_id
        if file_id not in ref_segments:
            for key in ref_segments:
                if key in file_id or file_id in key:
                    file_id = key
                    break
        
        ref = ref_segments.get(file_id, [])
        reference = [(seg['start'], seg['end'], seg['speaker']) for seg in ref]
        
        # Run inference
        hypothesis = self._run_inference(str(audio_path), num_speakers)
        
        # Compute metrics
        return self.metrics(reference, hypothesis)
    
    def _run_inference(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """Run model inference."""
        # Load audio
        waveform, sr = self.audio_processor.load(audio_path)
        waveform = waveform.to(self.device)
        
        # Get predictions
        with torch.amp.autocast('cuda', enabled=self.device == 'cuda'):
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
        return self._predictions_to_segments(preds, threshold=0.5)
    
    def _predictions_to_segments(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
        frame_duration: float = 0.016,
    ) -> List[Tuple[float, float, str]]:
        """Convert frame predictions to segments."""
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
            
            if in_segment:
                start_time = start_frame * frame_duration
                end_time = num_frames * frame_duration
                
                if end_time - start_time >= self.min_segment_duration:
                    segments.append((start_time, end_time, f"speaker_{spk}"))
        
        return sorted(segments, key=lambda x: x[0])
    
    def evaluate_dataset(
        self,
        audio_dir: str,
        rttm_dir: str,
        output_dir: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        num_speakers: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, float]:
        """Evaluate entire dataset."""
        audio_dir = Path(audio_dir)
        rttm_dir = Path(rttm_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files
        if file_list:
            audio_files = [audio_dir / f"{f}.wav" for f in file_list]
        else:
            audio_files = sorted(audio_dir.glob("*.wav"))
        
        # Evaluate
        all_metrics = defaultdict(list)
        evaluated = []
        
        for audio_path in tqdm(audio_files, desc="Evaluating"):
            file_id = audio_path.stem
            rttm_path = rttm_dir / f"{file_id}.rttm"
            
            if not audio_path.exists() or not rttm_path.exists():
                continue
            
            try:
                metrics = self.evaluate_file(str(audio_path), str(rttm_path), num_speakers)
                
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                evaluated.append(file_id)
                
            except Exception as e:
                print(f"Error evaluating {file_id}: {e}")
                continue
        
        # Aggregate
        results = {}
        for key, values in all_metrics.items():
            results[f"{key}_mean"] = float(np.mean(values))
            results[f"{key}_std"] = float(np.std(values))
        
        results['num_files'] = len(evaluated)
        
        # Save
        if output_dir:
            with open(output_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2)
        
        return results


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to use project's load_pretrained_model
    if HAS_MODEL and load_pretrained_model is not None:
        try:
            return load_pretrained_model(str(checkpoint_path), device=device)
        except Exception as e:
            print(f"Could not use load_pretrained_model: {e}")
    
    # Fallback: create simple model
    config = checkpoint.get('config', {})
    
    # Import or create model
    try:
        from models.segmentation import PyanNet
        model = PyanNet(
            num_speakers=config.get('num_speakers', 4),
            input_dim=config.get('n_mels', 80),
        )
    except ImportError:
        # Use simple model from train.py
        from train import SimpleDiarizationModel
        model = SimpleDiarizationModel(
            n_mels=config.get('n_mels', 80),
            max_speakers=config.get('num_speakers', config.get('max_speakers', 4)),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speaker diarization model")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-audio", type=str, required=True, help="Test audio directory")
    parser.add_argument("--test-rttm", type=str, required=True, help="Test RTTM directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--collar", type=float, default=0.25, help="Forgiveness collar (seconds)")
    parser.add_argument("--skip-overlap", action="store_true", help="Skip overlapping speech")
    parser.add_argument("--num-speakers", type=int, default=None, help="Oracle number of speakers")
    parser.add_argument("--clustering-method", type=str, default="ahc", 
                       choices=["ahc", "spectral", "vbx"], help="Clustering method")
    parser.add_argument("--clustering-threshold", type=float, default=0.5, help="Clustering threshold")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--file-list", type=str, default=None, help="File with list of files to evaluate")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_level="INFO", log_file=output_dir / "evaluate.log")
    
    logger.info("=" * 60)
    logger.info("Speaker Diarization Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test audio: {args.test_audio}")
    logger.info(f"Test RTTM: {args.test_rttm}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path, device=args.device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create evaluator
    if HAS_EVALUATOR and Evaluator is not None:
        try:
            evaluator = Evaluator(
                model=model,
                device=args.device,
                collar=args.collar,
                skip_overlap=args.skip_overlap,
                clustering_method=args.clustering_method,
                clustering_threshold=args.clustering_threshold,
            )
            logger.info("Using project Evaluator")
        except Exception as e:
            logger.warning(f"Could not create project Evaluator: {e}")
            evaluator = SimpleEvaluator(
                model=model,
                device=args.device,
                collar=args.collar,
                skip_overlap=args.skip_overlap,
            )
            logger.info("Using SimpleEvaluator")
    else:
        evaluator = SimpleEvaluator(
            model=model,
            device=args.device,
            collar=args.collar,
            skip_overlap=args.skip_overlap,
        )
        logger.info("Using SimpleEvaluator")
    
    # Get file list if provided
    file_list = None
    if args.file_list:
        with open(args.file_list) as f:
            file_list = [line.strip() for line in f if line.strip()]
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    results = evaluator.evaluate_dataset(
        audio_dir=args.test_audio,
        rttm_dir=args.test_rttm,
        output_dir=str(output_dir),
        file_list=file_list,
        num_speakers=args.num_speakers,
        save_predictions=True,
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    if 'der_mean' in results:
        logger.info(f"  DER: {results['der_mean']:.2f}% ± {results.get('der_std', 0):.2f}%")
    
    if 'false_alarm_mean' in results:
        logger.info(f"  False Alarm: {results['false_alarm_mean']:.2f}%")
    
    if 'missed_mean' in results:
        logger.info(f"  Missed: {results['missed_mean']:.2f}%")
    
    if 'confusion_mean' in results:
        logger.info(f"  Confusion: {results['confusion_mean']:.2f}%")
    
    logger.info(f"  Files evaluated: {results.get('num_files', 0)}")
    logger.info("=" * 60)
    logger.info(f"Results saved to {output_dir}")
    
    # Save detailed results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()