#!/usr/bin/env python3
"""
Inference script for speaker diarization.

Usage:
    # Single file
    python inference.py --model-path ./outputs/checkpoints/best_model.pt \
                        --audio-path ./test_audio.wav \
                        --output-path ./output.rttm
    
    # Batch inference
    python inference.py --model-path ./outputs/checkpoints/best_model.pt \
                        --audio-dir ./audio_files/ \
                        --output-dir ./results/
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
from collections import defaultdict

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Robust imports
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from models.diarization_model import SpeakerDiarizationModel, DiarizationPipeline, load_pretrained_model
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    load_pretrained_model = None
    DiarizationPipeline = None

try:
    from utils.rttm_utils import RTTMWriter, RTTMSegment
    from utils.audio_utils import AudioProcessor
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

try:
    from utils.helpers import setup_logging as _setup_logging
except ImportError:
    _setup_logging = None


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging."""
    if _setup_logging is not None:
        try:
            return _setup_logging(log_level=log_level)
        except Exception:
            pass
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE IMPLEMENTATIONS
# ============================================================================

class SimpleAudioProcessor:
    """Simple audio processor."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load(self, path: str) -> Tuple[torch.Tensor, int]:
        if not HAS_TORCHAUDIO:
            raise ImportError("torchaudio required")
        
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, self.sample_rate


class SimpleRTTMWriter:
    """Simple RTTM writer."""
    
    @staticmethod
    def write(segments: Dict[str, List], path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for file_id, segs in segments.items():
                for seg in segs:
                    if hasattr(seg, 'to_rttm_line'):
                        f.write(seg.to_rttm_line() + '\n')
                    else:
                        start = seg.get('start', seg[0] if isinstance(seg, tuple) else 0)
                        duration = seg.get('duration', seg[1] - seg[0] if isinstance(seg, tuple) else 0)
                        speaker = seg.get('speaker_id', seg[2] if isinstance(seg, tuple) else 'speaker_0')
                        f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


class SimpleRTTMSegment:
    """Simple RTTM segment."""
    
    def __init__(self, file_id: str, channel: int, start: float, duration: float, speaker_id: str):
        self.file_id = file_id
        self.channel = channel
        self.start = start
        self.duration = duration
        self.speaker_id = speaker_id
    
    def to_rttm_line(self) -> str:
        return f"SPEAKER {self.file_id} {self.channel} {self.start:.3f} {self.duration:.3f} <NA> <NA> {self.speaker_id} <NA> <NA>"


class SimpleDiarizationPipeline:
    """Simple diarization pipeline."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        window_duration: float = 5.0,
        window_step: float = 2.5,
        min_segment_duration: float = 0.1,
        merge_threshold: float = 0.3,
        activation_threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.window_duration = window_duration
        self.window_step = window_step
        self.min_segment_duration = min_segment_duration
        self.merge_threshold = merge_threshold
        self.activation_threshold = activation_threshold
        self.sample_rate = sample_rate
        
        # Get sample rate from model if available
        if hasattr(model, 'sample_rate'):
            self.sample_rate = model.sample_rate
    
    @torch.no_grad()
    def __call__(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
        num_speakers: Optional[int] = None,
    ) -> List[Tuple[float, float, str]]:
        """Run diarization."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        
        waveform = waveform.to(self.device)
        
        # Process in windows
        window_samples = int(self.window_duration * self.sample_rate)
        step_samples = int(self.window_step * self.sample_rate)
        total_samples = waveform.shape[-1]
        
        all_predictions = []
        
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            window = waveform[:, start:end]
            
            # Pad if needed
            if window.shape[-1] < window_samples:
                pad = window_samples - window.shape[-1]
                window = torch.nn.functional.pad(window, (0, pad))
            
            # Get predictions
            with torch.amp.autocast('cuda', enabled=self.device == 'cuda'):
                outputs = self.model(window.unsqueeze(0))
            
            if isinstance(outputs, dict):
                preds = outputs.get('speakers', outputs.get('predictions'))
            elif isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs
            
            all_predictions.append({
                'start': start / self.sample_rate,
                'preds': preds.squeeze(0).cpu().numpy(),
            })
            
            start += step_samples
        
        # Aggregate predictions
        total_duration = total_samples / self.sample_rate
        segments = self._aggregate_and_segment(all_predictions, total_duration)
        
        return segments
    
    def _aggregate_and_segment(
        self,
        predictions: List[Dict],
        total_duration: float,
    ) -> List[Tuple[float, float, str]]:
        """Aggregate window predictions and convert to segments."""
        frame_duration = 0.016  # 16ms
        num_frames = int(total_duration / frame_duration) + 1
        
        if not predictions:
            return []
        
        num_speakers = predictions[0]['preds'].shape[-1]
        
        # Aggregate
        pred_sum = np.zeros((num_frames, num_speakers))
        pred_count = np.zeros(num_frames)
        
        for pred_info in predictions:
            start_frame = int(pred_info['start'] / frame_duration)
            preds = pred_info['preds']
            
            for i, frame_pred in enumerate(preds):
                frame_idx = start_frame + i
                if frame_idx < num_frames:
                    pred_sum[frame_idx] += frame_pred
                    pred_count[frame_idx] += 1
        
        # Average
        pred_count = np.maximum(pred_count, 1)
        frame_preds = pred_sum / pred_count[:, np.newaxis]
        
        # Convert to segments
        segments = []
        binary_pred = frame_preds > self.activation_threshold
        
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
                    
                    if end_time - start_time >= self.min_segment_duration:
                        segments.append((start_time, end_time, f"speaker_{spk}"))
                    
                    in_segment = False
            
            if in_segment:
                start_time = start_frame * frame_duration
                end_time = len(spk_active) * frame_duration
                
                if end_time - start_time >= self.min_segment_duration:
                    segments.append((start_time, end_time, f"speaker_{spk}"))
        
        # Merge nearby segments
        segments = self._merge_segments(segments)
        
        return sorted(segments, key=lambda x: x[0])
    
    def _merge_segments(
        self,
        segments: List[Tuple[float, float, str]],
    ) -> List[Tuple[float, float, str]]:
        """Merge nearby segments from same speaker."""
        if not segments:
            return []
        
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


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try project's loader
    if HAS_MODEL and load_pretrained_model is not None:
        try:
            return load_pretrained_model(str(checkpoint_path), device=device)
        except Exception as e:
            logger.warning(f"Could not use load_pretrained_model: {e}")
    
    # Fallback
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Try different model types
    model = None
    
    try:
        from models.segmentation import PyanNet
        model = PyanNet(
            num_speakers=config.get('num_speakers', config.get('max_speakers', 4)),
            input_dim=config.get('n_mels', 80),
        )
        logger.info("Created PyanNet model")
    except ImportError:
        pass
    
    if model is None:
        try:
            from train import SimpleDiarizationModel
            model = SimpleDiarizationModel(
                n_mels=config.get('n_mels', 80),
                max_speakers=config.get('num_speakers', config.get('max_speakers', 4)),
                hidden_dim=config.get('hidden_dim', 256),
                num_layers=config.get('num_layers', 4),
            )
            logger.info("Created SimpleDiarizationModel")
        except ImportError:
            raise RuntimeError("Could not create any model type")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Set sample_rate attribute
    model.sample_rate = config.get('sample_rate', 16000)
    
    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def segments_to_json(
    segments: List[Tuple[float, float, str]],
    audio_duration: float,
) -> Dict:
    """Convert segments to JSON format."""
    speakers = {}
    
    for start, end, speaker in segments:
        if speaker not in speakers:
            speakers[speaker] = {"segments": [], "total_duration": 0.0}
        
        speakers[speaker]["segments"].append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
        speakers[speaker]["total_duration"] += end - start
    
    for speaker in speakers:
        speakers[speaker]["total_duration"] = round(speakers[speaker]["total_duration"], 3)
        speakers[speaker]["num_segments"] = len(speakers[speaker]["segments"])
    
    return {
        "audio_duration": round(audio_duration, 3),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": [
            {"start": round(s, 3), "end": round(e, 3), "speaker": spk}
            for s, e, spk in sorted(segments, key=lambda x: x[0])
        ],
    }


def print_segments(segments: List[Tuple[float, float, str]]):
    """Print segments nicely."""
    print("\n" + "=" * 60)
    print("Speaker Diarization Results")
    print("=" * 60)
    
    for start, end, speaker in sorted(segments, key=lambda x: x[0]):
        duration = end - start
        print(f"  {format_time(start)} --> {format_time(end)}  [{speaker}]  ({duration:.2f}s)")
    
    print("\n" + "-" * 60)
    print("Summary:")
    
    speakers = {}
    for start, end, speaker in segments:
        speakers[speaker] = speakers.get(speaker, 0) + (end - start)
    
    total_speech = sum(speakers.values())
    
    for speaker, duration in sorted(speakers.items()):
        pct = 100 * duration / total_speech if total_speech > 0 else 0
        print(f"  {speaker}: {duration:.2f}s ({pct:.1f}%)")
    
    print(f"\n  Total speech: {total_speech:.2f}s")
    print(f"  Number of speakers: {len(speakers)}")
    print("=" * 60 + "\n")


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def run_single_file(
    pipeline,
    audio_path: str,
    output_path: Optional[str] = None,
    output_format: str = "rttm",
    num_speakers: Optional[int] = None,
    verbose: bool = False,
) -> List[Tuple[float, float, str]]:
    """Run diarization on single file."""
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    logger.info(f"Processing: {audio_path}")
    start_time = time.time()
    
    # Load audio
    audio_processor = SimpleAudioProcessor(sample_rate=pipeline.sample_rate)
    waveform, sr = audio_processor.load(str(audio_path))
    audio_duration = waveform.shape[-1] / sr
    
    logger.info(f"  Duration: {format_time(audio_duration)}")
    
    # Run diarization
    segments = pipeline(waveform, sample_rate=sr, num_speakers=num_speakers)
    
    elapsed = time.time() - start_time
    rtf = elapsed / audio_duration
    
    logger.info(f"  Completed in {elapsed:.2f}s (RTF: {rtf:.2f}x)")
    logger.info(f"  Found {len(segments)} segments, {len(set(s[2] for s in segments))} speakers")
    
    if verbose:
        print_segments(segments)
    
    # Save output
    if output_path is None:
        output_path = audio_path.with_suffix('.rttm')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = audio_path.stem
    
    # Use appropriate writer
    if HAS_UTILS:
        try:
            rttm_segments = {
                file_id: [
                    RTTMSegment(
                        file_id=file_id, channel=1, start=s,
                        duration=e-s, speaker_id=spk
                    )
                    for s, e, spk in segments
                ]
            }
            RTTMWriter.write(rttm_segments, str(output_path))
        except Exception:
            SimpleRTTMWriter.write(
                {file_id: [SimpleRTTMSegment(file_id, 1, s, e-s, spk) for s, e, spk in segments]},
                str(output_path)
            )
    else:
        SimpleRTTMWriter.write(
            {file_id: [SimpleRTTMSegment(file_id, 1, s, e-s, spk) for s, e, spk in segments]},
            str(output_path)
        )
    
    logger.info(f"  Saved: {output_path}")
    
    # Save JSON if requested
    if output_format in ["json", "both"]:
        json_path = output_path.with_suffix('.json')
        json_data = segments_to_json(segments, audio_duration)
        json_data["file"] = str(audio_path)
        json_data["processing_time"] = round(elapsed, 3)
        json_data["rtf"] = round(rtf, 3)
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"  Saved: {json_path}")
    
    return segments


def run_batch(
    pipeline,
    audio_paths: List[str],
    output_dir: str,
    output_format: str = "rttm",
    num_speakers: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, List[Tuple[float, float, str]]]:
    """Run diarization on multiple files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    total_audio_duration = 0.0
    total_processing_time = 0.0
    
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.warning(f"Not found: {audio_path}")
            continue
        
        try:
            output_path = output_dir / f"{audio_path.stem}.rttm"
            
            start_time = time.time()
            segments = run_single_file(
                pipeline, str(audio_path), str(output_path),
                output_format, num_speakers, verbose
            )
            elapsed = time.time() - start_time
            
            # Get duration
            audio_processor = SimpleAudioProcessor(sample_rate=pipeline.sample_rate)
            waveform, sr = audio_processor.load(str(audio_path))
            audio_duration = waveform.shape[-1] / sr
            
            total_audio_duration += audio_duration
            total_processing_time += elapsed
            
            all_results[audio_path.stem] = segments
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Batch Processing Summary")
    logger.info("=" * 60)
    logger.info(f"  Files processed: {len(all_results)}/{len(audio_paths)}")
    logger.info(f"  Total audio: {format_time(total_audio_duration)}")
    logger.info(f"  Total time: {total_processing_time:.2f}s")
    if total_audio_duration > 0:
        logger.info(f"  Average RTF: {total_processing_time / total_audio_duration:.2f}x")
    logger.info(f"  Results: {output_dir}")
    logger.info("=" * 60)
    
    # Save summary
    summary = {
        "num_files": len(all_results),
        "total_audio_duration": round(total_audio_duration, 3),
        "total_processing_time": round(total_processing_time, 3),
        "average_rtf": round(total_processing_time / max(total_audio_duration, 1e-8), 3),
        "files": {
            fid: {"num_segments": len(segs), "num_speakers": len(set(s[2] for s in segs))}
            for fid, segs in all_results.items()
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Speaker diarization inference")
    
    # Model
    parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint")
    
    # Input
    parser.add_argument("--audio-path", type=str, default=None, help="Single audio file")
    parser.add_argument("--audio-dir", type=str, default=None, help="Audio directory")
    parser.add_argument("--file-list", type=str, default=None, help="File list")
    
    # Output
    parser.add_argument("--output-path", type=str, default=None, help="Output file")
    parser.add_argument("--output-dir", type=str, default="./diarization_results", help="Output directory")
    parser.add_argument("--output-format", type=str, default="rttm", choices=["rttm", "json", "both"])
    
    # Diarization
    parser.add_argument("--num-speakers", type=int, default=None, help="Number of speakers")
    parser.add_argument("--window-duration", type=float, default=5.0, help="Window duration")
    parser.add_argument("--window-step", type=float, default=2.5, help="Window step")
    parser.add_argument("--min-segment-duration", type=float, default=0.1, help="Min segment duration")
    parser.add_argument("--merge-threshold", type=float, default=0.3, help="Merge threshold")
    parser.add_argument("--activation-threshold", type=float, default=0.5, help="Activation threshold")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    global logger
    logger = setup_logging(log_level=log_level)
    
    # Validate input
    if args.audio_path is None and args.audio_dir is None and args.file_list is None:
        logger.error("Must specify --audio-path, --audio-dir, or --file-list")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device=args.device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create pipeline
    if HAS_MODEL and DiarizationPipeline is not None:
        try:
            pipeline = DiarizationPipeline(
                model=model,
                device=args.device,
                window_duration=args.window_duration,
                window_step=args.window_step,
                min_segment_duration=args.min_segment_duration,
                merge_threshold=args.merge_threshold,
            )
            logger.info("Using DiarizationPipeline")
        except Exception as e:
            logger.warning(f"Could not create DiarizationPipeline: {e}")
            pipeline = SimpleDiarizationPipeline(
                model=model,
                device=args.device,
                window_duration=args.window_duration,
                window_step=args.window_step,
                min_segment_duration=args.min_segment_duration,
                merge_threshold=args.merge_threshold,
                activation_threshold=args.activation_threshold,
            )
            logger.info("Using SimpleDiarizationPipeline")
    else:
        pipeline = SimpleDiarizationPipeline(
            model=model,
            device=args.device,
            window_duration=args.window_duration,
            window_step=args.window_step,
            min_segment_duration=args.min_segment_duration,
            merge_threshold=args.merge_threshold,
            activation_threshold=args.activation_threshold,
        )
        logger.info("Using SimpleDiarizationPipeline")
    
    # Run inference
    if args.audio_path:
        # Single file
        run_single_file(
            pipeline, args.audio_path, args.output_path,
            args.output_format, args.num_speakers, args.verbose
        )
    else:
        # Batch
        audio_paths = []
        
        if args.audio_dir:
            audio_dir = Path(args.audio_dir)
            audio_paths = sorted(
                list(audio_dir.glob("*.wav")) +
                list(audio_dir.glob("*.mp3")) +
                list(audio_dir.glob("*.flac"))
            )
        
        if args.file_list:
            with open(args.file_list) as f:
                audio_paths = [Path(line.strip()) for line in f if line.strip()]
        
        if not audio_paths:
            logger.error("No audio files found")
            sys.exit(1)
        
        logger.info(f"Found {len(audio_paths)} audio files")
        
        run_batch(
            pipeline, [str(p) for p in audio_paths],
            args.output_dir, args.output_format,
            args.num_speakers, args.verbose
        )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()