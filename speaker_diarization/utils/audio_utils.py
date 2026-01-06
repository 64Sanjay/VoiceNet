"""
Audio utilities for speaker diarization.
Handles audio I/O, resampling, and basic processing.
"""

from typing import List, Optional, Tuple, Union

import torch
import torchaudio
import numpy as np
from pathlib import Path
import warnings

# Set torchaudio backend for compatibility with version 2.x
try:
    # Try to use soundfile backend (requires soundfile package)
    torchaudio.set_audio_backend("soundfile")
except Exception:
    try:
        # Try sox_io backend
        torchaudio.set_audio_backend("sox_io")
    except Exception:
        # If all else fails, we'll handle it in the load function
        pass


def load_audio(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """
    Load audio file with fallback methods.
    Compatible with different torchaudio versions.
    """
    path = str(path)
    
    # Try torchaudio.load first
    try:
        waveform, sr = torchaudio.load(path)
        return waveform, sr
    except Exception as e1:
        pass
    
    # Try with soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        # Convert to torch tensor and reshape
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T  # soundfile returns [samples, channels]
        return waveform, sr
    except ImportError:
        pass
    except Exception as e2:
        pass
    
    # Try with scipy
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        waveform = torch.from_numpy(data).float()
        # Normalize to [-1, 1]
        if data.dtype == np.int16:
            waveform = waveform / 32768.0
        elif data.dtype == np.int32:
            waveform = waveform / 2147483648.0
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        return waveform, sr
    except Exception as e3:
        pass
    
    raise RuntimeError(f"Could not load audio file: {path}. Please install soundfile: pip install soundfile")


class AudioProcessor:
    """Handles audio loading, saving, and basic processing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
    
    def load(
        self,
        path: Union[str, Path],
        start: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int]:
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Load the whole file using our robust loader
        waveform, original_sr = load_audio(path)
        
        # Handle start/duration
        if start is not None or duration is not None:
            start_sample = int((start or 0) * original_sr)
            if duration is not None:
                end_sample = start_sample + int(duration * original_sr)
            else:
                end_sample = waveform.shape[-1]
            waveform = waveform[:, start_sample:end_sample]
        
        # Convert to mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if original_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, original_sr, self.sample_rate
            )
        
        # Normalize
        if self.normalize:
            waveform = self._normalize(waveform)
        
        return waveform, self.sample_rate
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def save(
        self,
        waveform: torch.Tensor,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = sample_rate or self.sample_rate
        
        try:
            torchaudio.save(str(path), waveform, sr)
        except Exception:
            # Fallback to soundfile
            import soundfile as sf
            # Convert to numpy and transpose for soundfile
            data = waveform.numpy()
            if data.shape[0] <= 2:  # channels first
                data = data.T
            sf.write(str(path), data, sr)
    
    def get_duration(self, path: Union[str, Path]) -> float:
        """Get audio duration in seconds."""
        waveform, sr = load_audio(path)
        return waveform.shape[-1] / sr
    
    def segment(
        self,
        waveform: torch.Tensor,
        segment_duration: float,
        segment_step: Optional[float] = None,
        drop_last: bool = False,
    ) -> List[torch.Tensor]:
        if segment_step is None:
            segment_step = segment_duration
        
        segment_samples = int(segment_duration * self.sample_rate)
        step_samples = int(segment_step * self.sample_rate)
        
        total_samples = waveform.shape[-1]
        segments = []
        
        start = 0
        while start < total_samples:
            end = start + segment_samples
            segment = waveform[..., start:end]
            
            if segment.shape[-1] < segment_samples:
                if drop_last:
                    break
                padding = segment_samples - segment.shape[-1]
                segment = torch.nn.functional.pad(segment, (0, padding))
            
            segments.append(segment)
            start += step_samples
        
        return segments
    
    def concat(self, segments: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(segments, dim=-1)


def compute_energy(
    waveform: torch.Tensor,
    frame_length: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    frames = waveform.unfold(-1, frame_length, hop_length)
    energy = (frames ** 2).sum(dim=-1)
    return energy.squeeze(0) if energy.shape[0] == 1 else energy


def simple_vad(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    energy_threshold: float = 0.01,
    frame_length: float = 0.025,
    hop_length: float = 0.010,
) -> torch.Tensor:
    frame_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    energy = compute_energy(waveform, frame_samples, hop_samples)
    max_energy = energy.max()
    if max_energy > 0:
        energy = energy / max_energy
    vad_mask = energy > energy_threshold
    return vad_mask


def apply_vad_mask(
    waveform: torch.Tensor,
    vad_mask: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: float = 0.010,
) -> List[Tuple[float, float]]:
    hop_samples = int(hop_length * sample_rate)
    segments = []
    in_speech = False
    start = 0
    
    for i, is_speech in enumerate(vad_mask):
        if is_speech and not in_speech:
            start = i * hop_samples / sample_rate
            in_speech = True
        elif not is_speech and in_speech:
            end = i * hop_samples / sample_rate
            segments.append((start, end))
            in_speech = False
    
    if in_speech:
        end = len(vad_mask) * hop_samples / sample_rate
        segments.append((start, end))
    
    return segments


def mix_audio(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    if noise.shape[-1] < signal.shape[-1]:
        repeats = signal.shape[-1] // noise.shape[-1] + 1
        noise = noise.repeat(1, repeats)[..., :signal.shape[-1]]
    else:
        noise = noise[..., :signal.shape[-1]]
    
    signal_power = (signal ** 2).mean()
    noise_power = (noise ** 2).mean()
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
    mixed = signal + scale * noise
    return mixed
