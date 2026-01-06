"""
RTTM (Rich Transcription Time Marked) file utilities.
Standard format for speaker diarization annotations.

RTTM Format:
SPEAKER <file_id> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class RTTMSegment:
    """Represents a single RTTM segment."""
    file_id: str
    channel: int
    start: float
    duration: float
    speaker_id: str
    
    @property
    def end(self) -> float:
        """Get segment end time."""
        return self.start + self.duration
    
    def overlaps(self, other: "RTTMSegment") -> bool:
        """Check if this segment overlaps with another."""
        return self.start < other.end and other.start < self.end
    
    def overlap_duration(self, other: "RTTMSegment") -> float:
        """Compute overlap duration with another segment."""
        if not self.overlaps(other):
            return 0.0
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        return overlap_end - overlap_start
    
    def to_rttm_line(self) -> str:
        """Convert to RTTM format line."""
        return f"SPEAKER {self.file_id} {self.channel} {self.start:.3f} {self.duration:.3f} <NA> <NA> {self.speaker_id} <NA> <NA>"
    
    @classmethod
    def from_rttm_line(cls, line: str) -> "RTTMSegment":
        """Parse RTTM line to RTTMSegment."""
        parts = line.strip().split()
        if len(parts) < 9:
            raise ValueError(f"Invalid RTTM line: {line}")
        
        return cls(
            file_id=parts[1],
            channel=int(parts[2]),
            start=float(parts[3]),
            duration=float(parts[4]),
            speaker_id=parts[7],
        )


class RTTMReader:
    """Read and parse RTTM files."""
    
    def __init__(self):
        self.segments: Dict[str, List[RTTMSegment]] = defaultdict(list)
    
    def read(self, path: Union[str, Path]) -> Dict[str, List[RTTMSegment]]:
        """
        Read RTTM file.
        
        Args:
            path: Path to RTTM file
            
        Returns:
            Dictionary mapping file_id to list of segments
        """
        path = Path(path)
        self.segments = defaultdict(list)
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('SPEAKER'):
                    segment = RTTMSegment.from_rttm_line(line)
                    self.segments[segment.file_id].append(segment)
        
        # Sort segments by start time
        for file_id in self.segments:
            self.segments[file_id].sort(key=lambda x: x.start)
        
        return dict(self.segments)
    
    def read_directory(
        self,
        directory: Union[str, Path],
        extension: str = ".rttm"
    ) -> Dict[str, List[RTTMSegment]]:
        """Read all RTTM files in a directory."""
        directory = Path(directory)
        all_segments = defaultdict(list)
        
        for rttm_file in directory.glob(f"*{extension}"):
            segments = self.read(rttm_file)
            for file_id, segs in segments.items():
                all_segments[file_id].extend(segs)
        
        return dict(all_segments)


class RTTMWriter:
    """Write RTTM files."""
    
    @staticmethod
    def write(
        segments: Dict[str, List[RTTMSegment]],
        path: Union[str, Path],
    ) -> None:
        """
        Write segments to RTTM file.
        
        Args:
            segments: Dictionary mapping file_id to list of segments
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for file_id in sorted(segments.keys()):
                for segment in sorted(segments[file_id], key=lambda x: x.start):
                    f.write(segment.to_rttm_line() + '\n')
    
    @staticmethod
    def write_from_predictions(
        predictions: Dict[str, List[Tuple[float, float, str]]],
        path: Union[str, Path],
        channel: int = 1,
    ) -> None:
        """
        Write predictions to RTTM file.
        
        Args:
            predictions: Dict mapping file_id to list of (start, end, speaker_id) tuples
            path: Output path
            channel: Channel number
        """
        segments = defaultdict(list)
        
        for file_id, preds in predictions.items():
            for start, end, speaker_id in preds:
                segment = RTTMSegment(
                    file_id=file_id,
                    channel=channel,
                    start=start,
                    duration=end - start,
                    speaker_id=speaker_id,
                )
                segments[file_id].append(segment)
        
        RTTMWriter.write(segments, path)


def segments_to_frames(
    segments: List[RTTMSegment],
    total_duration: float,
    frame_duration: float = 0.01,
    num_speakers: Optional[int] = None,
) -> np.ndarray:
    """
    Convert RTTM segments to frame-level labels.
    
    Args:
        segments: List of RTTM segments
        total_duration: Total audio duration in seconds
        frame_duration: Frame duration in seconds
        num_speakers: Maximum number of speakers (auto-detected if None)
        
    Returns:
        Frame-level labels [num_frames, num_speakers] with binary values
    """
    num_frames = int(np.ceil(total_duration / frame_duration))
    
    # Get unique speakers
    speakers = sorted(set(seg.speaker_id for seg in segments))
    if num_speakers is None:
        num_speakers = len(speakers)
    
    speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
    
    # Create frame labels
    labels = np.zeros((num_frames, num_speakers), dtype=np.float32)
    
    for segment in segments:
        if segment.speaker_id not in speaker_to_idx:
            continue
        
        spk_idx = speaker_to_idx[segment.speaker_id]
        start_frame = int(segment.start / frame_duration)
        end_frame = int(segment.end / frame_duration)
        
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        labels[start_frame:end_frame, spk_idx] = 1.0
    
    return labels


def frames_to_segments(
    frame_labels: np.ndarray,
    frame_duration: float = 0.01,
    threshold: float = 0.5,
    min_duration: float = 0.1,
    file_id: str = "audio",
    speaker_prefix: str = "speaker_",
) -> List[RTTMSegment]:
    """
    Convert frame-level predictions to RTTM segments.
    
    Args:
        frame_labels: Frame-level predictions [num_frames, num_speakers]
        frame_duration: Frame duration in seconds
        threshold: Threshold for binary classification
        min_duration: Minimum segment duration
        file_id: File identifier
        speaker_prefix: Prefix for speaker IDs
        
    Returns:
        List of RTTM segments
    """
    num_frames, num_speakers = frame_labels.shape
    segments = []
    
    # Binarize
    binary_labels = (frame_labels > threshold).astype(np.int32)
    
    for spk_idx in range(num_speakers):
        speaker_id = f"{speaker_prefix}{spk_idx}"
        spk_labels = binary_labels[:, spk_idx]
        
        # Find contiguous regions
        in_segment = False
        start_frame = 0
        
        for frame_idx in range(num_frames):
            if spk_labels[frame_idx] == 1 and not in_segment:
                start_frame = frame_idx
                in_segment = True
            elif spk_labels[frame_idx] == 0 and in_segment:
                end_frame = frame_idx
                start_time = start_frame * frame_duration
                end_time = end_frame * frame_duration
                
                if end_time - start_time >= min_duration:
                    segment = RTTMSegment(
                        file_id=file_id,
                        channel=1,
                        start=start_time,
                        duration=end_time - start_time,
                        speaker_id=speaker_id,
                    )
                    segments.append(segment)
                
                in_segment = False
        
        # Handle last segment
        if in_segment:
            end_frame = num_frames
            start_time = start_frame * frame_duration
            end_time = end_frame * frame_duration
            
            if end_time - start_time >= min_duration:
                segment = RTTMSegment(
                    file_id=file_id,
                    channel=1,
                    start=start_time,
                    duration=end_time - start_time,
                    speaker_id=speaker_id,
                )
                segments.append(segment)
    
    return segments


def merge_adjacent_segments(
    segments: List[RTTMSegment],
    max_gap: float = 0.3,
) -> List[RTTMSegment]:
    """
    Merge adjacent segments of the same speaker.
    
    Args:
        segments: List of segments
        max_gap: Maximum gap between segments to merge
        
    Returns:
        Merged segments
    """
    if not segments:
        return []
    
    # Group by speaker
    speaker_segments = defaultdict(list)
    for seg in segments:
        speaker_segments[seg.speaker_id].append(seg)
    
    merged = []
    for speaker_id, spk_segs in speaker_segments.items():
        spk_segs = sorted(spk_segs, key=lambda x: x.start)
        
        current = spk_segs[0]
        for next_seg in spk_segs[1:]:
            gap = next_seg.start - current.end
            
            if gap <= max_gap:
                # Merge
                current = RTTMSegment(
                    file_id=current.file_id,
                    channel=current.channel,
                    start=current.start,
                    duration=next_seg.end - current.start,
                    speaker_id=current.speaker_id,
                )
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
    
    return sorted(merged, key=lambda x: x.start)


def compute_overlap_regions(
    segments: List[RTTMSegment],
) -> List[Tuple[float, float, List[str]]]:
    """
    Find overlapping speech regions.
    
    Args:
        segments: List of segments
        
    Returns:
        List of (start, end, [speakers]) tuples for overlap regions
    """
    if not segments:
        return []
    
    # Create events
    events = []
    for seg in segments:
        events.append((seg.start, 'start', seg.speaker_id))
        events.append((seg.end, 'end', seg.speaker_id))
    
    events.sort(key=lambda x: (x[0], x[1] == 'start'))
    
    # Process events
    active_speakers = set()
    overlaps = []
    prev_time = 0
    
    for time, event_type, speaker_id in events:
        if len(active_speakers) > 1 and time > prev_time:
            overlaps.append((prev_time, time, list(active_speakers)))
        
        if event_type == 'start':
            active_speakers.add(speaker_id)
        else:
            active_speakers.discard(speaker_id)
        
        prev_time = time
    
    return overlaps


def get_speaker_statistics(
    segments: List[RTTMSegment],
) -> Dict[str, Dict[str, float]]:
    """
    Compute speaker statistics from segments.
    
    Args:
        segments: List of segments
        
    Returns:
        Dictionary with speaker statistics
    """
    speaker_stats = defaultdict(lambda: {
        'total_duration': 0.0,
        'num_segments': 0,
        'avg_segment_duration': 0.0,
    })
    
    for seg in segments:
        speaker_stats[seg.speaker_id]['total_duration'] += seg.duration
        speaker_stats[seg.speaker_id]['num_segments'] += 1
    
    for speaker_id in speaker_stats:
        stats = speaker_stats[speaker_id]
        if stats['num_segments'] > 0:
            stats['avg_segment_duration'] = (
                stats['total_duration'] / stats['num_segments']
            )
    
    return dict(speaker_stats)