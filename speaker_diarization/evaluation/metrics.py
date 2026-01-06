"""
Evaluation metrics for speaker diarization.

Primary metric: Diarization Error Rate (DER)
Additional metrics: JER, Coverage, Purity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from pyannote.core import Segment, Annotation, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.metrics.detection import DetectionErrorRate
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


@dataclass
class DERComponents:
    """Components of Diarization Error Rate."""
    false_alarm: float  # Speaker detected when none present
    missed_detection: float  # Speaker present but not detected
    confusion: float  # Wrong speaker detected
    total_speech: float  # Total speech duration
    
    @property
    def der(self) -> float:
        """Compute DER as percentage."""
        if self.total_speech == 0:
            return 0.0
        return 100 * (
            self.false_alarm + self.missed_detection + self.confusion
        ) / self.total_speech


def compute_der_simple(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> DERComponents:
    """
    Compute Diarization Error Rate without pyannote.
    
    Args:
        reference: Ground truth segments [(start, end, speaker), ...]
        hypothesis: Predicted segments [(start, end, speaker), ...]
        collar: Forgiveness collar in seconds
        skip_overlap: Skip overlapping speech regions
        
    Returns:
        DER components
    """
    if not reference:
        return DERComponents(0, 0, 0, 0)
    
    # Get time boundaries
    all_times = set()
    for start, end, _ in reference + hypothesis:
        all_times.add(start)
        all_times.add(end)
        # Add collar boundaries
        all_times.add(start + collar)
        all_times.add(start - collar)
        all_times.add(end + collar)
        all_times.add(end - collar)
    
    all_times = sorted(filter(lambda x: x >= 0, all_times))
    
    false_alarm = 0.0
    missed_detection = 0.0
    confusion = 0.0
    total_speech = 0.0
    
    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        duration = end - start
        
        # Get active speakers in this segment
        ref_speakers = set()
        hyp_speakers = set()
        
        for seg_start, seg_end, spk in reference:
            if seg_start + collar <= start and end <= seg_end - collar:
                ref_speakers.add(spk)
        
        for seg_start, seg_end, spk in hypothesis:
            if seg_start <= start and end <= seg_end:
                hyp_speakers.add(spk)
        
        # Skip overlap if requested
        if skip_overlap and len(ref_speakers) > 1:
            continue
        
        # Count errors
        n_ref = len(ref_speakers)
        n_hyp = len(hyp_speakers)
        
        if n_ref > 0:
            total_speech += duration * n_ref
        
        if n_ref == 0 and n_hyp > 0:
            false_alarm += duration * n_hyp
        elif n_ref > 0 and n_hyp == 0:
            missed_detection += duration * n_ref
        elif n_ref > 0 and n_hyp > 0:
            # Use optimal mapping (simplified)
            n_correct = min(n_ref, n_hyp)
            if n_hyp > n_ref:
                false_alarm += duration * (n_hyp - n_ref)
            elif n_ref > n_hyp:
                missed_detection += duration * (n_ref - n_hyp)
            
            # Confusion (simplified - assumes no matching)
            # Full implementation would use Hungarian matching
            confusion += duration * max(0, n_correct - len(ref_speakers & hyp_speakers))
    
    return DERComponents(
        false_alarm=false_alarm,
        missed_detection=missed_detection,
        confusion=confusion,
        total_speech=total_speech,
    )


def compute_der_pyannote(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.25,
    skip_overlap: bool = False,
    uri: str = "audio",
) -> Dict[str, float]:
    """
    Compute DER using pyannote.metrics.
    
    Args:
        reference: Ground truth segments
        hypothesis: Predicted segments
        collar: Forgiveness collar
        skip_overlap: Skip overlapping speech
        uri: Audio identifier
        
    Returns:
        Dictionary with DER and components
    """
    if not PYANNOTE_AVAILABLE:
        raise ImportError("pyannote.metrics required for compute_der_pyannote")
    
    # Create pyannote Annotations
    ref_annotation = Annotation(uri=uri)
    hyp_annotation = Annotation(uri=uri)
    
    for start, end, speaker in reference:
        ref_annotation[Segment(start, end)] = speaker
    
    for start, end, speaker in hypothesis:
        hyp_annotation[Segment(start, end)] = speaker
    
    # Compute DER
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    der = metric(ref_annotation, hyp_annotation)
    
    return {
        'der': der * 100,
        'false_alarm': metric['false alarm'] * 100,
        'missed_detection': metric['missed detection'] * 100,
        'confusion': metric['confusion'] * 100,
    }


def compute_jer(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute Jaccard Error Rate (JER).
    
    JER = 1 - Jaccard Index
    
    Args:
        reference: Ground truth segments
        hypothesis: Predicted segments
        
    Returns:
        JER as percentage
    """
    if not reference:
        return 0.0 if not hypothesis else 100.0
    
    # Group segments by speaker
    ref_by_speaker = defaultdict(list)
    hyp_by_speaker = defaultdict(list)
    
    for start, end, spk in reference:
        ref_by_speaker[spk].append((start, end))
    
    for start, end, spk in hypothesis:
        hyp_by_speaker[spk].append((start, end))
    
    # Compute optimal matching using Hungarian algorithm (simplified)
    total_jer = 0.0
    n_speakers = 0
    
    for ref_spk, ref_segs in ref_by_speaker.items():
        best_jer = 1.0
        
        for hyp_spk, hyp_segs in hyp_by_speaker.items():
            # Compute intersection and union
            intersection = 0.0
            for r_start, r_end in ref_segs:
                for h_start, h_end in hyp_segs:
                    i_start = max(r_start, h_start)
                    i_end = min(r_end, h_end)
                    if i_end > i_start:
                        intersection += i_end - i_start
            
            ref_total = sum(end - start for start, end in ref_segs)
            hyp_total = sum(end - start for start, end in hyp_segs)
            union = ref_total + hyp_total - intersection
            
            if union > 0:
                jaccard = intersection / union
                jer = 1 - jaccard
                best_jer = min(best_jer, jer)
        
        total_jer += best_jer
        n_speakers += 1
    
    return 100 * total_jer / max(n_speakers, 1)


def compute_coverage(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute coverage (recall) of speaker detection.
    
    Args:
        reference: Ground truth segments
        hypothesis: Predicted segments
        
    Returns:
        Coverage as percentage
    """
    if not reference:
        return 100.0
    
    # Compute total reference duration
    ref_total = sum(end - start for start, end, _ in reference)
    
    # Compute covered duration
    covered = 0.0
    for r_start, r_end, r_spk in reference:
        for h_start, h_end, _ in hypothesis:
            i_start = max(r_start, h_start)
            i_end = min(r_end, h_end)
            if i_end > i_start:
                covered += i_end - i_start
    
    return 100 * covered / ref_total


def compute_purity(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute purity (precision) of speaker clustering.
    
    Args:
        reference: Ground truth segments
        hypothesis: Predicted segments
        
    Returns:
        Purity as percentage
    """
    if not hypothesis:
        return 100.0
    
    # For each hypothesis segment, find dominant reference speaker
    total_duration = 0.0
    correct_duration = 0.0
    
    for h_start, h_end, h_spk in hypothesis:
        h_duration = h_end - h_start
        total_duration += h_duration
        
        # Count overlap with each reference speaker
        speaker_overlap = defaultdict(float)
        
        for r_start, r_end, r_spk in reference:
            i_start = max(h_start, r_start)
            i_end = min(h_end, r_end)
            if i_end > i_start:
                speaker_overlap[r_spk] += i_end - i_start
        
        if speaker_overlap:
            correct_duration += max(speaker_overlap.values())
    
    return 100 * correct_duration / max(total_duration, 1e-8)


class DiarizationMetrics:
    """
    Compute all diarization metrics.
    """
    
    def __init__(
        self,
        collar: float = 0.25,
        skip_overlap: bool = False,
        use_pyannote: bool = True,
    ):
        """
        Initialize metrics.
        
        Args:
            collar: Forgiveness collar in seconds
            skip_overlap: Skip overlapping speech
            use_pyannote: Use pyannote.metrics if available
        """
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.use_pyannote = use_pyannote and PYANNOTE_AVAILABLE
    
    def __call__(
        self,
        reference: List[Tuple[float, float, str]],
        hypothesis: List[Tuple[float, float, str]],
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            reference: Ground truth segments
            hypothesis: Predicted segments
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # DER
        if self.use_pyannote:
            try:
                der_results = compute_der_pyannote(
                    reference, hypothesis,
                    collar=self.collar,
                    skip_overlap=self.skip_overlap,
                )
                results.update(der_results)
            except Exception:
                der_components = compute_der_simple(
                    reference, hypothesis,
                    collar=self.collar,
                    skip_overlap=self.skip_overlap,
                )
                results['der'] = der_components.der
                results['false_alarm'] = der_components.false_alarm
                results['missed_detection'] = der_components.missed_detection
                results['confusion'] = der_components.confusion
        else:
            der_components = compute_der_simple(
                reference, hypothesis,
                collar=self.collar,
                skip_overlap=self.skip_overlap,
            )
            results['der'] = der_components.der
            results['false_alarm'] = der_components.false_alarm
            results['missed_detection'] = der_components.missed_detection
            results['confusion'] = der_components.confusion
        
        # JER
        results['jer'] = compute_jer(reference, hypothesis)
        
        # Coverage and Purity
        results['coverage'] = compute_coverage(reference, hypothesis)
        results['purity'] = compute_purity(reference, hypothesis)
        
        return results