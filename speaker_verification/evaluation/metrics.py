# speaker_verification/evaluation/metrics.py
"""
Evaluation metrics for speaker verification.
Implements EER and MinDCF as per the paper.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from typing import Tuple, Dict


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        labels: Ground truth labels (0 or 1)
        scores: Similarity scores
        
    Returns:
        Tuple of (EER, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # Find EER
    try:
        eer_threshold = brentq(
            lambda x: interp1d(thresholds, fpr)(x) - interp1d(thresholds, fnr)(x),
            thresholds[0], thresholds[-1]
        )
        eer = interp1d(thresholds, fpr)(eer_threshold)
    except ValueError:
        # Fallback
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        eer_threshold = thresholds[idx]
    
    return float(eer), float(eer_threshold)


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0
) -> Tuple[float, float]:
    """
    Compute Minimum Detection Cost Function (MinDCF).
    
    From the paper:
    "minimum detection cost function (MinDCF) with 0.01 target probability"
    
    Args:
        labels: Ground truth labels
        scores: Similarity scores
        p_target: Prior probability of target
        c_miss: Cost of miss
        c_fa: Cost of false alarm
        
    Returns:
        Tuple of (MinDCF, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # DCF = C_miss * P_miss * P_target + C_fa * P_fa * (1 - P_target)
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    
    # Minimum DCF
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]
    
    # Normalize by default DCF (always accept or always reject)
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf_norm = min_dcf / default_dcf
    
    return float(min_dcf_norm), float(min_dcf_threshold)


def compute_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """Compute accuracy at given threshold."""
    predictions = (scores >= threshold).astype(int)
    return float(np.mean(predictions == labels) * 100)


def compute_verification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.01
) -> Dict[str, float]:
    """
    Compute all verification metrics.
    
    Args:
        labels: Ground truth labels
        scores: Similarity scores
        p_target: Prior probability for MinDCF
        
    Returns:
        Dictionary with EER, MinDCF, and thresholds
    """
    eer, eer_threshold = compute_eer(labels, scores)
    min_dcf, min_dcf_threshold = compute_min_dcf(labels, scores, p_target)
    accuracy = compute_accuracy(labels, scores, eer_threshold)
    
    return {
        'eer': eer * 100,  # Percentage
        'eer_threshold': eer_threshold,
        'min_dcf': min_dcf,
        'min_dcf_threshold': min_dcf_threshold,
        'accuracy': accuracy
    }