# speaker_identification/evaluation/metrics.py
"""
Evaluation metrics for speaker identification.
Implements EER and AUC computation from Section 2.3.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Tuple, Dict


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    From Equation 11:
    EER = FPR(t*) = FNR(t*)
    t* = arg min_t |FPR(t) - FNR(t)|
    
    Args:
        labels: Ground truth labels (0 or 1)
        scores: Similarity scores
        
    Returns:
        Tuple of (EER, threshold)
    """
    # Compute FPR and TPR
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    
    # FNR = 1 - TPR
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR
    # Using interpolation for more accurate EER
    try:
        eer_threshold = brentq(lambda x: interp1d(thresholds, fpr)(x) - interp1d(thresholds, fnr)(x), 
                               thresholds[0], thresholds[-1])
        eer = interp1d(thresholds, fpr)(eer_threshold)
    except ValueError:
        # Fallback: find closest point
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        eer_threshold = thresholds[idx]
    
    return float(eer), float(eer_threshold)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        labels: Ground truth labels (0 or 1)
        scores: Similarity scores
        
    Returns:
        AUC score
    """
    return float(roc_auc_score(labels, scores))


def compute_far_frr(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR).
    
    From Equation 12:
    FPR(t) = FP / (FP + TN)
    FNR(t) = FN / (FN + TP)
    
    Args:
        labels: Ground truth labels
        scores: Similarity scores
        threshold: Decision threshold
        
    Returns:
        Tuple of (FAR, FRR)
    """
    predictions = (scores >= threshold).astype(int)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # FAR = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # FRR = FN / (FN + TP)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return float(far), float(frr)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute all speaker verification metrics.
    
    Args:
        labels: Ground truth labels (0 for different, 1 for same speaker)
        scores: Similarity scores
        
    Returns:
        Dictionary containing EER, AUC, and threshold
    """
    eer, threshold = compute_eer(labels, scores)
    auc_score = compute_auc(labels, scores)
    far, frr = compute_far_frr(labels, scores, threshold)
    
    return {
        'eer': eer * 100,  # Convert to percentage
        'eer_threshold': threshold,
        'auc': auc_score,
        'far': far * 100,
        'frr': frr * 100
    }


def compute_verification_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """
    Compute verification accuracy at a given threshold.
    
    Args:
        labels: Ground truth labels
        scores: Similarity scores
        threshold: Decision threshold
        
    Returns:
        Accuracy as a percentage
    """
    predictions = (scores >= threshold).astype(int)
    accuracy = np.mean(predictions == labels) * 100
    return float(accuracy)