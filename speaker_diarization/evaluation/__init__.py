"""Evaluation module for speaker diarization."""

from .metrics import (
    DERComponents,
    DiarizationMetrics,
    compute_der_simple,
    compute_der_pyannote,
    compute_jer,
    compute_coverage,
    compute_purity,
)
from .evaluator import (
    Evaluator,
    OracleEvaluator,
    evaluate_model,
)

__all__ = [
    # Metrics
    "DERComponents",
    "DiarizationMetrics",
    "compute_der_simple",
    "compute_der_pyannote",
    "compute_jer",
    "compute_coverage",
    "compute_purity",
    # Evaluator
    "Evaluator",
    "OracleEvaluator",
    "evaluate_model",
]