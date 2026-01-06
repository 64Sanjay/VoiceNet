# speaker_verification/evaluation/__init__.py
from .metrics import compute_eer, compute_min_dcf, compute_verification_metrics
from .evaluator import SpeakerVerificationEvaluator