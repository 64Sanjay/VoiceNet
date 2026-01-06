# speaker_verification/evaluation/evaluator.py
"""
Evaluator for CAM++ Speaker Verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import itertools

from models.cam_plus_plus import CAMPlusPlus, CAMPlusPlusClassifier
from .metrics import compute_verification_metrics


class SpeakerVerificationEvaluator:
    """
    Evaluator for speaker verification.
    
    Supports:
    1. Embedding extraction
    2. Cosine similarity scoring
    3. EER and MinDCF computation
    """
    
    def __init__(
        self,
        model: CAMPlusPlus,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Handle both CAMPlusPlus and CAMPlusPlusClassifier
        if isinstance(model, CAMPlusPlusClassifier):
            self.model = model.encoder
        else:
            self.model = model
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all samples.
        
        Args:
            dataloader: DataLoader with (features, labels)
            
        Returns:
            Tuple of (embeddings, labels)
        """
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            features, labels = batch[0], batch[-1]
            features = features.to(self.device)
            
            embeddings = self.model.extract_embedding(features)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels
    
    def compute_cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        From the paper:
        "We use cosine similarity scoring for evaluation"
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def create_verification_pairs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_positive: int = 5000,
        num_negative: int = 5000
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Create positive and negative pairs for verification."""
        
        # Group by speaker
        speaker_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in speaker_to_indices:
                speaker_to_indices[label] = []
            speaker_to_indices[label].append(idx)
        
        pairs = []
        pair_labels = []
        
        # Positive pairs
        positive_candidates = []
        for speaker, indices in speaker_to_indices.items():
            if len(indices) >= 2:
                for i, j in itertools.combinations(indices, 2):
                    positive_candidates.append((i, j))
        
        np.random.shuffle(positive_candidates)
        for pair in positive_candidates[:num_positive]:
            pairs.append(pair)
            pair_labels.append(1)
        
        # Negative pairs
        speakers = list(speaker_to_indices.keys())
        negative_candidates = []
        for s1, s2 in itertools.combinations(speakers, 2):
            for i in speaker_to_indices[s1][:3]:
                for j in speaker_to_indices[s2][:3]:
                    negative_candidates.append((i, j))
        
        np.random.shuffle(negative_candidates)
        for pair in negative_candidates[:num_negative]:
            pairs.append(pair)
            pair_labels.append(0)
        
        return pairs, pair_labels
    
    def evaluate(
        self,
        dataloader: DataLoader,
        num_pairs: int = 10000
    ) -> Dict[str, float]:
        """
        Evaluate speaker verification performance.
        
        Args:
            dataloader: DataLoader with test data
            num_pairs: Number of pairs to evaluate
            
        Returns:
            Dictionary with EER, MinDCF, etc.
        """
        # Extract embeddings
        embeddings, labels = self.extract_embeddings(dataloader)
        
        print(f"Extracted {len(embeddings)} embeddings")
        print(f"Unique speakers: {len(np.unique(labels))}")
        
        # Create pairs
        pairs, pair_labels = self.create_verification_pairs(
            embeddings, labels,
            num_positive=num_pairs // 2,
            num_negative=num_pairs // 2
        )
        
        print(f"Created {len(pairs)} verification pairs")
        
        # Compute similarities
        scores = []
        for idx1, idx2 in tqdm(pairs, desc="Computing similarities"):
            sim = self.compute_cosine_similarity(embeddings[idx1], embeddings[idx2])
            scores.append(sim)
        
        scores = np.array(scores)
        pair_labels = np.array(pair_labels)
        
        # Compute metrics
        metrics = compute_verification_metrics(pair_labels, scores)
        
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"  EER: {metrics['eer']:.2f}%")
        print(f"  MinDCF: {metrics['min_dcf']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Threshold: {metrics['eer_threshold']:.4f}")
        print(f"{'='*50}")
        
        return metrics
    
    def evaluate_trial_file(
        self,
        trial_file: str,
        embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate using a trial file.
        
        Args:
            trial_file: Path to trial file (label enroll_id test_id)
            embeddings: Dictionary mapping audio_id to embedding
            
        Returns:
            Evaluation metrics
        """
        labels = []
        scores = []
        
        with open(trial_file, 'r') as f:
            for line in tqdm(f, desc="Evaluating trials"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    enroll_id = parts[1]
                    test_id = parts[2]
                    
                    if enroll_id in embeddings and test_id in embeddings:
                        sim = self.compute_cosine_similarity(
                            embeddings[enroll_id],
                            embeddings[test_id]
                        )
                        labels.append(label)
                        scores.append(sim)
        
        labels = np.array(labels)
        scores = np.array(scores)
        
        return compute_verification_metrics(labels, scores)