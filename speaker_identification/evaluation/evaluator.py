# evaluation/evaluator.py
"""
WSI Evaluator for speaker verification evaluation.
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

from models.wsi_model import WSIModel
from evaluation.metrics import compute_metrics, compute_eer, compute_auc


class WSIEvaluator:
    """
    Evaluator for WSI speaker verification.
    
    Implements the evaluation procedure from Section 2.3:
    1. Extract embeddings for all utterances
    2. Compute cosine similarity between pairs
    3. Calculate EER and AUC
    """
    
    def __init__(
        self,
        model: WSIModel,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained WSI model
            device: Device to use for evaluation
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for all samples in dataloader.
        
        Args:
            dataloader: DataLoader with (features, speaker_ids) tuples
            
        Returns:
            Tuple of (embeddings, speaker_ids) as numpy arrays
        """
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            features, labels = batch[0], batch[-1]  # Handle both 2-tuple and 4-tuple
            features = features.to(self.device)
            
            # Get normalized embeddings (Equation 10)
            embeddings = self.model.get_embedding(features)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels
    
    def compute_cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Implements Equation 10:
        sim(z_1, z_2) = (z_1 · z_2) / (||z_1|| * ||z_2||)
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def create_verification_pairs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        num_positive_pairs: int = 5000,
        num_negative_pairs: int = 5000
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Create positive and negative pairs for verification.
        
        Args:
            embeddings: All embeddings
            labels: Speaker labels
            num_positive_pairs: Number of positive pairs to sample
            num_negative_pairs: Number of negative pairs to sample
            
        Returns:
            Tuple of (pairs, pair_labels)
            pairs: List of (idx1, idx2) tuples
            pair_labels: 1 for same speaker, 0 for different
        """
        # Group samples by speaker
        speaker_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in speaker_to_indices:
                speaker_to_indices[label] = []
            speaker_to_indices[label].append(idx)
        
        pairs = []
        pair_labels = []
        
        # Generate positive pairs (same speaker)
        positive_candidates = []
        for speaker, indices in speaker_to_indices.items():
            if len(indices) >= 2:
                for i, j in itertools.combinations(indices, 2):
                    positive_candidates.append((i, j))
        
        # Sample positive pairs
        np.random.shuffle(positive_candidates)
        for pair in positive_candidates[:num_positive_pairs]:
            pairs.append(pair)
            pair_labels.append(1)
        
        # Generate negative pairs (different speakers)
        speakers = list(speaker_to_indices.keys())
        negative_candidates = []
        for s1, s2 in itertools.combinations(speakers, 2):
            for i in speaker_to_indices[s1]:
                for j in speaker_to_indices[s2]:
                    negative_candidates.append((i, j))
        
        # Sample negative pairs
        np.random.shuffle(negative_candidates)
        for pair in negative_candidates[:num_negative_pairs]:
            pairs.append(pair)
            pair_labels.append(0)
        
        return pairs, pair_labels
    
    def evaluate(
        self,
        dataloader: DataLoader,
        num_pairs: int = 10000
    ) -> Dict[str, float]:
        """
        Evaluate the model on speaker verification.
        
        Args:
            dataloader: Evaluation data loader
            num_pairs: Total number of pairs to evaluate
            
        Returns:
            Dictionary with EER, AUC, and other metrics
        """
        # Extract embeddings
        embeddings, labels = self.extract_embeddings(dataloader)
        
        print(f"Extracted {len(embeddings)} embeddings")
        print(f"Unique speakers: {len(np.unique(labels))}")
        
        # Create verification pairs
        pairs, pair_labels = self.create_verification_pairs(
            embeddings, labels,
            num_positive_pairs=num_pairs // 2,
            num_negative_pairs=num_pairs // 2
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
        metrics = compute_metrics(pair_labels, scores)
        
        print(f"\nResults:")
        print(f"  EER: {metrics['eer']:.2f}%")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Threshold: {metrics['eer_threshold']:.4f}")
        
        return metrics