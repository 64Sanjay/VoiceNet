"""
Clustering algorithms for speaker diarization.

Includes:
- Agglomerative Hierarchical Clustering (AHC)
- Spectral Clustering
- VBx (Variational Bayes HMM x-vectors)
- Online clustering for streaming
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Union
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.ndimage import gaussian_filter1d
import warnings

try:
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Some clustering methods will be disabled.")


class SpeakerClustering:
    """
    Base class for speaker clustering.
    """
    
    def __init__(self, metric: str = "cosine"):
        """
        Initialize clustering.
        
        Args:
            metric: Distance metric (cosine, euclidean)
        """
        self.metric = metric
    
    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster embeddings.
        
        Args:
            embeddings: Speaker embeddings [N, D]
            num_clusters: Number of clusters (optional)
            
        Returns:
            labels: Cluster labels [N]
        """
        raise NotImplementedError
    
    def _compute_affinity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute affinity matrix from embeddings."""
        if self.metric == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)
            # Cosine similarity
            affinity = np.dot(normalized, normalized.T)
            # Convert to [0, 1] range
            affinity = (affinity + 1) / 2
        else:
            # Euclidean distance -> affinity
            distances = squareform(pdist(embeddings, metric='euclidean'))
            sigma = np.median(distances)
            affinity = np.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))
        
        return affinity
    
    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        if self.metric == "cosine":
            return squareform(pdist(embeddings, metric='cosine'))
        else:
            return squareform(pdist(embeddings, metric='euclidean'))


class AgglomerativeHierarchicalClustering(SpeakerClustering):
    """
    Agglomerative Hierarchical Clustering (AHC).
    
    Most commonly used clustering method for speaker diarization.
    Builds a hierarchy of clusters by progressively merging.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        linkage_type: str = "average",
        metric: str = "cosine",
        min_clusters: int = 1,
        max_clusters: int = 20,
    ):
        """
        Initialize AHC.
        
        Args:
            threshold: Distance threshold for clustering
            linkage_type: Linkage type (single, complete, average, ward)
            metric: Distance metric
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
        """
        super().__init__(metric)
        
        self.threshold = threshold
        self.linkage_type = linkage_type
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
    
    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster embeddings using AHC.
        
        Args:
            embeddings: Speaker embeddings [N, D]
            num_clusters: Number of clusters (if known)
            
        Returns:
            labels: Cluster labels [N]
        """
        n_samples = embeddings.shape[0]
        
        if n_samples == 0:
            return np.array([])
        
        if n_samples == 1:
            return np.array([0])
        
        # Use sklearn if available and num_clusters is specified
        if num_clusters is not None and SKLEARN_AVAILABLE:
            affinity = 'cosine' if self.metric == 'cosine' else 'euclidean'
            link = 'average' if self.linkage_type == 'average' else self.linkage_type
            
            # Ward requires euclidean
            if self.linkage_type == 'ward':
                affinity = 'euclidean'
                link = 'ward'
            
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                metric=affinity,
                linkage=link,
            )
            return clustering.fit_predict(embeddings)
        
        # Compute condensed distance matrix
        if self.metric == "cosine":
            distances = pdist(embeddings, metric='cosine')
        else:
            distances = pdist(embeddings, metric='euclidean')
        
        # Handle NaN values
        distances = np.nan_to_num(distances, nan=1.0)
        
        # Hierarchical clustering
        Z = linkage(distances, method=self.linkage_type)
        
        if num_clusters is not None:
            # Fixed number of clusters
            labels = fcluster(Z, t=num_clusters, criterion='maxclust') - 1
        else:
            # Cut dendrogram at threshold
            labels = fcluster(Z, t=self.threshold, criterion='distance') - 1
            
            # Check cluster count constraints
            n_clusters = len(np.unique(labels))
            
            if n_clusters < self.min_clusters:
                labels = fcluster(Z, t=self.min_clusters, criterion='maxclust') - 1
            elif n_clusters > self.max_clusters:
                labels = fcluster(Z, t=self.max_clusters, criterion='maxclust') - 1
        
        return labels
    
    def find_optimal_threshold(
        self,
        embeddings: np.ndarray,
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        num_steps: int = 20,
    ) -> float:
        """
        Find optimal threshold using silhouette score.
        
        Args:
            embeddings: Speaker embeddings
            threshold_range: Range of thresholds to try
            num_steps: Number of threshold values to try
            
        Returns:
            Optimal threshold
        """
        if not SKLEARN_AVAILABLE:
            return self.threshold
        
        if embeddings.shape[0] < 3:
            return self.threshold
        
        thresholds = np.linspace(*threshold_range, num_steps)
        best_score = -1
        best_threshold = self.threshold
        
        # Compute distances once
        if self.metric == "cosine":
            distances = pdist(embeddings, metric='cosine')
        else:
            distances = pdist(embeddings, metric='euclidean')
        
        distances = np.nan_to_num(distances, nan=1.0)
        Z = linkage(distances, method=self.linkage_type)
        
        for threshold in thresholds:
            labels = fcluster(Z, t=threshold, criterion='distance') - 1
            n_clusters = len(np.unique(labels))
            
            if n_clusters < 2 or n_clusters >= embeddings.shape[0]:
                continue
            
            try:
                score = silhouette_score(embeddings, labels, metric=self.metric)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except ValueError:
                continue
        
        return best_threshold


class SpectralClusteringWrapper(SpeakerClustering):
    """
    Spectral Clustering for speaker diarization.
    
    Works well when clusters have complex shapes.
    Uses affinity matrix eigen-decomposition.
    """
    
    def __init__(
        self,
        metric: str = "cosine",
        min_clusters: int = 2,
        max_clusters: int = 10,
        n_neighbors: int = 10,
    ):
        """
        Initialize Spectral Clustering.
        
        Args:
            metric: Distance metric
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            n_neighbors: Number of neighbors for affinity
        """
        super().__init__(metric)
        
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_neighbors = n_neighbors
    
    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster embeddings using Spectral Clustering.
        
        Args:
            embeddings: Speaker embeddings [N, D]
            num_clusters: Number of clusters
            
        Returns:
            labels: Cluster labels [N]
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for SpectralClustering")
        
        n_samples = embeddings.shape[0]
        
        if n_samples == 0:
            return np.array([])
        
        if n_samples == 1:
            return np.array([0])
        
        # Compute affinity matrix
        affinity = self._compute_affinity(embeddings)
        
        # Estimate number of clusters if not provided
        if num_clusters is None:
            num_clusters = self._estimate_num_clusters(affinity)
        
        num_clusters = max(self.min_clusters, min(num_clusters, self.max_clusters, n_samples))
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
        )
        
        labels = clustering.fit_predict(affinity)
        
        return labels
    
    def _estimate_num_clusters(self, affinity: np.ndarray) -> int:
        """
        Estimate number of clusters using eigenvalue analysis.
        
        Args:
            affinity: Affinity matrix
            
        Returns:
            Estimated number of clusters
        """
        n = affinity.shape[0]
        
        if n <= self.min_clusters:
            return self.min_clusters
        
        # Compute normalized Laplacian eigenvalues
        degree = np.sum(affinity, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
        L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
        
        try:
            eigenvalues = np.linalg.eigvalsh(L_norm)
            eigenvalues = np.sort(eigenvalues)
            
            # Find largest gap
            gaps = np.diff(eigenvalues[:self.max_clusters + 1])
            num_clusters = np.argmax(gaps) + 1
            
            return max(self.min_clusters, min(num_clusters, self.max_clusters))
        except np.linalg.LinAlgError:
            return self.min_clusters


class VBxClustering(SpeakerClustering):
    """
    Variational Bayes HMM x-vector Clustering (VBx).
    
    State-of-the-art clustering for speaker diarization.
    Uses Bayesian HMM to model speaker sequences.
    
    Based on: https://github.com/BUTSpeechFIT/VBx
    """
    
    def __init__(
        self,
        fa: float = 0.3,
        fb: float = 17.0,
        loop_prob: float = 0.9,
        metric: str = "cosine",
        max_iters: int = 10,
        min_clusters: int = 1,
        max_clusters: int = 20,
    ):
        """
        Initialize VBx clustering.
        
        Args:
            fa: False alarm rate prior
            fb: Speaker factor prior
            loop_prob: Self-loop probability
            metric: Distance metric
            max_iters: Maximum iterations
            min_clusters: Minimum clusters
            max_clusters: Maximum clusters
        """
        super().__init__(metric)
        
        self.fa = fa
        self.fb = fb
        self.loop_prob = loop_prob
        self.max_iters = max_iters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
    
    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster embeddings using VBx.
        
        Args:
            embeddings: Speaker embeddings [N, D]
            num_clusters: Number of clusters (optional)
            
        Returns:
            labels: Cluster labels [N]
        """
        n_samples, dim = embeddings.shape
        
        if n_samples == 0:
            return np.array([])
        
        if n_samples == 1:
            return np.array([0])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Initialize with AHC
        ahc = AgglomerativeHierarchicalClustering(
            threshold=0.5,
            metric=self.metric,
            min_clusters=self.min_clusters,
            max_clusters=self.max_clusters,
        )
        initial_labels = ahc.cluster(embeddings, num_clusters)
        
        # Run VB-HMM refinement
        labels = self._vb_hmm_refinement(embeddings, initial_labels)
        
        return labels
    
    def _vb_hmm_refinement(
        self,
        embeddings: np.ndarray,
        initial_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Refine clustering using VB-HMM.
        
        Args:
            embeddings: Normalized embeddings
            initial_labels: Initial cluster labels
            
        Returns:
            Refined labels
        """
        n_samples, dim = embeddings.shape
        n_clusters = len(np.unique(initial_labels))
        
        if n_clusters <= 1:
            return initial_labels
        
        # Initialize cluster means
        means = np.zeros((n_clusters, dim))
        for k in range(n_clusters):
            mask = initial_labels == k
            if np.sum(mask) > 0:
                means[k] = embeddings[mask].mean(axis=0)
        
        # Normalize means
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        means = means / (norms + 1e-8)
        
        # VB-HMM iterations
        labels = initial_labels.copy()
        
        for iteration in range(self.max_iters):
            # E-step: Compute responsibilities
            # Cosine similarity to cluster means
            similarities = embeddings @ means.T  # [N, K]
            
            # Add transition prior (favor staying in same cluster)
            log_probs = similarities * self.fb
            
            # Apply forward-backward smoothing
            gamma = self._forward_backward(log_probs)
            
            # M-step: Update labels
            new_labels = np.argmax(gamma, axis=1)
            
            # Update means
            for k in range(n_clusters):
                mask = new_labels == k
                if np.sum(mask) > 0:
                    means[k] = embeddings[mask].mean(axis=0)
                    means[k] = means[k] / (np.linalg.norm(means[k]) + 1e-8)
            
            # Check convergence
            if np.array_equal(new_labels, labels):
                break
            
            labels = new_labels
        
        return labels
    
    def _forward_backward(self, log_probs: np.ndarray) -> np.ndarray:
        """
        Forward-backward algorithm for HMM smoothing.
        
        Args:
            log_probs: Log probabilities [N, K]
            
        Returns:
            Smoothed posteriors [N, K]
        """
        n_samples, n_clusters = log_probs.shape
        
        # Simple smoothing using Gaussian filter
        gamma = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-8)
        
        # Temporal smoothing
        for k in range(n_clusters):
            gamma[:, k] = gaussian_filter1d(gamma[:, k], sigma=2)
        
        # Renormalize
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-8)
        
        return gamma


class OnlineClustering(SpeakerClustering):
    """
    Online clustering for streaming speaker diarization.
    
    Maintains cluster centroids and assigns new embeddings
    to existing clusters or creates new ones.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        metric: str = "cosine",
        max_clusters: int = 20,
        update_rate: float = 0.1,
    ):
        """
        Initialize online clustering.
        
        Args:
            threshold: Similarity threshold for new cluster
            metric: Distance metric
            max_clusters: Maximum number of clusters
            update_rate: Centroid update rate
        """
        super().__init__(metric)
        
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.update_rate = update_rate
        
        # State
        self.centroids: List[np.ndarray] = []
        self.counts: List[int] = []
    
    def reset(self):
        """Reset clustering state."""
        self.centroids = []
        self.counts = []
    
    def cluster_single(self, embedding: np.ndarray) -> int:
        """
        Assign a single embedding to a cluster.
        
        Args:
            embedding: Single embedding [D]
            
        Returns:
            Cluster label
        """
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        if len(self.centroids) == 0:
            self.centroids.append(embedding.copy())
            self.counts.append(1)
            return 0
        
        # Compute similarities to existing centroids
        similarities = np.array([
            np.dot(embedding, c) for c in self.centroids
        ])
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim >= self.threshold:
            # Assign to existing cluster and update centroid
            label = best_idx
            self._update_centroid(label, embedding)
        elif len(self.centroids) < self.max_clusters:
            # Create new cluster
            self.centroids.append(embedding.copy())
            self.counts.append(1)
            label = len(self.centroids) - 1
        else:
            # Force assignment to closest cluster
            label = best_idx
            self._update_centroid(label, embedding)
        
        return label
    
    def _update_centroid(self, idx: int, embedding: np.ndarray):
        """Update cluster centroid with new embedding."""
        self.counts[idx] += 1
        
        # Exponential moving average
        alpha = self.update_rate
        self.centroids[idx] = (1 - alpha) * self.centroids[idx] + alpha * embedding
        
        # Renormalize
        self.centroids[idx] = self.centroids[idx] / (
            np.linalg.norm(self.centroids[idx]) + 1e-8
        )
    
    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """
        Cluster all embeddings sequentially.
        
        Args:
            embeddings: Speaker embeddings [N, D]
            num_clusters: Ignored for online clustering
            
        Returns:
            labels: Cluster labels [N]
        """
        self.reset()
        labels = []
        
        for embedding in embeddings:
            label = self.cluster_single(embedding)
            labels.append(label)
        
        return np.array(labels)


class PLDAScoring:
    """
    Probabilistic Linear Discriminant Analysis (PLDA) scoring.
    
    Used for computing speaker similarity scores.
    """
    
    def __init__(
        self,
        dim: int = 192,
        speaker_dim: int = 128,
    ):
        """
        Initialize PLDA.
        
        Args:
            dim: Embedding dimension
            speaker_dim: Speaker subspace dimension
        """
        self.dim = dim
        self.speaker_dim = speaker_dim
        
        # Parameters (would normally be trained)
        self.mean = np.zeros(dim)
        self.transform = np.eye(dim)[:speaker_dim]  # LDA transform
        self.psi = np.eye(speaker_dim)  # Between-class covariance
        
        self.trained = False
    
    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Train PLDA model.
        
        Args:
            embeddings: Training embeddings [N, D]
            labels: Speaker labels [N]
        """
        # Compute global mean
        self.mean = embeddings.mean(axis=0)
        
        # Center embeddings
        centered = embeddings - self.mean
        
        # Compute within-class and between-class scatter
        unique_labels = np.unique(labels)
        
        Sw = np.zeros((self.dim, self.dim))
        Sb = np.zeros((self.dim, self.dim))
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = centered[mask]
            class_mean = class_embeddings.mean(axis=0)
            
            # Within-class scatter
            diff = class_embeddings - class_mean
            Sw += diff.T @ diff
            
            # Between-class scatter
            Sb += len(class_embeddings) * np.outer(class_mean, class_mean)
        
        # Solve generalized eigenvalue problem
        try:
            Sw_inv = np.linalg.inv(Sw + 1e-6 * np.eye(self.dim))
            eigvals, eigvecs = np.linalg.eigh(Sw_inv @ Sb)
            
            # Sort by eigenvalue
            idx = np.argsort(eigvals)[::-1]
            self.transform = eigvecs[:, idx[:self.speaker_dim]].T
            
            # Compute speaker covariance in reduced space
            reduced = (centered @ self.transform.T)
            self.psi = np.cov(reduced.T) + 1e-6 * np.eye(self.speaker_dim)
            
            self.trained = True
        except np.linalg.LinAlgError:
            warnings.warn("PLDA training failed, using identity transform")
    
    def score(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute PLDA score between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            PLDA score (log-likelihood ratio)
        """
        if not self.trained:
            # Fall back to cosine similarity
            e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            return np.dot(e1, e2)
        
        # Center and transform
        e1 = (embedding1 - self.mean) @ self.transform.T
        e2 = (embedding2 - self.mean) @ self.transform.T
        
        # Simplified PLDA scoring
        # Full implementation would use proper likelihood ratio
        score = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        
        return score
    
    def score_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise PLDA scores.
        
        Args:
            embeddings: Embeddings [N, D]
            
        Returns:
            Score matrix [N, N]
        """
        n = embeddings.shape[0]
        scores = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                s = self.score(embeddings[i], embeddings[j])
                scores[i, j] = s
                scores[j, i] = s
        
        return scores


def create_clustering(
    method: str = "ahc",
    **kwargs,
) -> SpeakerClustering:
    """
    Factory function to create clustering algorithm.
    
    Args:
        method: Clustering method (ahc, spectral, vbx, online)
        **kwargs: Method-specific arguments
        
    Returns:
        Clustering instance
    """
    method = method.lower()
    
    if method == "ahc" or method == "agglomerative":
        return AgglomerativeHierarchicalClustering(**kwargs)
    elif method == "spectral":
        return SpectralClusteringWrapper(**kwargs)
    elif method == "vbx":
        return VBxClustering(**kwargs)
    elif method == "online":
        return OnlineClustering(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")