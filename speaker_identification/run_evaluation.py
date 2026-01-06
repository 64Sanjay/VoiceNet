# run_evaluation.py
"""
Evaluate trained WSI model.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import numpy as np
from tqdm import tqdm
import itertools


def main():
    parser = argparse.ArgumentParser(description="Evaluate WSI Model")
    parser.add_argument('--checkpoint', type=str, default='outputs_v2/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='data/librispeech_prepared/test',
                        help='Path to test data')
    parser.add_argument('--num_pairs', type=int, default=5000,
                        help='Number of pairs to evaluate')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WSI Model Evaluation")
    print("=" * 60)
    
    # Check paths
    checkpoint_path = Path(args.checkpoint)
    test_path = Path(args.test_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Run training first: python run_training_v2.py")
        return
    
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("   Run: python prepare_librispeech.py")
        return
    
    # Import modules
    from config.config import WSIConfig
    from data.preprocessing import AudioPreprocessor
    from data.dataset import SpeakerDataset, create_dataloader
    from models.wsi_model import WSIModel
    from evaluation.metrics import compute_metrics
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Config
    config = WSIConfig()
    
    # Load model
    print("\n1. Loading model...")
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim
    )
    
    # Fix for PyTorch 2.6+ - use weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"   Loaded from: {checkpoint_path}")
    print(f"   Trained epochs: {checkpoint.get('epoch', 'unknown') + 1}")
    
    # Load test data
    print("\n2. Loading test data...")
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    test_dataset = SpeakerDataset(
        data_path=str(test_path),
        preprocessor=preprocessor,
        augmentor=None,
        split="test",
        return_augmented=False
    )
    
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Test speakers: {test_dataset.num_speakers}")
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # Extract embeddings
    print("\n3. Extracting embeddings...")
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            features, labels = batch[0], batch[-1]
            features = features.to(device)
            
            embeddings = model.get_embedding(features)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"   Extracted {len(embeddings)} embeddings")
    print(f"   Unique speakers: {len(np.unique(labels))}")
    
    # Create verification pairs
    print("\n4. Creating verification pairs...")
    
    # Group samples by speaker
    speaker_to_indices = {}
    for idx, label in enumerate(labels):
        if label not in speaker_to_indices:
            speaker_to_indices[label] = []
        speaker_to_indices[label].append(idx)
    
    pairs = []
    pair_labels = []
    
    # Positive pairs (same speaker)
    positive_candidates = []
    for speaker, indices in speaker_to_indices.items():
        if len(indices) >= 2:
            for i, j in itertools.combinations(indices, 2):
                positive_candidates.append((i, j))
    
    np.random.shuffle(positive_candidates)
    num_positive = min(args.num_pairs // 2, len(positive_candidates))
    for pair in positive_candidates[:num_positive]:
        pairs.append(pair)
        pair_labels.append(1)
    
    # Negative pairs (different speakers)
    speakers = list(speaker_to_indices.keys())
    negative_candidates = []
    for s1, s2 in itertools.combinations(speakers, 2):
        for i in speaker_to_indices[s1][:5]:  # Limit per speaker
            for j in speaker_to_indices[s2][:5]:
                negative_candidates.append((i, j))
    
    np.random.shuffle(negative_candidates)
    num_negative = min(args.num_pairs // 2, len(negative_candidates))
    for pair in negative_candidates[:num_negative]:
        pairs.append(pair)
        pair_labels.append(0)
    
    print(f"   Created {len(pairs)} pairs ({num_positive} positive, {num_negative} negative)")
    
    # Compute similarities
    print("\n5. Computing similarities...")
    scores = []
    for idx1, idx2 in tqdm(pairs, desc="Similarities"):
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        
        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(emb1, emb2) / (norm1 * norm2)
        else:
            sim = 0.0
        scores.append(sim)
    
    scores = np.array(scores)
    pair_labels = np.array(pair_labels)
    
    # Compute metrics
    print("\n6. Computing metrics...")
    metrics = compute_metrics(pair_labels, scores)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"   Equal Error Rate (EER): {metrics['eer']:.2f}%")
    print(f"   Area Under Curve (AUC): {metrics['auc']:.4f}")
    print(f"   Optimal Threshold: {metrics['eer_threshold']:.4f}")
    print(f"   False Accept Rate (FAR): {metrics['far']:.2f}%")
    print(f"   False Reject Rate (FRR): {metrics['frr']:.2f}%")
    print("=" * 60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("WSI Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Data: {test_path}\n")
        f.write(f"Number of Pairs: {len(pairs)}\n")
        f.write("=" * 40 + "\n")
        f.write(f"EER: {metrics['eer']:.2f}%\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Threshold: {metrics['eer_threshold']:.4f}\n")
        f.write(f"FAR: {metrics['far']:.2f}%\n")
        f.write(f"FRR: {metrics['frr']:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()