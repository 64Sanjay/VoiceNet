# speaker_identification/inference.py
"""
Inference script for speaker verification.
"""

import argparse
import torch
from pathlib import Path

from config.config import get_default_config
from data.preprocessing import AudioPreprocessor
from models.wsi_model import WSIModel


def parse_args():
    parser = argparse.ArgumentParser(description="Speaker verification inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--audio1", type=str, required=True, help="First audio file")
    parser.add_argument("--audio2", type=str, required=True, help="Second audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = get_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Preprocess audio
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    features1 = preprocessor.preprocess(args.audio1).unsqueeze(0).to(device)
    features2 = preprocessor.preprocess(args.audio2).unsqueeze(0).to(device)
    
    # Get embeddings and similarity
    with torch.no_grad():
        emb1 = model.get_embedding(features1)
        emb2 = model.get_embedding(features2)
        similarity = model.compute_similarity(emb1, emb2).item()
    
    # Decision
    same_speaker = similarity >= args.threshold
    
    print(f"\nSpeaker Verification Result:")
    print(f"  Audio 1: {args.audio1}")
    print(f"  Audio 2: {args.audio2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Same Speaker: {'YES' if same_speaker else 'NO'}")


if __name__ == "__main__":
    main()