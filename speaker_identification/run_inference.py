# run_inference.py
"""
Run speaker verification inference.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch


def main():
    parser = argparse.ArgumentParser(description="Speaker Verification Inference")
    parser.add_argument('--checkpoint', type=str, default='outputs_v2/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--audio1', type=str, required=True,
                        help='Path to first audio file')
    parser.add_argument('--audio2', type=str, required=True,
                        help='Path to second audio file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WSI Speaker Verification")
    print("=" * 60)
    
    # Check paths
    checkpoint_path = Path(args.checkpoint)
    audio1_path = Path(args.audio1)
    audio2_path = Path(args.audio2)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not audio1_path.exists():
        print(f"❌ Audio file not found: {audio1_path}")
        return
    
    if not audio2_path.exists():
        print(f"❌ Audio file not found: {audio2_path}")
        return
    
    # Import modules
    from config.config import WSIConfig
    from data.preprocessing import AudioPreprocessor
    from models.wsi_model import WSIModel
    
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
    print("   ✅ Model loaded")
    
    # Initialize preprocessor
    print("\n2. Processing audio files...")
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    # Process audio files
    features1 = preprocessor.preprocess(str(audio1_path)).unsqueeze(0).to(device)
    features2 = preprocessor.preprocess(str(audio2_path)).unsqueeze(0).to(device)
    print(f"   Audio 1: {audio1_path.name}")
    print(f"   Audio 2: {audio2_path.name}")
    
    # Get embeddings and compute similarity
    print("\n3. Computing similarity...")
    with torch.no_grad():
        emb1 = model.get_embedding(features1)
        emb2 = model.get_embedding(features2)
        similarity = model.compute_similarity(emb1, emb2).item()
    
    # Make decision
    same_speaker = similarity >= args.threshold
    
    # Print results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    print(f"   Audio 1: {audio1_path}")
    print(f"   Audio 2: {audio2_path}")
    print(f"   Similarity Score: {similarity:.4f}")
    print(f"   Threshold: {args.threshold:.4f}")
    print(f"   Decision: {'✅ SAME SPEAKER' if same_speaker else '❌ DIFFERENT SPEAKERS'}")
    print("=" * 60)


if __name__ == "__main__":
    main()