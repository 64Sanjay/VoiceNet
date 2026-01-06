# test_model.py
"""
Quick test script for the trained WSI model.
Tests same-speaker and different-speaker scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import random


def main():
    print("=" * 60)
    print("WSI Model Quick Test")
    print("=" * 60)
    
    # Import modules
    from config.config import WSIConfig
    from data.preprocessing import AudioPreprocessor
    from models.wsi_model import WSIModel
    
    # Paths - use the improved model
    checkpoint_path = Path("outputs_v2/best_model.pt")
    test_path = Path("data/librispeech_prepared/test")
    
    if not checkpoint_path.exists():
        # Fallback to original outputs
        checkpoint_path = Path("outputs/final_model.pt")
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found")
            return
    
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    
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
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    # Get all speakers and their audio files
    speakers = {}
    for speaker_dir in sorted(test_path.iterdir()):
        if speaker_dir.is_dir():
            audio_files = sorted(list(speaker_dir.glob("*.wav")))
            if len(audio_files) >= 2:
                speakers[speaker_dir.name] = audio_files
    
    print(f"\n2. Found {len(speakers)} speakers with multiple audio files")
    
    # Function to compute similarity
    def compute_similarity(audio1_path, audio2_path):
        features1 = preprocessor.preprocess(str(audio1_path)).unsqueeze(0).to(device)
        features2 = preprocessor.preprocess(str(audio2_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb1 = model.get_embedding(features1)
            emb2 = model.get_embedding(features2)
            similarity = model.compute_similarity(emb1, emb2).item()
        
        return similarity
    
    # Test 1: Same Speaker Pairs
    print("\n" + "=" * 60)
    print("TEST 1: SAME SPEAKER PAIRS")
    print("=" * 60)
    
    same_speaker_scores = []
    speaker_list = list(speakers.keys())
    
    for i, speaker in enumerate(speaker_list[:5]):  # Test 5 speakers
        audio_files = speakers[speaker]
        audio1 = audio_files[0]
        audio2 = audio_files[1]
        
        similarity = compute_similarity(audio1, audio2)
        same_speaker_scores.append(similarity)
        
        print(f"\n   {speaker}:")
        print(f"      Audio 1: {audio1.name}")
        print(f"      Audio 2: {audio2.name}")
        print(f"      Similarity: {similarity:.4f} {'✅' if similarity > 0.5 else '❌'}")
    
    avg_same = sum(same_speaker_scores) / len(same_speaker_scores)
    print(f"\n   Average Same-Speaker Similarity: {avg_same:.4f}")
    
    # Test 2: Different Speaker Pairs
    print("\n" + "=" * 60)
    print("TEST 2: DIFFERENT SPEAKER PAIRS")
    print("=" * 60)
    
    diff_speaker_scores = []
    
    for i in range(min(5, len(speaker_list) - 1)):  # Test 5 pairs
        speaker1 = speaker_list[i]
        speaker2 = speaker_list[i + 1]
        
        audio1 = speakers[speaker1][0]
        audio2 = speakers[speaker2][0]
        
        similarity = compute_similarity(audio1, audio2)
        diff_speaker_scores.append(similarity)
        
        print(f"\n   {speaker1} vs {speaker2}:")
        print(f"      Audio 1: {audio1.name}")
        print(f"      Audio 2: {audio2.name}")
        print(f"      Similarity: {similarity:.4f} {'✅' if similarity < 0.5 else '❌'}")
    
    avg_diff = sum(diff_speaker_scores) / len(diff_speaker_scores)
    print(f"\n   Average Different-Speaker Similarity: {avg_diff:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Same Speaker Avg Similarity:      {avg_same:.4f}")
    print(f"   Different Speaker Avg Similarity: {avg_diff:.4f}")
    print(f"   Gap (Same - Different):           {avg_same - avg_diff:.4f}")
    
    # Determine threshold
    threshold = (avg_same + avg_diff) / 2
    print(f"\n   Suggested Threshold: {threshold:.4f}")
    
    # Accuracy estimation
    correct_same = sum(1 for s in same_speaker_scores if s >= threshold)
    correct_diff = sum(1 for s in diff_speaker_scores if s < threshold)
    total = len(same_speaker_scores) + len(diff_speaker_scores)
    accuracy = (correct_same + correct_diff) / total * 100
    
    print(f"   Estimated Accuracy: {accuracy:.1f}%")
    
    # Performance assessment
    print("\n" + "=" * 60)
    if avg_same > 0.7 and avg_diff < 0.3:
        print("🎉 EXCELLENT! Model is performing well.")
    elif avg_same > 0.5 and avg_diff < 0.4:
        print("👍 GOOD! Model shows clear discrimination.")
    elif avg_same > avg_diff:
        print("⚠️ FAIR. Model learns some patterns but needs improvement.")
    else:
        print("❌ POOR. Model needs more training or data.")
    print("=" * 60)


if __name__ == "__main__":
    main()