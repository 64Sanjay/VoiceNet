# speaker_verification/inference.py
"""
Inference script for CAM++ Speaker Verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import torch.nn.functional as F

from config.config import get_config
from data.preprocessing import AudioPreprocessor
from models.cam_plus_plus import CAMPlusPlusClassifier


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model with correct architecture from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    if 'model_config' in checkpoint and checkpoint['model_config']:
        print("   ✅ Found model_config in checkpoint")
        model_config = checkpoint['model_config']
    else:
        # Infer from state_dict
        print("   ⚠️ Inferring config from checkpoint weights...")
        state_dict = checkpoint['model_state_dict']
        model_config = infer_config_from_state_dict(state_dict)
    
    # Print config
    print(f"   Model Configuration:")
    print(f"      fcm_channels: {model_config['fcm_channels']}")
    print(f"      fcm_num_blocks: {model_config['fcm_num_blocks']}")
    print(f"      init_channels: {model_config['init_channels']}")
    print(f"      dtdnn_blocks: {model_config['dtdnn_blocks']}")
    print(f"      growth_rate: {model_config['growth_rate']}")
    print(f"      bn_size: {model_config['bn_size']}")
    
    # Create model with correct config
    model = CAMPlusPlusClassifier(
        num_classes=model_config['num_classes'],
        n_mels=model_config.get('n_mels', 80),
        embedding_dim=model_config.get('embedding_dim', 192),
        scale=30.0,
        margin=0.1,
        easy_margin=True,
        fcm_channels=model_config['fcm_channels'],
        fcm_num_blocks=model_config['fcm_num_blocks'],
        dtdnn_blocks=model_config['dtdnn_blocks'],
        growth_rate=model_config['growth_rate'],
        bn_size=model_config['bn_size'],
        init_channels=model_config['init_channels'],
        use_cam=True,
        cam_reduction=model_config.get('cam_reduction', 2),
        segment_length=model_config.get('segment_length', 100)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_config


def infer_config_from_state_dict(state_dict):
    """Infer model configuration from checkpoint state_dict."""
    config = {}
    
    # Infer num_classes
    if 'weight' in state_dict:
        config['num_classes'] = state_dict['weight'].shape[0]
    else:
        config['num_classes'] = 100
    
    # Infer embedding_dim
    if 'encoder.embedding.weight' in state_dict:
        config['embedding_dim'] = state_dict['encoder.embedding.weight'].shape[0]
    else:
        config['embedding_dim'] = 192
    
    # Infer fcm_channels
    if 'encoder.fcm.conv1.weight' in state_dict:
        config['fcm_channels'] = state_dict['encoder.fcm.conv1.weight'].shape[0]
    else:
        config['fcm_channels'] = 32
    
    # Count fcm blocks
    fcm_num_blocks = 0
    for key in state_dict.keys():
        if key.startswith('encoder.fcm.blocks.') and '.conv1.weight' in key:
            block_idx = int(key.split('.')[3])
            fcm_num_blocks = max(fcm_num_blocks, block_idx + 1)
    config['fcm_num_blocks'] = fcm_num_blocks if fcm_num_blocks > 0 else 4
    
    # Infer init_channels
    if 'encoder.input_tdnn.0.weight' in state_dict:
        config['init_channels'] = state_dict['encoder.input_tdnn.0.weight'].shape[0]
    else:
        config['init_channels'] = 128
    
    # Infer growth_rate
    if 'encoder.dtdnn_blocks.0.layers.0.tdnn.weight' in state_dict:
        config['growth_rate'] = state_dict['encoder.dtdnn_blocks.0.layers.0.tdnn.weight'].shape[0]
    else:
        config['growth_rate'] = 32
    
    # Infer bn_size
    if 'encoder.dtdnn_blocks.0.layers.0.fc1.weight' in state_dict:
        fc1_out = state_dict['encoder.dtdnn_blocks.0.layers.0.fc1.weight'].shape[0]
        config['bn_size'] = fc1_out // config['growth_rate'] if config['growth_rate'] > 0 else 4
    else:
        config['bn_size'] = 4
    
    # Count dtdnn blocks and layers
    dtdnn_blocks = []
    for block_idx in range(10):
        layer_count = 0
        for key in state_dict.keys():
            if key.startswith(f'encoder.dtdnn_blocks.{block_idx}.layers.') and '.bn1.weight' in key:
                layer_idx = int(key.split('.')[4])
                layer_count = max(layer_count, layer_idx + 1)
        if layer_count > 0:
            dtdnn_blocks.append(layer_count)
        else:
            break
    config['dtdnn_blocks'] = tuple(dtdnn_blocks) if dtdnn_blocks else (12, 24, 16)
    
    # Defaults
    config['n_mels'] = 80
    config['cam_reduction'] = 2
    config['segment_length'] = 100
    
    return config


def main():
    parser = argparse.ArgumentParser(description="CAM++ Speaker Verification Inference")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--audio1', type=str, required=True,
                        help='Path to first audio file')
    parser.add_argument('--audio2', type=str, required=True,
                        help='Path to second audio file')
    parser.add_argument('--threshold', type=float, default=0.6694,
                        help='Decision threshold (from evaluation)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAM++ Speaker Verification")
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model with correct architecture
    print("\n1. Loading model...")
    model, model_config = load_model_from_checkpoint(str(checkpoint_path), device)
    print("   ✅ Model loaded successfully")
    
    # Initialize preprocessor
    print("\n2. Processing audio files...")
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=80
    )
    
    # Process audio files
    waveform1, _ = preprocessor.load_audio(str(audio1_path))
    features1 = preprocessor.extract_features(waveform1)
    features1 = preprocessor.normalize_features(features1)
    features1 = features1.unsqueeze(0).to(device)
    
    waveform2, _ = preprocessor.load_audio(str(audio2_path))
    features2 = preprocessor.extract_features(waveform2)
    features2 = preprocessor.normalize_features(features2)
    features2 = features2.unsqueeze(0).to(device)
    
    print(f"   Audio 1: {audio1_path.name} -> {features1.shape}")
    print(f"   Audio 2: {audio2_path.name} -> {features2.shape}")
    
    # Get embeddings
    print("\n3. Computing embeddings...")
    with torch.no_grad():
        emb1 = model.extract_embedding(features1)
        emb2 = model.extract_embedding(features2)
    
    print(f"   Embedding 1 shape: {emb1.shape}")
    print(f"   Embedding 2 shape: {emb2.shape}")
    
    # Compute similarity
    similarity = F.cosine_similarity(emb1, emb2).item()
    
    # Decision
    same_speaker = similarity >= args.threshold
    
    # Confidence calculation
    if same_speaker:
        confidence = min((similarity - args.threshold) / (1.0 - args.threshold) * 100, 100)
    else:
        confidence = min((args.threshold - similarity) / args.threshold * 100, 100)
    
    # Print results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    print(f"   Audio 1: {audio1_path}")
    print(f"   Audio 2: {audio2_path}")
    print("-" * 60)
    print(f"   Similarity Score: {similarity:.4f}")
    print(f"   Threshold: {args.threshold:.4f}")
    print(f"   Confidence: {confidence:.1f}%")
    print("-" * 60)
    if same_speaker:
        print(f"   ✅ DECISION: SAME SPEAKER")
    else:
        print(f"   ❌ DECISION: DIFFERENT SPEAKERS")
    print("=" * 60)


if __name__ == "__main__":
    main()