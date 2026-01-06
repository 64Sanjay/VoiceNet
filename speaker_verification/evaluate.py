# speaker_verification/evaluate.py
"""
Evaluation script for CAM++ Speaker Verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import numpy as np
from tqdm import tqdm

from config.config import get_config, get_small_config
from data.preprocessing import AudioPreprocessor
from data.dataset import SpeakerVerificationDataset, create_dataloader
from models.cam_plus_plus import CAMPlusPlusClassifier
from evaluation.evaluator import SpeakerVerificationEvaluator
from utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAM++ Speaker Verification")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, default='data/prepared/test',
                        help='Path to test data')
    parser.add_argument('--num_pairs', type=int, default=10000,
                        help='Number of verification pairs')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    set_seed(42)
    
    print("=" * 60)
    print("CAM++ Speaker Verification Evaluation")
    print("=" * 60)
    
    # Check paths
    checkpoint_path = Path(args.checkpoint)
    test_path = Path(args.test_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Train the model first: python train.py")
        return
    
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("   Prepare data first: python scripts/prepare_data.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    print("\n1. Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if model_config is saved in checkpoint
    if 'model_config' in checkpoint and checkpoint['model_config']:
        print("   ✅ Found model_config in checkpoint")
        model_config = checkpoint['model_config']
        print(f"   Model type: {'small' if model_config.get('small', False) else 'full'}")
    else:
        print("   ⚠️ No model_config in checkpoint, inferring from state_dict...")
        # Fallback: try to infer from state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Infer num_classes
        num_classes = state_dict['weight'].shape[0] if 'weight' in state_dict else 100
        
        # Infer fcm_channels from first conv
        fcm_channels = state_dict['encoder.fcm.conv1.weight'].shape[0]
        
        # Count fcm blocks
        fcm_num_blocks = 0
        for key in state_dict.keys():
            if key.startswith('encoder.fcm.blocks.') and '.conv1.weight' in key:
                block_idx = int(key.split('.')[3])
                fcm_num_blocks = max(fcm_num_blocks, block_idx + 1)
        
        # Infer init_channels
        init_channels = state_dict['encoder.input_tdnn.0.weight'].shape[0]
        
        # Infer growth_rate
        growth_rate = state_dict['encoder.dtdnn_blocks.0.layers.0.tdnn.weight'].shape[0]
        
        # Infer bn_size
        fc1_out = state_dict['encoder.dtdnn_blocks.0.layers.0.fc1.weight'].shape[0]
        bn_size = fc1_out // growth_rate if growth_rate > 0 else 2
        
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
        
        model_config = {
            'num_classes': num_classes,
            'n_mels': 80,
            'embedding_dim': 192,
            'fcm_channels': fcm_channels,
            'fcm_num_blocks': fcm_num_blocks,
            'dtdnn_blocks': tuple(dtdnn_blocks),
            'growth_rate': growth_rate,
            'bn_size': bn_size,
            'init_channels': init_channels,
            'cam_reduction': 2,
            'segment_length': 100,
        }
    
    # Print model config
    print(f"\n   Model Configuration:")
    print(f"      num_classes: {model_config['num_classes']}")
    print(f"      embedding_dim: {model_config.get('embedding_dim', 192)}")
    print(f"      fcm_channels: {model_config['fcm_channels']}")
    print(f"      fcm_num_blocks: {model_config['fcm_num_blocks']}")
    print(f"      init_channels: {model_config['init_channels']}")
    print(f"      dtdnn_blocks: {model_config['dtdnn_blocks']}")
    print(f"      growth_rate: {model_config['growth_rate']}")
    print(f"      bn_size: {model_config['bn_size']}")
    
    # Create model with the saved config
    print("\n2. Creating model...")
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
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best val acc: {checkpoint.get('best_val_acc', 'unknown')}")
    print(f"   Total parameters: {total_params:,}")
    
    # Load test data
    print("\n3. Loading test data...")
    config = get_config()  # Use default audio config
    preprocessor = AudioPreprocessor(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels
    )
    
    test_dataset = SpeakerVerificationDataset(
        data_path=str(test_path),
        preprocessor=preprocessor,
        augmentor=None,
        train=False
    )
    
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Test speakers: {test_dataset.num_speakers}")
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print("\n4. Running evaluation...")
    evaluator = SpeakerVerificationEvaluator(model=model, device=str(device))
    metrics = evaluator.evaluate(test_loader, num_pairs=args.num_pairs)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"   EER: {metrics['eer']:.2f}%")
    print(f"   MinDCF: {metrics['min_dcf']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Threshold: {metrics['eer_threshold']:.4f}")
    print("=" * 60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("CAM++ Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Data: {test_path}\n")
        f.write(f"Number of Pairs: {args.num_pairs}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model Config:\n")
        for k, v in model_config.items():
            f.write(f"   {k}: {v}\n")
        f.write("=" * 40 + "\n")
        f.write(f"EER: {metrics['eer']:.2f}%\n")
        f.write(f"MinDCF: {metrics['min_dcf']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Threshold: {metrics['eer_threshold']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()