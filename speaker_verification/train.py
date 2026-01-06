# speaker_verification/train.py
"""
Main training script for CAM++ Speaker Verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
from tqdm import tqdm

from config.config import get_config, get_small_config
from data.preprocessing import AudioPreprocessor
from data.augmentation import SpeakerAugmentor
from data.dataset import SpeakerVerificationDataset, create_dataloader
from models.cam_plus_plus import CAMPlusPlusClassifier
from training.trainer import CAMPlusPlusTrainer
from utils.helpers import set_seed, count_parameters, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train CAM++ Speaker Verification")
    parser.add_argument('--data_path', type=str, default='data/prepared/train',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='data/prepared/val',
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--small', action='store_true',
                        help='Use smaller model for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = get_logger("cam++_train")
    
    print("=" * 60)
    print("CAM++ Speaker Verification Training")
    print("=" * 60)
    
    # Check data path
    train_path = Path(args.data_path)
    val_path = Path(args.val_path)
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        print("   Run: python scripts/prepare_data.py --dataset synthetic")
        return
    
    # Config
    config = get_small_config() if args.small else get_config()
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.output_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = AudioPreprocessor(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels
    )
    
    # Initialize augmentor
    print("2. Initializing augmentor...")
    augmentor = SpeakerAugmentor(
        sample_rate=config.audio.sample_rate,
        speed_perturb=config.augmentation.speed_perturb,
        noise_aug=False,  # Disable if MUSAN not available
        reverb_aug=False,  # Disable if RIR not available
        spec_augment=config.augmentation.spec_augment
    )
    
    # Create datasets
    print("\n3. Loading datasets...")
    train_dataset = SpeakerVerificationDataset(
        data_path=str(train_path),
        preprocessor=preprocessor,
        augmentor=augmentor,
        duration=config.audio.training_duration,
        train=True
    )
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Train speakers: {train_dataset.num_speakers}")
    
    val_dataset = None
    val_loader = None
    if val_path.exists():
        val_dataset = SpeakerVerificationDataset(
            data_path=str(val_path),
            preprocessor=preprocessor,
            augmentor=None,
            train=False
        )
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Val speakers: {val_dataset.num_speakers}")
    
    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print(f"   Train batches: {len(train_loader)}")
    
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        print(f"   Val batches: {len(val_loader)}")
    
    # Create model with adjusted AAM-Softmax parameters
    print("\n5. Creating CAM++ model...")
    
    # Define model config dict for saving in checkpoint
    model_config = {
        'num_classes': train_dataset.num_speakers,
        'n_mels': config.audio.n_mels,
        'embedding_dim': config.model.embedding_dim,
        'fcm_channels': config.model.fcm_channels,
        'fcm_num_blocks': config.model.fcm_num_blocks,
        'dtdnn_blocks': config.model.dtdnn_blocks,
        'growth_rate': config.model.growth_rate,
        'bn_size': config.model.bn_size,
        'init_channels': config.model.init_channels,
        'cam_reduction': config.model.cam_reduction,
        'segment_length': config.model.segment_length,
        'small': args.small,
    }
    
    # Create the model using the config
    model = CAMPlusPlusClassifier(
        num_classes=model_config['num_classes'],
        n_mels=model_config['n_mels'],
        embedding_dim=model_config['embedding_dim'],
        scale=30.0,   # Slightly lower scale for stability
        margin=0.1,   # Lower margin for initial training
        easy_margin=True,  # Use easy margin
        fcm_channels=model_config['fcm_channels'],
        fcm_num_blocks=model_config['fcm_num_blocks'],
        dtdnn_blocks=model_config['dtdnn_blocks'],
        growth_rate=model_config['growth_rate'],
        bn_size=model_config['bn_size'],
        init_channels=model_config['init_channels'],
        use_cam=True,
        cam_reduction=model_config['cam_reduction'],
        segment_length=model_config['segment_length']
    )
    
    num_params = count_parameters(model)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Model size: ~{num_params * 4 / 1e6:.1f} MB")
    
    # Training config
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")
    print(f"   Optimizer: AdamW")
    print(f"   AAM-Softmax: margin=0.1, scale=30.0")
    print(f"   Model type: {'small' if args.small else 'full'}")
    print(f"   Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Create trainer with model config
    print("\n6. Initializing trainer...")
    trainer = CAMPlusPlusTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_config=model_config  # Pass model config for checkpoint saving
    )
    
    # Train
    print("\n7. Starting training...")
    print("=" * 60)
    
    history = trainer.train()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Best val accuracy: {trainer.best_val_acc:.2f}%")
    print(f"   Checkpoints saved to: {args.output_dir}")
    print("=" * 60)
    
    print("\nNext Steps:")
    print(f"1. Evaluate: python evaluate.py --checkpoint {args.output_dir}/best_model.pt")
    print(f"2. Demo: python demo/demo_gradio.py --checkpoint {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()