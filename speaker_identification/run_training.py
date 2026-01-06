# run_training.py
"""
Main training script for WSI with LibriSpeech dataset.
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import logging
from tqdm import tqdm  # <-- Add this import


def main():
    parser = argparse.ArgumentParser(description="Train WSI Speaker Identification Model")
    parser.add_argument('--train_path', type=str, default='data/librispeech_prepared/train',
                        help='Path to training data')
    parser.add_argument('--val_path', type=str, default='data/librispeech_prepared/val',
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("WSI Speaker Identification Training")
    print("=" * 60)
    
    # Check paths
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        print("   Run: python prepare_librispeech.py")
        return
    
    if not val_path.exists():
        print(f"⚠️ Validation data not found: {val_path}")
        val_path = None
    
    # Import modules (after adding to path)
    from config.config import WSIConfig
    from data.preprocessing import AudioPreprocessor
    from data.augmentation import AudioAugmentor
    from data.dataset import SpeakerDataset, create_dataloader
    from models.wsi_model import WSIModel
    from losses.joint_loss import WSIJointLoss
    from utils.helpers import set_seed, count_parameters
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = WSIConfig()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.output_dir
    config.data.num_workers = args.num_workers
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    print("   ✅ Preprocessor ready")
    
    # Initialize augmentor
    print("\n2. Initializing augmentor...")
    augmentor = AudioAugmentor(
        noise_snr_db=config.data.noise_snr_db,
        time_stretch_range=config.data.time_stretch_range,
        sample_rate=config.data.sample_rate
    )
    print("   ✅ Augmentor ready")
    
    # Create datasets
    print("\n3. Loading datasets...")
    train_dataset = SpeakerDataset(
        data_path=str(train_path),
        preprocessor=preprocessor,
        augmentor=augmentor,
        split="train",
        return_augmented=True
    )
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Train speakers: {train_dataset.num_speakers}")
    
    val_loader = None
    if val_path:
        val_dataset = SpeakerDataset(
            data_path=str(val_path),
            preprocessor=preprocessor,
            augmentor=None,
            split="val",
            return_augmented=False
        )
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Val speakers: {val_dataset.num_speakers}")
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    print(f"   Train batches: {len(train_loader)}")
    if val_loader:
        print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print("\n5. Creating model...")
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim,
        freeze_encoder=config.model.freeze_encoder
    )
    model = model.to(device)
    
    total_params = count_parameters(model)
    print(f"   Model: {config.model.whisper_model_name}")
    print(f"   Embedding dim: {config.model.embedding_dim}")
    print(f"   Total parameters: {total_params:,}")
    
    # Create loss function
    criterion = WSIJointLoss(
        triplet_margin=config.loss.triplet_margin,
        nt_xent_temperature=config.loss.nt_xent_temperature,
        lambda_weight=config.loss.self_supervised_weight
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Print training config
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"   Epochs: {config.training.epochs}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Learning rate: {config.training.learning_rate}")
    print(f"   Triplet margin: {config.loss.triplet_margin}")
    print(f"   NT-Xent temperature: {config.loss.nt_xent_temperature}")
    print(f"   Self-supervised weight (λ): {config.loss.self_supervised_weight}")
    print(f"   Output dir: {args.output_dir}")
    print("=" * 60)
    
    # Training loop
    print("\n6. Starting training...")
    print("=" * 60)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(config.training.epochs):
        # Training
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.training.epochs}")
        
        for batch in pbar:
            # Unpack batch
            if len(batch) == 4:
                original, noise_aug, time_aug, labels = batch
            else:
                original, labels = batch
                noise_aug = time_aug = None
            
            # Move to device
            original = original.to(device)
            labels = labels.to(device)
            
            if noise_aug is not None:
                noise_aug = noise_aug.to(device)
                time_aug = time_aug.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(original, noise_aug, time_aug)
            z_original = outputs['original']
            z_noise = outputs.get('noise_augmented', z_original)
            z_time = outputs.get('time_stretched', z_original)
            
            # Compute loss
            loss, loss_dict = criterion(z_original, z_noise, z_time, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'triplet': f"{loss_dict['loss_triplet']:.4f}",
                'nt_xent': f"{loss_dict['loss_nt_xent']:.4f}"
            })
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    if len(batch) == 4:
                        original, noise_aug, time_aug, labels = batch
                    else:
                        original, labels = batch
                        noise_aug = time_aug = None
                    
                    original = original.to(device)
                    labels = labels.to(device)
                    
                    if noise_aug is not None:
                        noise_aug = noise_aug.to(device)
                        time_aug = time_aug.to(device)
                    
                    outputs = model(original, noise_aug, time_aug)
                    z_original = outputs['original']
                    z_noise = outputs.get('noise_augmented', z_original)
                    z_time = outputs.get('time_stretched', z_original)
                    
                    loss, _ = criterion(z_original, z_noise, z_time, labels)
                    val_losses.append(loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, Path(args.output_dir) / 'best_model.pt')
                print(f"   ✅ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            print(f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
        }, Path(args.output_dir) / f'checkpoint_epoch_{epoch + 1}.pt')
    
    # Save final model
    torch.save({
        'epoch': config.training.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'history': history,
    }, Path(args.output_dir) / 'final_model.pt')
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {args.output_dir}")
    print("=" * 60)
    
    # Next steps
    print("\nNext Steps:")
    print(f"1. Evaluate: python run_evaluation.py --checkpoint {args.output_dir}/final_model.pt")
    print(f"2. Inference: python run_inference.py --checkpoint {args.output_dir}/final_model.pt --audio1 path/to/audio1.wav --audio2 path/to/audio2.wav")


if __name__ == "__main__":
    main()