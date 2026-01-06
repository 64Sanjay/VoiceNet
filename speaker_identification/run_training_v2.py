# run_training_v2.py
"""
Improved training script for WSI with better hyperparameters.
"""

import sys
import os
import argparse
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train WSI Speaker Identification Model (Improved)")
    parser.add_argument('--train_path', type=str, default='data/librispeech_prepared/train')
    parser.add_argument('--val_path', type=str, default='data/librispeech_prepared/val')
    parser.add_argument('--output_dir', type=str, default='outputs_v2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("WSI Speaker Identification Training (Improved v2)")
    print("=" * 60)
    
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return
    
    from config.config import WSIConfig
    from data.preprocessing import AudioPreprocessor
    from data.augmentation import AudioAugmentor
    from data.dataset import SpeakerDataset, create_dataloader
    from models.wsi_model import WSIModel
    from utils.helpers import set_seed, count_parameters
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = WSIConfig()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor and augmentor
    print("\n1. Initializing components...")
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    augmentor = AudioAugmentor(
        noise_snr_db=(5.0, 25.0),  # Wider SNR range
        time_stretch_range=(0.9, 1.1),  # Smaller stretch range
        sample_rate=config.data.sample_rate
    )
    print("   ✅ Components ready")
    
    # Create datasets
    print("\n2. Loading datasets...")
    train_dataset = SpeakerDataset(
        data_path=str(train_path),
        preprocessor=preprocessor,
        augmentor=augmentor,
        split="train",
        return_augmented=True
    )
    print(f"   Train samples: {len(train_dataset)}, Speakers: {train_dataset.num_speakers}")
    
    val_dataset = None
    val_loader = None
    if val_path.exists():
        val_dataset = SpeakerDataset(
            data_path=str(val_path),
            preprocessor=preprocessor,
            augmentor=None,
            split="val",
            return_augmented=False
        )
        print(f"   Val samples: {len(val_dataset)}, Speakers: {val_dataset.num_speakers}")
        val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"\n   Train batches: {len(train_loader)}")
    
    # Create model
    print("\n3. Creating model...")
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=512,
        freeze_encoder=False
    )
    model = model.to(device)
    print(f"   Total parameters: {count_parameters(model):,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    print("\n" + "=" * 60)
    print("Training Configuration (Improved)")
    print("=" * 60)
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Optimizer: AdamW with weight decay 0.01")
    print(f"   Scheduler: CosineAnnealingLR")
    print(f"   Triplet margin: 0.3 (reduced)")
    print(f"   NT-Xent temperature: 0.07 (reduced)")
    print("=" * 60)
    
    # Training loop
    print("\n4. Starting training...")
    print("=" * 60)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': []}
    best_val_loss = float('inf')
    
    # Loss parameters (tuned)
    triplet_margin = 0.3  # Reduced from 1.0
    nt_xent_temp = 0.07   # Reduced from 0.5
    lambda_weight = 0.5   # Reduced from 1.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_triplet_losses = []
        epoch_nt_xent_losses = []
        correct_pairs = 0
        total_pairs = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch in pbar:
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
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(original, noise_aug, time_aug)
            z_original = outputs['original']
            z_noise = outputs.get('noise_augmented', z_original)
            z_time = outputs.get('time_stretched', z_original)
            
            # Normalize embeddings
            z_original = F.normalize(z_original, p=2, dim=1)
            z_noise = F.normalize(z_noise, p=2, dim=1)
            z_time = F.normalize(z_time, p=2, dim=1)
            
            # Compute improved triplet loss with semi-hard mining
            triplet_loss = compute_triplet_loss_improved(z_original, labels, margin=triplet_margin)
            
            # Compute NT-Xent loss
            nt_xent_loss = compute_nt_xent_loss(z_original, z_noise, z_time, temperature=nt_xent_temp)
            
            # Combined loss
            loss = triplet_loss + lambda_weight * nt_xent_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_triplet_losses.append(triplet_loss.item())
            epoch_nt_xent_losses.append(nt_xent_loss.item())
            
            # Compute training accuracy (same speaker pairs should have high similarity)
            with torch.no_grad():
                sim_matrix = torch.mm(z_original, z_original.t())
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        if labels[i] == labels[j]:
                            if sim_matrix[i, j] > 0.5:
                                correct_pairs += 1
                            total_pairs += 1
                        else:
                            if sim_matrix[i, j] < 0.5:
                                correct_pairs += 1
                            total_pairs += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'triplet': f"{triplet_loss.item():.4f}",
                'nt_xent': f"{nt_xent_loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        scheduler.step()
        
        avg_train_loss = np.mean(epoch_losses)
        avg_triplet = np.mean(epoch_triplet_losses)
        avg_nt_xent = np.mean(epoch_nt_xent_losses)
        train_acc = correct_pairs / max(total_pairs, 1) * 100
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
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
                    z_original = F.normalize(outputs['original'], p=2, dim=1)
                    z_noise = F.normalize(outputs.get('noise_augmented', z_original), p=2, dim=1)
                    z_time = F.normalize(outputs.get('time_stretched', z_original), p=2, dim=1)
                    
                    triplet_loss = compute_triplet_loss_improved(z_original, labels, margin=triplet_margin)
                    nt_xent_loss = compute_nt_xent_loss(z_original, z_noise, z_time, temperature=nt_xent_temp)
                    loss = triplet_loss + lambda_weight * nt_xent_loss
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} (triplet={avg_triplet:.4f}, nt_xent={avg_nt_xent:.4f}), "
                  f"Val Loss = {avg_val_loss:.4f}, Train Acc = {train_acc:.1f}%")
            
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
            print(f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.1f}%")
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, Path(args.output_dir) / 'final_model.pt')
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {args.output_dir}")
    print("=" * 60)


def compute_triplet_loss_improved(embeddings, labels, margin=0.3):
    """
    Improved triplet loss with all valid triplets.
    Uses batch-all strategy for better gradient signal.
    """
    device = embeddings.device
    batch_size = embeddings.size(0)
    
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Create masks
    labels = labels.unsqueeze(0)
    same_identity_mask = (labels == labels.t()).float()
    diff_identity_mask = 1 - same_identity_mask
    
    # Remove diagonal (self-comparisons)
    eye = torch.eye(batch_size, device=device)
    same_identity_mask = same_identity_mask - eye
    
    # Compute loss for all valid triplets
    # For each anchor-positive pair, compute loss with all negatives
    loss = 0.0
    num_triplets = 0
    
    for i in range(batch_size):
        # Get positive indices
        pos_mask = same_identity_mask[i] > 0
        neg_mask = diff_identity_mask[i] > 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        
        # Get distances
        anchor_pos_dist = distances[i][pos_mask]  # distances to positives
        anchor_neg_dist = distances[i][neg_mask]  # distances to negatives
        
        # Compute triplet loss for all combinations
        # Using broadcasting: (num_pos, 1) - (1, num_neg) + margin
        triplet_losses = anchor_pos_dist.unsqueeze(1) - anchor_neg_dist.unsqueeze(0) + margin
        triplet_losses = F.relu(triplet_losses)
        
        # Use mean of hard triplets (positive loss)
        hard_triplets = triplet_losses[triplet_losses > 0]
        if len(hard_triplets) > 0:
            loss += hard_triplets.mean()
            num_triplets += 1
    
    if num_triplets == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss / num_triplets


def compute_nt_xent_loss(z_original, z_noise, z_time, temperature=0.07):
    """
    Compute NT-Xent loss between original and augmented views.
    """
    batch_size = z_original.size(0)
    device = z_original.device
    
    # Combine augmented views
    z_aug = (z_noise + z_time) / 2  # Average of augmented views
    
    # Concatenate original and augmented
    z = torch.cat([z_original, z_aug], dim=0)  # (2B, D)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Create labels: positive pairs are (i, i+B) and (i+B, i)
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, float('-inf'))
    
    # Compute cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


if __name__ == "__main__":
    main()