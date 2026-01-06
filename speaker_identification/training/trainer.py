# training/trainer.py
"""
WSI Trainer implementing Algorithm 1 from the paper.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging

from models.wsi_model import WSIModel
from losses.joint_loss import WSIJointLoss
from config.config import WSIConfig


class WSITrainer:
    """
    Trainer for WSI model.
    
    Implements Algorithm 1: Training Procedure for Online Triplet Mining
    with Multi-View Self-Supervision
    """
    
    def __init__(
        self,
        model: WSIModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[WSIConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: WSI model instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
        """
        self.config = config or WSIConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function (Equation 9)
        self.criterion = WSIJointLoss(
            triplet_margin=self.config.loss.triplet_margin,
            nt_xent_temperature=self.config.loss.nt_xent_temperature,
            lambda_weight=self.config.loss.self_supervised_weight
        )
        
        # Optimizer (Adam with lr=1e-5 from Table 3)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.training.use_amp else None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Implements the inner loop of Algorithm 1.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch: (original, noise_aug, time_aug, labels)
            if len(batch) == 4:
                original, noise_aug, time_aug, labels = batch
            else:
                # Fallback for evaluation-only batches
                original, labels = batch
                noise_aug = time_aug = None
            
            # Move to device
            original = original.to(self.device)
            labels = labels.to(self.device)
            
            if noise_aug is not None:
                noise_aug = noise_aug.to(self.device)
                time_aug = time_aug.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.training.use_amp and self.scaler is not None:
                with autocast():
                    loss, loss_dict = self._forward_step(
                        original, noise_aug, time_aug, labels
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self._forward_step(
                    original, noise_aug, time_aug, labels
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'triplet': f"{loss_dict['loss_triplet']:.4f}",
                'nt_xent': f"{loss_dict['loss_nt_xent']:.4f}"
            })
            
            # Logging
            if self.global_step % self.config.training.log_every_n_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"triplet={loss_dict['loss_triplet']:.4f}, "
                    f"nt_xent={loss_dict['loss_nt_xent']:.4f}"
                )
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_metrics['train_loss'] = sum(epoch_losses) / len(epoch_losses)
        
        return epoch_metrics
    
    def _forward_step(
        self,
        original: torch.Tensor,
        noise_aug: Optional[torch.Tensor],
        time_aug: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform forward pass and compute loss.
        
        Implements Equations 4-9 from the paper.
        
        Args:
            original: Original features (B, 80, T)
            noise_aug: Noise-augmented features (B, 80, T)
            time_aug: Time-stretched features (B, 80, T)
            labels: Speaker labels (B,)
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Get embeddings (Equations 4, 5, 6)
        outputs = self.model(original, noise_aug, time_aug)
        
        z_original = outputs['original']
        z_noise = outputs.get('noise_augmented', z_original)
        z_time = outputs.get('time_stretched', z_original)
        
        # Compute joint loss (Equation 9)
        loss, loss_dict = self.criterion(z_original, z_noise, z_time, labels)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            if len(batch) == 4:
                original, noise_aug, time_aug, labels = batch
            else:
                original, labels = batch
                noise_aug = time_aug = None
            
            original = original.to(self.device)
            labels = labels.to(self.device)
            
            if noise_aug is not None:
                noise_aug = noise_aug.to(self.device)
                time_aug = time_aug.to(self.device)
            
            outputs = self.model(original, noise_aug, time_aug)
            z_original = outputs['original']
            z_noise = outputs.get('noise_augmented', z_original)
            z_time = outputs.get('time_stretched', z_original)
            
            loss, _ = self.criterion(z_original, z_noise, z_time, labels)
            val_losses.append(loss.item())
        
        val_metrics = {
            'val_loss': sum(val_losses) / len(val_losses)
        }
        
        return val_metrics
    
    def train(self) -> Dict[str, list]:
        """
        Full training loop.
        
        Implements Algorithm 1 from the paper.
        
        Returns:
            Dictionary of training history
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.training.epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate()
            if 'val_loss' in val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}"
                + (f" - Val Loss: {val_metrics.get('val_loss', 'N/A')}" if val_metrics else "")
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Save best model
            if val_metrics and val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best_model.pt")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        self.logger.info("Training completed!")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")