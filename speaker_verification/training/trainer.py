# speaker_verification/training/trainer.py
"""
Trainer for CAM++ Speaker Verification - FIXED VERSION
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import logging
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cam_plus_plus import CAMPlusPlusClassifier
from config.config import CAMPlusPlusSystemConfig


class CosineAnnealingWarmupLR:
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        max_lr: float = 0.1,
        min_lr: float = 1e-4
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            lr = self.max_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class CAMPlusPlusTrainer:
    """Trainer for CAM++ model - Fixed version."""
    
    def __init__(
        self,
        model: CAMPlusPlusClassifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[CAMPlusPlusSystemConfig] = None,
        model_config: Optional[Dict] = None  # Add model config parameter
    ):
        self.config = config or CAMPlusPlusSystemConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Store model config for checkpoint saving
        self.model_config = model_config or {}
        
        # Use Adam optimizer instead of SGD for more stable training
        # SGD with high LR can be unstable on small/synthetic datasets
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Lower learning rate for Adam
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda')
        self.use_amp = torch.cuda.is_available()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
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
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(features, labels)
                    logits = outputs['logits']
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features, labels)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += features.size(0)
            
            acc = 100. * total_correct / total_samples
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * total_correct / total_samples
        
        return {'train_loss': avg_loss, 'train_acc': accuracy}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for features, labels in tqdm(self.val_loader, desc="Validation"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features, labels)
            logits = outputs['logits']
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += features.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = 100. * total_correct / total_samples
        
        return {'val_loss': avg_loss, 'val_acc': accuracy}
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            self.scheduler.step()
            
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_acc'].append(val_metrics['val_acc'])
            
            log_msg = (
                f"Epoch {epoch + 1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.2f}%"
            )
            if val_metrics:
                log_msg += (
                    f", Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}%"
                )
            self.logger.info(log_msg)
            
            # Always save checkpoint every N epochs
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Save best model based on validation accuracy or loss
            if val_metrics:
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"  ✅ Saved best model (val_acc: {self.best_val_acc:.2f}%)")
                elif val_metrics['val_loss'] < self.best_val_loss and self.best_val_acc == 0:
                    # If accuracy is 0, save based on loss improvement
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"  ✅ Saved best model (val_loss: {self.best_val_loss:.4f})")
            else:
                # Save based on training metrics if no validation
                self.save_checkpoint("best_model.pt")
        
        # Always save final model
        self.save_checkpoint("final_model.pt")
        self.logger.info("Training completed!")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint with model configuration."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            # Save model configuration for proper loading later
            'model_config': self.model_config,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.model_config = checkpoint.get('model_config', {})
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")