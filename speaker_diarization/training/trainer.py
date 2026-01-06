"""
Training pipeline for speaker diarization models.
Fixed for proper GPU usage with PyTorch 2.x
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Any, Union
from pathlib import Path
import logging
import json
import time
from tqdm import tqdm
from collections import defaultdict
import os

# Robust imports
try:
    from ..models.diarization_model import SpeakerDiarizationModel
    from ..models.losses import DiarizationLoss, PermutationInvariantTrainingLoss
    from ..utils.helpers import (
        AverageMeter,
        EarlyStopping,
        save_checkpoint,
        load_checkpoint,
        set_seed,
    )
except ImportError:
    from models.diarization_model import SpeakerDiarizationModel
    from models.losses import DiarizationLoss, PermutationInvariantTrainingLoss
    from utils.helpers import (
        AverageMeter,
        EarlyStopping,
        save_checkpoint,
        load_checkpoint,
        set_seed,
    )

# Optional evaluator import
try:
    from ..evaluation.evaluator import Evaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False
    Evaluator = None


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for speaker diarization models.
    
    Features:
    - Mixed precision training (PyTorch 2.x compatible)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Logging (tensorboard/wandb)
    - Proper GPU utilization
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        
        # Optimizer settings
        optimizer: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        
        # Scheduler settings
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        
        # Training settings
        num_epochs: int = 100,
        gradient_clip_val: float = 5.0,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        
        # Loss settings
        criterion: Optional[nn.Module] = None,
        use_pit: bool = True,
        pit_weight: float = 1.0,
        embedding_weight: float = 0.1,
        vad_weight: float = 0.5,
        num_speakers: int = 4,
        
        # Checkpoint settings
        output_dir: str = "./outputs",
        save_top_k: int = 3,
        checkpoint_every_n_epochs: int = 1,
        
        # Early stopping
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        
        # Device
        device: Union[str, torch.device] = None,
        
        # Logging
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: str = "speaker-diarization",
        experiment_name: str = "experiment",
        log_every: int = 50,
        
        # Seed
        seed: int = 42,
        
        # Performance
        compile_model: bool = False,
    ):
        """Initialize trainer with proper GPU setup."""
        
        # Set seed first
        set_seed(seed)
        
        # Setup device properly
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Log device info
        self._log_device_info()
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Get num_speakers from model if available
        if hasattr(model, 'num_speakers'):
            num_speakers = model.num_speakers
        elif hasattr(model, 'max_speakers'):
            num_speakers = model.max_speakers
        self.num_speakers = num_speakers
        
        # Optionally compile model (PyTorch 2.x)
        if compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        self.log_every = log_every
        
        # Mixed precision - use new API
        self.use_amp = use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision training (AMP)")
        else:
            self.scaler = None
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = DiarizationLoss(
                num_speakers=num_speakers,
                use_pit=use_pit,
                pit_weight=pit_weight,
                embedding_weight=embedding_weight,
                vad_weight=vad_weight,
            )
        
        # Optimizer
        self.optimizer = self._create_optimizer(
            optimizer, learning_rate, weight_decay, betas
        )
        
        # Scheduler
        self.scheduler_name = scheduler
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.scheduler = self._create_scheduler(
            scheduler, warmup_epochs, min_lr
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode="min" if "loss" in early_stopping_metric else "max",
        )
        self.early_stopping_metric = early_stopping_metric
        
        # Checkpoint management
        self.save_top_k = save_top_k
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.best_checkpoints: List[Dict] = []
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('inf')
        
        # TensorBoard setup
        self.use_tensorboard = use_tensorboard
        self.writer = None
        self._setup_tensorboard()
        
        # Wandb setup
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name
        self.wandb = None
        self._setup_wandb(wandb_project, learning_rate)
        
        # History
        self.history = defaultdict(list)
        
        # Log model info
        self._log_model_info()
    
    def _log_device_info(self):
        """Log device information."""
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    def _log_model_info(self):
        """Log model information."""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.output_dir / 'tensorboard')
                logger.info(f"TensorBoard logging to {self.output_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.use_tensorboard = False
    
    def _setup_wandb(self, wandb_project: str, learning_rate: float):
        """Setup Weights & Biases logging."""
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=self.experiment_name,
                    config={
                        'learning_rate': learning_rate,
                        'num_epochs': self.num_epochs,
                        'batch_size': self.train_loader.batch_size if self.train_loader else None,
                        'num_speakers': self.num_speakers,
                        'use_amp': self.use_amp,
                        'model_params': sum(p.numel() for p in self.model.parameters()),
                    }
                )
                self.wandb = wandb
                logger.info("Wandb logging enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling")
                self.use_wandb = False
    
    def _create_optimizer(
        self,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float,
        betas: tuple,
    ) -> optim.Optimizer:
        """Create optimizer with proper parameter groups."""
        
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to bias and normalization layers
            if 'bias' in name or 'norm' in name.lower() or 'bn' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == "adam":
            return optim.Adam(param_groups, lr=learning_rate, betas=betas)
        elif optimizer_name == "adamw":
            return optim.AdamW(param_groups, lr=learning_rate, betas=betas)
        elif optimizer_name == "sgd":
            return optim.SGD(param_groups, lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(
        self,
        scheduler_name: str,
        warmup_epochs: int,
        min_lr: float,
    ) -> Optional[Any]:
        """Create learning rate scheduler."""
        
        if self.train_loader is None:
            return None
        
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = len(self.train_loader) * warmup_epochs
        
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name == "cosine":
            # Cosine annealing with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(min_lr / self.optimizer.defaults['lr'], 
                          0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.defaults['lr'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
                anneal_strategy='cos',
                final_div_factor=self.optimizer.defaults['lr'] / min_lr if min_lr > 0 else 1000,
            )
        
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=min_lr,
            )
        
        elif scheduler_name == "none" or scheduler_name is None:
            return None
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor items in batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def _forward_pass(
        self, 
        features: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that handles different model output formats.
        
        Returns:
            Dict with 'speakers' and optionally 'vad', 'embeddings'
        """
        # Call model
        outputs = self.model(features, lengths)
        
        # Normalize output format
        if isinstance(outputs, dict):
            # Already a dict - ensure 'speakers' key exists
            if 'speakers' not in outputs and 'predictions' in outputs:
                outputs['speakers'] = outputs['predictions']
            return outputs
        elif isinstance(outputs, torch.Tensor):
            # Single tensor output
            return {'speakers': outputs}
        elif isinstance(outputs, tuple):
            # Tuple output (predictions, embeddings) or (predictions, vad)
            result = {'speakers': outputs[0]}
            if len(outputs) > 1:
                if outputs[1].shape[-1] == self.num_speakers or outputs[1].dim() == 2:
                    result['embeddings'] = outputs[1]
                else:
                    result['vad'] = outputs[1]
            return result
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")
    
    def train(self) -> Dict[str, List[float]]:
        """Run training loop."""
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            epoch_time = time.time() - epoch_start
            samples_per_sec = len(self.train_loader.dataset) / epoch_time
            
            # Validation epoch
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
            
            # Update scheduler (for epoch-based schedulers)
            self._update_scheduler_epoch(val_metrics, train_metrics)
            
            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time, samples_per_sec)
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            metric_value = val_metrics.get(
                self.early_stopping_metric,
                train_metrics.get('train_loss', 0)
            )
            if self.early_stopping(metric_value, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        # Save final model
        self._save_checkpoint(self.current_epoch, val_metrics, final=True)
        
        # Cleanup
        self._cleanup()
        
        return dict(self.history)
    
    def _update_scheduler_epoch(self, val_metrics: Dict, train_metrics: Dict):
        """Update scheduler at end of epoch."""
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
        elif not isinstance(self.scheduler, (optim.lr_scheduler.OneCycleLR, 
                                              optim.lr_scheduler.LambdaLR)):
            self.scheduler.step()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with proper GPU usage."""
        self.model.train()
        
        # Metrics
        loss_meter = AverageMeter("Loss")
        speaker_loss_meter = AverageMeter("SpeakerLoss")
        vad_loss_meter = AverageMeter("VADLoss")
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]",
            leave=True,
        )
        
        # Zero gradients at start
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to GPU
            batch = self._move_batch_to_device(batch)
            
            features = batch['features']
            labels = batch['labels']
            lengths = batch.get('lengths', None)
            vad = batch.get('vad', None)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self._forward_pass(features, lengths)
                    loss, loss_dict = self.criterion(outputs, labels, vad, lengths)
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Step-based scheduler update
                    if self.scheduler is not None and isinstance(
                        self.scheduler, (optim.lr_scheduler.OneCycleLR, 
                                        optim.lr_scheduler.LambdaLR)
                    ):
                        self.scheduler.step()
            else:
                # Without AMP
                outputs = self._forward_pass(features, lengths)
                loss, loss_dict = self.criterion(outputs, labels, vad, lengths)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.scheduler is not None and isinstance(
                        self.scheduler, (optim.lr_scheduler.OneCycleLR,
                                        optim.lr_scheduler.LambdaLR)
                    ):
                        self.scheduler.step()
            
            # Update meters
            loss_meter.update(loss.item() * self.accumulation_steps)
            speaker_loss_meter.update(loss_dict.get('speaker_loss', 0))
            vad_loss_meter.update(loss_dict.get('vad_loss', 0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'spk': f"{speaker_loss_meter.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # TensorBoard logging
            if self.writer and self.global_step % self.log_every == 0:
                self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/speaker_loss', speaker_loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/vad_loss', vad_loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return {
            'train_loss': loss_meter.avg,
            'train_speaker_loss': speaker_loss_meter.avg,
            'train_vad_loss': vad_loss_meter.avg,
        }
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        loss_meter = AverageMeter("Loss")
        speaker_loss_meter = AverageMeter("SpeakerLoss")
        accuracy_meter = AverageMeter("Accuracy")
        precision_meter = AverageMeter("Precision")
        recall_meter = AverageMeter("Recall")
        f1_meter = AverageMeter("F1")
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]",
            leave=True,
        )
        
        for batch in pbar:
            # Move to device
            batch = self._move_batch_to_device(batch)
            
            features = batch['features']
            labels = batch['labels']
            lengths = batch.get('lengths', None)
            vad = batch.get('vad', None)
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self._forward_pass(features, lengths)
                loss, loss_dict = self.criterion(outputs, labels, vad, lengths)
            
            # Get predictions
            preds = outputs.get('speakers', outputs.get('predictions'))
            
            # Compute metrics
            metrics = self._compute_metrics(preds, labels, lengths)
            
            # Update meters
            loss_meter.update(loss.item())
            speaker_loss_meter.update(loss_dict.get('speaker_loss', 0))
            accuracy_meter.update(metrics['accuracy'])
            precision_meter.update(metrics['precision'])
            recall_meter.update(metrics['recall'])
            f1_meter.update(metrics['f1'])
            
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc': f"{accuracy_meter.avg:.2f}%",
                'f1': f"{f1_meter.avg:.2f}%",
            })
        
        return {
            'val_loss': loss_meter.avg,
            'val_speaker_loss': speaker_loss_meter.avg,
            'val_accuracy': accuracy_meter.avg,
            'val_precision': precision_meter.avg,
            'val_recall': recall_meter.avg,
            'val_f1': f1_meter.avg,
        }
    
    def _compute_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pred_binary = (preds > 0.5).float()
        
        if lengths is not None:
            batch_size, time_steps = labels.shape[:2]
            mask = torch.arange(time_steps, device=self.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(labels)
            
            correct = ((pred_binary == labels) & mask).sum()
            total = mask.sum()
            accuracy = (correct / total).item() * 100
            
            tp = ((pred_binary == 1) & (labels == 1) & mask).sum().float()
            fp = ((pred_binary == 1) & (labels == 0) & mask).sum().float()
            fn = ((pred_binary == 0) & (labels == 1) & mask).sum().float()
        else:
            accuracy = (pred_binary == labels).float().mean().item() * 100
            tp = ((pred_binary == 1) & (labels == 1)).sum().float()
            fp = ((pred_binary == 1) & (labels == 0)).sum().float()
            fn = ((pred_binary == 0) & (labels == 1)).sum().float()
        
        precision = (tp / (tp + fp + 1e-8)).item() * 100
        recall = (tp / (tp + fn + 1e-8)).item() * 100
        f1 = (2 * precision * recall / (precision + recall + 1e-8))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
        samples_per_sec: float,
    ):
        """Log epoch metrics."""
        # Console logging
        log_str = f"Epoch {epoch + 1}/{self.num_epochs}"
        log_str += f" | Time: {epoch_time:.1f}s | Speed: {samples_per_sec:.1f} samples/sec"
        
        log_str += f" | Train Loss: {train_metrics.get('train_loss', 0):.4f}"
        
        if val_metrics:
            log_str += f" | Val Loss: {val_metrics.get('val_loss', 0):.4f}"
            log_str += f" | Val F1: {val_metrics.get('val_f1', 0):.2f}%"
        
        logger.info(log_str)
        
        # History
        for key, value in {**train_metrics, **val_metrics}.items():
            self.history[key].append(value)
        self.history['epoch'].append(epoch + 1)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.history['epoch_time'].append(epoch_time)
        
        # TensorBoard
        if self.writer:
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            self.writer.add_scalar('epoch/time', epoch_time, epoch)
            self.writer.add_scalar('epoch/samples_per_sec', samples_per_sec, epoch)
        
        # Wandb logging
        if self.wandb:
            self.wandb.log({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
            })
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        final: bool = False,
    ):
        """Save model checkpoint."""
        val_loss = val_metrics.get('val_loss', float('inf'))
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self._get_model_config(),
            'history': dict(self.history),
        }
        
        if final:
            path = self.checkpoint_dir / "final_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved final model to {path}")
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"✓ New best model saved! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_info = {
            'epoch': epoch,
            'val_loss': val_loss,
            'path': checkpoint_path,
        }
        
        self.best_checkpoints.append(checkpoint_info)
        self.best_checkpoints.sort(key=lambda x: x['val_loss'])
        
        # Keep only top-k
        while len(self.best_checkpoints) > self.save_top_k:
            removed = self.best_checkpoints.pop()
            if removed['path'].exists() and removed['path'] != checkpoint_path:
                removed['path'].unlink()
        
        # Save current if in top-k
        if any(c['path'] == checkpoint_path for c in self.best_checkpoints):
            torch.save(checkpoint, checkpoint_path)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        config = {
            'num_speakers': self.num_speakers,
        }
        
        # Try to get additional config from model
        for attr in ['sample_rate', 'n_mels', 'embedding_dim', 'hidden_dim']:
            if hasattr(self.model, attr):
                config[attr] = getattr(self.model, attr)
        
        return config
    
    def _cleanup(self):
        """Cleanup resources."""
        # Close tensorboard
        if self.writer:
            self.writer.close()
        
        # Close wandb
        if self.wandb:
            self.wandb.finish()
        
        # Save history
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        logger.info(f"Saved training history to {history_path}")
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        
        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}, best val loss: {self.best_val_loss:.4f}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict] = None,
    resume_from: Optional[str] = None,
    **kwargs,
) -> Dict[str, List[float]]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        resume_from: Checkpoint path to resume from
        **kwargs: Additional trainer arguments
        
    Returns:
        Training history
    """
    # Merge config and kwargs
    if config is None:
        config = {}
    
    trainer_kwargs = {**config, **kwargs}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **trainer_kwargs,
    )
    
    # Resume if specified
    if resume_from:
        trainer.resume(resume_from)
    
    # Train
    history = trainer.train()
    
    return history