# speaker_identification/train.py
"""
Main training script for WSI model.
"""

import argparse
import logging
from pathlib import Path

from config.config import WSIConfig, get_default_config
from data.preprocessing import AudioPreprocessor
from data.augmentation import AudioAugmentor
from data.dataset import SpeakerDataset, create_dataloader
from models.wsi_model import WSIModel
from training.trainer import WSITrainer
from utils.helpers import set_seed, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train WSI model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = get_logger("wsi_train", log_file=f"{args.output_dir}/train.log")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Config
    config = get_default_config()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.output_dir
    
    logger.info(f"Config: {config}")
    
    # Data
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    augmentor = AudioAugmentor(
        noise_snr_db=config.data.noise_snr_db,
        time_stretch_range=config.data.time_stretch_range,
        sample_rate=config.data.sample_rate
    )
    
    train_dataset = SpeakerDataset(
        data_path=args.data_path,
        preprocessor=preprocessor,
        augmentor=augmentor,
        split="train",
        return_augmented=True
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_loader = None
    if args.val_path:
        val_dataset = SpeakerDataset(
            data_path=args.val_path,
            preprocessor=preprocessor,
            augmentor=None,
            split="val",
            return_augmented=False
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Train speakers: {train_dataset.num_speakers}")
    
    # Model
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim,
        freeze_encoder=config.model.freeze_encoder
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = WSITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train
    history = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()