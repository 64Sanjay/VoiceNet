# speaker_identification/evaluate.py
"""
Evaluation script for WSI model.
"""

import argparse
import torch
from pathlib import Path

from config.config import get_default_config
from data.preprocessing import AudioPreprocessor
from data.dataset import SpeakerDataset, create_dataloader
from models.wsi_model import WSIModel
from evaluation.evaluator import WSIEvaluator
from utils.helpers import set_seed, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WSI model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=10000, help="Number of pairs to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = get_logger("wsi_eval")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = get_default_config()
    
    # Load model
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Data
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    test_dataset = SpeakerDataset(
        data_path=args.data_path,
        preprocessor=preprocessor,
        augmentor=None,
        split="test",
        return_augmented=False
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Test speakers: {test_dataset.num_speakers}")
    
    # Evaluate
    evaluator = WSIEvaluator(model=model, device=config.device)
    metrics = evaluator.evaluate(test_loader, num_pairs=args.num_pairs)
    
    # Save results
    results_file = Path(args.output_dir) / "results.txt"
    with open(results_file, "w") as f:
        f.write(f"EER: {metrics['eer']:.2f}%\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Threshold: {metrics['eer_threshold']:.4f}\n")
    
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()