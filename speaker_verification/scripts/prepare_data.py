# speaker_verification/scripts/prepare_data.py
"""
Prepare datasets for CAM++ speaker verification training.
Supports VoxCeleb, CN-Celeb, LibriSpeech, and synthetic data.
"""

import os
import sys
from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def prepare_cnceleb(data_dir: str, output_dir: str):
    """
    Prepare CN-Celeb dataset.
    
    From the paper:
    "For CN-Celeb, the development sets of CN-Celeb1 and CN-Celeb2 are used for training, 
    which contain 2785 speakers."
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("Preparing CN-Celeb dataset...")
    
    # Look for CN-Celeb directories
    cnceleb1_dir = data_dir / "CN-Celeb_flac"
    cnceleb2_dir = data_dir / "CN-Celeb2_flac"
    
    samples = []
    speaker_files = defaultdict(list)
    
    # Process CN-Celeb1
    if cnceleb1_dir.exists():
        print(f"Processing CN-Celeb1 from {cnceleb1_dir}...")
        for speaker_dir in tqdm(list(cnceleb1_dir.iterdir()), desc="CN-Celeb1"):
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                for audio_file in speaker_dir.rglob("*.flac"):
                    samples.append({
                        'audio_path': str(audio_file),
                        'speaker_id': f"cn1_{speaker_id}"
                    })
                    speaker_files[f"cn1_{speaker_id}"].append(str(audio_file))
    else:
        print(f"⚠️ CN-Celeb1 not found at {cnceleb1_dir}")
    
    # Process CN-Celeb2
    if cnceleb2_dir.exists():
        print(f"Processing CN-Celeb2 from {cnceleb2_dir}...")
        for speaker_dir in tqdm(list(cnceleb2_dir.iterdir()), desc="CN-Celeb2"):
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                for audio_file in speaker_dir.rglob("*.flac"):
                    samples.append({
                        'audio_path': str(audio_file),
                        'speaker_id': f"cn2_{speaker_id}"
                    })
                    speaker_files[f"cn2_{speaker_id}"].append(str(audio_file))
    else:
        print(f"⚠️ CN-Celeb2 not found at {cnceleb2_dir}")
    
    if len(samples) == 0:
        print("❌ No CN-Celeb data found!")
        print("   Please download from: http://cnceleb.org/")
        return
    
    print(f"Found {len(samples)} samples from {len(speaker_files)} speakers")
    
    # Split speakers (80% train, 10% val, 10% test)
    speakers = list(speaker_files.keys())
    random.shuffle(speakers)
    
    n_train = int(0.8 * len(speakers))
    n_val = int(0.1 * len(speakers))
    
    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train:n_train + n_val])
    test_speakers = set(speakers[n_train + n_val:])
    
    # Create splits
    for split, split_speakers in [
        ('train', train_speakers),
        ('val', val_speakers),
        ('test', test_speakers)
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        split_samples = [s for s in samples if s['speaker_id'] in split_speakers]
        
        manifest_path = output_dir / f"{split}_manifest.csv"
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_path', 'speaker_id'])
            for sample in split_samples:
                writer.writerow([sample['audio_path'], sample['speaker_id']])
        
        print(f"  {split}: {len(split_samples)} samples, {len(split_speakers)} speakers")
    
    print("\n✅ CN-Celeb preparation complete!")


def prepare_voxceleb(data_dir: str, output_dir: str):
    """Prepare VoxCeleb dataset."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Preparing VoxCeleb dataset...")
    
    samples = []
    speaker_ids = set()
    
    for speaker_dir in tqdm(list(data_dir.iterdir()), desc="Scanning speakers"):
        if speaker_dir.is_dir() and speaker_dir.name.startswith("id"):
            speaker_id = speaker_dir.name
            speaker_ids.add(speaker_id)
            
            for video_dir in speaker_dir.iterdir():
                if video_dir.is_dir():
                    for audio_file in video_dir.glob("*.wav"):
                        samples.append({
                            'audio_path': str(audio_file),
                            'speaker_id': speaker_id
                        })
    
    print(f"Found {len(samples)} samples from {len(speaker_ids)} speakers")
    
    manifest_path = output_dir / "train_manifest.csv"
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_path', 'speaker_id'])
        for sample in samples:
            writer.writerow([sample['audio_path'], sample['speaker_id']])
    
    print(f"Saved manifest to {manifest_path}")


def prepare_librispeech(data_dir: str, output_dir: str):
    """Prepare LibriSpeech dataset."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("Preparing LibriSpeech dataset...")
    
    raw_dir = data_dir / "raw" / "LibriSpeech"
    if not raw_dir.exists():
        # Try speaker_identification path
        alt_dir = Path("../speaker_identification/data/librispeech/raw/LibriSpeech")
        if alt_dir.exists():
            raw_dir = alt_dir
        else:
            print(f"❌ LibriSpeech not found")
            print("   Generating synthetic data instead...")
            prepare_synthetic(str(output_dir), 100)
            return
    
    subsets = []
    for subset in ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean']:
        if (raw_dir / subset).exists():
            subsets.append(subset)
    
    print(f"Found subsets: {subsets}")
    
    samples = defaultdict(list)
    
    for subset in subsets:
        subset_dir = raw_dir / subset
        split = "train" if "train" in subset else ("val" if "dev" in subset else "test")
        
        for speaker_dir in subset_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                
                for chapter_dir in speaker_dir.iterdir():
                    if chapter_dir.is_dir():
                        for audio_file in chapter_dir.glob("*.flac"):
                            samples[split].append({
                                'audio_path': str(audio_file),
                                'speaker_id': speaker_id
                            })
    
    for split, split_samples in samples.items():
        speakers = set(s['speaker_id'] for s in split_samples)
        print(f"  {split}: {len(split_samples)} samples, {len(speakers)} speakers")
    
    for split, split_samples in samples.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = output_dir / f"{split}_manifest.csv"
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_path', 'speaker_id'])
            for sample in split_samples:
                writer.writerow([sample['audio_path'], sample['speaker_id']])
    
    print("\n✅ LibriSpeech preparation complete!")


def prepare_synthetic(output_dir: str, num_speakers: int = 100, samples_per_speaker: int = 100):
    """Generate more realistic synthetic dataset for testing."""
    
    output_dir = Path(output_dir)
    
    print(f"Generating synthetic dataset with {num_speakers} speakers...")
    print("(Using more diverse audio patterns for better training)")
    
    # Different base frequencies and characteristics for each speaker
    np.random.seed(42)
    speaker_params = []
    for i in range(num_speakers):
        speaker_params.append({
            'base_freq': np.random.uniform(100, 300),
            'freq_var': np.random.uniform(5, 20),
            'harmonics': np.random.randint(3, 8),
            'vibrato_rate': np.random.uniform(3, 8),
            'vibrato_depth': np.random.uniform(0.02, 0.1)
        })
    
    for split, num_samples in [
        ('train', samples_per_speaker),
        ('val', max(10, samples_per_speaker // 5)),
        ('test', max(10, samples_per_speaker // 5))
    ]:
        split_dir = output_dir / split
        samples = []
        
        for spk_idx in tqdm(range(num_speakers), desc=f"Generating {split}"):
            speaker_id = f"speaker_{spk_idx:04d}"
            speaker_dir = split_dir / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            params = speaker_params[spk_idx]
            
            for sample_idx in range(num_samples):
                duration = np.random.uniform(2.0, 5.0)
                sample_rate = 16000
                t = np.linspace(0, duration, int(duration * sample_rate))
                
                # Base frequency with natural variation
                f0 = params['base_freq'] + np.random.uniform(-params['freq_var'], params['freq_var'])
                
                # Add vibrato
                vibrato = params['vibrato_depth'] * np.sin(2 * np.pi * params['vibrato_rate'] * t)
                f_t = f0 * (1 + vibrato)
                
                # Generate with harmonics
                audio = np.zeros_like(t)
                for h in range(1, params['harmonics'] + 1):
                    amplitude = 1.0 / h
                    audio += amplitude * np.sin(2 * np.pi * h * np.cumsum(f_t) / sample_rate)
                
                # Add formants (speaker-specific resonances)
                formant_freqs = [
                    params['base_freq'] * np.random.uniform(2.5, 3.5),
                    params['base_freq'] * np.random.uniform(4.5, 5.5),
                ]
                for ff in formant_freqs:
                    audio += 0.3 * np.sin(2 * np.pi * ff * t) * np.exp(-t * 0.5)
                
                # Amplitude envelope (speech-like)
                envelope = np.ones_like(t)
                num_syllables = int(duration * np.random.uniform(2, 4))
                for _ in range(num_syllables):
                    center = np.random.uniform(0, duration)
                    width = np.random.uniform(0.1, 0.3)
                    envelope *= (1 + 0.5 * np.exp(-((t - center) / width) ** 2))
                
                audio = audio * envelope
                
                # Add some noise
                audio += 0.02 * np.random.randn(len(t))
                
                # Normalize
                audio = (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)
                
                audio_path = speaker_dir / f"audio_{sample_idx:04d}.wav"
                
                if HAS_SOUNDFILE:
                    sf.write(audio_path, audio, sample_rate)
                
                samples.append({
                    'audio_path': str(audio_path),
                    'speaker_id': speaker_id
                })
        
        manifest_path = output_dir / f"{split}_manifest.csv"
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_path', 'speaker_id'])
            for sample in samples:
                writer.writerow([sample['audio_path'], sample['speaker_id']])
        
        print(f"  Created {len(samples)} samples for {split}")
    
    print("\n✅ Synthetic dataset generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for CAM++")
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['voxceleb', 'cnceleb', 'librispeech', 'synthetic'],
                        help='Dataset to prepare')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='data/prepared',
                        help='Output directory')
    parser.add_argument('--num_speakers', type=int, default=100,
                        help='Number of speakers for synthetic dataset')
    parser.add_argument('--samples_per_speaker', type=int, default=100,
                        help='Samples per speaker for synthetic dataset')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Preparing {args.dataset} dataset")
    print("=" * 60)
    
    if args.dataset == 'voxceleb':
        prepare_voxceleb(args.data_dir, args.output_dir)
    elif args.dataset == 'cnceleb':
        prepare_cnceleb(args.data_dir, args.output_dir)
    elif args.dataset == 'librispeech':
        prepare_librispeech(args.data_dir, args.output_dir)
    elif args.dataset == 'synthetic':
        prepare_synthetic(args.output_dir, args.num_speakers, args.samples_per_speaker)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Train the model:")
    print(f"   python train.py --data_path {args.output_dir}/train --val_path {args.output_dir}/val --epochs 50")
    print(f"\n2. Evaluate the model:")
    print(f"   python evaluate.py --checkpoint checkpoints/best_model.pt")


if __name__ == "__main__":
    main()