# prepare_librispeech.py
"""
Prepare LibriSpeech dataset for WSI training.
Converts the raw LibriSpeech format to speaker-organized directories.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import csv
from tqdm import tqdm
import subprocess

# LibriSpeech structure:
# raw/LibriSpeech/dev-clean/SPEAKER_ID/CHAPTER_ID/SPEAKER_ID-CHAPTER_ID-UTTERANCE_ID.flac

def convert_flac_to_wav(flac_path, wav_path):
    """Convert FLAC to WAV using ffmpeg or soundfile."""
    try:
        import soundfile as sf
        data, sr = sf.read(flac_path)
        sf.write(wav_path, data, sr)
        return True
    except Exception as e:
        # Try ffmpeg as fallback
        try:
            subprocess.run(
                ['ffmpeg', '-i', str(flac_path), '-ar', '16000', '-ac', '1', str(wav_path), '-y'],
                capture_output=True,
                check=True
            )
            return True
        except:
            print(f"Error converting {flac_path}: {e}")
            return False


def prepare_librispeech():
    """
    Prepare LibriSpeech dataset for WSI training.
    """
    
    print("=" * 60)
    print("Preparing LibriSpeech Dataset for WSI")
    print("=" * 60)
    
    # Paths
    raw_dir = Path("data/librispeech/raw/LibriSpeech")
    output_dir = Path("data/librispeech_prepared")
    
    # Check if raw data exists
    if not raw_dir.exists():
        print(f"❌ Raw data not found at: {raw_dir}")
        print("   Please run download_librispeech.py first")
        return
    
    # Find all available subsets
    subsets = []
    for subset_name in ['dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360']:
        subset_path = raw_dir / subset_name
        if subset_path.exists():
            subsets.append(subset_name)
            print(f"Found subset: {subset_name}")
    
    if not subsets:
        print("❌ No LibriSpeech subsets found!")
        return
    
    # Collect all audio files grouped by speaker
    print("\n1. Scanning for audio files...")
    speaker_files = defaultdict(list)
    
    for subset in subsets:
        subset_path = raw_dir / subset
        
        # LibriSpeech structure: subset/speaker_id/chapter_id/*.flac
        for speaker_dir in subset_path.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                
                for chapter_dir in speaker_dir.iterdir():
                    if chapter_dir.is_dir():
                        for audio_file in chapter_dir.glob("*.flac"):
                            speaker_files[speaker_id].append(audio_file)
    
    print(f"   Found {len(speaker_files)} speakers")
    total_files = sum(len(files) for files in speaker_files.values())
    print(f"   Found {total_files} audio files")
    
    # Filter speakers with enough samples
    min_samples = 5
    valid_speakers = {
        spk: files for spk, files in speaker_files.items() 
        if len(files) >= min_samples
    }
    print(f"   Speakers with >= {min_samples} samples: {len(valid_speakers)}")
    
    # Split speakers into train/val/test (70/15/15)
    print("\n2. Splitting speakers into train/val/test...")
    speakers = list(valid_speakers.keys())
    random.seed(42)
    random.shuffle(speakers)
    
    n_speakers = len(speakers)
    n_train = int(0.70 * n_speakers)
    n_val = int(0.15 * n_speakers)
    
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]
    
    print(f"   Train speakers: {len(train_speakers)}")
    print(f"   Val speakers: {len(val_speakers)}")
    print(f"   Test speakers: {len(test_speakers)}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    def process_split(speakers_list, split_name, max_samples_per_speaker=50):
        """Process speakers for a split."""
        print(f"\n3. Processing {split_name} split...")
        
        split_dir = output_dir / split_name
        total_processed = 0
        
        for speaker_id in tqdm(speakers_list, desc=f"{split_name}"):
            speaker_out_dir = split_dir / f"speaker_{speaker_id}"
            speaker_out_dir.mkdir(exist_ok=True)
            
            # Get files for this speaker
            files = valid_speakers[speaker_id][:max_samples_per_speaker]
            
            for idx, flac_path in enumerate(files):
                wav_path = speaker_out_dir / f"audio_{idx:04d}.wav"
                
                if not wav_path.exists():
                    if convert_flac_to_wav(flac_path, wav_path):
                        total_processed += 1
                else:
                    total_processed += 1
        
        print(f"   Processed {total_processed} files for {split_name}")
        return total_processed
    
    # Process all splits
    train_count = process_split(train_speakers, 'train', max_samples_per_speaker=50)
    val_count = process_split(val_speakers, 'val', max_samples_per_speaker=20)
    test_count = process_split(test_speakers, 'test', max_samples_per_speaker=20)
    
    # Create manifest files
    print("\n4. Creating manifest files...")
    
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        manifest_path = output_dir / f"{split}_manifest.csv"
        
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['audio_path', 'speaker_id', 'language'])
            
            for speaker_dir in sorted(split_dir.iterdir()):
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name
                    for audio_file in sorted(speaker_dir.glob("*.wav")):
                        writer.writerow([str(audio_file), speaker_id, 'en'])
        
        # Count entries
        with open(manifest_path, 'r') as f:
            count = sum(1 for _ in f) - 1  # Subtract header
        print(f"   Created {manifest_path.name} with {count} entries")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_speakers)} speakers, {train_count} files")
    print(f"  Val:   {len(val_speakers)} speakers, {val_count} files")
    print(f"  Test:  {len(test_speakers)} speakers, {test_count} files")
    
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── speaker_XXX/")
    print(f"  │   │   ├── audio_0000.wav")
    print(f"  │   │   └── ...")
    print(f"  │   └── ...")
    print(f"  ├── val/")
    print(f"  ├── test/")
    print(f"  ├── train_manifest.csv")
    print(f"  ├── val_manifest.csv")
    print(f"  └── test_manifest.csv")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Run training:")
    print(f"   python run_training.py --train_path {output_dir}/train --val_path {output_dir}/val")
    print("\n2. Or use the quick training script:")
    print("   python run_quick_training.py")
    print("=" * 60)


if __name__ == "__main__":
    prepare_librispeech()