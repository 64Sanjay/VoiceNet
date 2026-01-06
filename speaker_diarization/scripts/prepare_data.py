#!/usr/bin/env python3
"""
Prepare data for speaker diarization training.

Supports multiple dataset formats:
- AISHELL-4
- AMI Corpus
- VoxConverse
- Custom format
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_custom_dataset(
    audio_dir: str,
    rttm_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Prepare custom dataset with audio and RTTM files.
    
    Args:
        audio_dir: Directory containing audio files
        rttm_dir: Directory containing RTTM files
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    random.seed(seed)
    
    audio_dir = Path(audio_dir)
    rttm_dir = Path(rttm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching audio and RTTM files
    audio_files = {f.stem: f for f in audio_dir.glob("*.wav")}
    rttm_files = {f.stem: f for f in rttm_dir.glob("*.rttm")}
    
    # Get common files
    common_ids = sorted(set(audio_files.keys()) & set(rttm_files.keys()))
    
    if not common_ids:
        logger.error("No matching audio and RTTM files found!")
        return
    
    logger.info(f"Found {len(common_ids)} matching files")
    
    # Shuffle and split
    random.shuffle(common_ids)
    
    n_total = len(common_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = common_ids[:n_train]
    val_ids = common_ids[n_train:n_train + n_val]
    test_ids = common_ids[n_train + n_val:]
    
    # Write file lists
    with open(output_dir / "train.txt", 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(output_dir / "val.txt", 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(output_dir / "test.txt", 'w') as f:
        f.write('\n'.join(test_ids))
    
    # Create symlinks or copy files
    for split_name, split_ids in [
        ('train', train_ids),
        ('val', val_ids),
        ('test', test_ids),
    ]:
        split_audio_dir = output_dir / split_name / "audio"
        split_rttm_dir = output_dir / split_name / "rttm"
        
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        split_rttm_dir.mkdir(parents=True, exist_ok=True)
        
        for file_id in split_ids:
            audio_src = audio_files[file_id]
            rttm_src = rttm_files[file_id]
            
            audio_dst = split_audio_dir / audio_src.name
            rttm_dst = split_rttm_dir / rttm_src.name
            
            # Create symlinks
            if not audio_dst.exists():
                audio_dst.symlink_to(audio_src.absolute())
            if not rttm_dst.exists():
                rttm_dst.symlink_to(rttm_src.absolute())
    
    # Compute statistics
    stats = compute_dataset_statistics(output_dir)
    
    with open(output_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Dataset prepared:")
    logger.info(f"  Train: {len(train_ids)} files")
    logger.info(f"  Val: {len(val_ids)} files")
    logger.info(f"  Test: {len(test_ids)} files")


def compute_dataset_statistics(data_dir: Path) -> Dict:
    """Compute statistics for the dataset."""
    from ..utils.rttm_utils import RTTMReader, get_speaker_statistics
    
    stats = {
        'splits': {},
        'total': {
            'num_files': 0,
            'total_duration_hours': 0,
            'num_speakers': 0,
        }
    }
    
    rttm_reader = RTTMReader()
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        rttm_dir = split_dir / "rttm"
        
        if not rttm_dir.exists():
            continue
        
        split_stats = {
            'num_files': 0,
            'total_duration_hours': 0,
            'speakers_per_file': [],
            'speech_duration_hours': 0,
        }
        
        all_speakers = set()
        
        for rttm_file in rttm_dir.glob("*.rttm"):
            segments = rttm_reader.read(rttm_file)
            
            for file_id, segs in segments.items():
                split_stats['num_files'] += 1
                
                # Compute duration
                if segs:
                    max_end = max(seg.end for seg in segs)
                    split_stats['total_duration_hours'] += max_end / 3600
                
                # Count speakers
                file_speakers = set(seg.speaker_id for seg in segs)
                split_stats['speakers_per_file'].append(len(file_speakers))
                all_speakers.update(file_speakers)
                
                # Speech duration
                speech_dur = sum(seg.duration for seg in segs)
                split_stats['speech_duration_hours'] += speech_dur / 3600
        
        split_stats['num_unique_speakers'] = len(all_speakers)
        split_stats['avg_speakers_per_file'] = (
            sum(split_stats['speakers_per_file']) / 
            max(len(split_stats['speakers_per_file']), 1)
        )
        
        stats['splits'][split] = split_stats
        stats['total']['num_files'] += split_stats['num_files']
        stats['total']['total_duration_hours'] += split_stats['total_duration_hours']
    
    return stats


def validate_rttm_files(rttm_dir: str) -> List[str]:
    """Validate RTTM files and report issues."""
    from ..utils.rttm_utils import RTTMReader
    
    rttm_dir = Path(rttm_dir)
    rttm_reader = RTTMReader()
    
    issues = []
    
    for rttm_file in rttm_dir.glob("*.rttm"):
        try:
            segments = rttm_reader.read(rttm_file)
            
            for file_id, segs in segments.items():
                # Check for overlapping segments from same speaker
                segs_by_speaker = defaultdict(list)
                for seg in segs:
                    segs_by_speaker[seg.speaker_id].append(seg)
                
                for speaker, speaker_segs in segs_by_speaker.items():
                    speaker_segs.sort(key=lambda x: x.start)
                    for i in range(len(speaker_segs) - 1):
                        if speaker_segs[i].end > speaker_segs[i + 1].start:
                            issues.append(
                                f"{rttm_file.name}: Overlapping segments for speaker {speaker}"
                            )
                            break
                
                # Check for negative durations
                for seg in segs:
                    if seg.duration <= 0:
                        issues.append(
                            f"{rttm_file.name}: Non-positive duration for segment"
                        )
        
        except Exception as e:
            issues.append(f"{rttm_file.name}: Parse error - {e}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for speaker diarization"
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Custom dataset
    custom_parser = subparsers.add_parser('custom', help='Prepare custom dataset')
    custom_parser.add_argument('--audio-dir', type=str, required=True)
    custom_parser.add_argument('--rttm-dir', type=str, required=True)
    custom_parser.add_argument('--output-dir', type=str, required=True)
    custom_parser.add_argument('--train-ratio', type=float, default=0.8)
    custom_parser.add_argument('--val-ratio', type=float, default=0.1)
    custom_parser.add_argument('--test-ratio', type=float, default=0.1)
    custom_parser.add_argument('--seed', type=int, default=42)
    
    # Validate
    validate_parser = subparsers.add_parser('validate', help='Validate RTTM files')
    validate_parser.add_argument('--rttm-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'custom':
        prepare_custom_dataset(
            audio_dir=args.audio_dir,
            rttm_dir=args.rttm_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    
    elif args.command == 'validate':
        issues = validate_rttm_files(args.rttm_dir)
        if issues:
            logger.warning(f"Found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("All RTTM files valid!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()