#!/usr/bin/env python3
"""
Download and prepare AISHELL-4 dataset for speaker diarization.

AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by
8-channel circular microphone array for speech processing in conference scenarios.

Website: https://www.aishelltech.com/aishell_4
Paper: https://arxiv.org/abs/2104.03603

Audio format: FLAC
Annotations: RTTM and TextGrid
"""

import os
import sys
import argparse
import subprocess
import tarfile
import zipfile
from pathlib import Path
import shutil
import logging
from typing import Optional, List, Dict, Tuple
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# AISHELL-4 download URLs from OpenSLR
OPENSLR_BASE = "https://www.openslr.org/resources/111/"

SPLITS = {
    'train_L': 'train_L.tar.gz',
    'train_M': 'train_M.tar.gz', 
    'train_S': 'train_S.tar.gz',
    'test': 'test.tar.gz',
}

# Audio file extensions to look for (AISHELL-4 uses FLAC)
AUDIO_EXTENSIONS = ['.flac', '.wav', '.mp3', '.m4a']


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except ImportError:
        logger.info("requests not available, using wget")
        result = subprocess.run(
            ['wget', '-O', str(output_path), url],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract tar.gz or zip archive."""
    logger.info(f"Extracting {archive_path}")
    
    if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
    elif str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def download_aishell4(
    output_dir: str,
    splits: Optional[List[str]] = None,
    keep_archives: bool = False,
):
    """
    Download AISHELL-4 dataset.
    
    Args:
        output_dir: Output directory
        splits: Splits to download (train_L, train_M, train_S, test)
        keep_archives: Keep downloaded archives
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archives_dir = output_dir / "archives"
    archives_dir.mkdir(exist_ok=True)
    
    if splits is None:
        splits = list(SPLITS.keys())
    
    # Download each split
    for split in splits:
        if split not in SPLITS:
            logger.warning(f"Unknown split: {split}")
            continue
            
        logger.info(f"Downloading {split} split...")
        
        filename = SPLITS[split]
        url = f"{OPENSLR_BASE}{filename}"
        archive_path = archives_dir / filename
        
        if archive_path.exists():
            logger.info(f"{archive_path} already exists, skipping download")
        else:
            success = download_file(url, archive_path)
            if not success:
                logger.error(f"Failed to download {split}")
                continue
        
        # Extract
        extract_archive(archive_path, output_dir)
    
    # Clean up archives
    if not keep_archives and archives_dir.exists():
        shutil.rmtree(archives_dir)
    
    logger.info(f"AISHELL-4 downloaded to {output_dir}")


def find_audio_files(data_dir: Path) -> List[Path]:
    """Find all audio files in directory."""
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(audio_files)


def find_matching_rttm(audio_path: Path, data_dir: Path) -> Optional[Path]:
    """Find matching RTTM file for an audio file."""
    file_id = audio_path.stem
    
    # Search patterns
    search_patterns = [
        # Same directory
        audio_path.with_suffix('.rttm'),
        # TextGrid directory (sibling to wav directory)
        audio_path.parent.parent / 'TextGrid' / f"{file_id}.rttm",
        # Same parent with TextGrid subdirectory
        audio_path.parent / 'TextGrid' / f"{file_id}.rttm",
    ]
    
    for pattern in search_patterns:
        if pattern.exists():
            return pattern
    
    # Search entire data directory
    rttm_files = list(data_dir.rglob(f"{file_id}.rttm"))
    if rttm_files:
        return rttm_files[0]
    
    return None


def convert_flac_to_wav(flac_path: Path, wav_path: Path) -> bool:
    """Convert FLAC to WAV using ffmpeg or soundfile."""
    try:
        # Try using soundfile (Python library)
        import soundfile as sf
        data, samplerate = sf.read(str(flac_path))
        sf.write(str(wav_path), data, samplerate)
        return True
    except ImportError:
        pass
    
    try:
        # Try using torchaudio
        import torchaudio
        waveform, sr = torchaudio.load(str(flac_path))
        torchaudio.save(str(wav_path), waveform, sr)
        return True
    except Exception:
        pass
    
    try:
        # Try using ffmpeg
        result = subprocess.run(
            ['ffmpeg', '-i', str(flac_path), '-y', str(wav_path)],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Try using sox
        result = subprocess.run(
            ['sox', str(flac_path), str(wav_path)],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return False


def copy_or_convert_audio(src_path: Path, dst_path: Path) -> bool:
    """Copy audio file, converting FLAC to WAV if necessary."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if src_path.suffix.lower() == '.flac':
        # Convert FLAC to WAV
        wav_dst = dst_path.with_suffix('.wav')
        if wav_dst.exists():
            return True
        
        logger.debug(f"Converting {src_path.name} to WAV...")
        success = convert_flac_to_wav(src_path, wav_dst)
        
        if not success:
            # If conversion fails, just copy the FLAC file
            logger.warning(f"Could not convert {src_path.name} to WAV, copying FLAC instead")
            flac_dst = dst_path.with_suffix('.flac')
            shutil.copy(src_path, flac_dst)
            return True
        
        return success
    else:
        # Copy directly
        if dst_path.exists():
            return True
        shutil.copy(src_path, dst_path)
        return True


def prepare_aishell4(
    data_dir: str,
    output_dir: str,
    convert_to_wav: bool = True,
    max_files: Optional[int] = None,
):
    """
    Prepare AISHELL-4 for training.
    
    Args:
        data_dir: Raw AISHELL-4 directory
        output_dir: Output directory  
        convert_to_wav: Convert FLAC files to WAV
        max_files: Maximum number of files to process (for testing)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    audio_dir = output_dir / "audio"
    rttm_dir = output_dir / "rttm"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing AISHELL-4 from {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find all audio files
    audio_files = find_audio_files(data_dir)
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logger.error("No audio files found!")
        logger.info("Directory contents:")
        for item in sorted(data_dir.rglob('*'))[:30]:
            logger.info(f"  {item.relative_to(data_dir)}")
        return
    
    # Limit files if specified
    if max_files:
        audio_files = audio_files[:max_files]
        logger.info(f"Processing first {max_files} files")
    
    train_files = []
    test_files = []
    processed = 0
    skipped = 0
    
    from tqdm import tqdm
    
    for audio_path in tqdm(audio_files, desc="Processing files"):
        # Determine if train or test based on path
        rel_path = audio_path.relative_to(data_dir)
        path_str = str(rel_path).lower()
        
        is_test = 'test' in path_str
        
        # Get file ID
        file_id = audio_path.stem
        
        # Find matching RTTM
        rttm_path = find_matching_rttm(audio_path, data_dir)
        
        if rttm_path is None:
            logger.debug(f"No RTTM found for {file_id}, skipping")
            skipped += 1
            continue
        
        # Copy/convert audio
        dst_audio = audio_dir / f"{file_id}.wav"
        success = copy_or_convert_audio(audio_path, dst_audio)
        
        if not success:
            logger.warning(f"Failed to process audio: {audio_path}")
            skipped += 1
            continue
        
        # Copy RTTM
        dst_rttm = rttm_dir / f"{file_id}.rttm"
        if not dst_rttm.exists():
            shutil.copy(rttm_path, dst_rttm)
        
        # Add to file list
        if is_test:
            test_files.append(file_id)
        else:
            train_files.append(file_id)
        
        processed += 1
    
    logger.info(f"Processed: {processed}, Skipped: {skipped}")
    
    # Remove duplicates and sort
    train_files = sorted(list(set(train_files)))
    test_files = sorted(list(set(test_files)))
    
    # Split train into train/val (90/10)
    import random
    random.seed(42)
    random.shuffle(train_files)
    
    val_size = max(1, len(train_files) // 10)
    val_files = train_files[:val_size]
    train_files = train_files[val_size:]
    
    # Write file lists
    with open(output_dir / "train.txt", 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(output_dir / "val.txt", 'w') as f:
        f.write('\n'.join(val_files))
    
    with open(output_dir / "test.txt", 'w') as f:
        f.write('\n'.join(test_files))
    
    # Print summary
    logger.info("=" * 60)
    logger.info("AISHELL-4 Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"  Train: {len(train_files)} sessions")
    logger.info(f"  Val: {len(val_files)} sessions")
    logger.info(f"  Test: {len(test_files)} sessions")
    logger.info(f"  Audio directory: {audio_dir}")
    logger.info(f"  RTTM directory: {rttm_dir}")
    
    # Verify output
    audio_count = len(list(audio_dir.glob('*.wav'))) + len(list(audio_dir.glob('*.flac')))
    rttm_count = len(list(rttm_dir.glob('*.rttm')))
    logger.info(f"  Total audio files: {audio_count}")
    logger.info(f"  Total RTTM files: {rttm_count}")
    logger.info("=" * 60)
    
    # Save metadata
    metadata = {
        'train_files': len(train_files),
        'val_files': len(val_files),
        'test_files': len(test_files),
        'total_audio': audio_count,
        'total_rttm': rttm_count,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def explore_directory(directory: Path, max_depth: int = 3) -> Dict:
    """Explore directory structure."""
    structure = {
        'directories': [],
        'audio_files': [],
        'rttm_files': [],
        'textgrid_files': [],
        'other_files': [],
    }
    
    for item in directory.rglob('*'):
        rel_path = item.relative_to(directory)
        depth = len(rel_path.parts)
        
        if depth > max_depth:
            continue
            
        if item.is_dir():
            structure['directories'].append(str(rel_path))
        elif item.suffix.lower() in AUDIO_EXTENSIONS:
            structure['audio_files'].append(str(rel_path))
        elif item.suffix == '.rttm':
            structure['rttm_files'].append(str(rel_path))
        elif item.suffix == '.TextGrid':
            structure['textgrid_files'].append(str(rel_path))
        else:
            structure['other_files'].append(str(rel_path))
    
    return structure


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare AISHELL-4 dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/aishell4",
        help="Output directory",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, don't prepare",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true", 
        help="Only prepare (assume already downloaded)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw data directory (for prepare-only)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=None,
        help="Splits to download (train_L, train_M, train_S, test)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Only explore directory structure",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to process (for testing)",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Don't convert FLAC to WAV",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw"
    
    if args.explore:
        logger.info(f"Exploring {raw_dir}")
        structure = explore_directory(raw_dir)
        logger.info(f"Directories: {len(structure['directories'])}")
        for d in structure['directories'][:20]:
            logger.info(f"  {d}")
        logger.info(f"\nAudio files ({', '.join(AUDIO_EXTENSIONS)}): {len(structure['audio_files'])}")
        for f in structure['audio_files'][:10]:
            logger.info(f"  {f}")
        logger.info(f"\nRTTM files: {len(structure['rttm_files'])}")
        for f in structure['rttm_files'][:10]:
            logger.info(f"  {f}")
        logger.info(f"\nTextGrid files: {len(structure['textgrid_files'])}")
        for f in structure['textgrid_files'][:10]:
            logger.info(f"  {f}")
        return
    
    if not args.prepare_only:
        download_aishell4(
            output_dir=str(raw_dir),
            splits=args.splits,
            keep_archives=args.keep_archives,
        )
    
    if not args.download_only:
        prepare_aishell4(
            data_dir=str(raw_dir),
            output_dir=str(output_dir),
            convert_to_wav=not args.no_convert,
            max_files=args.max_files,
        )


if __name__ == "__main__":
    main()