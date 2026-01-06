# download_librispeech.py
"""
Download LibriSpeech dataset for speaker identification.
"""

import os
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(tar_path, extract_path):
    """Extract tar.gz file."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete!")

def download_librispeech():
    """Download LibriSpeech dev-clean subset."""
    
    # URLs for different subsets
    datasets = {
        'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
        # 'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',  # Larger
    }
    
    base_dir = Path('data/librispeech')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for name, url in datasets.items():
        tar_path = base_dir / f'{name}.tar.gz'
        
        if not tar_path.exists():
            print(f"\nDownloading {name}...")
            download_file(url, tar_path)
        else:
            print(f"\n{name}.tar.gz already exists, skipping download.")
        
        # Extract
        extract_path = base_dir / 'raw'
        extract_path.mkdir(exist_ok=True)
        
        if not (extract_path / 'LibriSpeech' / name).exists():
            extract_tar(tar_path, extract_path)
        else:
            print(f"{name} already extracted.")
    
    print("\nLibriSpeech download complete!")
    print(f"Data location: {base_dir}")

if __name__ == "__main__":
    download_librispeech()