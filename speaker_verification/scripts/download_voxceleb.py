# speaker_verification/scripts/download_voxceleb.py
"""
Download VoxCeleb dataset for speaker verification.

VoxCeleb1 and VoxCeleb2 are the standard benchmarks used in the paper.
"""

import os
import sys
from pathlib import Path
import subprocess
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=Path(output_path).name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    print("=" * 60)
    print("VoxCeleb Dataset Download Guide")
    print("=" * 60)
    
    print("""
VoxCeleb requires registration to download. Please follow these steps:

1. REGISTER at VoxCeleb website:
   https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

2. After registration, download:
   - VoxCeleb1 (for testing): ~30GB
   - VoxCeleb2 (for training): ~300GB

3. Alternatively, use these tools:
   
   a) voxceleb_trainer (recommended):
      git clone https://github.com/clovaai/voxceleb_trainer.git
      cd voxceleb_trainer
      python dataprep.py --save_path data/voxceleb1 --download --user USERNAME --password PASSWORD
   
   b) Direct links (after authentication):
      - VoxCeleb1 dev: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
      - VoxCeleb2 dev: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa

4. For quick testing, use LibriSpeech instead (no registration needed):
   python scripts/prepare_librispeech.py

""")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    (data_dir / "voxceleb1").mkdir(exist_ok=True)
    (data_dir / "voxceleb2").mkdir(exist_ok=True)
    (data_dir / "musan").mkdir(exist_ok=True)
    (data_dir / "rir_noises").mkdir(exist_ok=True)
    
    print("\nDirectory structure created:")
    print("  data/")
    print("  ├── voxceleb1/   (test set)")
    print("  ├── voxceleb2/   (training set)")
    print("  ├── musan/       (noise augmentation)")
    print("  └── rir_noises/  (reverb augmentation)")
    
    # Download MUSAN (freely available)
    print("\n" + "=" * 60)
    print("Downloading MUSAN dataset (for noise augmentation)...")
    print("=" * 60)
    
    musan_url = "https://www.openslr.org/resources/17/musan.tar.gz"
    musan_path = data_dir / "musan.tar.gz"
    
    if not musan_path.exists() and not (data_dir / "musan" / "noise").exists():
        try:
            print(f"Downloading from {musan_url}...")
            download_file(musan_url, str(musan_path))
            print("Extracting MUSAN...")
            subprocess.run(["tar", "-xzf", str(musan_path), "-C", str(data_dir)], check=True)
            print("✅ MUSAN downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download MUSAN: {e}")
            print("   You can download it manually from: https://www.openslr.org/17/")
    else:
        print("MUSAN already exists or downloaded.")
    
    # Download RIR (freely available)
    print("\n" + "=" * 60)
    print("Downloading RIR dataset (for reverb augmentation)...")
    print("=" * 60)
    
    rir_url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    rir_path = data_dir / "rirs_noises.zip"
    
    if not rir_path.exists() and not (data_dir / "rir_noises" / "simulated_rirs").exists():
        try:
            print(f"Downloading from {rir_url}...")
            download_file(rir_url, str(rir_path))
            print("Extracting RIR...")
            subprocess.run(["unzip", "-o", str(rir_path), "-d", str(data_dir / "rir_noises")], check=True)
            print("✅ RIR downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download RIR: {e}")
            print("   You can download it manually from: https://www.openslr.org/28/")
    else:
        print("RIR already exists or downloaded.")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. For VoxCeleb, register and download from:
   https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

2. Or use LibriSpeech for testing:
   python scripts/prepare_librispeech.py

3. After downloading, prepare the data:
   python scripts/prepare_data.py --dataset voxceleb

4. Start training:
   python train.py
""")


if __name__ == "__main__":
    main()