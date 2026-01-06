# speaker_verification/scripts/download_cnceleb.py
"""
Download and prepare CN-Celeb dataset for CAM++ training.

CN-Celeb is a challenging Chinese speaker recognition dataset.
From the paper:
"For CN-Celeb, the development sets of CN-Celeb1 and CN-Celeb2 are used for training, 
which contain 2785 speakers."

Website: http://cnceleb.org/
"""

import os
import sys
from pathlib import Path
import subprocess
from tqdm import tqdm


def main():
    print("=" * 60)
    print("CN-Celeb Dataset Download Guide")
    print("=" * 60)
    
    print("""
CN-Celeb is a large-scale Chinese speaker recognition dataset.

📊 Dataset Statistics:
    - CN-Celeb1: 1,000 speakers, ~130,000 utterances
    - CN-Celeb2: 2,000 speakers, ~500,000 utterances
    - Total: ~3,000 speakers for training

📥 Download Instructions:

1. REGISTER at the official website:
   http://cnceleb.org/
   
2. After registration, you'll receive download links for:
   - CN-Celeb1 (training + evaluation)
   - CN-Celeb2 (training only)
   
3. Download the datasets and extract to:
   data/cnceleb/
   ├── CN-Celeb_flac/
   │   ├── id00001/
   │   ├── id00002/
   │   └── ...
   └── CN-Celeb2_flac/
       ├── id00001/
       ├── id00002/
       └── ...

4. Prepare the dataset:
   python scripts/prepare_data.py --dataset cnceleb --data_dir data/cnceleb

📋 Alternative: Use VoxCeleb (larger, English)
   - VoxCeleb1: 1,251 speakers
   - VoxCeleb2: 6,112 speakers
   - Website: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

For quick testing without downloading large datasets:
   python scripts/prepare_data.py --dataset synthetic --num_speakers 100
""")
    
    # Create directories
    data_dir = Path("data/cnceleb")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Directory Structure Created")
    print("=" * 60)
    print(f"""
    {data_dir}/
    ├── CN-Celeb_flac/    (Put CN-Celeb1 here)
    └── CN-Celeb2_flac/   (Put CN-Celeb2 here)
    """)
    
    # Check if data exists
    cnceleb1 = data_dir / "CN-Celeb_flac"
    cnceleb2 = data_dir / "CN-Celeb2_flac"
    
    if cnceleb1.exists():
        speakers1 = len([d for d in cnceleb1.iterdir() if d.is_dir()])
        print(f"✅ CN-Celeb1 found: {speakers1} speakers")
    else:
        print("❌ CN-Celeb1 not found")
    
    if cnceleb2.exists():
        speakers2 = len([d for d in cnceleb2.iterdir() if d.is_dir()])
        print(f"✅ CN-Celeb2 found: {speakers2} speakers")
    else:
        print("❌ CN-Celeb2 not found")


if __name__ == "__main__":
    main()