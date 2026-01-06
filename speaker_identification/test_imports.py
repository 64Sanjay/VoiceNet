# test_imports.py
"""Test all imports"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    from config.config import WSIConfig
    print("✅ config.config")
except Exception as e:
    print(f"❌ config.config: {e}")

try:
    from data.preprocessing import AudioPreprocessor
    print("✅ data.preprocessing")
except Exception as e:
    print(f"❌ data.preprocessing: {e}")

try:
    from data.augmentation import AudioAugmentor
    print("✅ data.augmentation")
except Exception as e:
    print(f"❌ data.augmentation: {e}")

try:
    from data.dataset import SpeakerDataset
    print("✅ data.dataset")
except Exception as e:
    print(f"❌ data.dataset: {e}")

try:
    from models.wsi_model import WSIModel
    print("✅ models.wsi_model")
except Exception as e:
    print(f"❌ models.wsi_model: {e}")

try:
    from losses.joint_loss import WSIJointLoss
    print("✅ losses.joint_loss")
except Exception as e:
    print(f"❌ losses.joint_loss: {e}")

try:
    from evaluation.metrics import compute_metrics
    print("✅ evaluation.metrics")
except Exception as e:
    print(f"❌ evaluation.metrics: {e}")

try:
    from utils.helpers import set_seed
    print("✅ utils.helpers")
except Exception as e:
    print(f"❌ utils.helpers: {e}")

print("\nAll imports tested!")