#!/usr/bin/env python3
"""Fix missing typing imports in all Python files."""

import os
from pathlib import Path

# Files that need typing imports
files_to_check = [
    "data/preprocessing.py",
    "data/dataset.py", 
    "data/augmentation.py",
    "models/speaker_encoder.py",
    "models/segmentation.py",
    "models/clustering.py",
    "models/losses.py",
    "models/diarization_model.py",
    "evaluation/metrics.py",
    "evaluation/evaluator.py",
    "training/trainer.py",
    "utils/audio_utils.py",
    "utils/rttm_utils.py",
    "utils/helpers.py",
]

typing_import = "from typing import List, Optional, Tuple, Dict, Union, Any\n"

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        continue
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if typing import already exists
    if "from typing import" in content:
        print(f"Skipping {filepath} (already has typing import)")
        continue
    
    # Find where to insert (after docstring and before other imports)
    lines = content.split('\n')
    insert_idx = 0
    
    # Skip shebang and docstring
    in_docstring = False
    for i, line in enumerate(lines):
        if line.startswith('#!'):
            insert_idx = i + 1
            continue
        if line.startswith('"""') or line.startswith("'''"):
            if in_docstring:
                insert_idx = i + 1
                in_docstring = False
                break
            else:
                in_docstring = True
                continue
        if not in_docstring and line.strip() and not line.startswith('#'):
            insert_idx = i
            break
    
    # Insert typing import
    lines.insert(insert_idx, typing_import.strip())
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Fixed {filepath}")

print("\nDone!")
