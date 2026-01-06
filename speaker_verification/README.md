# Speaker Verification

A modular, research-oriented toolkit for modern **speaker verification** tasks. This project provides state-of-the-art speaker embedding models (x-vector, ECAPA-TDNN, ResNet, SincNet), robust pair-based data handling, configurable training pipelines, flexible scoring backends (Cosine, PLDA), and standard evaluation metrics (EER, minDCF).

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Verification Pipeline](#verification-pipeline)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
    
---

## Features

- **Pair-based data handling** for verification-specific experiments (positive/negative pairs)
- **State-of-the-art embedding architectures**: x-vector, ECAPA-TDNN, ResNet, SincNet
- **Modular backends**: Cosine similarity and PLDA scoring supported
- **Standard metrics**: EER, minDCF, ROC
- **Flexible & extensible config** system for rapid research and production deployment
- **Seamless pipeline** for training, evaluation, and inference

---

## Repository Structure

```text
speaker_verification/
│
├── config/                # Experiment & model configuration
│   └── config.py
│
├── data/                  # Dataset & preprocessing
│   ├── aishell/           # AISHELL verification splits
│   │   ├── metadata.json
│   │   ├── train_pairs.txt
│   │   ├── train_small.txt
│   │   ├── train_full.txt
│   │   ├── val_pairs.txt
│   │   ├── test_pairs.txt
│   │   └── test_trials.txt
│   ├── augmentation.py    # Noise, reverb, channel effects
│   ├── preprocessing.py   # FBANK, MFCC, etc.
│   ├── dataset.py         # Pair-based loader
│   └── __init__.py
│
├── models/                # Speaker embedding models
│   ├── tdnn.py            # x-vector / ECAPA-TDNN
│   ├── resnet.py          # ResNet encoder
│   ├── sincnet.py         # SincNet raw waveform
│   ├── pooling.py         # Stats / attentive pooling
│   ├── losses.py          # Contrastive/Triplet/AAM-Softmax
│   └── __init__.py
│
├── training/              # Training logic
│   ├── trainer.py
│   └── __init__.py
│
├── scoring/               # Scoring backends
│   ├── cosine.py
│   ├── plda.py
│   ├── score_normalization.py
│   └── __init__.py
│
├── evaluation/            # Evaluation metrics & logic
│   ├── evaluator.py
│   ├── metrics.py
│   └── __init__.py
│
├── outputs/               # Model checkpoints, logs
│   ├── verification_*/    # Experiment runs
│   └── simple_model/
│
├── utils/                 # Utility functions
│   ├── audio_utils.py
│   ├── embedding_utils.py
│   ├── helpers.py
│   └── __init__.py
│
├── scripts/               # Dataset utilities
│   ├── download_dataset.py
│   └── prepare_trials.py
│
├── inference.py           # Run inference on audio pairs
├── evaluate.py            # Evaluate system on trials
├── train.py               # Train speaker verification system
└── __init__.py
```

---

## Why this structure is correct

- ✔ **Pair-based data handling**: Designed for true speaker verification tasks (same/different judgment on audio pairs).
- ✔ **Modular embedding backbone architectures**: Use, extend, or replace models easily.
- ✔ **Clear separation of scoring logic**: Try multiple scoring methods (Cosine, PLDA) with no codebase tangle.
- ✔ **Standardized evaluation**: Built-in, reproducible verification metrics and reports.
- ✔ **Production & research ready**: Easily scriptable, config-driven, experiment-friendly.

---

## Verification Pipeline (Conceptual)

```
Audio₁, Audio₂
   ↓
Speaker Encoder (x-vector/ECAPA/ResNet/SincNet)
   ↓
Embedding₁, Embedding₂
   ↓
Scoring (Cosine / PLDA)
   ↓
Accept / Reject
```

---

## Getting Started

### Requirements

- Python 3.8+ and PyTorch >= 1.8
- numpy, scipy, scikit-learn
- librosa, soundfile, tqdm
- (see `requirements.txt` for complete list)

### Installation

```bash
git clone https://github.com/<your-username>/speaker_verification.git
cd speaker_verification
pip install -r requirements.txt
```

### Dataset Preparation

1. **Download datasets** (example: AISHELL):
   ```bash
   python scripts/download_dataset.py --dataset aishell
   ```
2. **Prepare verification pairs/trials:**
   ```bash
   python scripts/prepare_trials.py
   ```

### Training

```bash
python train.py --config config/config.py
```

### Evaluation

```bash
python evaluate.py --config config/config.py
```

### Inference

```bash
python inference.py --audio1 path/to/audio1.wav --audio2 path/to/audio2.wav --checkpoint path/to/model.pt
```
