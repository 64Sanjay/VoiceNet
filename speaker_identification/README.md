# Speaker Identification

A modular and extensible PyTorch-based framework for **closed-set speaker identification**. This repository supports state-of-the-art neural architectures (CNN, TDNN/x-vector, ResNet), robust training pipelines, configurable data handling, and evaluation metrics. Designed for research, benchmarking, and practical deployment.

---

## 🌟 Features

- **Neural architectures:** CNN (incl. 1D), TDNN/x-vector, ResNet
- **Flexible pipelines:** Complete separation of data, models, training, and evaluation
- **Data augmentation:** Noise, channel robustification
- **Configurable:** Plug-and-play experiment configs
- **Easy extensibility:** Add new datasets, models, or metrics with minimal refactoring
- **Research & reproducibility:** GitHub and research-paper friendly structure

---

## 🔍 Task Pipeline (Conceptual Overview)

```
Audio → Feature Extraction → Speaker Encoder → Classifier → Speaker ID
```
- **Audio**: raw speech data
- **Feature Extraction**: MFCC, FBANK, etc.
- **Speaker Encoder**: Neural network (CNN/TDNN/ResNet)
- **Classifier**: Softmax, AM-Softmax, AAM-Softmax head
- **Speaker ID**: Closed-set speaker classification

---

## 📁 Repository Structure

```
speaker_identification/
│
├── config/                         # Experiment & model configuration
│   └── config.py
│
├── data/                           # Dataset & preprocessing
│   ├── aishell/                    # AISHELL speaker ID dataset
│   │   ├── metadata.json
│   │   ├── train.txt
│   │   ├── train_small.txt
│   │   ├── train_full.txt
│   │   ├── val.txt
│   │   ├── val_small.txt
│   │   ├── val_full.txt
│   │   └── test.txt
│   ├── augmentation.py             # Noise & channel augmentation
│   ├── preprocessing.py            # Feature extraction (MFCC, FBANK)
│   ├── dataset.py                  # Speaker ID dataset loader
│   └── __init__.py
│
├── models/                         # Speaker identification models
│   ├── cnn.py                      # CNN / CNN1D models
│   ├── tdnn.py                     # TDNN / x-vector models
│   ├── resnet.py                   # ResNet-based speaker models
│   ├── losses.py                   # Softmax, AM-Softmax, AAM-Softmax
│   ├── classifier.py               # Speaker classifier head
│   └── __init__.py
│
├── training/                       # Training pipeline
│   ├── trainer.py
│   └── __init__.py
│
├── evaluation/                     # Evaluation & metrics
│   ├── evaluator.py
│   ├── metrics.py                  # Accuracy, Top-K
│   └── __init__.py
│
├── outputs/                        # Training outputs
│   ├── speaker_id_*/               # Experiment runs
│   │   ├── best_model.pt
│   │   ├── checkpoint_epoch_*.pt
│   │   ├── config.json
│   │   └── train.log
│   └── simple_model/
│       └── model.pt
│
├── utils/                          # Utility functions
│   ├── audio_utils.py
│   ├── feature_utils.py
│   ├── helpers.py
│   └── __init__.py
│
├── scripts/                        # Dataset utilities
│   ├── download_dataset.py
│   └── prepare_data.py
│
├── inference.py                    # Speaker ID inference
├── evaluate.py                     # Evaluation entry point
├── train.py                        # Training entry point
└── __init__.py
```

### ✅ Why this structure is correct

- ✔ Matches speaker identification task (closed-set classification)
- ✔ Supports CNN, TDNN, ResNet, x-vector
- ✔ Clean separation of data → model → training → evaluation
- ✔ Research-paper & GitHub-friendly
- ✔ Easily extensible to multilingual speakers

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/speaker_identification.git
cd speaker_identification
```

### 2. Install Dependencies

*Python 3.8+ recommended*

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

- Download supported datasets (e.g. AISHELL):

```bash
python scripts/download_dataset.py --dataset aishell
```

- Prepare train/val/test splits:

```bash
python scripts/prepare_data.py --dataset aishell
```

### 4. Train a Speaker ID Model

```bash
python train.py --config config/config.py
```

### 5. Evaluate

```bash
python evaluate.py --model outputs/speaker_id_*/best_model.pt --data data/aishell/test.txt
```

### 6. Inference

```bash
python inference.py --model outputs/speaker_id_*/best_model.pt --audio example.wav
```

---

## 📑 Configuration

Experiment and model hyperparameters are set in `config/config.py`. Supports easy tweaking of:
- Model type (`cnn`, `tdnn`, `resnet`)
- Feature type (MFCC, FBANK)
- Loss function
- Optimizer, learning rate, batch size

---

## 📈 Evaluation Metrics

- **Accuracy** (Top-1, Top-K)
- **Confusion Matrix**
- **Per-speaker breakdown**

Custom metrics can be added in `evaluation/metrics.py`.

---

## 🛠️ Utilities

- **Feature extraction**: `data/preprocessing.py`
- **Augmentation**: `data/augmentation.py`
- **Audio helpers**: `utils/audio_utils.py`

---

## 🤝 How to Contribute

1. Fork this repo, create your branch.
2. Add new models, datasets, or metrics in their respective folders.
3. Submit a pull request with description.

---

## 📖 References

- [x-vector: Robust Speaker Embedding Extraction](https://arxiv.org/abs/1710.10468)
- [ResNet for Speaker Recognition](https://arxiv.org/abs/1908.10234)
- [AISHELL-1 Dataset](https://www.openslr.org/33)
