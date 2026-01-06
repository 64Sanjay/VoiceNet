# VoiceNet

VoiceNet is an advanced, open-source speaker recognition framework designed to handle speaker verification, identification, and diarization tasks. Leveraging state-of-the-art deep learning architectures, VoiceNet offers robust tools for extracting speaker embeddings, performing classification, and analyzing speech data.

---

## Project Purpose

The purpose of VoiceNet is to provide an accessible and modular platform for speaker recognition research and development, enabling both academic and industry professionals to:

- Accurately verify and identify speakers in real time or batch audio.
- Automate speaker diarization for meeting, call, broadcast, and surveillance audio.
- Experiment with and improve deep learning models for audio and speech analytics.
- Enable reproducible research with clear benchmarks and pretrained models.
- Support learning and teaching in audio AI through well-documented code and demos.

---

## Requirements

To use VoiceNet, you will need:

- **Python**: Version 3.7 or above
- **Machine Learning Framework**: [PyTorch](https://pytorch.org/) (recommended) or [TensorFlow](https://www.tensorflow.org/) (optional backend support)
- **Audio and Data Libraries**:
  - `numpy`
  - `scipy`
  - `librosa`
  - `scikit-learn` (`sklearn`)
- **Others**:
  - `tqdm` for progress bars (optional)
  - `pandas` for data handling (optional)
- **Gradio** (for demo UI, optional): `gradio`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## FINAL REPOSITORY STRUCTURE

```
VoxLab/
│
├── speaker_verification/                 # Speaker Verification (1:1)
│   ├── checkpoints/                      # Saved models
│   ├── config/                           # Configurations
│   ├── data/                             # Dataset, preprocessing, augmentation
│   ├── models/                           # CAM++, DTDNN, pooling, losses
│   ├── training/                         # Training logic
│   ├── evaluation/                       # Metrics & evaluators
│   ├── demo/                             # Verification demos
│   ├── inference.py                      # Verification inference
│   ├── evaluate.py                       # Evaluation entry
│   └── train.py                          # Training entry
│
├── speaker_identification/               # Speaker Identification (1:N)
│   ├── config/
│   ├── data/
│   ├── models/                           # Whisper encoder, projection head
│   ├── losses/                           # Triplet, NT-Xent, Joint loss
│   ├── training/                         # Trainer & mining
│   ├── evaluation/                       # Metrics
│   ├── demo/                             # CLI / API / Gradio demos
│   ├── outputs/                          # Model checkpoints
│   ├── inference.py                      # Identification inference
│   ├── evaluate.py                       # Evaluation entry
│   └── train.py                          # Training entry
│
├── speaker_diarization/                  # Speaker Diarization (Who spoke when)
│   ├── config/
│   ├── data/
│   │   └── aishell4/                     # AISHELL-4 dataset splits
│   ├── models/
│   │   ├── speaker_encoder.py
│   │   ├── segmentation.py
│   │   └── clustering.py
│   ├── training/                         # Diarization trainer
│   ├── evaluation/                       # DER metrics
│   ├── outputs/                          # Logs & checkpoints
│   ├── demo/                             # Gradio demo
│   ├── inference.py                      # Diarization inference
│   └── train.py                          # Training entry
│
├── demo/                                 # Unified multi-task demo
│   ├── demo_gradio.py
│   ├── verification_tab.py
│   ├── identification_tab.py
│   ├── diarization_tab.py
│   ├── utils.py
│   └── __init__.py
│
├── run_demo.py                           # Launch full demo
├── requirements.txt                      # Dependencies
├── .gitignore                            # Ignore rules
└── README.md                             # Project description
```

---

## Features

- **Speaker Verification**: Accurately verify if two voice samples belong to the same speaker.
- **Speaker Identification**: Recognize and assign speech to known speakers from a set.
- **Speaker Diarization**: Automatically segment audio into unique speaker turns.
- **Deep Learning Integration**: Utilizes modern architectures for superior performance.
- **Easy-to-Use API**: Simple, consistent Python interface for all core functionalities.
- **Custom Training**: Supports training on user-provided or popular public datasets.

---

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/sanjay423bel/VoiceNet.git
cd VoiceNet
```

(Optional) Install as a package:

```bash
pip install .
```

---

## Usage

### Speaker Verification Example

```python
from voicenet import SpeakerVerifier

verifier = SpeakerVerifier(model_path="pretrained_model.pth")
result = verifier.verify("audio1.wav", "audio2.wav")
print(f"Match: {result}")
```

### Speaker Identification Example

```python
from voicenet import SpeakerIdentifier

identifier = SpeakerIdentifier(model_path="pretrained_model.pth")
speaker = identifier.identify("test_audio.wav")
print(f"Speaker ID: {speaker}")
```

### Speaker Diarization Example

```python
from voicenet import SpeakerDiarizer

diarizer = SpeakerDiarizer(model_path="pretrained_model.pth")
diarization = diarizer.diarize("multi_speaker_audio.wav")
print(f"Diarization: {diarization}")
```

See [`examples/`](./examples/) for detailed scripts and advanced customizations.

---

## Models

- [x] Pretrained models provided or train your own using open datasets (e.g., VoxCeleb, LibriSpeech).
- [x] Support for model export, inference, and fine-tuning.

---

## Dataset Preparation

VoiceNet is compatible with common datasets such as VoxCeleb, LibriSpeech, and more. See [`docs/datasets.md`](./docs/datasets.md) for guides on preparing your data.

---

## Documentation

- Complete API reference: [`docs/`](./docs/)
- Tutorials and example notebooks: [`examples/`](./examples/)
- Model specification and training details: [`docs/model.md`](./docs/model.md)

---

## Contributing

We welcome contributions! Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) to get started.

---

*VoiceNet: Powerful, scalable, and open-source voice recognition for modern applications.*
