# Speaker Diarization

Speaker diarization is the process of partitioning an audio stream into segments according to the identity of the speaker. This repository provides tools and code for automatic speaker diarization — figuring out "who spoke when" in an audio recording.

## Features

- **Automatic Speaker Segmentation:** Splits audio into segments each corresponding to a single speaker.
- **Speaker Identification:** Assigns unique labels (Speaker 1, Speaker 2, etc.) to each speaker detected, optionally linking to known identities if available.
- **Visualization and Reporting:** Generates easy-to-read visualizations of speaker timelines.

---

## Repository Structure

```
speaker_diarization/
│
├── config/                         # Configuration files
│   └── config.py
│
├── data/                           # Dataset & preprocessing
│   ├── aishell4/                   # AISHELL-4 dataset
│   │   ├── metadata.json
│   │   ├── segments/
│   │   │   └── segments.json
│   │   ├── train.txt
│   │   ├── train_small.txt
│   │   ├── train_full.txt
│   │   ├── val.txt
│   │   ├── val_small.txt
│   │   ├── val_full.txt
│   │   └── test.txt
│   ├── augmentation.py             # Audio augmentation
│   ├── preprocessing.py            # Feature extraction & segmentation
│   ├── dataset.py                  # Dataset loader
│   └── __init__.py
│
├── models/                         # Diarization models
│   ├── diarization_model.py        # End-to-end diarization model
│   ├── speaker_encoder.py          # Speaker embedding extractor
│   ├── segmentation.py             # Speech segmentation model
│   ├── clustering.py               # Speaker clustering logic
│   ├── losses.py                   # Loss functions
│   └── __init__.py
│
├── training/                       # Training pipeline
│   ├── trainer.py
│   └── __init__.py
│
├── evaluation/                     # Evaluation logic
│   ├── evaluator.py
│   ├── metrics.py                  # DER, JER, etc.
│   └── __init__.py
│
├── outputs/                        # Training outputs
│   ├── diarization_*/              # Experiment runs
│   │   ├── best_model.pt
│   │   ├── checkpoint_epoch_*.pt
│   │   ├── config.json
│   │   └── train.log
│   └── simple_model/
│       └── model.pt
│
├── demo/                           # Interactive demo
│   └── demo_gradio.py
│
├── utils/                          # Utility functions
│   ├── audio_utils.py
│   ├── rttm_utils.py
│   ├── helpers.py
│   └── __init__.py
│
├── scripts/                        # Dataset scripts
│   ├── download_aishell4.py
│   └── prepare_data.py
│
├── inference.py                    # Diarization inference
├── evaluate.py                     # Evaluation entry script
├── train.py                        # Training entry script
├── fix_imports.py                  # Import fixes
└── __init__.py
```

### ✅ Why this structure is ideal

- ✔ Matches your actual filesystem  
- ✔ Fully modular (data → model → training → evaluation)  
- ✔ Research-grade diarization pipeline  
- ✔ Compatible with AISHELL-4  
- ✔ Ready for papers, thesis, and production demos  

**🔍 Conceptual flow (for understanding)**  
Audio → Segmentation → Speaker Encoder → Clustering → RTTM Output  

---

## Installation

Clone the repository:
```bash
git clone https://github.com/<owner>/speaker_diarization.git
cd speaker_diarization
```

Install dependencies (using pip or conda):
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your audio file
Place your audio file (e.g., `meeting.wav`) in the `data/` directory.

### 2. Run diarization

```bash
python diarize.py --audio data/meeting.wav --output results.json
```

### 3. View results

- Speaker segments and identities are saved to the specified output.
- Example output:
    ```json
    [
      {"speaker": "Speaker 1", "start": 0.0, "end": 15.2},
      {"speaker": "Speaker 2", "start": 15.2, "end": 30.8}
    ]
    ```

### 4. Visualization

Optionally, visualize diarization:
```bash
python visualize.py --input results.json --show
```

## Example

```python
from diarization import diarize_audio

segments = diarize_audio("data/meeting.wav")
for segment in segments:
    print(f"{segment['speaker']} spoke from {segment['start']}s to {segment['end']}s")
```

## Requirements

- Python 3.7+
- [List of specific dependencies, e.g., PyTorch, librosa, numpy]

## Model

This repository uses [mention model or algorithm, such as pyAudioAnalysis, pyannote.audio, etc.], see `model/` directory for details. You can change model settings in `config.yaml`.

## Contributing

Pull requests, issues, and feature suggestions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
