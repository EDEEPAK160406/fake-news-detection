Video Authenticity Training README

Overview

This folder provides scripts to prepare, train, and evaluate a multimodal video authenticity detector.

Prerequisites

- Python 3.9+
- A virtualenv activated for the project (recommended):

  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install -U pip
  ```

- Install core dependencies (if not already installed):

  ```powershell
  pip install -r requirements.txt
  pip install -r requirements-dl.txt
  ```

- Install ffmpeg on your system (used to extract audio):
  - Windows: use winget or download from ffmpeg.org

Prepare dataset

- Place labeled videos into:
  - `data/video_authenticity/real/` and
  - `data/video_authenticity/ai_generated/`

- Run preprocessing (face crops + optional MFCC extraction):

  ```powershell
  python scripts/prepare_video_dataset.py --input data/video_authenticity --out data/video_authenticity/processed --sample-rate 2 --audio
  ```

Train

- Basic training scaffold (adjust batch-size/epochs for your machine):

  ```powershell
  python scripts/train_video_multimodal.py --data-dir data/video_authenticity/processed --epochs 8 --batch-size 2 --audio
  ```

Evaluate

- Evaluator now reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a tuned threshold:

  ```powershell
  python scripts/evaluate_video_model.py --data-dir data/video_authenticity/processed_audio --model artifacts/video_multimodal_audio.pt --audio --output artifacts/video_eval_metrics.json
  ```

Notes

- Training on CPU is slow. Use a GPU if available for practical runs.
- The audio branch is optional. If you don't want audio, omit `--audio`.
- The training scaffold is a starting point—tune augmentations, model size, and training schedules for best performance.
- The live API now prefers the trained checkpoint at `artifacts/video_multimodal.pt` and falls back to the conservative heuristic only when no checkpoint is available.
