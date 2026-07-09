Image Authenticity - Training README

This project includes two training flows for the image-authenticity detector:

1) Feature-based classifier (no DL runtime required)
   - Implemented in `app/services/image_authenticity.py`
   - Train with: `python scripts/train_image_authenticity.py --augment --min-per-class 200`

2) Transfer-learning CNN (higher accuracy, requires PyTorch)
   - Script: `scripts/train_image_cnn.py`
   - Install DL deps (example):

```powershell
# Windows example — prefer official PyTorch instructions from https://pytorch.org/get-started/locally/
python -m pip install -U pip
pip install -r requirements-dl.txt
```

- Train:

```powershell
python scripts/train_image_cnn.py --data-dir data/image_authenticity --epochs 10 --batch-size 16 --output artifacts/image_authenticity_cnn.pt
```

Notes:
- The CNN expects `data/image_authenticity/ai_generated` and `data/image_authenticity/real` folders.
- After training, the app will auto-load `artifacts/image_authenticity_cnn.pt` if present and PyTorch is installed; otherwise the service falls back to the feature-based classifier.
- Use `scripts/train_image_authenticity.py --augment` to expand small classes without a DL runtime.
