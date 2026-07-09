# OCR System - Installation and Usage Guide

## Quick Installation

### Windows
```bash
setup.bat
venv\Scripts\activate
python examples.py
```

### Linux/Mac
```bash
bash setup.sh
source venv/bin/activate
python examples.py
```

## Project Structure

```
imgtotext/
├── config/
│   ├── config.py          # Main configuration
│   ├── config.yaml        # YAML configuration (optional)
│   └── __init__.py        # Package init
├── src/
│   ├── __init__.py        # Package init
│   ├── preprocessing.py   # Image preprocessing
│   ├── model.py           # Model architectures
│   ├── decoder.py         # Decoding strategies
│   ├── ocr_extractor.py   # Main OCR module
│   ├── utils.py           # Utilities
│   ├── train.py           # Training script
│   └── inference.py       # Inference script
├── tests/
│   └── test_ocr.py        # Unit tests
├── examples.py            # Example demonstrations
├── quickstart.py          # Interactive quickstart
├── requirements.txt       # Dependencies
├── setup.sh              # Linux/Mac setup
├── setup.bat             # Windows setup
├── Dockerfile            # Docker image
└── README.md             # Full documentation
```

## System Requirements

- Python 3.8+
- RAM: 4GB minimum (8GB recommended)
- GPU: NVIDIA CUDA 11.0+ (optional, for faster training)

## Features

### Text Recognition
- CNN-LSTM-CTC architecture
- Bidirectional LSTM for context modeling
- Beam search decoding for improved accuracy
- Support for uppercase, lowercase, digits, and punctuation

### Image Preprocessing
- Grayscale conversion
- Adaptive binarization
- Noise removal (bilateral filtering)
- Aspect-ratio-aware resizing
- Multiple normalization methods

### Data Augmentation
- Rotation (±10°)
- Elastic deformation
- Random noise
- Gaussian blur
- Shifting/translation

### Fake News Integration
- Text feature extraction
- Readability metrics
- Suspicious pattern detection
- Risk level classification

## Usage Examples

### 1. Extract Text from Image
```python
from src.ocr_extractor import create_ocr_extractor

extractor = create_ocr_extractor('models/best_ocr_model.h5')
result = extractor.extract_text('image.jpg')
print(result['extracted_text'])
```

### 2. Batch Processing
```python
extractor = create_ocr_extractor('models/best_ocr_model.h5')
results = extractor.extract_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### 3. Fake News Detection
```python
from src.ocr_extractor import FakeNewsTextExtractor

extractor = FakeNewsTextExtractor('models/best_ocr_model.h5')
result = extractor.extract_for_analysis('news_screenshot.jpg')
print(f"Risk Level: {result['text_flags']['risk_level']}")
```

### 4. Train Custom Model
```python
from src.train import OCRTrainer

trainer = OCRTrainer()
trainer.build_model()
trainer.train(images, labels, augment=True)
trainer.save_model('custom_model.h5')
```

## Command Line Interface

### Extract from single image:
```bash
python -m src.inference --image path/to/image.jpg --output result.json
```

### Batch processing:
```bash
python -m src.inference --batch path/to/images/ --output results.json --format csv
```

### Fake news detection:
```bash
python -m src.inference --batch path/to/images/ --fake-news --output analysis.json
```

## Configuration

Edit `config/config.py` to customize:
- Image dimensions (256×32)
- Model architecture parameters
- Training parameters (batch size, epochs, learning rate)
- Augmentation settings
- Decoding parameters

## Performance

- **Character Error Rate**: ~5-10%
- **Word Error Rate**: ~15-20%
- **Processing Speed**: 100-200 ms/image (GPU)
- **Memory**: ~2.5GB (GPU) / 500MB (CPU)

## Troubleshooting

### Model loading fails
```bash
pip install --force-reinstall tensorflow
```

### Low confidence scores
- Check image quality
- Adjust preprocessing parameters
- Retrain with more similar data

### Memory errors
- Reduce batch size
- Use GPU with CUDA
- Process smaller batches

## Testing

Run unit tests:
```bash
python -m tests.test_ocr
```

## Docker Deployment

Build and run with Docker:
```bash
docker build -t ocr-system .
docker run -v $(pwd)/data:/app/data ocr-system python examples.py
```

## Development

Structure for adding new features:
1. Add preprocessing in `src/preprocessing.py`
2. Add model components in `src/model.py`
3. Add decoding logic in `src/decoder.py`
4. Update tests in `tests/test_ocr.py`
5. Add examples in `examples.py`

## Literature & References

- Shi et al. (2016): "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"
- Graves et al. (2006): "Connectionist Temporal Classification"
- Vaswani et al. (2017): "Attention Is All You Need"

## Support

- Check README.md for detailed documentation
- Review test files for usage examples
- Run examples.py for demonstrations
- Read config/config.py for parameter descriptions

## Version

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Status**: Production Ready

---

For additional help, refer to the full README.md documentation.
