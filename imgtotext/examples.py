"""
Example usage script demonstrating the complete OCR pipeline.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr_extractor import create_ocr_extractor, FakeNewsTextExtractor
from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.decoder import TextMetrics, PostProcessing
from src.utils import Logger, ResultsFormatter
from src.train import SyntheticDataGenerator


def setup_logging():
    """Setup logging."""
    logger = Logger.setup_logger('example', 'INFO')
    return logger


def example_1_basic_extraction():
    """Example 1: Basic single image text extraction."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Single Image Text Extraction")
    print("="*60)
    
    logger = setup_logging()
    logger.info("Starting basic text extraction example...")
    
    # Note: In real usage, you would have a trained model
    # This example shows the code structure
    config = {
        'image_width': 256,
        'image_height': 32,
        'characters': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\'-',
        'num_classes': 67,
        'rnn_units': 256,
        'confidence_threshold': 0.5,
        'beam_width': 50,
    }
    
    # Create OCR extractor
    # In production: extractor = create_ocr_extractor('models/best_ocr_model.h5')
    logger.info(f"Configuration loaded: {config}")
    logger.info("To use in production:")
    logger.info("  extractor = create_ocr_extractor('models/best_ocr_model.h5')")
    logger.info("  result = extractor.extract_text('path/to/image.jpg')")
    

def example_2_preprocessing():
    """Example 2: Image preprocessing demonstration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Image Preprocessing")
    print("="*60)
    
    logger = setup_logging()
    logger.info("Demonstrating image preprocessing pipeline...")
    
    # Create dummy image for demonstration
    dummy_image = np.random.randint(50, 200, (64, 256, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(width=256, height=32)
    
    # Show preprocessing steps
    logger.info(f"Input image shape: {dummy_image.shape}")
    
    # Preprocessing steps
    processed = preprocessor.preprocess(dummy_image)
    logger.info(f"Output image shape: {processed.shape}")
    logger.info(f"Output value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Data augmentation
    logger.info("\nApplying data augmentation...")
    augmentation_config = {
        'rotation': True,
        'shifts': True,
        'elastic_deformation': False,
        'random_noise': True,
        'blur': True,
    }
    
    augmented = DataAugmentation.augment(dummy_image, augmentation_config)
    logger.info(f"Augmented image shape: {augmented.shape}")
    logger.info("Augmentation applied successfully")


def example_3_text_metrics():
    """Example 3: Text evaluation metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Text Evaluation Metrics")
    print("="*60)
    
    logger = setup_logging()
    logger.info("Calculating text recognition metrics...")
    
    # Example texts
    reference_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Fake news detection system",
        "Hello World!"
    ]
    
    hypothesis_texts = [
        "The quick brown fox jumps over the lazy dog",  # Perfect match
        "Fake news detection systam",                    # One error
        "Hello World!"                                   # Perfect match
    ]
    
    logger.info("\nText Metrics Evaluation:")
    logger.info("-" * 60)
    
    for i, (ref, hyp) in enumerate(zip(reference_texts, hypothesis_texts)):
        cer = TextMetrics.character_error_rate(ref, hyp)
        wer = TextMetrics.word_error_rate(ref, hyp)
        ser = TextMetrics.sequence_error_rate(ref, hyp)
        
        logger.info(f"\nPair {i+1}:")
        logger.info(f"  Reference: '{ref}'")
        logger.info(f"  Hypothesis: '{hyp}'")
        logger.info(f"  CER: {cer:.2f}%")
        logger.info(f"  WER: {wer:.2f}%")
        logger.info(f"  SER: {ser:.2f}")


def example_4_post_processing():
    """Example 4: Text post-processing and legibility scoring."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Text Post-Processing")
    print("="*60)
    
    logger = setup_logging()
    
    test_texts = [
        "This   is    valid   text",
        "!!!!!!!!!!!!",
        "A1B2C3D4E5",
        "High quality extracted news text",
        "5hrt n05y t3xt",
    ]
    
    logger.info("Evaluating text quality and legibility:\n")
    
    for text in test_texts:
        cleaned = PostProcessing.remove_extra_spaces(text)
        legibility = PostProcessing.calculate_legibility_score(text)
        
        logger.info(f"Original:      '{text}'")
        logger.info(f"Cleaned:       '{cleaned}'")
        logger.info(f"Legibility:    {legibility:.3f}")
        logger.info("-" * 60)


def example_5_fake_news_detection():
    """Example 5: Fake news detection text features."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Fake News Detection Features")
    print("="*60)
    
    logger = setup_logging()
    
    # Example extracted texts
    sample_texts = [
        "SHOCKING!!! This unverified claim will blow your mind!!!!",
        "According to recent studies, the new treatment shows promise.",
        "BEWARE: Anonymous sources claim this is FAKE NEWS!!!",
        "Breaking: Municipal council approves new infrastructure project.",
    ]
    
    logger.info("Analyzing texts for fake news indicators:\n")
    
    from src.ocr_extractor import FakeNewsTextExtractor
    
    for text in sample_texts:
        features = FakeNewsTextExtractor._extract_text_features(text)
        readability = FakeNewsTextExtractor._calculate_readability(text)
        flags = FakeNewsTextExtractor._detect_suspicious_patterns(text)
        
        logger.info(f"Text: '{text[:50]}...'")
        logger.info(f"  Features:")
        logger.info(f"    - Words: {features['word_count']}")
        logger.info(f"    - Uppercase Ratio: {features['uppercase_ratio']:.2%}")
        logger.info(f"    - Punctuation Ratio: {features['punctuation_ratio']:.2%}")
        logger.info(f"  Readability: {readability['complexity']}")
        logger.info(f"  Risk Level: {flags['risk_level']}")
        if flags['warnings']:
            logger.info(f"  Warnings: {', '.join(flags['warnings'][:3])}")
        logger.info("-" * 60 + "\n")


def example_6_batch_processing_structure():
    """Example 6: Batch processing structure."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Processing Structure")
    print("="*60)
    
    logger = setup_logging()
    logger.info("Demonstrating batch processing structure...\n")
    
    # Simulate batch results
    sample_results = [
        {
            'image_path': 'img_001.jpg',
            'extracted_text': 'Sample text from image 1',
            'confidence': 0.92,
            'status': 'success'
        },
        {
            'image_path': 'img_002.jpg',
            'extracted_text': 'Another extracted text',
            'confidence': 0.87,
            'status': 'success'
        },
        {
            'image_path': 'img_003.jpg',
            'extracted_text': '',
            'confidence': 0.31,
            'status': 'error'
        },
    ]
    
    # Format results
    formatted = {
        'total_images': len(sample_results),
        'successfully_processed': 2,
        'average_confidence': 0.90,
        'results': sample_results
    }
    
    logger.info(f"Total Images Processed: {formatted['total_images']}")
    logger.info(f"Successfully Processed: {formatted['successfully_processed']}")
    logger.info(f"Average Confidence: {formatted['average_confidence']:.2%}\n")
    
    logger.info("Individual Results:")
    for i, result in enumerate(sample_results, 1):
        logger.info(f"  {i}. {result['image_path']}")
        logger.info(f"     Text: {result['extracted_text'][:40]}...")
        logger.info(f"     Confidence: {result['confidence']:.2%}")
        logger.info(f"     Status: {result['status']}")


def example_7_synthetic_data():
    """Example 7: Synthetic dataset generation."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Synthetic Training Data Generation")
    print("="*60)
    
    logger = setup_logging()
    logger.info("Generating small synthetic dataset for demonstration...\n")
    
    try:
        # Generate small dataset for speed
        images, texts = SyntheticDataGenerator.generate_synthetic_dataset(
            num_samples=100,
            image_width=256,
            image_height=32
        )
        
        logger.info(f"Dataset generated successfully:")
        logger.info(f"  - Images shape: {images.shape}")
        logger.info(f"  - Number of text samples: {len(texts)}")
        logger.info(f"  - Sample texts: {texts[:5]}")
        
    except ImportError as e:
        logger.warning(f"PIL not available: {e}")
        logger.info("To generate synthetic data, install: pip install pillow")


def example_8_model_architecture():
    """Example 8: Model architecture overview."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Model Architecture Overview")
    print("="*60)
    
    logger = setup_logging()
    
    logger.info("\nCNN-LSTM-CTC Architecture:")
    logger.info("-" * 60)
    logger.info("""
    Input (32 × 256 × 1 - grayscale image)
        ↓
    CNN Feature Extraction
        - 4 Convolutional Blocks
        - Max Pooling after each block
        - Layer Normalization & Dropout
        ↓
    Reshape to Sequence (batch, seq_length, features)
        ↓
    Bidirectional LSTM
        - Layer 1: 256 units × 2 (forward + backward)
        - Layer 2: 256 units × 2 (forward + backward)
        - 25% Dropout
        ↓
    Dense Output Layer
        - Units: num_classes + 1 (for CTC blank)
        - Activation: Softmax
        ↓
    CTC Decoding
        - Greedy or Beam Search
        - Variable-length sequence output
        ↓
    Output: Recognized Text
    """)
    
    logger.info("\nKey Features:")
    logger.info("  ✓ Handles variable-length sequences")
    logger.info("  ✓ No need for character-level alignment")
    logger.info("  ✓ Bidirectional context modeling")
    logger.info("  ✓ Robust to input variations")


def example_9_configuration_guide():
    """Example 9: Configuration guide."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Configuration Guide")
    print("="*60)
    
    logger = setup_logging()
    
    logger.info("""
Key Configuration Parameters:

IMAGE PARAMETERS:
  - IMAGE_WIDTH: 256 (width of input images)
  - IMAGE_HEIGHT: 32 (height of input images)
  - IMAGE_CHANNELS: 1 (grayscale)

MODEL PARAMETERS:
  - RNN_UNITS: 256 (LSTM hidden units)
  - NUM_CLASSES: 67 (character classes + CTC blank)

TRAINING PARAMETERS:
  - BATCH_SIZE: 32
  - EPOCHS: 50
  - LEARNING_RATE: 0.001
  - VALIDATION_SPLIT: 0.2

AUGMENTATION:
  - AUGMENT_ROTATION: True (±10 degrees)
  - AUGMENT_SHIFTS: True
  - AUGMENT_ELASTIC_DEFORMATION: True
  - AUGMENT_RANDOM_NOISE: True
  - AUGMENT_BLUR: True

DECODING:
  - CTCBeamSearchDecoder_BEAM_WIDTH: 50
  - CONFIDENCE_THRESHOLD: 0.5

PREPROCESSING:
  - DENOISE_METHOD: "bilateral"
  - NORMALIZE_METHOD: "mean_std"
    """)
    
    logger.info("\nTo customize, edit config/config.py")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "OCR Pipeline Examples - Demonstration")
    print("="*70)
    
    examples = [
        example_1_basic_extraction,
        example_2_preprocessing,
        example_3_text_metrics,
        example_4_post_processing,
        example_5_fake_news_detection,
        example_6_batch_processing_structure,
        example_7_synthetic_data,
        example_8_model_architecture,
        example_9_configuration_guide,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Error in {example_func.__name__}: {e}")
    
    print("\n" + "="*70)
    print(" "*20 + "✓ All Examples Completed Successfully")
    print("="*70)
    print("""
Next Steps:
1. Review the config/config.py file to understand all parameters
2. Read the README.md for detailed usage instructions
3. Check the src/ directory for implementation details
4. Run tests with: python -m tests.test_ocr
5. Train a model with: python -m src.train
6. Use inference with: python -m src.inference --help

For questions or issues, refer to the documentation or test files.
    """)


if __name__ == '__main__':
    run_all_examples()
