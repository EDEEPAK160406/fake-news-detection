"""
Unit tests for OCR module.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.decoder import CTCDecoder, PostProcessing, TextMetrics
from src.model import CNNLSTMCTCModel
from config.config import IMAGE_WIDTH, IMAGE_HEIGHT, CHARACTERS


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing pipeline."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor(
            width=IMAGE_WIDTH, 
            height=IMAGE_HEIGHT
        )
    
    def test_preprocess_shape(self):
        """Test that preprocessing produces correct output shape."""
        # Create dummy image
        image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        
        processed = self.preprocessor.preprocess(image)
        
        self.assertEqual(processed.shape, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        grayscale = self.preprocessor._to_grayscale(image)
        
        self.assertEqual(len(grayscale.shape), 2)
    
    def test_normalization_range(self):
        """Test that normalization produces correct value range."""
        image = np.random.randint(0, 255, (64, 128, 1), dtype=np.uint8)
        normalized = self.preprocessor._normalize(image)
        
        # Values should be roughly in [-2, 2] for mean_std normalization
        self.assertLess(np.max(normalized), 5)
        self.assertGreater(np.min(normalized), -5)


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation techniques."""
    
    def test_rotation(self):
        """Test image rotation."""
        image = np.random.randint(0, 255, (64, 128), dtype=np.uint8)
        rotated = DataAugmentation.rotate(image, angle_range=(-5, 5))
        
        self.assertEqual(rotated.shape, image.shape)
    
    def test_shift(self):
        """Test image shifting."""
        image = np.random.randint(0, 255, (64, 128), dtype=np.uint8)
        shifted = DataAugmentation.shift(image, shift_range=(-0.1, 0.1))
        
        self.assertEqual(shifted.shape, image.shape)
    
    def test_noise_addition(self):
        """Test noise addition."""
        image = np.random.randint(0, 255, (64, 128), dtype=np.uint8)
        noisy = DataAugmentation.add_noise(image, noise_level=0.05)
        
        # Check that noise was added (values changed)
        self.assertFalse(np.allclose(image, noisy))


class TestCTCDecoder(unittest.TestCase):
    """Test CTC decoding."""
    
    def setUp(self):
        self.decoder = CTCDecoder(CHARACTERS)
    
    def test_greedy_decoding(self):
        """Test greedy CTC decoding."""
        # Create dummy predictions
        batch_size = 2
        seq_length = 50
        num_classes = len(CHARACTERS) + 1
        
        predictions = np.random.rand(batch_size, seq_length, num_classes)
        
        decoded = self.decoder.decode_greedy(predictions)
        
        self.assertEqual(len(decoded), batch_size)
        for text in decoded:
            self.assertIsInstance(text, str)
    
    def test_beam_search_decoding(self):
        """Test beam search decoding."""
        batch_size = 2
        seq_length = 50
        num_classes = len(CHARACTERS) + 1
        
        predictions = np.random.rand(batch_size, seq_length, num_classes)
        
        decoded = self.decoder.decode_beam_search(predictions, beam_width=10)
        
        self.assertEqual(len(decoded), batch_size)
        for text, confidence in decoded:
            self.assertIsInstance(text, str)
            self.assertIsInstance(confidence, float)


class TestPostProcessing(unittest.TestCase):
    """Test post-processing utilities."""
    
    def test_remove_extra_spaces(self):
        """Test space normalization."""
        text = "hello   world    test"
        cleaned = PostProcessing.remove_extra_spaces(text)
        
        self.assertEqual(cleaned, "hello world test")
    
    def test_legibility_score(self):
        """Test legibility scoring."""
        text1 = "This is a valid text"
        text2 = "!!!!!!!!!!!!!"
        
        score1 = PostProcessing.calculate_legibility_score(text1)
        score2 = PostProcessing.calculate_legibility_score(text2)
        
        self.assertGreater(score1, score2)
    
    def test_filter_by_confidence(self):
        """Test confidence-based filtering."""
        results = [
            {'text': 'a', 'confidence': 0.9},
            {'text': 'b', 'confidence': 0.3},
            {'text': 'c', 'confidence': 0.8},
        ]
        
        filtered = PostProcessing.filter_by_confidence(results, min_confidence=0.5)
        
        self.assertEqual(len(filtered), 2)


class TestTextMetrics(unittest.TestCase):
    """Test text evaluation metrics."""
    
    def test_character_error_rate(self):
        """Test CER calculation."""
        reference = "hello"
        hypothesis = "hello"
        
        cer = TextMetrics.character_error_rate(reference, hypothesis)
        self.assertEqual(cer, 0.0)
        
        hypothesis = "hallo"
        cer = TextMetrics.character_error_rate(reference, hypothesis)
        self.assertGreater(cer, 0.0)
    
    def test_word_error_rate(self):
        """Test WER calculation."""
        reference = "hello world"
        hypothesis = "hello world"
        
        wer = TextMetrics.word_error_rate(reference, hypothesis)
        self.assertEqual(wer, 0.0)
    
    def test_sequence_error_rate(self):
        """Test SER calculation."""
        reference = "hello"
        hypothesis = "hello"
        
        ser = TextMetrics.sequence_error_rate(reference, hypothesis)
        self.assertEqual(ser, 0.0)
        
        hypothesis = "world"
        ser = TextMetrics.sequence_error_rate(reference, hypothesis)
        self.assertEqual(ser, 1.0)


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture."""
    
    def test_cnn_lstm_ctc_model_creation(self):
        """Test CNN-LSTM-CTC model creation."""
        input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        num_classes = len(CHARACTERS) + 1
        
        model = CNNLSTMCTCModel(
            input_shape=input_shape,
            num_classes=num_classes
        )
        
        model.build_model()
        
        self.assertIsNotNone(model.model)
        self.assertEqual(model.model.input_shape, (None, *input_shape))


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
