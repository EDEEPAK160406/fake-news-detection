"""
Main Image-to-Text Extraction Module for Fake News Detection System.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.model import CNNLSTMCTCModel, TransformerOCRModel
from src.decoder import CTCDecoder, PostProcessing, TextMetrics
from src.utils import ImageLoader, ResultsFormatter, Logger

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Main OCR module for extracting text from images."""
    
    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize OCR Extractor.
        
        Args:
            model_path: Path to pretrained model
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.preprocessor = ImagePreprocessor(
            width=self.config['image_width'],
            height=self.config['image_height'],
            normalize_method=self.config.get('normalize_method', 'mean_std')
        )
        self.model = None
        self.decoder = CTCDecoder(self.config['characters'])
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    @staticmethod
    def _get_default_config() -> Dict:
        """Get default configuration."""
        return {
            'image_width': 256,
            'image_height': 32,
            'characters': "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-",
            'num_classes': 66,  # len(characters) + 1
            'rnn_units': 256,
            'normalize_method': 'mean_std',
            'confidence_threshold': 0.5,
            'beam_width': 50,
            'model_type': 'cnn_lstm_ctc'
        }
    
    def _initialize_model(self) -> None:
        """Initialize model architecture."""
        input_shape = (
            self.config['image_height'],
            self.config['image_width'],
            1  # Grayscale
        )
        
        if self.config['model_type'] == 'cnn_lstm_ctc':
            self.model = CNNLSTMCTCModel(
                input_shape=input_shape,
                num_classes=self.config['num_classes'],
                rnn_units=self.config['rnn_units']
            )
            self.model.build_model()
            self.model.compile_model()
        elif self.config['model_type'] == 'transformer':
            self.model = TransformerOCRModel(
                input_shape=input_shape,
                num_classes=self.config['num_classes']
            )
            self.model.build_model()
        
        logger.info(f"Model initialized: {self.config['model_type']}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load pretrained model.
        
        Args:
            model_path: Path to saved model
        """
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        
        try:
            # Custom objects for CTC loss
            custom_objects = {
                'ctc_loss': CNNLSTMCTCModel.ctc_loss,
                'character_error_rate': CNNLSTMCTCModel.character_error_rate
            }
            
            self.model = load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_text(self, image_path: str, return_confidence: bool = True) -> Dict:
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to input image
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Load and preprocess image
            image = ImageLoader.load_image(image_path)
            processed_image = self.preprocessor.preprocess(image)
            
            # Add batch dimension
            input_batch = np.expand_dims(processed_image, axis=0)
            
            # Get model predictions
            predictions = self.model.predict(input_batch, verbose=0)
            
            # Decode using beam search for better accuracy
            decoded_results = self.decoder.decode_beam_search(
                predictions,
                beam_width=self.config['beam_width']
            )
            
            text, confidence = decoded_results[0]
            
            # Post-processing
            text = PostProcessing.remove_extra_spaces(text)
            legibility = PostProcessing.calculate_legibility_score(text)
            
            result = {
                'image_path': str(image_path),
                'extracted_text': text,
                'confidence': float(confidence),
                'legibility_score': legibility,
                'status': 'success'
            }
            
            if return_confidence:
                result['character_confidences'] = self._get_char_confidences(predictions[0])
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'extracted_text': '',
                'confidence': 0.0,
                'legibility_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def extract_batch(self, image_paths: List[str], 
                     batch_size: int = 32) -> List[Dict]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess batch
            for image_path in batch_paths:
                try:
                    image = ImageLoader.load_image(image_path)
                    processed_image = self.preprocessor.preprocess(image)
                    batch_images.append(processed_image)
                    valid_paths.append(image_path)
                except Exception as e:
                    logger.warning(f"Skipping {image_path}: {e}")
                    all_results.append({
                        'image_path': str(image_path),
                        'extracted_text': '',
                        'confidence': 0.0,
                        'status': 'error',
                        'error': str(e)
                    })
            
            if not batch_images:
                continue
            
            # Batch prediction
            batch_array = np.array(batch_images)
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Decode batch
            for j, (pred, image_path) in enumerate(zip(predictions, valid_paths)):
                decoded_results = self.decoder.decode_beam_search(
                    np.expand_dims(pred, axis=0),
                    beam_width=self.config['beam_width']
                )
                
                text, confidence = decoded_results[0]
                text = PostProcessing.remove_extra_spaces(text)
                legibility = PostProcessing.calculate_legibility_score(text)
                
                result = {
                    'image_path': str(image_path),
                    'extracted_text': text,
                    'confidence': float(confidence),
                    'legibility_score': legibility,
                    'status': 'success'
                }
                
                all_results.append(result)
        
        return all_results
    
    def extract_with_confidence_filtering(self, image_paths: List[str],
                                         min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract text and filter by confidence threshold.
        
        Args:
            image_paths: List of image paths
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of high-confidence results
        """
        results = self.extract_batch(image_paths)
        
        filtered_results = [
            r for r in results 
            if r.get('confidence', 0) >= min_confidence
        ]
        
        return filtered_results
    
    def _get_char_confidences(self, predictions: np.ndarray) -> List[float]:
        """Extract per-character confidence scores."""
        confidences = []
        argmax_indices = np.argmax(predictions, axis=1)
        
        for idx in argmax_indices:
            if idx > 0 and idx < len(predictions[0]):
                confidences.append(float(predictions[idx, int(idx)]))
        
        return confidences


class FakeNewsTextExtractor(OCRExtractor):
    """
    Specialized OCR extractor for fake news detection.
    Optimized for extracting text from news articles, headlines, and text-based images.
    """
    
    def extract_for_analysis(self, image_path: str) -> Dict:
        """
        Extract text optimized for fake news detection.
        
        Args:
            image_path: Path to image containing text
            
        Returns:
            Dictionary with extracted text and metadata for fake news analysis
        """
        base_result = self.extract_text(image_path)
        
        # Enhance result with fake news detection metadata
        extracted_text = base_result['extracted_text']
        
        analysis_result = {
            **base_result,
            'text_features': self._extract_text_features(extracted_text),
            'readability_metrics': self._calculate_readability(extracted_text),
            'text_flags': self._detect_suspicious_patterns(extracted_text)
        }
        
        return analysis_result
    
    @staticmethod
    def _extract_text_features(text: str) -> Dict:
        """Extract linguistic features from text."""
        words = text.split()
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'punctuation_ratio': sum(1 for c in text if c in '.!?,;:') / len(text) if text else 0,
        }
    
    @staticmethod
    def _calculate_readability(text: str) -> Dict:
        """Calculate readability metrics."""
        sentences = [s for s in text.split('.') if s.strip()]
        words = text.split()
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'complexity': 'high' if (len(words) / len(sentences) > 15) else 'low',
        }
    
    @staticmethod
    def _detect_suspicious_patterns(text: str) -> Dict:
        """Detect patterns commonly associated with fake news/misinformation."""
        suspicious_keywords = [
            'fake', 'hoax', 'unverified', 'rumor', 'alleged',
            'supposedly', 'apparently', 'claims', 'anonymous sources'
        ]
        
        warnings = []
        text_lower = text.lower()
        
        # Check for multiple exclamation marks
        if text.count('!') > 2:
            warnings.append('excessive_exclamation')
        
        # Check for ALL CAPS
        if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            warnings.append('excessive_caps')
        
        # Check for suspicious keywords
        for keyword in suspicious_keywords:
            if keyword in text_lower:
                warnings.append(f'keyword_{keyword}')
        
        return {
            'has_warnings': len(warnings) > 0,
            'warnings': warnings,
            'risk_level': 'high' if len(warnings) > 2 else 'medium' if len(warnings) > 0 else 'low'
        }


def create_ocr_extractor(model_path: Optional[str] = None,
                         config: Optional[Dict] = None,
                         for_fake_news: bool = True) -> OCRExtractor:
    """
    Factory function to create OCR extractor.
    
    Args:
        model_path: Path to pretrained model
        config: Configuration dictionary
        for_fake_news: Whether to use specialized fake news extractor
        
    Returns:
        OCR extractor instance
    """
    ExtractorClass = FakeNewsTextExtractor if for_fake_news else OCRExtractor
    return ExtractorClass(model_path=model_path, config=config)
