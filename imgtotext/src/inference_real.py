"""
Real OCR Inference - Loads trained model and performs text extraction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from typing import Dict, Optional
import logging

try:
    import easyocr
except Exception:
    easyocr = None

logger = logging.getLogger(__name__)

class RealOCRInference:
    """Inference using trained OCR model."""
    
    def __init__(self, model_path: str):
        """Initialize with model path."""
        self.model_path = model_path
        self.model = None
        self.character_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-"
        self.int_to_char = {idx: char for idx, char in enumerate(self.character_set)}
        self.easyocr_reader = None
        self.load_model()

    def _get_easyocr_reader(self):
        """Lazy-initialize EasyOCR reader."""
        if easyocr is None:
            return None
        if self.easyocr_reader is None:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        return self.easyocr_reader

    def _is_text_low_quality(self, text: str) -> bool:
        """Heuristic for unreliable OCR outputs."""
        cleaned = text.strip()
        if not cleaned:
            return True
        if len(cleaned) < 3:
            return True

        alpha_count = sum(char.isalpha() for char in cleaned)
        punctuation_count = sum(not char.isalnum() and not char.isspace() for char in cleaned)
        if alpha_count == 0:
            return True
        if punctuation_count > alpha_count:
            return True
        return False

    def _easyocr_extract(self, image_path: str) -> Optional[Dict]:
        """Fallback extraction using EasyOCR for readable outputs."""
        reader = self._get_easyocr_reader()
        if reader is None:
            return None

        try:
            results = reader.readtext(image_path, detail=1, paragraph=True)
            if not results:
                return None

            text_parts = []
            confidences = []
            for item in results:
                if len(item) < 2:
                    continue
                if len(item) >= 3:
                    _, text, confidence = item
                else:
                    _, text = item
                    confidence = 0.75
                if text and text.strip():
                    text_parts.append(text.strip())
                    confidences.append(float(confidence))

            if not text_parts:
                return None

            merged_text = ' '.join(text_parts).strip()
            mean_conf = float(np.mean(confidences)) if confidences else 0.0
            return {
                'extracted_text': merged_text,
                'confidence': mean_conf,
                'source': 'easyocr_fallback'
            }
        except Exception as error:
            logger.warning(f"EasyOCR fallback failed: {error}")
            return None
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def extract_text(self, image_path: str) -> Dict:
        """Extract text from image."""
        try:
            # Primary OCR path: EasyOCR (more reliable for varied images)
            easyocr_result = self._easyocr_extract(image_path)
            if easyocr_result is not None and easyocr_result.get('extracted_text'):
                return easyocr_result

            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess
            img = cv2.GaussianBlur(img, (3, 3), 0)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.resize(img, (256, 32))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.model.predict(img, verbose=0)
            
            # Extract text
            text = self._decode_predictions(predictions[0])
            
            # Calculate confidence
            confidence = float(np.mean(np.max(predictions, axis=2)))
            
            result = {
                'extracted_text': text,
                'confidence': confidence,
                'source': 'real_model'
            }

            return result
        
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            fallback_result = self._easyocr_extract(image_path)
            if fallback_result is not None:
                return fallback_result
            return {
                'extracted_text': f'(Error: {str(e)})',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _decode_predictions(self, prediction: np.ndarray) -> str:
        """Decode model predictions to text."""
        if prediction.ndim != 2:
            return '(Unable to extract text)'

        pred_batch = np.expand_dims(prediction, axis=0)
        input_len = np.ones((1, 1), dtype=np.int32) * prediction.shape[0]

        decoded, _ = keras.backend.ctc_decode(pred_batch, input_length=input_len[:, 0], greedy=True)
        decoded_seq = decoded[0].numpy()[0]

        chars = []
        for idx in decoded_seq:
            if idx < 0:
                continue
            if 0 <= idx < len(self.character_set):
                chars.append(self.character_set[idx])
        text = ''.join(chars).strip()
        if text:
            return text

        # Fallback: direct argmax collapse (ignore blank class)
        best = np.argmax(prediction, axis=1)
        blank_idx = prediction.shape[1] - 1
        fallback_chars = []
        prev_idx = -1
        for idx in best:
            if idx == prev_idx:
                continue
            prev_idx = idx
            if idx == blank_idx:
                continue
            if 0 <= idx < len(self.character_set):
                fallback_chars.append(self.character_set[idx])

        fallback_text = ''.join(fallback_chars).strip()
        if fallback_text:
            return fallback_text

        # Last resort: if model predicts blank everywhere, use second-best class
        top2 = np.argsort(prediction, axis=1)[:, -2:]
        alt_indices = []
        for row in top2:
            best_idx = row[-1]
            second_idx = row[-2]
            alt_indices.append(second_idx if best_idx == blank_idx else best_idx)

        alt_chars = []
        prev_idx = -1
        for idx in alt_indices:
            if idx == prev_idx:
                continue
            prev_idx = idx
            if idx == blank_idx:
                continue
            if 0 <= idx < len(self.character_set):
                alt_chars.append(self.character_set[idx])

        alt_text = ''.join(alt_chars).strip()
        return alt_text if alt_text else '(Unable to extract text)'
    
    def extract_batch(self, image_paths: list) -> list:
        """Extract from multiple images."""
        results = []
        for path in image_paths:
            result = self.extract_text(path)
            result['image_path'] = path
            results.append(result)
        return results
