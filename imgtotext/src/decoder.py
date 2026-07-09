"""
Decoding strategies for OCR predictions.
Includes CTC decoding and beam search.
"""

import numpy as np
from typing import List, Tuple, Dict
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class CTCDecoder:
    """CTC (Connectionist Temporal Classification) decoder."""

    @staticmethod
    def _edit_distance(ref, hyp):
        """Pure Python implementation of edit distance (Levenshtein distance)."""
        r_len, h_len = len(ref), len(hyp)

        dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

        for i in range(r_len + 1):
            dp[i][0] = i
        for j in range(h_len + 1):
            dp[0][j] = j

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1]
                    )

        return dp[r_len][h_len]
    
    def __init__(self, character_list: str, blank_index: int = 0):
        """
        Initialize CTC decoder.
        
        Args:
            character_list: String of all possible characters
            blank_index: Index of CTC blank token
        """
        self.character_list = character_list
        self.blank_index = blank_index
        self.char_to_int = {char: idx + 1 for idx, char in enumerate(character_list)}
        self.int_to_char = {idx + 1: char for idx, char in enumerate(character_list)}
        self.int_to_char[0] = ''  # Blank token
    
    def decode_greedy(self, predictions: np.ndarray) -> List[str]:
        """
        Greedy CTC decoding - select highest probability at each timestep.
        
        Args:
            predictions: Model output (batch_size, sequence_length, num_classes)
            
        Returns:
            List of decoded strings
        """
        batch_size = predictions.shape[0]
        decoded_texts = []
        
        for b in range(batch_size):
            # Get argmax predictions for this sequence
            argmax_indices = np.argmax(predictions[b], axis=1)
            
            # Remove consecutive duplicates and blanks
            decoded = []
            prev_index = -1
            
            for idx in argmax_indices:
                if idx != prev_index and idx != self.blank_index:
                    if idx in self.int_to_char:
                        decoded.append(self.int_to_char[idx])
                prev_index = idx
            
            decoded_texts.append(''.join(decoded))
        
        return decoded_texts
    
    def decode_beam_search(self, predictions: np.ndarray, 
                          beam_width: int = 50) -> List[Tuple[str, float]]:
        """
        Beam search CTC decoding for better accuracy.
        
        Args:
            predictions: Model output (batch_size, sequence_length, num_classes)
            beam_width: Number of beams to maintain
            
        Returns:
            List of (text, confidence) tuples
        """
        batch_size = predictions.shape[0]
        decoded_texts = []
        
        for b in range(batch_size):
            seq_probs = predictions[b]  # (sequence_length, num_classes)
            
            # Initialize beams: (text, prob)
            beams = [('', 1.0)]
            
            for t in range(seq_probs.shape[0]):
                probs = seq_probs[t]
                new_beams = {}
                
                for text, curr_prob in beams:
                    for char_idx, char_prob in enumerate(probs):
                        if char_idx == self.blank_index:
                            # Blank: text stays same but probability extends
                            key = text
                        else:
                            # Add character
                            if char_idx in self.int_to_char:
                                char = self.int_to_char[char_idx]
                                # Avoid consecutive duplicates
                                if len(text) > 0 and text[-1] == char:
                                    key = text
                                else:
                                    key = text + char
                            else:
                                continue
                        
                        new_prob = curr_prob * char_prob
                        
                        # Keep track of best hypothesis for each text
                        if key not in new_beams or new_beams[key] < new_prob:
                            new_beams[key] = new_prob
                
                # Keep only top beam_width beams
                beams = sorted(new_beams.items(), 
                              key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Return best hypothesis with confidence
            if beams:
                best_text, best_prob = beams[0]
                decoded_texts.append((best_text, float(best_prob)))
            else:
                decoded_texts.append(('', 0.0))
        
        return decoded_texts
    
    def decode_with_language_model(self, predictions: np.ndarray,
                                   language_model=None) -> List[str]:
        """
        CTC decoding with optional language model for re-scoring.
        
        Args:
            predictions: Model output
            language_model: Optional LM for rescoring
            
        Returns:
            List of decoded strings
        """
        # First perform beam search
        candidates = self.decode_beam_search(predictions, beam_width=10)
        
        if language_model is None:
            return [text for text, _ in candidates]
        
        # Re-score with language model
        rescored = []
        for text, prob in candidates:
            lm_score = language_model.score(text)
            combined_score = prob * (lm_score ** 0.5)  # Combine scores
            rescored.append((text, combined_score))
        
        # Return best hypothesis
        return [sorted(rescored, key=lambda x: x[1], reverse=True)[0][0]]


class AttentionDecoder:
    """Attention-based decoder for sequence-to-sequence models."""
    
    def __init__(self, character_list: str):
        """
        Initialize attention decoder.
        
        Args:
            character_list: String of all possible characters
        """
        self.character_list = character_list
        self.char_to_int = {char: idx for idx, char in enumerate(character_list)}
        self.int_to_char = {idx: char for idx, char in enumerate(character_list)}
    
    def decode(self, predictions: np.ndarray, 
               confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Decode attention-based model predictions.
        
        Args:
            predictions: Model output (batch_size, sequence_length, num_classes)
            confidence_threshold: Minimum confidence for characters
            
        Returns:
            List of dicts with text and confidence scores
        """
        batch_size = predictions.shape[0]
        results = []
        
        for b in range(batch_size):
            seq_probs = predictions[b]  # (sequence_length, num_classes)
            
            decoded_chars = []
            char_confidences = []
            
            for probs in seq_probs:
                char_idx = np.argmax(probs)
                confidence = probs[char_idx]
                
                if confidence >= confidence_threshold and char_idx in self.int_to_char:
                    char = self.int_to_char[char_idx]
                    # Avoid end-of-sequence token typically at high indices
                    if char != '<END>' and char != '<PAD>':
                        decoded_chars.append(char)
                        char_confidences.append(float(confidence))
            
            results.append({
                'text': ''.join(decoded_chars),
                'confidence': float(np.mean(char_confidences)) if char_confidences else 0.0,
                'character_confidences': char_confidences
            })
        
        return results


class PostProcessing:
    """Post-processing utilities for OCR output."""
    
    @staticmethod
    def correct_common_mistakes(text: str, 
                                mistake_map: Dict[str, str] = None) -> str:
        """
        Correct common OCR mistakes using a mistake dictionary.
        
        Args:
            text: OCR output text
            mistake_map: Dict mapping common mistakes to corrections
            
        Returns:
            Corrected text
        """
        if mistake_map is None:
            # Common OCR errors
            mistake_map = {
                '0': 'O',  # Zero to O
                '1': 'l',  # One to lowercase L
                '5': 'S',  # Five to S
                '8': 'B',  # Eight to B
                '|': 'I',  # Pipe to I
                '`': "'",  # Backtick to apostrophe
            }
        
        corrected = text
        for mistake, correction in mistake_map.items():
            corrected = corrected.replace(mistake, correction)
        
        return corrected
    
    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Remove extra spaces and normalize whitespace."""
        return ' '.join(text.split())
    
    @staticmethod
    def calculate_legibility_score(text: str) -> float:
        """
        Calculate a legibility score for extracted text.
        Higher score = more likely to be valid text.
        
        Args:
            text: Extracted text
            
        Returns:
            Score between 0 and 1
        """
        if not text or len(text) < 2:
            return 0.0
        
        # Factors:
        # 1. Has reasonable length
        # 2. Mostly consists of valid characters
        # 3. Has proper spacing
        
        score = 0.0
        
        # Length factor (prefer medium-length text)
        length_factor = min(len(text) / 50.0, 1.0)  # Max at 50 chars
        score += 0.3 * length_factor
        
        # Character validity
        valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?-\'"')
        char_factor = valid_chars / len(text)
        score += 0.4 * char_factor
        
        # Space distribution - not too many spaces
        space_ratio = text.count(' ') / len(text)
        space_factor = 1.0 - abs(space_ratio - 0.2)  # Expect ~20% spaces
        score += 0.3 * max(space_factor, 0.0)
        
        return min(score, 1.0)
    
    @staticmethod
    def filter_by_confidence(results: List[Dict], 
                             min_confidence: float = 0.5) -> List[Dict]:
        """
        Filter results by minimum confidence threshold.
        
        Args:
            results: List of result dicts with confidence scores
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.get('confidence', 0) >= min_confidence]


class TextMetrics:
    """Metrics for evaluating text recognition quality."""
    
    @staticmethod
    def character_error_rate(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER).
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            CER as percentage (0-100)
        """
        if not reference:
            return 100.0 if hypothesis else 0.0
        
        distance = CTCDecoder._edit_distance(reference, hypothesis)
        return (distance / len(reference)) * 100
    
    @staticmethod
    def word_error_rate(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER).
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            WER as percentage (0-100)
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 100.0 if hyp_words else 0.0
        
        distance = CTCDecoder._edit_distance(ref_words, hyp_words)
        return (distance / len(ref_words)) * 100
    
    @staticmethod
    def sequence_error_rate(reference: str, hypothesis: str) -> float:
        """
        Calculate Sequence Error Rate (SER).
        1 if sequences differ, 0 if identical.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            SER (0 or 1)
        """
        return 0.0 if reference == hypothesis else 1.0
