"""
Demo OCR Extractor - Generates realistic synthetic text for demonstration.
Used when no pre-trained model is available.
"""

import numpy as np
import random
import string
from typing import Dict, Tuple

class DemoOCRExtractor:
    """Simulates OCR output for demonstration purposes."""
    
    def __init__(self):
        """Initialize demo extractor."""
        self.sample_texts = [
            "Breaking News: Scientists discover new renewable energy source",
            "Local government approves new infrastructure project",
            "Market reaches record high amid economic growth",
            "New medical breakthrough offers hope for patients",
            "Community rallies for environmental conservation",
            "Technology advances improve accessibility for all",
            "Education initiative launches next month",
            "Sports team wins championship title",
            "Local business celebrates 50 years of service",
            "Weather forecast predicts sunny skies this week",
            "Transportation improvements announced for region",
            "Students excel in national competition",
            "Museum opens new permanent exhibition",
            "Annual festival returns to downtown area",
            "Hospital expands emergency department",
            "Library launches digital collection program",
            "City council discusses housing development",
            "Environmental group awards conservation prizes",
            "University receives major research grant",
            "Community center opens new fitness facility"
        ]
        
        self.suspicious_texts = [
            "SHOCKING!!! Unverified claim will BLOW YOUR MIND!!!",
            "BEWARE: Anonymous sources reveal FAKE NEWS conspiracy",
            "URGENT: This MUST be shared immediately!!!",
            "You won't BELIEVE what celebrities are hiding!",
            "DESTROYS mainstream media narrative!!!",
            "Doctors HATE this one weird trick...",
            "This will CHANGE EVERYTHING you know!!!",
            "Government DOESN'T want you to see this!!!",
            "ALERT: Shocking truth about vaccines revealed",
            "EXCLUSIVE: Deep state operatives exposed!!!"
        ]
    
    def extract_text(self, image_path: str) -> Dict:
        """
        Simulate text extraction from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with extracted text and confidence
        """
        # Randomly choose between normal and suspicious text
        if random.random() < 0.7:
            text = random.choice(self.sample_texts)
            confidence = random.uniform(0.75, 0.98)
        else:
            text = random.choice(self.suspicious_texts)
            confidence = random.uniform(0.65, 0.85)
        
        return {
            'extracted_text': text,
            'confidence': confidence,
            'timestamp': self._get_timestamp(),
            'source': 'demo_mode'
        }
    
    def extract_batch(self, image_paths: list) -> list:
        """Extract text from multiple images."""
        results = []
        for path in image_paths:
            result = self.extract_text(path)
            result['image_path'] = path
            results.append(result)
        return results
    
    @staticmethod
    def _get_timestamp():
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class DemoFakeNewsExtractor:
    """Simulates fake news detection for demo purposes."""
    
    def __init__(self, model_path: str = None):
        """Initialize demo fake news extractor."""
        self.demo_extractor = DemoOCRExtractor()
    
    def extract_for_analysis(self, image_path: str) -> Dict:
        """Extract and analyze for fake news indicators."""
        result = self.demo_extractor.extract_text(image_path)
        text = result['extracted_text']
        
        return {
            'extracted_text': text,
            'confidence': result['confidence'],
            'text_features': self._extract_text_features(text),
            'readability_metrics': self._calculate_readability(text),
            'text_flags': self._detect_suspicious_patterns(text)
        }
    
    @staticmethod
    def _extract_text_features(text: str) -> Dict:
        """Extract text features."""
        words = text.split()
        chars = len(text)
        
        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_count = sum(1 for c in text if c in '!?,.\'-"')
        
        return {
            'word_count': len(words),
            'char_count': chars,
            'avg_word_length': chars / len(words) if words else 0,
            'uppercase_ratio': uppercase_count / chars if chars else 0,
            'digit_ratio': digit_count / chars if chars else 0,
            'punctuation_ratio': punctuation_count / chars if chars else 0
        }
    
    @staticmethod
    def _calculate_readability(text: str) -> Dict:
        """Calculate readability metrics."""
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences == 0:
            sentences = 1
        
        words = len(text.split())
        avg_length = words / sentences if sentences else 0
        
        # Complexity based on average sentence length
        if avg_length < 10:
            complexity = 'very_easy'
        elif avg_length < 15:
            complexity = 'easy'
        elif avg_length < 20:
            complexity = 'moderate'
        elif avg_length < 25:
            complexity = 'difficult'
        else:
            complexity = 'very_difficult'
        
        return {
            'sentence_count': sentences,
            'avg_sentence_length': avg_length,
            'complexity': complexity
        }
    
    @staticmethod
    def _detect_suspicious_patterns(text: str) -> Dict:
        """Detect fake news indicators."""
        suspicious_keywords = [
            'shocking', 'exclusive', 'unverified', 'anonymous', 'fake news',
            'destroy', 'hate', 'weird trick', 'doctors', 'believe'
        ]
        
        warnings = []
        score = 0
        
        # Check for excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count >= 3:
            warnings.append('excessive_exclamation')
            score += 2
        
        # Check for ALL CAPS words
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        if caps_words > len(words) * 0.2:
            warnings.append('excessive_caps')
            score += 1
        
        # Check for suspicious keywords
        text_lower = text.lower()
        for keyword in suspicious_keywords:
            if keyword in text_lower:
                warnings.append(f'keyword_{keyword}')
                score += 1
        
        # Determine risk level
        if score >= 4:
            risk_level = 'high'
        elif score >= 2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'warnings': list(set(warnings)),  # Remove duplicates
            'suspicion_score': score
        }
