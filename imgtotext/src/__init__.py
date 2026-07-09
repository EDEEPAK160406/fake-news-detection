"""
Image-to-Text Extraction Module for Fake News Detection System.
"""

from src.ocr_extractor import (
    OCRExtractor, 
    FakeNewsTextExtractor, 
    create_ocr_extractor
)
from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.model import CNNLSTMCTCModel, TransformerOCRModel, AttentionMechanism
from src.decoder import CTCDecoder, AttentionDecoder, PostProcessing, TextMetrics
from src.utils import (
    Logger, 
    ConfigLoader, 
    ImageLoader, 
    ResultsFormatter,
    ModelSaver,
    BatchProcessor
)

__version__ = '1.0.0'
__author__ = 'OCR Development Team'

__all__ = [
    'OCRExtractor',
    'FakeNewsTextExtractor',
    'create_ocr_extractor',
    'ImagePreprocessor',
    'DataAugmentation',
    'CNNLSTMCTCModel',
    'TransformerOCRModel',
    'AttentionMechanism',
    'CTCDecoder',
    'AttentionDecoder',
    'PostProcessing',
    'TextMetrics',
    'Logger',
    'ConfigLoader',
    'ImageLoader',
    'ResultsFormatter',
    'ModelSaver',
    'BatchProcessor',
]
