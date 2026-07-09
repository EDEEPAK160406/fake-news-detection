"""
Utility functions for OCR system.
"""

import json
import yaml
import logging
from typing import Dict, List, Any
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime


class Logger:
    """Logging utility."""
    
    @staticmethod
    def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
        """
        Setup a logger with console and file handlers.
        
        Args:
            name: Logger name
            log_level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        # Add handlers
        if not logger.handlers:
            logger.addHandler(ch)
        
        return logger


class ConfigLoader:
    """Configuration file loader."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to JSON or YAML.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save config
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix == '.json':
                json.dump(config, f, indent=2)
            elif config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")


class ImageLoader:
    """Image loading utilities."""
    
    @staticmethod
    def load_image(image_path: str, color_mode: str = 'bgr') -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            color_mode: 'bgr' or 'rgb'
            
        Returns:
            Image array
        """
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if color_mode == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    @staticmethod
    def load_batch(image_paths: List[str], color_mode: str = 'bgr') -> List[np.ndarray]:
        """
        Load batch of images.
        
        Args:
            image_paths: List of image paths
            color_mode: 'bgr' or 'rgb'
            
        Returns:
            List of image arrays
        """
        images = []
        for path in image_paths:
            try:
                img = ImageLoader.load_image(path, color_mode)
                images.append(img)
            except Exception as e:
                logging.warning(f"Error loading image {path}: {e}")
                continue
        
        return images
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """
        Save image to file.
        
        Args:
            image: Image array
            output_path: Path to save image
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not cv2.imwrite(str(output_path), image):
            raise ValueError(f"Could not save image: {output_path}")


class ResultsFormatter:
    """Format OCR results for output."""
    
    @staticmethod
    def format_single_result(image_path: str, text: str, 
                            confidence: float = 0.0) -> Dict[str, Any]:
        """
        Format single OCR result.
        
        Args:
            image_path: Path to input image
            text: Extracted text
            confidence: Confidence score
            
        Returns:
            Formatted result dictionary
        """
        return {
            'image': str(image_path),
            'extracted_text': text,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'length': len(text),
            'words': len(text.split())
        }
    
    @staticmethod
    def format_batch_results(image_paths: List[str], 
                            texts: List[str],
                            confidences: List[float] = None) -> Dict[str, Any]:
        """
        Format batch OCR results.
        
        Args:
            image_paths: List of input image paths
            texts: List of extracted texts
            confidences: List of confidence scores
            
        Returns:
            Formatted results dictionary
        """
        if confidences is None:
            confidences = [0.0] * len(texts)
        
        results = {
            'total_images': len(image_paths),
            'successfully_processed': len(texts),
            'average_confidence': np.mean(confidences),
            'results': []
        }
        
        for img_path, text, conf in zip(image_paths, texts, confidences):
            results['results'].append(
                ResultsFormatter.format_single_result(img_path, text, conf)
            )
        
        return results
    
    @staticmethod
    def save_results_json(results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def save_results_csv(results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if 'results' not in results:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'extracted_text', 
                                                    'confidence', 'length', 'words'])
            writer.writeheader()
            
            for result in results['results']:
                writer.writerow({
                    'image': result['image'],
                    'extracted_text': result['extracted_text'],
                    'confidence': result['confidence'],
                    'length': result['length'],
                    'words': result['words']
                })


class ModelSaver:
    """Save and load models."""
    
    @staticmethod
    def save_model(model, model_path: str, save_format: str = 'h5') -> None:
        """
        Save model to file.
        
        Args:
            model: Keras model
            model_path: Path to save model
            save_format: 'h5' or 'tf'
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'h5':
            model.save(str(model_path))
        elif save_format == 'tf':
            model.save(str(model_path), save_format='tf')
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
    
    @staticmethod
    def load_model(model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        from tensorflow.keras.models import load_model
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return load_model(str(model_path))


class BatchProcessor:
    """Process images in batches efficiently."""
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
        """
        self.batch_size = batch_size
    
    def create_batches(self, items: List[Any]) -> List[List[Any]]:
        """
        Create batches from list of items.
        
        Args:
            items: List of items to batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        
        return batches
    
    def process_batches(self, items: List[Any], 
                       process_func, *args, **kwargs) -> List[Any]:
        """
        Process items in batches using provided function.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each batch
            *args, **kwargs: Arguments to pass to process_func
            
        Returns:
            List of processed results
        """
        batches = self.create_batches(items)
        results = []
        
        for batch in batches:
            try:
                batch_results = process_func(batch, *args, **kwargs)
                results.extend(batch_results if isinstance(batch_results, list) 
                             else [batch_results])
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                continue
        
        return results
