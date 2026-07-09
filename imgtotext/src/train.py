"""
Training script for CNN-LSTM-CTC OCR model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from pathlib import Path
import logging
from typing import Tuple, List
import sys

from src.preprocessing import ImagePreprocessor, DataAugmentation
from src.model import CNNLSTMCTCModel
from src.decoder import CTCDecoder
from src.utils import Logger, ConfigLoader
from config.config import *

# Setup logging
logger = Logger.setup_logger(__name__)


class OCRTrainer:
    """Trainer for OCR models."""
    
    def __init__(self, config_dict: dict = None):
        """
        Initialize trainer.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or {
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'characters': CHARACTERS,
            'num_classes': NUM_CLASSES,
            'rnn_units': RNN_UNITS,
        }
        
        self.preprocessor = ImagePreprocessor(
            width=self.config['image_width'],
            height=self.config['image_height']
        )
        
        self.model = None
        self.history = None
    
    def build_model(self) -> CNNLSTMCTCModel:
        """Build OCR model."""
        input_shape = (
            self.config['image_height'],
            self.config['image_width'],
            1  # Grayscale
        )
        
        self.model = CNNLSTMCTCModel(
            input_shape=input_shape,
            num_classes=self.config['num_classes'],
            rnn_units=self.config['rnn_units']
        )
        
        self.model.build_model()
        self.model.compile_model(learning_rate=self.config['learning_rate'])
        
        logger.info("Model built and compiled successfully")
        return self.model
    
    def _setup_callbacks(self, checkpoint_dir: str = 'models/checkpoints') -> List:
        """Setup training callbacks."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpointing
            ModelCheckpoint(
                filepath=f'{checkpoint_dir}/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train(self, train_images: np.ndarray, train_labels: np.ndarray,
              val_images: np.ndarray = None, val_labels: np.ndarray = None,
              augment: bool = True) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_images: Training images (N, H, W, C)
            train_labels: Training labels (N, max_length)
            val_images: Validation images
            val_labels: Validation labels
            augment: Whether to use data augmentation
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Setup data augmentation
        if augment:
            augmentation_config = {
                'rotation': AUGMENT_ROTATION,
                'shifts': AUGMENT_SHIFTS,
                'elastic_deformation': AUGMENT_ELASTIC_DEFORMATION,
                'random_noise': AUGMENT_RANDOM_NOISE,
                'blur': AUGMENT_BLUR,
            }
            
            # Create augmented dataset using DataAugmentation
            logger.info("Applying data augmentation...")
            augmented_images = []
            for img in train_images:
                # Denormalize for augmentation
                if np.max(img) <= 1.0:
                    img_denorm = (img * 255).astype(np.uint8)
                else:
                    img_denorm = img.astype(np.uint8)
                
                # Apply augmentation
                aug_img = DataAugmentation.augment(img_denorm, augmentation_config)
                
                # Normalize back
                aug_img = self.preprocessor._normalize(aug_img)
                augmented_images.append(aug_img)
            
            train_images = np.array(augmented_images)
        
        # Setup validation split
        if val_images is None:
            split_idx = int(len(train_images) * (1 - VALIDATION_SPLIT))
            val_images = train_images[split_idx:]
            val_labels = train_labels[split_idx:]
            train_images = train_images[:split_idx]
            train_labels = train_labels[:split_idx]
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train model
        logger.info("Starting training...")
        self.history = self.model.get_model().fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def save_model(self, model_path: str = BEST_MODEL_NAME) -> None:
        """
        Save trained model.
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.get_model().save(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model.
        
        Args:
            model_path: Path to model file
        """
        from tensorflow.keras.models import load_model
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return
        
        custom_objects = {
            'ctc_loss': CNNLSTMCTCModel.ctc_loss,
            'character_error_rate': CNNLSTMCTCModel.character_error_rate
        }
        
        loaded_model = load_model(str(model_path), custom_objects=custom_objects)
        
        if self.model is None:
            self.model = CNNLSTMCTCModel(
                input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                num_classes=NUM_CLASSES
            )
        
        self.model.model = loaded_model
        logger.info(f"Model loaded from {model_path}")


class SyntheticDataGenerator:
    """Generate synthetic OCR training data for demonstration."""
    
    @staticmethod
    def generate_synthetic_dataset(num_samples: int = 1000,
                                   image_width: int = IMAGE_WIDTH,
                                   image_height: int = IMAGE_HEIGHT) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic text images for OCR training.
        
        Args:
            num_samples: Number of samples to generate
            image_width: Width of generated images
            image_height: Height of generated images
            
        Returns:
            Tuple of (images, labels)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random
            import string
        except ImportError:
            logger.error("PIL not installed. Install with: pip install pillow")
            return np.zeros((num_samples, image_height, image_width, 1)), np.zeros((num_samples, 50))
        
        images = []
        labels = []
        characters = CHARACTERS
        
        logger.info(f"Generating {num_samples} synthetic images...")
        
        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} images")
            
            # Generate random text
            text_length = random.randint(5, 15)
            text = ''.join(random.choices(characters, k=text_length))
            
            # Create image
            img = Image.new('RGB', (image_width, image_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add text with some variations
            font_size = random.randint(15, 20)
            try:
                # Try to use a system font (may not work on all systems)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Random position
            x = random.randint(5, image_width - 50)
            y = random.randint(5, image_height - 25)
            
            # Random color (dark text)
            text_color = tuple(random.randint(0, 100) for _ in range(3))
            
            draw.text((x, y), text, fill=text_color, font=font)
            
            # Convert to grayscale numpy array
            img_array = np.array(img.convert('L'))
            
            # Normalize
            img_array = img_array.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            
            images.append(img_array)
            labels.append(text)
        
        return np.array(images), np.array(labels)
    
    @staticmethod
    def text_to_labels(texts: List[str], character_set: str = CHARACTERS,
                      max_length: int = 50) -> np.ndarray:
        """
        Convert text strings to label arrays for CTC.
        
        Args:
            texts: List of text strings
            character_set: Character set used
            max_length: Maximum label length
            
        Returns:
            Label array (num_texts, max_length)
        """
        char_to_idx = {char: idx + 1 for idx, char in enumerate(character_set)}
        
        labels = np.zeros((len(texts), max_length), dtype=np.int32)
        
        for i, text in enumerate(texts):
            for j, char in enumerate(text[:max_length]):
                if char in char_to_idx:
                    labels[i, j] = char_to_idx[char]
        
        return labels


def main():
    """Main training script."""
    logger.info("OCR Model Training Pipeline")
    logger.info("=" * 50)
    
    # Generate synthetic dataset for demonstration
    logger.info("Generating synthetic dataset...")
    train_images, train_texts = SyntheticDataGenerator.generate_synthetic_dataset(
        num_samples=1000
    )
    
    # Convert texts to CTC labels
    train_labels = SyntheticDataGenerator.text_to_labels(train_texts)
    
    logger.info(f"Dataset size: {train_images.shape}")
    logger.info(f"Labels shape: {train_labels.shape}")
    
    # Initialize trainer
    trainer = OCRTrainer()
    
    # Build and train model
    trainer.build_model()
    trainer.train(train_images, train_labels, augment=True)
    
    # Save model
    trainer.save_model(BEST_MODEL_NAME)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == '__main__':
    main()
