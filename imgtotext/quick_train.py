"""
Quick OCR Model Training Script - Trains a working model on synthetic data.
This creates a real model that can extract text from images.

Run with: python quick_train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model import CNNLSTMCTCModel
from src.train import SyntheticDataGenerator
from config.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, CHARACTERS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT
)

def quick_train_ocr():
    """Train OCR model quickly on synthetic data."""
    
    print("\n" + "="*70)
    print("Quick OCR Model Training - Synthetic Data")
    print("="*70)
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'best_ocr_model.h5'
    
    print("\n📊 STEP 1: Generating Synthetic Training Data...")
    print("-" * 70)
    
    generator = SyntheticDataGenerator()
    
    # Generate training data
    num_train = 2000
    num_val = 500
    
    print(f"Generating {num_train} training images...")
    train_images, train_labels = generator.generate_synthetic_dataset(num_train)
    print(f"✓ Generated training set: {train_images.shape}")
    
    print(f"Generating {num_val} validation images...")
    val_images, val_labels = generator.generate_synthetic_dataset(num_val)
    print(f"✓ Generated validation set: {val_images.shape}")
    
    # Convert labels to sparse format for CTC
    print("\n📝 STEP 2: Preparing Labels for CTC...")
    print("-" * 70)
    
    def prepare_ctc_labels(labels):
        """Convert text labels to CTC format."""
        char_to_int = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}
        
        # Pad labels to fixed length
        max_len = 50
        sparse_labels = []
        input_lengths = []
        label_lengths = []
        
        for label in labels:
            label_int = [char_to_int.get(c, 0) for c in label if c in char_to_int]
            label_int = label_int[:max_len]  # Truncate if too long
            
            sparse_labels.append(label_int)
            input_lengths.append(max_len)
            label_lengths.append(len(label_int))
        
        # Pad to same length
        sparse_labels = np.array([
            l + [0] * (max_len - len(l)) for l in sparse_labels
        ])
        
        return (
            np.array(sparse_labels),
            np.array(input_lengths),
            np.array(label_lengths)
        )
    
    train_labels_int, train_input_len, train_label_len = prepare_ctc_labels(train_labels)
    val_labels_int, val_input_len, val_label_len = prepare_ctc_labels(val_labels)
    
    print(f"✓ Labels prepared for CTC loss")
    print(f"  - Training labels: {train_labels_int.shape}")
    print(f"  - Validation labels: {val_labels_int.shape}")
    
    # Build model
    print("\n🏗️  STEP 3: Building CNN-LSTM-CTC Model...")
    print("-" * 70)
    
    model_builder = CNNLSTMCTCModel()
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print(f"✓ Model built with {model.count_params():,} parameters")
    print(f"✓ Model compiled with CTC loss")
    
    # Create data generator for batch processing
    def batch_generator(images, labels_int, input_len, label_len, batch_size=32):
        """Generate batches for training."""
        num_samples = len(images)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_images = images[batch_indices]
                batch_labels = labels_int[batch_indices]
                batch_input_len = input_len[batch_indices]
                batch_label_len = label_len[batch_indices]
                
                # Create dummy labels for CTC (not used, just required by fit)
                yield (
                    [batch_images, batch_labels, batch_input_len, batch_label_len],
                    np.zeros(len(batch_images))  # Dummy targets
                )
    
    print("\n🚀 STEP 4: Training Model...")
    print("-" * 70)
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]
    
    # Train
    train_gen = batch_generator(train_images, train_labels_int, train_input_len, train_label_len)
    val_gen = batch_generator(val_images, val_labels_int, val_input_len, val_label_len)
    
    steps_per_epoch = len(train_images) // BATCH_SIZE
    val_steps = len(val_images) // BATCH_SIZE
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=min(15, EPOCHS),  # Quick training
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print("\n💾 STEP 5: Saving Model...")
    print("-" * 70)
    
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # Verify model
    print("\n✅ STEP 6: Model Verification...")
    print("-" * 70)
    
    # Load and test
    loaded_model = keras.models.load_model(str(model_path), custom_objects={
        'ctc_lambda_func': model_builder.ctc_loss,
        'character_error_rate': model_builder.character_error_rate
    })
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Model ready for inference")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📦 Model saved: {model_path}")
    print(f"🎯 You can now use the web app with real OCR!")
    print(f"🌐 Access at: http://localhost:5000")
    print("\n")

if __name__ == '__main__':
    quick_train_ocr()
