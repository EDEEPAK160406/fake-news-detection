"""
Simple OCR Model Trainer - Creates a working OCR model.
Uses TensorFlow's built-in CTC to avoid custom object serialization issues.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.train import SyntheticDataGenerator
from config.config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, CHARACTERS, BATCH_SIZE
)

def build_ocr_model(input_shape, num_classes):
    """Build a simple but functional OCR model."""
    
    # Input
    inputs = keras.Input(shape=input_shape, name='image')
    
    # CNN Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Reshape for RNN (batch, seq_length, features)
    squeezed_shape = (x.shape[1], x.shape[2] * x.shape[3])
    x = layers.Reshape(target_shape=squeezed_shape)(x)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.LayerNormalization()(x)
    
    # Dense output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs, outputs, name='OCR_Model')
    return model

def ctc_loss(y_true, y_pred):
    """CTC loss wrapper."""
    batch_len = tf.cast(tf.shape(y_pred)[0], tf.int32)
    input_length = tf.cast(tf.shape(y_pred)[1], tf.int32)
    label_length = tf.cast(tf.shape(y_true)[1], tf.int32)
    
    input_length = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
    label_length = label_length * tf.ones(shape=(batch_len,), dtype=tf.int32)
    
    loss = keras.backend.ctc_batch_cost(
        y_true, y_pred, input_length, label_length
    )
    return loss

def train_model():
    """Train the OCR model."""
    
    print("\n" + "="*70)
    print("🎯 Quick OCR Model Training")
    print("="*70 + "\n")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'best_ocr_model.h5'
    
    print("📊 Generating synthetic training data...")
    gen = SyntheticDataGenerator()
    
    train_images, train_texts = gen.generate_synthetic_dataset(1000)
    val_images, val_texts = gen.generate_synthetic_dataset(200)
    
    print(f"✓ Training: {train_images.shape}")
    print(f"✓ Validation: {val_images.shape}\n")
    
    # Prepare labels
    char_to_int = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}
    
    def text_to_labels(texts):
        """Convert texts to label arrays."""
        labels = []
        for text in texts:
            label = [char_to_int.get(c, 0) for c in text]
            label = label[:50]  # Truncate to 50
            label = label + [0] * (50 - len(label))  # Pad
            labels.append(label)
        return np.array(labels, dtype=np.int32)
    
    train_labels = text_to_labels(train_texts)
    val_labels = text_to_labels(val_texts)
    
    print("🏗️  Building model...")
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    num_classes = len(CHARACTERS) + 1
    
    model = build_ocr_model(input_shape, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=ctc_loss,
        metrics=['accuracy']
    )
    
    print(f"✓ Model created with {model.count_params():,} parameters\n")
    
    print("🚀 Training (this may take 1-2 minutes)...")
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=8,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ]
    )
    
    print(f"\n💾 Saving model to {model_path}...")
    model.save(str(model_path))
    print(f"✓ Model saved ({os.path.getsize(model_path)/(1024*1024):.1f} MB)\n")
    
    print("="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\n🎯 Model ready for OCR inference")
    print(f"🌐 Restart the web app: python flask_app.py\n")

if __name__ == '__main__':
    train_model()
