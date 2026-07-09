"""Simple but correct CTC training script for OCR."""

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
from config.config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, CHARACTERS, BATCH_SIZE


MAX_LABEL_LEN = 50


def build_prediction_model(input_shape, num_classes):
    """Build CNN-LSTM model that outputs character probabilities."""
    inputs = keras.Input(shape=input_shape, name='image')

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.Dropout(0.2)(x)

    shape = x.shape
    x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return keras.Model(inputs, outputs, name='OCR_PREDICTOR')


def build_ctc_training_model(prediction_model):
    """Wrap prediction model with CTC loss for training."""
    labels = keras.Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int32')
    input_length = keras.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = keras.Input(name='label_length', shape=(1,), dtype='int32')

    ctc_loss = layers.Lambda(
        lambda tensors: keras.backend.ctc_batch_cost(*tensors),
        name='ctc_loss'
    )([labels, prediction_model.output, input_length, label_length])

    training_model = keras.Model(
        inputs=[prediction_model.input, labels, input_length, label_length],
        outputs=ctc_loss,
        name='OCR_CTC_TRAINER'
    )
    training_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=lambda y_true, y_pred: y_pred)
    return training_model


def encode_texts(texts):
    """Encode text labels to integer sequences with padding."""
    char_to_int = {char: idx for idx, char in enumerate(CHARACTERS)}

    labels = np.zeros((len(texts), MAX_LABEL_LEN), dtype=np.int32)
    lengths = np.zeros((len(texts), 1), dtype=np.int32)

    for row_idx, text in enumerate(texts):
        encoded = [char_to_int[c] for c in text if c in char_to_int][:MAX_LABEL_LEN]
        if encoded:
            labels[row_idx, :len(encoded)] = encoded
        lengths[row_idx, 0] = len(encoded)

    return labels, lengths


def train():
    print("\n" + "=" * 70)
    print("🎯 Training Real OCR Model (CTC)")
    print("=" * 70 + "\n")

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'best_ocr_model.h5'

    print("📊 Generating data...")
    gen = SyntheticDataGenerator()
    train_imgs, train_txts = gen.generate_synthetic_dataset(2200)
    val_imgs, val_txts = gen.generate_synthetic_dataset(400)

    train_labels, train_label_len = encode_texts(train_txts)
    val_labels, val_label_len = encode_texts(val_txts)

    print(f"✓ Train images: {train_imgs.shape}")
    print(f"✓ Val images: {val_imgs.shape}")

    prediction_model = build_prediction_model(
        (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        len(CHARACTERS) + 1
    )
    training_model = build_ctc_training_model(prediction_model)

    train_input_len = np.full((train_imgs.shape[0], 1), prediction_model.output_shape[1], dtype=np.int32)
    val_input_len = np.full((val_imgs.shape[0], 1), prediction_model.output_shape[1], dtype=np.int32)

    y_dummy_train = np.zeros((train_imgs.shape[0], 1), dtype=np.float32)
    y_dummy_val = np.zeros((val_imgs.shape[0], 1), dtype=np.float32)

    print(f"🏗️  Predictor params: {prediction_model.count_params():,}")
    print("🚀 Training (this can take a few minutes)...\n")

    training_model.fit(
        x=[train_imgs, train_labels, train_input_len, train_label_len],
        y=y_dummy_train,
        validation_data=(
            [val_imgs, val_labels, val_input_len, val_label_len],
            y_dummy_val
        ),
        batch_size=BATCH_SIZE,
        epochs=12,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )

    prediction_model.save(str(model_path))
    print("\n" + "=" * 70)
    print("✅ Model trained and saved")
    print("=" * 70)
    print(f"📍 Model: {model_path}")
    print(f"📦 Size: {os.path.getsize(model_path)/(1024*1024):.1f} MB")


if __name__ == '__main__':
    train()
