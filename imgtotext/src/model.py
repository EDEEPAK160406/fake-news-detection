"""
CNN-LSTM-CTC model for text recognition/OCR.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Reshape, Dense, Dropout, 
    LSTM, Bidirectional, LayerNormalization, Input
)
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class CNNLSTMCTCModel:
    """CNN-LSTM-CTC model for OCR text recognition."""
    
    def __init__(self, input_shape: Tuple[int, int, int], 
                 num_classes: int, 
                 rnn_units: int = 256):
        """
        Initialize the OCR model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of character classes + 1 (for CTC blank)
            rnn_units: Number of LSTM units
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rnn_units = rnn_units
        self.model = None
        
    def build_model(self) -> Model:
        """
        Build the CNN-LSTM-CTC model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        
        # CNN Feature Extraction
        x = self._build_cnn_block(inputs)
        
        # Reshape for RNN input
        # From (batch, height, width, channels) to (batch, width, features)
        x = layers.Reshape((-1, self._get_cnn_output_channels(x)))(x)
        
        # Bidirectional LSTM for sequence modeling
        x = self._build_lstm_block(x)
        
        # Dense layer to map to character classes
        x = Dense(self.num_classes + 1, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        logger.info("Model architecture built successfully")
        
        return self.model
    
    def _build_cnn_block(self, inputs) -> layers.Layer:
        """Build CNN feature extraction block."""
        x = inputs
        
        # Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = LayerNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = LayerNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = LayerNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((1, 2))(x)
        x = Dropout(0.25)(x)
        
        # Block 4
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = LayerNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((1, 2))(x)
        x = Dropout(0.25)(x)
        
        return x
    
    def _get_cnn_output_channels(self, tensor) -> int:
        """Get output channels from CNN block."""
        return tensor.shape[-1]
    
    def _build_lstm_block(self, x) -> layers.Layer:
        """Build bidirectional LSTM sequence modeling block."""
        
        # Bidirectional LSTM - forward and backward
        x = Bidirectional(LSTM(self.rnn_units, return_sequences=True, 
                              dropout=0.25))(x)
        x = LayerNormalization()(x)
        
        # Second LSTM layer for deeper sequence modeling
        x = Bidirectional(LSTM(self.rnn_units, return_sequences=True, 
                              dropout=0.25))(x)
        x = LayerNormalization()(x)
        
        return x
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """
        Compile model with CTC loss and appropriate metrics.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.ctc_loss,
            metrics=[self.character_error_rate]
        )
        logger.info(f"Model compiled with learning rate: {learning_rate}")
    
    @staticmethod
    def ctc_loss(y_true, y_pred):
        """
        Connectionist Temporal Classification (CTC) loss.
        Handles variable-length sequences.
        """
        batch_size = tf.shape(y_pred)[0]
        max_label_length = tf.shape(y_true)[1]
        input_length = tf.shape(y_pred)[1]
        
        # Reduce input length by 2 (due to conv pooling)
        input_length = input_length * tf.ones(shape=(batch_size,), dtype='int64')
        label_length = tf.reduce_sum(tf.cast(y_true > 0, 'int64'), axis=1)
        
        return keras.backend.ctc_batch_cost(
            y_true, y_pred, input_length, label_length
        )
    
    @staticmethod
    def character_error_rate(y_true, y_pred):
        """Calculate character error rate metric."""
        # Decode predictions using CTC greedy decoder
        _, decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(y_pred, [1, 0, 2]), 
            sequence_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        )
        
        # Cast sparse tensor to dense
        decoded = tf.sparse.to_dense(decoded[0], default_value=-1)
        
        # Calculate error rate
        diff = tf.cast(
            tf.not_equal(y_true[:, :tf.shape(decoded)[1]], decoded), 
            tf.float32
        )
        
        return tf.reduce_mean(diff)
    
    def get_model(self) -> Model:
        """Get the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()


class AttentionMechanism(layers.Layer):
    """Attention mechanism for sequence-to-sequence models."""
    
    def __init__(self, units: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    
    def call(self, query, values):
        """
        Calculate attention weights and context.
        
        Args:
            query: Query vector (decoder state)
            values: Encoder outputs
            
        Returns:
            Context vector and attention weights
        """
        # Add dimensions for broadcasting
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Attention score
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))
        
        # Normalize attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class TransformerOCRModel:
    """Transformer-based OCR model (alternative to LSTM)."""
    
    def __init__(self, input_shape: Tuple[int, int, int],
                 num_classes: int,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4):
        """
        Initialize Transformer OCR model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of character classes
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
    
    def build_model(self) -> Model:
        """Build Transformer-based OCR model."""
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = self._build_cnn_backbone(inputs)
        
        # Reshape for transformer
        x = layers.Reshape((-1, self.embed_dim))(x)
        
        # Positional encoding
        x = x + self._positional_encoding(x)
        
        # Transformer encoder
        for _ in range(self.num_layers):
            x = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads
            )(x, x)
            x = layers.LayerNormalization()(x)
            x = layers.Dense(self.embed_dim * 4, activation='relu')(x)
            x = layers.Dense(self.embed_dim)(x)
            x = layers.LayerNormalization()(x)
        
        # Output layer
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        logger.info("Transformer OCR model built successfully")
        
        return self.model
    
    def _build_cnn_backbone(self, inputs) -> layers.Layer:
        """Build CNN backbone for feature extraction."""
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(self.embed_dim, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        return x
    
    @staticmethod
    def _positional_encoding(x) -> tf.Tensor:
        """Add positional encoding to embeddings."""
        seq_length = tf.shape(x)[1]
        d_model = x.shape[-1]
        
        pos = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        
        positional_encoding = pos * angle_rates
        positional_encoding = tf.concat([
            tf.sin(positional_encoding[:, 0::2]),
            tf.cos(positional_encoding[:, 1::2])
        ], axis=-1)
        
        return positional_encoding[tf.newaxis, :, :]
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile Transformer model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
