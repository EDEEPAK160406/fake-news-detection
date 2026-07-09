"""
Configuration file for the Image-to-Text OCR module.
"""

# Image preprocessing parameters
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 1  # Grayscale

# Character set for recognition
CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-"
NUM_CLASSES = len(CHARACTERS) + 1  # +1 for CTC blank token

# CNN Architecture
CNN_FILTERS = [32, 64, 128, 256]
CNN_KERNEL_SIZES = [(3, 3), (3, 3), (3, 3), (3, 3)]
CNN_POOL_SIZES = [(2, 2), (2, 2), (2, 2), (2, 2)]

# RNN Architecture
RNN_UNITS = 256
RNN_TYPE = "lstm"  # or "gru"

# CTC Loss parameters
CTC_LOSS_TYPE = "ctc"  # CTC decoding
CTCBeamSearchDecoder_BEAM_WIDTH = 50

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5

# Data augmentation
AUGMENT_ROTATION = True
AUGMENT_SHIFTS = True
AUGMENT_ELASTIC_DEFORMATION = True
AUGMENT_RANDOM_NOISE = True
AUGMENT_BLUR = True

# Preprocessing
DENOISE_METHOD = "bilateral"  # or "median", "morphological"
DENOISE_STRENGTH = 9
NORMALIZE_METHOD = "mean_std"  # or "minmax", "robust"

# Model paths
MODEL_DIR = "models"
BEST_MODEL_NAME = "best_ocr_model.h5"
CHECKPOINT_DIR = "models/checkpoints"

# Logging
LOG_DIR = "logs"
LOG_LEVEL = "INFO"

# Testing/Inference
CONFIDENCE_THRESHOLD = 0.5
MAX_TEXT_LENGTH = 50
