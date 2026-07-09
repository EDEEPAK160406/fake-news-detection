"""
Image preprocessing module for OCR.
Handles resizing, grayscale conversion, noise removal, and normalization.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses images for OCR model input."""
    
    def __init__(self, width: int = 256, height: int = 32, 
                 normalize_method: str = "mean_std"):
        """
        Initialize image preprocessor.
        
        Args:
            width: Target image width
            height: Target image height
            normalize_method: Normalization method ('mean_std', 'minmax', 'robust')
        """
        self.width = width
        self.height = height
        self.normalize_method = normalize_method
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Step 1: Convert to grayscale
        image = self._to_grayscale(image)
        
        # Step 2: Denoise
        image = self._denoise(image)
        
        # Step 3: Binarization (optional but helps with degraded images)
        image = self._binarize(image)
        
        # Step 4: Resize to target dimensions
        image = self._resize(image)
        
        # Step 5: Normalize
        image = self._normalize(image)
        
        # Step 6: Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        return image
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            # Assume BGR format
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _denoise(self, image: np.ndarray, strength: int = 9) -> np.ndarray:
        """
        Remove noise from image using bilateral filtering.
        Better than Gaussian blur for edge preservation.
        """
        # Bilateral filter: effective at noise removal while preserving edges
        denoised = cv2.bilateralFilter(image, strength, 75, 75)
        
        # Apply morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize image using Otsu's thresholding.
        Effective for handling different lighting conditions.
        """
        # Apply adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        return binary
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        Maintains aspect ratio by padding.
        """
        h, w = image.shape
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(self.width / w, self.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create canvas with target dimensions (white background)
        canvas = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Center the resized image on canvas
        y_offset = (self.height - new_h) // 2
        x_offset = (self.width - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        Options: mean_std, minmax, robust
        """
        image = image.astype(np.float32)
        
        if self.normalize_method == "mean_std":
            # Standardization: (x - mean) / std
            mean = np.mean(image)
            std = np.std(image)
            normalized = (image - mean) / (std + 1e-7)
            
        elif self.normalize_method == "minmax":
            # Min-Max normalization: (x - min) / (max - min)
            min_val = np.min(image)
            max_val = np.max(image)
            normalized = (image - min_val) / (max_val - min_val + 1e-7)
            
        elif self.normalize_method == "robust":
            # Robust scaling using quantiles
            q1 = np.percentile(image, 25)
            q3 = np.percentile(image, 75)
            iqr = q3 - q1
            normalized = (image - q1) / (iqr + 1e-7)
            
        else:
            normalized = image / 255.0
            
        return normalized
    
    def batch_preprocess(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays
            
        Returns:
            Batch of preprocessed images (N, H, W, C)
        """
        processed_images = []
        for image in images:
            try:
                processed = self.preprocess(image)
                processed_images.append(processed)
            except Exception as e:
                logger.warning(f"Error preprocessing image: {e}")
                continue
                
        return np.array(processed_images)


class DataAugmentation:
    """Image augmentation for improved model robustness."""
    
    @staticmethod
    def rotate(image: np.ndarray, angle_range: Tuple[int, int] = (-10, 10)) -> np.ndarray:
        """Randomly rotate image within angle range."""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    @staticmethod
    def shift(image: np.ndarray, shift_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """Randomly shift image horizontally or vertically."""
        h, w = image.shape[:2]
        dx = int(np.random.uniform(shift_range[0], shift_range[1]) * w)
        dy = int(np.random.uniform(shift_range[0], shift_range[1]) * h)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, matrix, (w, h), 
                                 borderMode=cv2.BORDER_REFLECT)
        return shifted
    
    @staticmethod
    def elastic_deformation(image: np.ndarray, alpha: float = 20, sigma: float = 3) -> np.ndarray:
        """Apply elastic deformation to simulate handwriting variations."""
        from scipy.ndimage import map_coordinates
        from scipy.ndimage import gaussian_filter
        
        h, w = image.shape[:2]
        
        # Create random displacement fields
        dx = gaussian_filter(np.random.uniform(-1, 1, image.shape) * alpha, sigma)
        dy = gaussian_filter(np.random.uniform(-1, 1, image.shape) * alpha, sigma)
        
        # Create coordinate arrays
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        coordinates = np.array([y + dy, x + dx])
        
        # Interpolate using map_coordinates
        deformed = map_coordinates(image, coordinates, order=1, mode='reflect')
        
        return deformed.astype(image.dtype)
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian random noise to image."""
        noise = np.random.normal(0, 1, image.shape)
        noisy = image + (noise_level * 255 * noise)
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(image.dtype)
    
    @staticmethod
    def blur(image: np.ndarray, kernel_size_range: Tuple[int, int] = (3, 5)) -> np.ndarray:
        """Apply Gaussian blur to simulate focus variations."""
        kernel_size = np.random.choice(range(kernel_size_range[0], 
                                             kernel_size_range[1] + 1, 2))
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    @staticmethod
    def augment(image: np.ndarray, augmentation_config: dict) -> np.ndarray:
        """Apply random augmentations based on config."""
        if augmentation_config.get("rotation", False) and np.random.random() > 0.5:
            image = DataAugmentation.rotate(image)
        
        if augmentation_config.get("shifts", False) and np.random.random() > 0.5:
            image = DataAugmentation.shift(image)
        
        if augmentation_config.get("elastic_deformation", False) and np.random.random() > 0.5:
            image = DataAugmentation.elastic_deformation(image)
        
        if augmentation_config.get("random_noise", False) and np.random.random() > 0.5:
            image = DataAugmentation.add_noise(image)
        
        if augmentation_config.get("blur", False) and np.random.random() > 0.5:
            image = DataAugmentation.blur(image)
        
        return image
