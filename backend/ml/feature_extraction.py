"""
SIGNAL / VISION: Feature Extraction Engine (HOG - Histogram of Oriented Gradients).
Extracts dense spatial gradient histograms for classical machine learning classifiers:
- Support Vector Machines (Linear & RBF)
- K-Nearest Neighbors
- Random Forests

Implementation uses skimage.feature.hog with StandardScaler normalization.
"""

import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Union
import joblib
from pathlib import Path

class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) Feature Extractor.
    Default configuration:
    - Target image size: 48x48
    - Orientations: 9
    - Pixels per cell: (8, 8)
    - Cells per block: (2, 2)
    - Block normalization: L2-Hys
    Feature vector length: (48//8 - 1) * (48//8 - 1) * 2 * 2 * 9 = 5 * 5 * 4 * 9 = 900 dimensions.
    """
    def __init__(self, image_size: int = 48, orientations: int = 9, pixels_per_cell: int = 8):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_single(self, img: Image.Image) -> np.ndarray:
        """
        Extracts 1D HOG feature vector for a single PIL image.
        Converts to grayscale and resizes to target window size.
        """
        # Convert to grayscale
        gray = np.array(img.convert("L"))
        if gray.shape[:2] != (self.image_size, self.image_size):
            gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            
        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )
        return features.astype(np.float32)

    def extract_batch(self, samples: List[Tuple[str, int]], max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts HOG feature vectors for a list of (image_path, label) samples.
        Returns X (N, n_features) and y (N,).
        """
        if max_samples and max_samples < len(samples):
            # Deterministic subset for reproducibility
            np.random.seed(42)
            indices = np.random.choice(len(samples), max_samples, replace=False)
            samples = [samples[i] for i in indices]
            
        X_list = []
        y_list = []
        
        for path, label in samples:
            try:
                img = Image.open(path)
                feat = self.extract_single(img)
                X_list.append(feat)
                y_list.append(label)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
                
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits StandardScaler and scales feature matrix X."""
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales feature matrix using previously fitted StandardScaler."""
        if not self.is_fitted:
            raise RuntimeError("HOGFeatureExtractor scaler has not been fitted yet.")
        return self.scaler.transform(X)

    def save_scaler(self, path: Union[str, Path]):
        """Persists fitted StandardScaler to disk."""
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: Union[str, Path]):
        """Loads fitted StandardScaler from disk."""
        self.scaler = joblib.load(path)
        self.is_fitted = True
