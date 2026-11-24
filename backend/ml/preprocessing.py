"""
SIGNAL / VISION: Image Preprocessing and Data Augmentation Pipeline.
Provides unified, deterministic transformation pipelines for:
- Classical ML (HOG on grayscale / RGB)
- Deep Learning (Custom CNN and ResNet-18)
- Real-time Inference (FastAPI endpoints)

Strict Safety Invariants:
- NEVER horizontally or vertically flip traffic signs (reverses directional semantics).
- Augmentation applied strictly to training split; validation and test splits are strictly untouched.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as T
from typing import Tuple, Union, Optional

# ImageNet normalization standard (used for ResNet-18 and general transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# GTSRB Empirical statistics across all 39,209 training images
GTSRB_MEAN = [0.3403, 0.3121, 0.3214]
GTSRB_STD = [0.2724, 0.2608, 0.2669]

class AddRandomShadow:
    """Simulates realistic roadway shadow variations across the traffic sign."""
    def __init__(self, shadow_ratio_range=(0.4, 0.8)):
        self.shadow_ratio_range = shadow_ratio_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > 0.4:
            return img
        np_img = np.array(img, dtype=np.float32)
        h, w = np_img.shape[:2]
        
        # Random polygon shadow mask
        mask = np.ones((h, w), dtype=np.float32)
        top_x = np.random.randint(0, w)
        bot_x = np.random.randint(0, w)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Linear boundary
        if np.random.rand() > 0.5:
            mask[cc < (top_x + (bot_x - top_x) * rr / h)] = np.random.uniform(*self.shadow_ratio_range)
        else:
            mask[cc >= (top_x + (bot_x - top_x) * rr / h)] = np.random.uniform(*self.shadow_ratio_range)
            
        np_img = np_img * mask[:, :, np.newaxis]
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

class MildGaussianBlur:
    """Simulates camera defocus / motion blur."""
    def __init__(self, p=0.25, max_radius=1.2):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            radius = np.random.uniform(0.3, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

def get_train_transforms(image_size: int = 48) -> T.Compose:
    """
    Controlled Training Augmentation Pipeline:
    - Small rotation (+/- 12 degrees)
    - Mild random translation (+/- 8%)
    - Mild scaling (0.9 to 1.1)
    - Random perspective distortion (p=0.25)
    - Color jitter: brightness, contrast, saturation
    - Shadow variation & mild blur
    - Normalization
    NO horizontal or vertical flipping!
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomApply([AddRandomShadow()], p=0.4),
        T.RandomApply([MildGaussianBlur(p=0.4)], p=0.3),
        T.RandomRotation(degrees=(-12, 12), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transforms(image_size: int = 48) -> T.Compose:
    """
    Standard deterministic validation, testing, and inference transform:
    - Resize to target size (32 or 48)
    - Convert to Float32 Tensor [0.0, 1.0]
    - Normalize with exact ImageNet mean and std
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_numpy_image(img_bgr_or_rgb: np.ndarray, image_size: int = 48, is_bgr: bool = True) -> np.ndarray:
    """
    Preprocess raw OpenCV numpy array for ML inference.
    Returns normalized float32 tensor of shape (3, image_size, image_size).
    """
    if is_bgr:
        img_rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr_or_rgb.copy()
        
    resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm = (norm - mean) / std
    
    # Transpose to Channel-First (C, H, W)
    chw = np.transpose(norm, (2, 0, 1))
    return chw

def get_pil_preprocessing(image_bytes: bytes, image_size: int = 48) -> Tuple[Image.Image, torch.Tensor]:
    """
    Used by FastAPI inference endpoint to validate, load, and convert uploaded image.
    """
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_eval_transforms(image_size=image_size)
    tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, H, W)
    return img, tensor
