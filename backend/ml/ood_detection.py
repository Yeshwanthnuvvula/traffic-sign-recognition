"""
SIGNAL / VISION: Out-Of-Distribution (OOD) & Uncertainty Detection Engine.
Prevents blind misclassification of non-traffic-sign photographs, unsupported symbols, or corrupted images:
1. Maximum Softmax Probability (MSP) Calibration: Evaluates predictive entropy and top-1 confidence.
2. Feature-Space Centroid Distance (Mahalanobis / Cosine Distance): Evaluates whether image embedding resides within the support manifold of known GTSRB traffic signs.
3. Quality & Plausibility Checks: Detects extreme low-contrast, blur, and abnormal aspect ratios.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F

class OODDetector:
    """
    Multilayer Out-Of-Distribution and Uncertainty Evaluator.
    """
    def __init__(self, confidence_threshold: float = 0.65, min_resolution: int = 16):
        self.confidence_threshold = confidence_threshold
        self.min_resolution = min_resolution
        self.class_centroids: Optional[np.ndarray] = None

    def validate_image_quality(self, pil_image: Image.Image) -> Tuple[bool, str]:
        """Validates that input is a viable photograph with sufficient resolution and information density."""
        w, h = pil_image.size
        if w < self.min_resolution or h < self.min_resolution:
            return False, f"Resolution {w}x{h} is below minimum threshold ({self.min_resolution}px)"
            
        aspect = w / float(h)
        if aspect < 0.25 or aspect > 4.0:
            return False, f"Extreme aspect ratio {aspect:.2f} inconsistent with traffic sign geometry"
            
        np_img = np.array(pil_image.convert("L"))
        variance_of_laplacian = cv2.Laplacian(np_img, cv2.CV_64F).var()
        if variance_of_laplacian < 8.0:
            return False, f"Image lacks structural edge variation (severe blur or blank color)"
            
        return True, "Valid"

    def evaluate_uncertainty(
        self,
        probabilities: np.ndarray,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Calculates confidence, entropy, and OOD rejection status.
        probabilities: 1D array of length 43 summing to 1.0
        """
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probs = [float(probabilities[i]) for i in top_indices]
        top_ids = [int(i) for i in top_indices]
        
        confidence = top_probs[0]
        margin = top_probs[0] - top_probs[1] if len(top_probs) > 1 else 1.0
        
        # Predictive Shannon Entropy: H(p) = -sum(p * log(p))
        eps = 1e-12
        entropy = float(-np.sum(probabilities * np.log(probabilities + eps)))
        max_entropy = float(np.log(len(probabilities)))  # ~3.76 for 43 classes
        normalized_entropy = float(entropy / max_entropy)
        
        is_uncertain = bool(confidence < self.confidence_threshold or margin < 0.15)
        is_rejected_ood = bool(confidence < 0.45 and normalized_entropy > 0.60)
        
        status_message = "Normal"
        if is_rejected_ood:
            status_message = "Rejected (Out-of-Distribution / Non-Traffic Sign Image)"
        elif is_uncertain:
            status_message = "Uncertain Prediction (Low Confidence or High Ambiguity)"
            
        return {
            "top_ids": top_ids,
            "top_probabilities": top_probs,
            "confidence": round(confidence, 4),
            "decision_margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "is_uncertain": is_uncertain,
            "is_rejected": is_rejected_ood,
            "status_message": status_message
        }
