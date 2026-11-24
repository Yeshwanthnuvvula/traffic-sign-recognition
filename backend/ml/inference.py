"""
SIGNAL / VISION: Unified Multi-Model Inference Engine.
Loads model checkpoints (SVM, KNN, Random Forest, Custom CNN, ResNet-18),
executes deterministic preprocessing, runs forward inference, and calculates
confidence, top-3 candidates, latency, and OOD uncertainty.
"""

import time
import io
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ml.dataset import GTSRB_CLASSES, get_class_category
from backend.ml.preprocessing import get_eval_transforms
from backend.ml.feature_extraction import HOGFeatureExtractor
from backend.ml.train_cnn import CustomTrafficSignCNN
from backend.ml.train_resnet import TrafficSignResNet18
from backend.ml.ood_detection import OODDetector

MODELS_DIR = BASE_DIR / "backend" / "models"

class InferenceEngine:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ood_detector = OODDetector(confidence_threshold=0.60)
        self.hog_extractor = HOGFeatureExtractor(image_size=48)
        self.scaler = None
        
        # Cached model instances
        self.loaded_models: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self):
        # Load Scaler if exists
        scaler_p = MODELS_DIR / "hog_scaler.joblib"
        if scaler_p.exists():
            self.hog_extractor.load_scaler(scaler_p)

    def get_available_models(self) -> List[Dict[str, str]]:
        models = []
        if (MODELS_DIR / "resnet18.pth").exists():
            models.append({"id": "resnet18", "name": "ResNet-18 (Deep Residual)", "type": "Deep Learning"})
        if (MODELS_DIR / "custom_cnn.pth").exists():
            models.append({"id": "custom_cnn", "name": "Custom TrafficSignCNN", "type": "Deep Learning"})
        if (MODELS_DIR / "svm_model.joblib").exists():
            models.append({"id": "svm", "name": "Support Vector Machine (SVM)", "type": "Classical ML"})
        if (MODELS_DIR / "rf_model.joblib").exists():
            models.append({"id": "rf", "name": "Random Forest", "type": "Ensemble ML"})
        if (MODELS_DIR / "knn_model.joblib").exists():
            models.append({"id": "knn", "name": "K-Nearest Neighbors (KNN)", "type": "Classical ML"})
        return models

    def load_model(self, model_id: str):
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        if model_id == "resnet18":
            pth = MODELS_DIR / "resnet18.pth"
            if not pth.exists():
                raise FileNotFoundError("ResNet-18 model weights not found. Please train model first.")
            net = TrafficSignResNet18(num_classes=43, pretrained=False).to(self.device)
            ckpt = torch.load(pth, map_location=self.device)
            net.load_state_dict(ckpt["model_state_dict"])
            net.eval()
            self.loaded_models[model_id] = net
            return net

        elif model_id == "custom_cnn":
            pth = MODELS_DIR / "custom_cnn.pth"
            if not pth.exists():
                raise FileNotFoundError("Custom CNN weights not found. Please train model first.")
            net = CustomTrafficSignCNN(num_classes=43).to(self.device)
            ckpt = torch.load(pth, map_location=self.device)
            net.load_state_dict(ckpt["model_state_dict"])
            net.eval()
            self.loaded_models[model_id] = net
            return net

        elif model_id in ["svm", "knn", "rf"]:
            pth = MODELS_DIR / f"{model_id}_model.joblib"
            if not pth.exists():
                raise FileNotFoundError(f"{model_id.upper()} model not found. Please train model first.")
            clf = joblib.load(pth)
            self.loaded_models[model_id] = clf
            return clf
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

    def predict(self, image_input: Any, model_id: str = "resnet18") -> Dict[str, Any]:
        """
        Runs full inference pipeline:
        1. Image validation & Quality check
        2. Model loading & Feature forward pass
        3. Probability distribution extraction
        4. OOD & Uncertainty detection
        5. Top 3 candidate assembly
        """
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError("Expected PIL Image or bytes.")

        # Quality & Plausibility Validation
        is_valid, quality_msg = self.ood_detector.validate_image_quality(pil_img)
        w, h = pil_img.size
        
        t0 = time.time()
        model = self.load_model(model_id)
        
        if model_id in ["custom_cnn", "resnet18"]:
            transform = get_eval_transforms(image_size=48)
            tensor = transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = model(tensor)
                probs_tensor = F.softmax(logits, dim=1)
                probabilities = probs_tensor.cpu().numpy()[0]
        else:
            # Classical HOG pipeline
            features = self.hog_extractor.extract_single(pil_img)
            scaled = self.hog_extractor.transform(features.reshape(1, -1))
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(scaled)[0]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(scaled)[0]
                exp_df = np.exp(df - np.max(df))
                probabilities = exp_df / np.sum(exp_df)
            else:
                pred = model.predict(scaled)[0]
                probabilities = np.zeros(43)
                probabilities[pred] = 1.0

        latency_ms = (time.time() - t0) * 1000.0
        
        # Uncertainty Analysis
        uncertainty = self.ood_detector.evaluate_uncertainty(probabilities, top_k=3)
        
        if not is_valid:
            uncertainty["is_uncertain"] = True
            uncertainty["is_rejected"] = True
            uncertainty["status_message"] = f"Image Quality Issue: {quality_msg}"

        top_predictions = []
        for class_id, prob in zip(uncertainty["top_ids"], uncertainty["top_probabilities"]):
            top_predictions.append({
                "class_id": class_id,
                "class_name": GTSRB_CLASSES[class_id],
                "category": get_class_category(class_id),
                "probability": round(prob, 4),
                "percentage": round(prob * 100.0, 2)
            })

        predicted_primary = top_predictions[0]

        return {
            "predicted_class_id": predicted_primary["class_id"],
            "predicted_class_name": predicted_primary["class_name"],
            "category": predicted_primary["category"],
            "confidence": round(uncertainty["confidence"], 4),
            "confidence_percentage": round(uncertainty["confidence"] * 100.0, 2),
            "top_3_predictions": top_predictions,
            "inference_time_ms": round(latency_ms, 2),
            "model_id": model_id,
            "uncertainty": uncertainty,
            "image_metadata": {
                "width": w,
                "height": h,
                "aspect_ratio": round(w / float(h), 2),
                "channels": 3,
                "is_valid_quality": is_valid,
                "quality_note": quality_msg
            }
        }

# Global singleton
inference_engine = InferenceEngine()
