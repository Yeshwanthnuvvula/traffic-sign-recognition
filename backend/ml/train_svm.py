"""
SIGNAL / VISION: Model 1 — Support Vector Machine (SVM) Classifier.
Features: Dense HOG (Histogram of Oriented Gradients) with StandardScaler.
Kernels evaluated: Linear and Radial Basis Function (RBF).
"""

import time
import json
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

from backend.ml.dataset import locate_dataset_paths, load_train_samples, get_stratified_train_val_split, GTSRB_CLASSES
from backend.ml.feature_extraction import HOGFeatureExtractor

MODELS_DIR = BASE_DIR / "backend" / "models"
RESULTS_DIR = BASE_DIR / "backend" / "results"
DATA_DIR = BASE_DIR / "data"

def train_svm(sample_limit: int = 12000, kernel: str = "rbf", C: float = 10.0, gamma: str = "scale"):
    print("==================================================")
    print(f"Training Support Vector Machine (Kernel={kernel}, C={C}, gamma={gamma})")
    print("==================================================")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dir, _, _ = locate_dataset_paths(DATA_DIR)
    train_samples = load_train_samples(train_dir)
    train_split, val_split = get_stratified_train_val_split(train_samples, val_ratio=0.2, random_state=42)

    extractor = HOGFeatureExtractor(image_size=48, orientations=9, pixels_per_cell=8)
    print(f"Extracting HOG features for training (subset: {min(sample_limit, len(train_split))})...")
    X_train_raw, y_train = extractor.extract_batch(train_split, max_samples=sample_limit)
    X_val_raw, y_val = extractor.extract_batch(val_split, max_samples=sample_limit // 4)

    print(f"Feature matrix dimensions: Train {X_train_raw.shape}, Val {X_val_raw.shape}")
    X_train = extractor.fit_transform(X_train_raw)
    X_val = extractor.transform(X_val_raw)
    extractor.save_scaler(MODELS_DIR / "hog_scaler.joblib")

    start_time = time.time()
    if kernel == "linear":
        base_clf = LinearSVC(C=C, max_iter=3000, random_state=42, dual=False)
        clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    else:
        base_clf = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
        clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")

    print(f"Fitting {kernel.upper()} SVM classifier with probability calibration...")
    clf.fit(X_train, y_train)
    train_duration = time.time() - start_time
    print(f"SVM training complete in {train_duration:.2f}s.")

    inf_start = time.time()
    y_pred = clf.predict(X_val)
    inf_duration = (time.time() - inf_start) / len(X_val) * 1000.0

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\nSVM Validation Results:")
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Inference:   {inf_duration:.4f} ms/sample")

    cm = confusion_matrix(y_val, y_pred, labels=list(range(43))).tolist()
    report = classification_report(y_val, y_pred, target_names=[GTSRB_CLASSES[i] for i in range(43)], output_dict=True, zero_division=0)

    model_path = MODELS_DIR / "svm_model.joblib"
    joblib.dump(clf, model_path)

    metrics = {
        "model_name": "Support Vector Machine (SVM)",
        "model_id": "svm",
        "kernel": kernel,
        "hyperparameters": {"C": C, "gamma": gamma, "kernel": kernel},
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "feature_dim": int(X_train.shape[1]),
        "train_time_seconds": round(train_duration, 2),
        "inference_latency_ms": round(inf_duration, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "confusion_matrix": cm,
        "classification_report": report
    }

    with open(RESULTS_DIR / "svm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Model saved to {model_path}")
    print(f"✓ Metrics saved to {RESULTS_DIR / 'svm_metrics.json'}")
    return metrics

if __name__ == "__main__":
    train_svm()
