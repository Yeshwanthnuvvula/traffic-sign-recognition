"""
SIGNAL / VISION: Model 3 — Random Forest Classifier.
Features: HOG gradient features scaled with StandardScaler.
Hyperparameter sweeps:
- Number of trees (n_estimators in [50, 100, 200])
- Max tree depth (max_depth in [15, 25, None])
- Minimum samples split (min_samples_split in [2, 5])
- Minimum samples leaf (min_samples_leaf in [1, 2])
"""

import time
import json
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

from backend.ml.dataset import locate_dataset_paths, load_train_samples, get_stratified_train_val_split, GTSRB_CLASSES
from backend.ml.feature_extraction import HOGFeatureExtractor

MODELS_DIR = BASE_DIR / "backend" / "models"
RESULTS_DIR = BASE_DIR / "backend" / "results"
DATA_DIR = BASE_DIR / "data"

def train_random_forest(
    sample_limit: int = 15000,
    n_estimators: int = 100,
    max_depth: int = 25,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
):
    print("==================================================")
    print(f"Training Random Forest (trees={n_estimators}, max_depth={max_depth})")
    print("==================================================")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_dir, _, _ = locate_dataset_paths(DATA_DIR)
    train_samples = load_train_samples(train_dir)
    train_split, val_split = get_stratified_train_val_split(train_samples, val_ratio=0.2, random_state=42)
    
    extractor = HOGFeatureExtractor(image_size=48, orientations=9, pixels_per_cell=8)
    
    scaler_path = MODELS_DIR / "hog_scaler.joblib"
    X_train_raw, y_train = extractor.extract_batch(train_split, max_samples=sample_limit)
    X_val_raw, y_val = extractor.extract_batch(val_split, max_samples=sample_limit // 4)
    
    if scaler_path.exists():
        extractor.load_scaler(scaler_path)
        X_train = extractor.transform(X_train_raw)
    else:
        X_train = extractor.fit_transform(X_train_raw)
        extractor.save_scaler(scaler_path)
    X_val = extractor.transform(X_val_raw)
    
    start_time = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    train_duration = time.time() - start_time
    print(f"Random Forest fit in {train_duration:.2f} seconds.")
    
    # Benchmark Inference Latency
    inf_start = time.time()
    y_pred = clf.predict(X_val)
    inf_duration = (time.time() - inf_start) / len(X_val) * 1000.0  # ms per sample
    
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    
    print(f"\nRandom Forest Validation Results:")
    print(f"  Accuracy:    {acc * 100:.2f}%")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Inference:   {inf_duration:.2f} ms/sample")
    
    cm = confusion_matrix(y_val, y_pred, labels=list(range(43))).tolist()
    report = classification_report(y_val, y_pred, target_names=[GTSRB_CLASSES[i] for i in range(43)], output_dict=True, zero_division=0)
    
    model_path = MODELS_DIR / "rf_model.joblib"
    joblib.dump(clf, model_path)
    
    metrics = {
        "model_name": "Random Forest",
        "model_id": "rf",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        },
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "train_time_seconds": round(train_duration, 2),
        "inference_latency_ms": round(inf_duration, 2),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    with open(RESULTS_DIR / "rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Metrics saved to {RESULTS_DIR / 'rf_metrics.json'}")
    return metrics

if __name__ == "__main__":
    train_random_forest()
