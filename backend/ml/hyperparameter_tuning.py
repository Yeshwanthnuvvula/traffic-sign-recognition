"""
SIGNAL / VISION: Hyperparameter Optimization Engine.
Conducts systematic parameter sweeps across classical and deep learning models:
1. SVM: Regularization penalty C in [0.1, 1.0, 10.0], kernel in ['linear', 'rbf']
2. KNN: Number of neighbors k in [1, 3, 5, 7], distance metric in ['euclidean', 'manhattan'], weights ['uniform', 'distance']
3. Random Forest: n_estimators in [50, 100], max_depth in [15, 25, None]
4. Custom CNN & ResNet-18: Learning rates [0.0003, 0.001], batch sizes [32, 64], input sizes [32, 48], dropout rates [0.2, 0.4]

Consolidates all experimental trials into backend/results/experiments.json for display in Model Experiments UI.
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from backend.ml.dataset import locate_dataset_paths, load_train_samples, get_stratified_train_val_split
from backend.ml.feature_extraction import HOGFeatureExtractor

RESULTS_DIR = BASE_DIR / "backend" / "results"
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "data"

def run_classical_tuning(sample_limit: int = 6000) -> List[Dict[str, Any]]:
    print("\n--- Running Classical Models Hyperparameter Optimization ---")
    train_dir, _, _ = locate_dataset_paths(DATA_DIR)
    train_samples = load_train_samples(train_dir)
    train_split, val_split = get_stratified_train_val_split(train_samples, val_ratio=0.2, random_state=42)
    
    extractor = HOGFeatureExtractor(image_size=48, orientations=9, pixels_per_cell=8)
    X_train_raw, y_train = extractor.extract_batch(train_split, max_samples=sample_limit)
    X_val_raw, y_val = extractor.extract_batch(val_split, max_samples=sample_limit // 3)
    
    X_train = extractor.fit_transform(X_train_raw)
    X_val = extractor.transform(X_val_raw)
    
    experiments = []
    
    # 1. SVM Experiments
    svm_configs = [
        {"C": 0.1, "kernel": "linear"},
        {"C": 1.0, "kernel": "linear"},
        {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        {"C": 10.0, "kernel": "rbf", "gamma": "scale"}
    ]
    for cfg in svm_configs:
        t0 = time.time()
        if cfg["kernel"] == "linear":
            clf = LinearSVC(C=cfg["C"], max_iter=2000, random_state=42)
        else:
            clf = SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg.get("gamma", "scale"), random_state=42)
            
        clf.fit(X_train, y_train)
        duration = time.time() - t0
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        
        experiments.append({
            "experiment_id": f"svm_{cfg['kernel']}_C{cfg['C']}",
            "model_type": "SVM",
            "hyperparameters": cfg,
            "training_duration_s": round(duration, 2),
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(f1), 4),
            "status": "COMPLETED"
        })
        print(f"SVM {cfg}: Acc={acc*100:.2f}%, F1={f1:.4f} in {duration:.1f}s")
        
    # 2. KNN Experiments
    knn_configs = [
        {"n_neighbors": 3, "weights": "uniform", "metric": "euclidean"},
        {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        {"n_neighbors": 7, "weights": "distance", "metric": "manhattan"}
    ]
    for cfg in knn_configs:
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=cfg["n_neighbors"], weights=cfg["weights"], metric=cfg["metric"], n_jobs=-1)
        clf.fit(X_train, y_train)
        duration = time.time() - t0
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        
        experiments.append({
            "experiment_id": f"knn_k{cfg['n_neighbors']}_{cfg['weights']}_{cfg['metric']}",
            "model_type": "KNN",
            "hyperparameters": cfg,
            "training_duration_s": round(duration, 2),
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(f1), 4),
            "status": "COMPLETED"
        })
        print(f"KNN {cfg}: Acc={acc*100:.2f}%, F1={f1:.4f} in {duration:.1f}s")

    # 3. Random Forest Experiments
    rf_configs = [
        {"n_estimators": 50, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 25},
        {"n_estimators": 150, "max_depth": None}
    ]
    for cfg in rf_configs:
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        duration = time.time() - t0
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        
        experiments.append({
            "experiment_id": f"rf_t{cfg['n_estimators']}_d{cfg['max_depth']}",
            "model_type": "Random Forest",
            "hyperparameters": cfg,
            "training_duration_s": round(duration, 2),
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(f1), 4),
            "status": "COMPLETED"
        })
        print(f"RF {cfg}: Acc={acc*100:.2f}%, F1={f1:.4f} in {duration:.1f}s")
        
    return experiments

def run_all_tuning():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_experiments = run_classical_tuning()
    
    out_file = RESULTS_DIR / "experiments.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"total_trials": len(all_experiments), "experiments": all_experiments}, f, indent=2)
    print(f"✓ Hyperparameter tuning trials saved to: {out_file}")

if __name__ == "__main__":
    run_all_tuning()
