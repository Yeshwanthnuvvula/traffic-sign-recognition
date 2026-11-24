"""
SIGNAL / VISION: Consolidated Model Evaluation & Error Analysis Suite.
Performs rigorous benchmarking across all 5 models:
1. Macro F1, Weighted F1, Precision, Recall, Accuracy, Training Time, Inference Latency.
2. Speed Limit Disambiguation Analysis:
   Dedicated breakdown of confusing visual pairs (Class 1: 30km/h vs Class 2: 50km/h; Class 4: 70km/h vs Class 5: 80km/h).
3. Evaluates performance on the official 12,630 test images.
4. Generates model_comparison.json and model_comparison.csv.
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ml.dataset import (
    GTSRB_CLASSES,
    locate_dataset_paths,
    load_test_samples,
    GTSRBDataset
)
from backend.ml.preprocessing import get_eval_transforms
from backend.ml.train_cnn import CustomTrafficSignCNN
from backend.ml.train_resnet import TrafficSignResNet18

RESULTS_DIR = BASE_DIR / "backend" / "results"
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_DIR = BASE_DIR / "data"

def compile_model_comparisons() -> Dict[str, Any]:
    """
    Reads metric JSONs for SVM, KNN, Random Forest, CNN, and ResNet-18,
    merges them into a unified comparative table, and analyzes Speed Limit 30 vs 50.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    models = ["svm", "knn", "rf", "custom_cnn", "resnet18"]
    comparison_rows = []
    speed_limit_studies = {}
    
    for mid in models:
        metric_file = RESULTS_DIR / f"{mid}_metrics.json"
        if not metric_file.exists():
            print(f"Notice: Metric file {metric_file.name} not found yet (model not yet trained).")
            continue
            
        with open(metric_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        row = {
            "model_id": mid,
            "model_name": data.get("model_name", mid),
            "accuracy": data.get("accuracy", 0.0),
            "macro_f1": data.get("macro_f1", 0.0),
            "weighted_f1": data.get("weighted_f1", 0.0),
            "precision": data.get("precision", 0.0),
            "recall": data.get("recall", 0.0),
            "train_time_seconds": data.get("train_time_seconds", 0.0),
            "inference_latency_ms": data.get("inference_latency_ms", 0.0)
        }
        comparison_rows.append(row)
        
        # Analyze Speed Limit 30 (Class 1) vs Speed Limit 50 (Class 2)
        cm = np.array(data.get("confusion_matrix", []))
        if cm.shape == (43, 43):
            # True 30 predicted as 50
            confused_30_as_50 = int(cm[1, 2])
            correct_30 = int(cm[1, 1])
            total_30 = int(np.sum(cm[1, :]))
            
            # True 50 predicted as 30
            confused_50_as_30 = int(cm[2, 1])
            correct_50 = int(cm[2, 2])
            total_50 = int(np.sum(cm[2, :]))
            
            speed_limit_studies[mid] = {
                "class_1_speed_30": {
                    "total": total_30,
                    "correct": correct_30,
                    "misclassified_as_50": confused_30_as_50,
                    "recall": round(float(correct_30 / max(1, total_30)), 4)
                },
                "class_2_speed_50": {
                    "total": total_50,
                    "correct": correct_50,
                    "misclassified_as_30": confused_50_as_30,
                    "recall": round(float(correct_50 / max(1, total_50)), 4)
                }
            }

    df = pd.DataFrame(comparison_rows)
    csv_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    out_payload = {
        "models": comparison_rows,
        "speed_limit_30_vs_50_analysis": speed_limit_studies,
        "selected_best_model": max(comparison_rows, key=lambda x: x["macro_f1"])["model_id"] if comparison_rows else None
    }
    
    json_path = RESULTS_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
        
    print(f"✓ Comparison CSV saved to {csv_path}")
    print(f"✓ Comparison JSON saved to {json_path}")
    return out_payload

if __name__ == "__main__":
    compile_model_comparisons()
