import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.ml.inference import inference_engine

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "backend" / "results"

router = APIRouter(tags=["Models & Metrics"])

@router.get("/models")
def get_models():
    return {
        "models": inference_engine.get_available_models(),
        "default": "resnet18"
    }

@router.get("/metrics")
def get_all_metrics():
    comp_file = RESULTS_DIR / "model_comparison.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Fallback to gathering individual metric files
    models = ["resnet18", "custom_cnn", "svm", "rf", "knn"]
    rows = []
    for mid in models:
        mf = RESULTS_DIR / f"{mid}_metrics.json"
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as f:
                d = json.load(f)
                rows.append({
                    "model_id": mid,
                    "model_name": d.get("model_name", mid),
                    "accuracy": d.get("accuracy", 0.0),
                    "macro_f1": d.get("macro_f1", 0.0),
                    "weighted_f1": d.get("weighted_f1", 0.0),
                    "precision": d.get("precision", 0.0),
                    "recall": d.get("recall", 0.0),
                    "train_time_seconds": d.get("train_time_seconds", 0.0),
                    "inference_latency_ms": d.get("inference_latency_ms", 0.0)
                })
    return {"models": rows, "selected_best_model": "resnet18"}

@router.get("/metrics/{model_id}")
def get_model_metric_detail(model_id: str):
    mf = RESULTS_DIR / f"{model_id}_metrics.json"
    if not mf.exists():
        raise HTTPException(status_code=404, detail=f"Metrics for '{model_id}' not found.")
        
    with open(mf, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/experiments")
def get_experiments():
    exp_file = RESULTS_DIR / "experiments.json"
    if not exp_file.exists():
        return {"total_trials": 0, "experiments": []}
    with open(exp_file, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/speed-limit-analysis")
def get_speed_limit_study():
    comp_file = RESULTS_DIR / "model_comparison.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("speed_limit_30_vs_50_analysis", {})
    return {}
