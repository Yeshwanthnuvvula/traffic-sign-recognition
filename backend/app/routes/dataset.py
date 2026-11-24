import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "backend" / "results" / "eda"

router = APIRouter(tags=["Dataset & EDA"])

@router.get("/dataset/summary")
def get_dataset_summary():
    eda_json = RESULTS_DIR / "eda_summary.json"
    if not eda_json.exists():
        raise HTTPException(status_code=404, detail="EDA summary not generated yet.")
        
    with open(eda_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return {
        "dataset_name": data["dataset_name"],
        "total_classes": data["total_classes"],
        "total_train_images": data["total_train_images"],
        "total_test_images": data["total_test_images"],
        "imbalance_metrics": data["imbalance_metrics"],
        "dimension_stats": data["dimension_stats"],
        "photometric_stats": data["photometric_stats"],
        "categories_breakdown": data["categories_breakdown"]
    }

@router.get("/eda")
def get_full_eda():
    eda_json = RESULTS_DIR / "eda_summary.json"
    if not eda_json.exists():
        raise HTTPException(status_code=404, detail="EDA summary not generated yet.")
        
    with open(eda_json, "r", encoding="utf-8") as f:
        return json.load(f)
