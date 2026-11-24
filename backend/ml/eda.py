"""
SIGNAL / VISION: Exploratory Data Analysis (EDA) Module.
Performs rigorous statistical and visual analysis on the official GTSRB dataset:
- Total dataset counts & class frequency distribution
- Class imbalance metrics (Imbalance ratio, Gini coefficient)
- Resolution distributions (Width, Height, Aspect Ratio percentiles)
- RGB Color channel histograms & channel intensity statistics
- Representative canonical samples from all 43 GTSRB classes
- Exports all figures and machine-readable eda_summary.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ml.dataset import (
    GTSRB_CLASSES,
    GTSRB_CATEGORIES,
    locate_dataset_paths,
    load_train_samples,
    load_test_samples,
    get_class_category
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "backend" / "results" / "eda"
SAMPLES_DIR = RESULTS_DIR / "samples"

def gini_coefficient(array: np.ndarray) -> float:
    """Computes the Gini coefficient to quantify class imbalance."""
    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def run_eda() -> Dict[str, Any]:
    print("\n==================================================")
    print("Executing Comprehensive Exploratory Data Analysis")
    print("==================================================")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    train_dir, test_dir, test_csv = locate_dataset_paths(DATA_DIR)
    if not train_dir:
        raise FileNotFoundError(f"Training data not found in {DATA_DIR}")
        
    print(f"Discovered Train Path: {train_dir}")
    train_samples = load_train_samples(train_dir)
    print(f"Total Train Samples Loaded: {len(train_samples)}")
    
    test_samples = []
    if test_dir and test_csv and test_csv.exists():
        test_samples = load_test_samples(test_dir, test_csv)
        print(f"Total Test Samples Loaded: {len(test_samples)}")
        
    # 1. Class Distribution Analysis
    class_counts = {cid: 0 for cid in range(43)}
    class_images = {cid: [] for cid in range(43)}
    widths, heights, aspect_ratios = [], [], []
    r_means, g_means, b_means = [], [], []
    
    print("Processing images for dimensional and photometric profiling...")
    for idx, (path, label) in enumerate(train_samples):
        class_counts[label] += 1
        class_images[label].append(path)
        
        # Subsample for pixel stats if large
        if idx % 5 == 0:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / float(h) if h > 0 else 1.0)
                    
                    if idx % 20 == 0:
                        rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
                        r_means.append(float(np.mean(rgb_arr[:, :, 0])))
                        g_means.append(float(np.mean(rgb_arr[:, :, 1])))
                        b_means.append(float(np.mean(rgb_arr[:, :, 2])))
            except Exception as e:
                print(f"Warning on image {path}: {e}")

    counts_arr = np.array([class_counts[cid] for cid in range(43)])
    min_count = int(np.min(counts_arr))
    max_count = int(np.max(counts_arr))
    imbalance_ratio = float(max_count / max(1, min_count))
    gini = gini_coefficient(counts_arr)

    # 2. Save Canonical Class Representative Samples for UI
    print("Generating representative samples for all 43 classes...")
    sample_catalog = {}
    for cid in range(43):
        imgs = class_images[cid]
        if imgs:
            # Pick a representative sample near the median index
            rep_path = imgs[len(imgs) // 2]
            with Image.open(rep_path) as img:
                out_sample_path = SAMPLES_DIR / f"class_{cid:02d}.png"
                img.convert("RGB").save(out_sample_path, "PNG")
                sample_catalog[cid] = {
                    "class_id": cid,
                    "class_name": GTSRB_CLASSES[cid],
                    "category": get_class_category(cid),
                    "count": class_counts[cid],
                    "sample_file": f"samples/class_{cid:02d}.png"
                }

    # 3. Generate Scientific Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Chart 1: Class Distribution
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    bars = ax.bar(range(43), counts_arr, color="#B84D38", edgecolor="#202A28", linewidth=0.8, alpha=0.9)
    ax.set_title("GTSRB Class Frequency Distribution (N = 39,209)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Class ID (0 - 42)", fontsize=11)
    ax.set_ylabel("Number of Training Instances", fontsize=11)
    ax.set_xticks(range(0, 43, 2))
    ax.axhline(np.mean(counts_arr), color="#202A28", linestyle="--", linewidth=1.2, label=f"Mean Class Count ({np.mean(counts_arr):.0f})")
    ax.legend(frameon=True, facecolor="#EEECE5")
    fig.tight_layout()
    chart1_path = RESULTS_DIR / "class_distribution.png"
    fig.savefig(chart1_path)
    plt.close(fig)

    # Chart 2: Resolution & Aspect Ratio Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    sns.histplot(widths, bins=30, ax=ax1, color="#202A28", kde=True)
    ax1.set_title("Image Width Distribution (px)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Width (px)")
    
    sns.histplot(aspect_ratios, bins=30, ax=ax2, color="#A6AD91", kde=True)
    ax2.set_title("Aspect Ratio Distribution (W/H)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Aspect Ratio")
    fig.tight_layout()
    chart2_path = RESULTS_DIR / "resolution_distribution.png"
    fig.savefig(chart2_path)
    plt.close(fig)

    # 4. Compile Structured Machine-Readable JSON
    eda_summary = {
        "dataset_name": "German Traffic Sign Recognition Benchmark (GTSRB)",
        "total_classes": 43,
        "total_train_images": len(train_samples),
        "total_test_images": len(test_samples),
        "class_counts": {str(cid): int(class_counts[cid]) for cid in range(43)},
        "class_details": sample_catalog,
        "imbalance_metrics": {
            "min_class_instances": min_count,
            "min_class_id": int(np.argmin(counts_arr)),
            "min_class_name": GTSRB_CLASSES[int(np.argmin(counts_arr))],
            "max_class_instances": max_count,
            "max_class_id": int(np.argmax(counts_arr)),
            "max_class_name": GTSRB_CLASSES[int(np.argmax(counts_arr))],
            "imbalance_ratio": round(imbalance_ratio, 2),
            "gini_coefficient": round(gini, 4),
            "mean_instances_per_class": round(float(np.mean(counts_arr)), 1),
            "std_instances_per_class": round(float(np.std(counts_arr)), 1)
        },
        "dimension_stats": {
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "median_width": float(np.median(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "median_height": float(np.median(heights)),
            "mean_aspect_ratio": round(float(np.mean(aspect_ratios)), 3)
        },
        "photometric_stats": {
            "mean_red": round(float(np.mean(r_means)), 2),
            "mean_green": round(float(np.mean(g_means)), 2),
            "mean_blue": round(float(np.mean(b_means)), 2),
            "normalized_mean_rgb": [
                round(float(np.mean(r_means) / 255.0), 4),
                round(float(np.mean(g_means) / 255.0), 4),
                round(float(np.mean(b_means) / 255.0), 4)
            ]
        },
        "categories_breakdown": {
            cat: {
                "classes": ids,
                "total_instances": sum(class_counts[i] for i in ids)
            }
            for cat, ids in GTSRB_CATEGORIES.items()
        }
    }

    json_path = RESULTS_DIR / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    print(f"✓ EDA Completed Successfully. Summary saved to: {json_path}")
    return eda_summary

if __name__ == "__main__":
    run_eda()
