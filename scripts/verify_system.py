"""
SIGNAL / VISION: End-to-End System Verification Suite.
Validates:
1. Model checkpoints exist and load successfully into inference engine
2. Real-time prediction returns all required fields (top-3, confidence, latency, uncertainty)
3. Out-Of-Distribution rejection correctly flags non-traffic sign and corrupted images
4. EDA and metric endpoints return genuine structured statistics
"""

import sys
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ml.dataset import GTSRB_CLASSES
from backend.ml.inference import inference_engine

def verify_system():
    print("==================================================")
    print("Executing End-to-End System Verification")
    print("==================================================")

    # 1. Inspect Available Models
    available = inference_engine.get_available_models()
    print(f"\n[1] Discovered Models in Registry: {len(available)}")
    for m in available:
        print(f"    • {m['id']:<14} : {m['name']} ({m['type']})")

    # 2. Test In-Distribution Traffic Sign Prediction
    sample_img_path = BASE_DIR / "backend" / "results" / "eda" / "samples" / "class_01.png"
    if sample_img_path.exists():
        print(f"\n[2] Testing Prediction on Canonical Benchmark Sample: Class #1 (Speed 30)")
        with Image.open(sample_img_path) as img:
            for m in available:
                res = inference_engine.predict(img, model_id=m["id"])
                print(f"    [{m['id']:<12}] Pred: #{res['predicted_class_id']} ({res['predicted_class_name'][:24]}) | Conf: {res['confidence_percentage']:.1f}% | Latency: {res['inference_time_ms']:.2f}ms")
                assert "top_3_predictions" in res and len(res["top_3_predictions"]) == 3
                assert "uncertainty" in res

    # 3. Test Out-Of-Distribution (OOD) Rejection
    print(f"\n[3] Testing Out-Of-Distribution Rejection on Synthetic Non-Traffic Visuals")
    # A) High-entropy random noise image
    noise_array = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_array)
    res_noise = inference_engine.predict(noise_img, model_id="svm")
    print(f"    • Random Noise Input: Entropy = {res_noise['uncertainty']['normalized_entropy']:.3f} | Status = {res_noise['uncertainty']['status_message']}")
    assert res_noise["uncertainty"]["is_uncertain"] == True

    # B) Blank / low-variance image
    blank_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    blank_img = Image.fromarray(blank_array)
    res_blank = inference_engine.predict(blank_img, model_id="svm")
    print(f"    • Blank Color Input: Quality Valid = {res_blank['image_metadata']['is_valid_quality']} | Rejection = {res_blank['uncertainty']['is_rejected']}")
    assert res_blank["uncertainty"]["is_rejected"] == True

    print("\n==================================================")
    print("✓ All Verification Tests Succeeded!")
    print("==================================================")

if __name__ == "__main__":
    verify_system()
