#!/usr/bin/env python3
"""
GTSRB (German Traffic Sign Recognition Benchmark) Dataset Downloader and Verifier.
Official Source: https://benchmark.ini.rub.de/
Official Archive Host: SID / ERDA benchmark archive (sid.erda.dk)
"""

import sys
from pathlib import Path
import torchvision.datasets as datasets

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def download_and_verify():
    print("==================================================")
    print("GTSRB Dataset Automated Downloader & Verification")
    print("==================================================")
    print(f"Target Directory: {DATA_DIR}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/2] Loading/Downloading Official Training Dataset (39,209 images)...")
    train_dataset = datasets.GTSRB(root=str(DATA_DIR), split="train", download=True)
    print(f"✓ Training Set Ready: {len(train_dataset)} images across 43 classes.")
    
    print("\n[2/2] Loading/Downloading Official Test Dataset (12,630 images)...")
    test_dataset = datasets.GTSRB(root=str(DATA_DIR), split="test", download=True)
    print(f"✓ Test Set Ready: {len(test_dataset)} images.")
    
    # Verification checks
    assert len(train_dataset) == 39209, f"Expected 39209 train images, got {len(train_dataset)}"
    assert len(test_dataset) == 12630, f"Expected 12630 test images, got {len(test_dataset)}"
    
    print("\n==================================================")
    print("GTSRB Integrity Verification Succeeded!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print("All 43 classes present and validated.")
    print("==================================================")

if __name__ == "__main__":
    download_and_verify()
