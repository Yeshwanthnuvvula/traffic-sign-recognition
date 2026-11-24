"""
GTSRB Dataset Management & Stratified Split Module.
Handles loading official GTSRB training and testing sets, mapping class IDs to human-readable names,
grouping categories, and computing stratified splits.
"""

import os
import csv
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# 43 Official GTSRB Class Mapping
GTSRB_CLASSES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing (No overtaking)",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 metric tons"
}

# Sign category taxonomy
GTSRB_CATEGORIES: Dict[str, List[int]] = {
    "Speed Limit": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "Prohibitory": [9, 10, 15, 16, 17],
    "Derestriction": [6, 32, 41, 42],
    "Mandatory": [33, 34, 35, 36, 37, 38, 39, 40],
    "Danger / Warning": [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Priority": [12, 13, 14]
}

def get_class_category(class_id: int) -> str:
    for cat_name, ids in GTSRB_CATEGORIES.items():
        if class_id in ids:
            return cat_name
    return "Other"

class GTSRBDataset(Dataset):
    """
    Unified PyTorch Dataset for GTSRB.
    Loads raw PPM/PNG files and applies transform pipeline.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

def locate_dataset_paths(base_data_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Auto-discovers training and testing directory structures.
    Prioritizes official full 39,209 images in Final_Training/Images.
    """
    possible_train = [
        base_data_dir / "GTSRB" / "Final_Training" / "Images",
        base_data_dir / "Final_Training" / "Images",
        base_data_dir / "gtsrb" / "GTSRB" / "Training",
        base_data_dir / "GTSRB" / "Training",
        base_data_dir / "Training"
    ]
    train_dir = None
    for p in possible_train:
        if p.exists() and p.is_dir():
            train_dir = p
            break

    possible_test = [
        base_data_dir / "gtsrb" / "GTSRB" / "Final_Test" / "Images",
        base_data_dir / "GTSRB" / "Final_Test" / "Images",
        base_data_dir / "GTSRB" / "GTSRB" / "Final_Test" / "Images",
        base_data_dir / "Final_Test" / "Images"
    ]
    test_dir = None
    for p in possible_test:
        if p.exists() and p.is_dir():
            test_dir = p
            break

    possible_test_csv = [
        base_data_dir / "gtsrb" / "GT-final_test.csv",
        base_data_dir / "GTSRB" / "GT-final_test.csv",
        base_data_dir / "GT-final_test.csv"
    ]
    test_csv = None
    for p in possible_test_csv:
        if p.exists() and p.is_file():
            test_csv = p
            break

    return train_dir, test_dir, test_csv

def load_train_samples(train_dir: Path) -> List[Tuple[str, int]]:
    """
    Scans class folders 00000..00042 in train_dir.
    Returns list of (filepath, class_id) tuples.
    """
    samples = []
    class_folders = sorted([d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    for folder in class_folders:
        try:
            class_id = int(folder.name)
        except ValueError:
            continue
            
        files = list(folder.glob("*.ppm")) + list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        for f in files:
            if not f.name.startswith('.'):
                samples.append((str(f), class_id))
                
    return samples

def load_test_samples(test_dir: Path, test_csv: Path) -> List[Tuple[str, int]]:
    """
    Loads official test set images and ground truth labels from GT-final_test.csv.
    """
    samples = []
    with open(test_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";", skipinitialspace=True)
        for row in reader:
            filename = row["Filename"]
            class_id = int(row["ClassId"])
            img_path = test_dir / filename
            if img_path.exists():
                samples.append((str(img_path), class_id))
    return samples

def get_stratified_train_val_split(samples: List[Tuple[str, int]], val_ratio: float = 0.2, random_state: int = 42):
    """
    Performs 80/20 stratified split on training samples.
    Returns train_samples, val_samples.
    """
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_ratio, random_state=random_state, stratify=labels
    )
    
    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    
    return train_samples, val_samples
