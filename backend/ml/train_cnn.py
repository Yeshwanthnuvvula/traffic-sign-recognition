"""
SIGNAL / VISION: Model 4 — Deep Custom Convolutional Neural Network (TrafficSignCNN).
Architecture:
- 3 VGG-style Convolutional Blocks with Batch Normalization and Spatial Dropout
- Max-Pooling downsampling
- Adaptive Average Pooling
- Fully Connected Classifier Head with Dense Dropout (0.4)
- 43-way Logits Output for CrossEntropyLoss (Softmax)
Regularization:
- Controlled training augmentation (affine, perspective, color jitter, shadow)
- Weight decay (L2 penalty)
- ReduceLROnPlateau learning rate scheduler
- Early stopping with best checkpoint restoration
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ml.dataset import (
    GTSRB_CLASSES,
    locate_dataset_paths,
    load_train_samples,
    get_stratified_train_val_split,
    GTSRBDataset
)
from backend.ml.preprocessing import get_train_transforms, get_eval_transforms

MODELS_DIR = BASE_DIR / "backend" / "models"
RESULTS_DIR = BASE_DIR / "backend" / "results"
DATA_DIR = BASE_DIR / "data"

class CustomTrafficSignCNN(nn.Module):
    """
    Modular deep CNN engineered specifically for multi-scale traffic sign classification.
    Input: (B, 3, H, W) where H, W in {32, 48}
    Output: (B, 43) raw logits
    """
    def __init__(self, num_classes: int = 43, dropout_rate: float = 0.3):
        super().__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.6)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.8)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((2, 2))
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 1.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate embedding for OOD detection."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        fc1 = self.classifier[0](x)
        bn1 = self.classifier[1](fc1)
        emb = self.classifier[2](bn1)
        return emb

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_cnn(
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    image_size: int = 48
) -> Dict[str, Any]:
    print("==================================================")
    print(f"Training Custom TrafficSignCNN (epochs={epochs}, lr={lr}, batch_size={batch_size})")
    print("==================================================")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Utilizing Compute Device: {device}")
    
    train_dir, _, _ = locate_dataset_paths(DATA_DIR)
    all_train_samples = load_train_samples(train_dir)
    train_samples, val_samples = get_stratified_train_val_split(all_train_samples, val_ratio=0.2, random_state=42)
    
    train_ds = GTSRBDataset(train_samples, transform=get_train_transforms(image_size=image_size))
    val_ds = GTSRBDataset(val_samples, transform=get_eval_transforms(image_size=image_size))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = CustomTrafficSignCNN(num_classes=43, dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }
    
    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "custom_cnn.pth"
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_acc)
        
        duration = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{duration:.1f}s] - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}% | LR: {current_lr:.6f}")
        
        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["lr"].append(current_lr)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "image_size": image_size
                }
            }, best_model_path)
            
    total_duration = time.time() - total_start_time
    print(f"\nCNN Training finished in {total_duration:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # Load best checkpoint for final evaluation metrics
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    inf_start = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    inf_time_per_sample = (time.time() - inf_start) / len(val_samples) * 1000.0  # ms
    
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(43))).tolist()
    report = classification_report(all_targets, all_preds, target_names=[GTSRB_CLASSES[i] for i in range(43)], output_dict=True, zero_division=0)
    
    metrics = {
        "model_name": "Custom TrafficSignCNN",
        "model_id": "custom_cnn",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "image_size": image_size,
            "best_epoch": checkpoint["epoch"]
        },
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_time_seconds": round(total_duration, 2),
        "inference_latency_ms": round(inf_time_per_sample, 2),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "history": history,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    with open(RESULTS_DIR / "custom_cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"✓ Best Model Checkpoint saved to: {best_model_path}")
    print(f"✓ Metrics and curves saved to: {RESULTS_DIR / 'custom_cnn_metrics.json'}")
    return metrics

if __name__ == "__main__":
    train_cnn(epochs=10)
