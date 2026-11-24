# SIGNAL / VISION: Traffic Sign Recognition Research Platform

**Traffic Sign Recognition Using Machine Learning and Hyperparameter Optimization**  
*A Computer Vision & Deep Learning Investigation on the Official GTSRB Benchmark (43 Classes)*

---

## 1. Project Overview

SIGNAL / VISION is an end-to-end, university-grade machine learning system designed to classify, analyze, and detect uncertainty across all **43 official German Traffic Sign Recognition Benchmark (GTSRB)** categories.

The platform bridges classical gradient descriptors with modern deep convolutional and residual neural networks, accompanied by a contemporary academic research publication interface and interactive recognition studio.

### Key Capabilities
- **43-Class Recognition**: Full coverage of all standard GTSRB traffic sign categories.
- **5 Benchmarked Algorithms**: Support Vector Machines (Linear/RBF), K-Nearest Neighbors, Random Forest, Custom 3-Block TrafficSignCNN, and ResNet-18 (Transfer Learning).
- **Interactive Computer Vision Studio**: Drag-and-drop upload, 48×48 preprocessing preview, top-3 candidates, confidence percentages, decision margins, and real-time inference latency.
- **Safety Invariant Pipeline**: Strictly prohibits horizontal or vertical sign inversion to preserve directional semantics.
- **Out-Of-Distribution (OOD) Rejection**: Rejects non-traffic sign photographs and corrupted images using predictive Shannon entropy and calibrated Maximum Softmax Probability (MSP).
- **Speed Limit Disambiguation**: Specialized confusion analysis resolving visually ambiguous speed limit pairs (e.g. 30 km/h vs. 50 km/h).
- **Academic Research Interface**: Warm paper (`#EEECE5`) editorial aesthetic, interactive 43×43 confusion matrix, Pareto accuracy/latency tradeoff, and PDF exportable thesis report.

---

## 2. System Architecture

```
signal-vision/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & static route mounts
│   │   ├── routes/              # Modular REST endpoints (health, predict, dataset, models)
│   │   ├── schemas/             # Pydantic data schemas
│   │   └── utils/               # Path resolution & system helpers
│   │
│   ├── ml/
│   │   ├── dataset.py           # Stratified 80/20 GTSRB split & 43-class taxonomic metadata
│   │   ├── preprocessing.py     # Deterministic 48x48 resize, ImageNet normalization, augmentation
│   │   ├── eda.py               # Statistical profiling, Gini coefficient, 43 canonical thumbnails
│   │   ├── feature_extraction.py# Dalal-Triggs HOG descriptor (900-dim) & StandardScaler
│   │   ├── train_svm.py         # Support Vector Machine (Linear & RBF)
│   │   ├── train_knn.py         # K-Nearest Neighbors classifier
│   │   ├── train_random_forest.py # Random Forest ensemble
│   │   ├── train_cnn.py         # Custom 3-block TrafficSignCNN (PyTorch MPS)
│   │   ├── train_resnet.py      # ResNet-18 Transfer Learning
│   │   ├── hyperparameter_tuning.py # Systematic parameter sweeps
│   │   ├── evaluation.py        # Comparative matrix & speed limit disambiguation
│   │   ├── ood_detection.py     # Predictive entropy & OOD rejection engine
│   │   └── inference.py         # Unified multi-model inference engine
│   │
│   ├── models/                  # Persisted checkpoints (.pth, .joblib)
│   ├── results/                 # Empirical metric JSONs, confusion matrices, charts
│   └── requirements.txt         # Documented Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── layouts/AppLayout.tsx # Editorial masthead header, navigation & footer
│   │   ├── pages/
│   │   │   ├── CoverPage.tsx        # Page 01: Academic Research Cover
│   │   │   ├── StudioPage.tsx       # Page 02: Computer Vision Recognition Studio
│   │   │   ├── DataExplorerPage.tsx # Page 03: Empirical Profiler & 43-Class Gallery
│   │   │   ├── ExperimentsPage.tsx  # Page 04: Controlled Model Experiments
│   │   │   ├── PerformancePage.tsx  # Page 05: Interactive 43x43 Confusion Matrix & Curves
│   │   │   ├── MethodologyPage.tsx  # Page 06: Scientific Workflow & Mathematical Formulations
│   │   │   └── ReportPage.tsx       # Page 07: Academic Monograph (PDF Exportable)
│   │   ├── services/api.ts          # Axios REST API client
│   │   └── types.ts                 # Full TypeScript schemas
│   ├── package.json
│   ├── tailwind.config.js       # Warm paper, deep ink, signal red, muted sage palette
│   └── vite.config.ts
│
├── data/                        # Official GTSRB training & holdout test partitions
├── scripts/
│   ├── download_gtsrb.py        # Automated GTSRB downloader & integrity verifier
│   ├── test_unseen_generalization.py # External perturbation benchmark
│   └── verify_system.py         # End-to-end verification suite
├── DEVELOPMENT_PROGRESS.md      # Detailed 10-stage research log
└── README.md
```

---

## 3. Dataset Specifications

- **Benchmark**: German Traffic Sign Recognition Benchmark (GTSRB).
- **Official Source**: `https://benchmark.ini.rub.de/` (Archive Host: SID/ERDA).
- **Categories**: 43 standardized sign classes (Speed limits, Prohibitory, Mandatory, Danger, Priority, Derestriction).
- **Training Set**: 39,209 photographs (stratified 80% train = 31,367; 20% holdout validation = 7,842).
- **Test Set**: 12,630 photographs (ground truth untouched).
- **Class Imbalance**: Imbalance ratio = 10.71× (Min: 210 in Speed 20; Max: 2,250 in Speed 50). Gini coefficient = 0.4124.

---

## 4. Empirical Model Benchmarks

All models trained on verified GTSRB data and evaluated on the stratified validation set:

| Model Architecture | Type | Input Representation | Top-1 Accuracy | Macro F1 | Weighted F1 | Inference Latency |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Custom TrafficSignCNN** | Deep Learning | 48×48 RGB | **99.83%** | **0.9981** | **0.9983** | **2.14 ms** |
| **ResNet-18 (Transfer)** | Residual Net | 48×48 RGB | **99.25%** | **0.9918** | **0.9924** | **3.85 ms** |
| **Support Vector Machine (RBF)** | Classical ML | 900-dim HOG | **97.13%** | **0.9752** | **0.9714** | **7.22 ms** |
| **K-Nearest Neighbors (k=5)** | Instance ML | 900-dim HOG | **92.73%** | **0.9357** | **0.9277** | **0.29 ms** |
| **Random Forest (100 Trees)** | Ensemble ML | 900-dim HOG | **91.97%** | **0.9207** | **0.9190** | **0.01 ms** |

---

## 5. Quickstart & Reproducibility

### 1. Backend Environment Setup
```bash
# Clone and enter workspace
cd "TRAFFIC SIGN RECOGNITION"

# Activate Python 3.12 virtual environment
source .venv/bin/activate

# Install dependencies (already completed in repo setup)
pip install -r backend/requirements.txt
```

### 2. Run API Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation: `http://localhost:8000/docs`

### 3. Run Frontend UI
```bash
cd frontend
npm run dev
```
Open browser at: `http://localhost:5173`

### 4. Run Verification Suite
```bash
python scripts/verify_system.py
```
