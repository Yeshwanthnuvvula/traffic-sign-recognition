# SIGNAL / VISION: Development Progress & Comprehensive Research Log

**Project Title:** SIGNAL / VISION — Traffic Sign Recognition Using Machine Learning and Hyperparameter Optimization  
**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB) — 43 Classes  
**Lead Architecture:** Multi-Model Laboratory (SVM, KNN, Random Forest, Custom CNN, ResNet-18) + FastAPI + React TypeScript Editorial UI  

---

## Stage-by-Stage Execution Log

| Stage | Milestone | Status | Key Deliverables & Results |
|:---|:---|:---:|:---|
| **STAGE 1** | Environment & Dataset Setup | ✅ Complete | Python 3.12 venv, PyTorch with Apple Silicon MPS, verified 39,209 train & 12,630 test images. |
| **STAGE 2** | Exploratory Data Analysis & Preprocessing | ✅ Complete | Gini = 0.4124, 10.71× class imbalance, 43 canonical thumbnails, ImageNet mean/std, zero-flip invariant. |
| **STAGE 3** | Classical Machine Learning Models | ✅ Complete | 900-dim HOG extraction; SVM (RBF): 97.13%; KNN (k=5): 92.73%; Random Forest: 91.97%. |
| **STAGE 4** | Deep Learning Architectures | ✅ Complete | Custom 3-block TrafficSignCNN: 99.83% val accuracy; ResNet-18 Transfer Learning: >97.25%. |
| **STAGE 5** | Hyperparameter Optimization | ✅ Complete | Parameter sweeps across C, gamma, k, weights, tree depth, learning rate; saved to `experiments.json`. |
| **STAGE 6** | Comprehensive Evaluation & Generalization | ✅ Complete | 43-class Confusion Matrix, Speed Limit 30 vs 50 dedicated analysis, OOD Shannon entropy rejection. |
| **STAGE 7** | High-Performance FastAPI Backend | ✅ Complete | Endpoints for `/api/predict`, `/api/metrics`, `/api/eda`, `/api/models`, static sample serving. |
| **STAGE 8** | Academic Editorial React Frontend | ✅ Complete | 7 distinct pages, Warm Paper `#EEECE5` palette, Cormorant Garamond typography, Recharts visualizations. |
| **STAGE 9** | End-to-End System Integration | ✅ Complete | Live prediction with top-3 posteriors, decision margins, latency bench, OOD rejection verification. |
| **STAGE 10** | Academic Monograph & Verification | ✅ Complete | Full academic research report with PDF export, verification test suite (`verify_system.py`). |

---

## Detailed Results & Architecture Profiles

### 1. Model 1: Support Vector Machine (SVM)
- **Feature Space:** 900-dimensional Dalal-Triggs HOG descriptor (48×48 window, 8×8 cells, 2×2 blocks).
- **Kernel:** Radial Basis Function (RBF), C = 10.0, gamma = 'scale'.
- **Validation Accuracy:** **97.13%**
- **Macro F1:** **0.9752** | **Weighted F1:** **0.9714**
- **Inference Latency:** 7.22 ms/image

### 2. Model 2: K-Nearest Neighbors (KNN)
- **Configuration:** k = 5, inverse distance weighting, Euclidean distance metric.
- **Validation Accuracy:** **92.73%**
- **Macro F1:** **0.9357** | **Weighted F1:** **0.9277**
- **Inference Latency:** 0.29 ms/image

### 3. Model 3: Random Forest Ensemble
- **Configuration:** 100 decision trees, max depth = 25, Gini impurity criterion.
- **Validation Accuracy:** **91.97%**
- **Macro F1:** **0.9207** | **Weighted F1:** **0.9190**
- **Inference Latency:** 0.01 ms/image (Ultra-fast CPU evaluation)

### 4. Model 4: Custom TrafficSignCNN
- **Architecture:** 3 Convolutional Blocks (Conv2d 32→64→128) with spatial Dropout2d and BatchNorm2d, followed by AdaptiveAvgPool2d((2,2)) and a 256-unit dense classifier head.
- **Regularization:** AdamW, weight decay = 1e-4, affine perturbation, simulated shadow masks.
- **Validation Accuracy:** **99.83%**
- **Inference Latency:** 2.14 ms/image on Apple Silicon MPS

### 5. Model 5: ResNet-18 (Transfer Learning)
- **Backbone:** ImageNet pre-trained residual layers with differential learning rates.
- **Classification Head:** 2-stage linear mapping with BatchNorm1d and dense dropout.
- **Convergence:** Rapid epoch-1 transfer learning convergence >97%.

### 6. Out-Of-Distribution (OOD) Rejection Protocol
- **Shannon Predictive Entropy:** \( H(p) = -\sum_{i=1}^{43} p_i \ln(p_i) \)
- **Rejection Criteria:**
  - Normalized Entropy > 0.60 combined with Top-1 Confidence < 0.45 flags non-traffic imagery.
  - Structural Laplacian variance < 8.0 detects unviable defocus blur or blank color fields.
- **Decision Margin Guardrail:** Margin < 0.15 between top-1 and top-2 candidates triggers an explicit ambiguity warning in the UI rather than confirmed output.
