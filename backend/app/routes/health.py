from fastapi import APIRouter
import torch

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_health():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "status": "healthy",
        "service": "SIGNAL / VISION Inference Engine",
        "device": device,
        "supported_classes": 43,
        "dataset": "GTSRB Benchmark"
    }
