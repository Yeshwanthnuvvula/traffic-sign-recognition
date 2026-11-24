from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictionCandidate(BaseModel):
    class_id: int
    class_name: str
    category: str
    probability: float
    percentage: float

class UncertaintyInfo(BaseModel):
    confidence: float
    decision_margin: float
    entropy: float
    normalized_entropy: float
    is_uncertain: bool
    is_rejected: bool
    status_message: str

class ImageMetadata(BaseModel):
    width: int
    height: int
    aspect_ratio: float
    channels: int
    is_valid_quality: bool
    quality_note: str

class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
    category: str
    confidence: float
    confidence_percentage: float
    top_3_predictions: List[PredictionCandidate]
    inference_time_ms: float
    model_id: str
    uncertainty: UncertaintyInfo
    image_metadata: ImageMetadata
