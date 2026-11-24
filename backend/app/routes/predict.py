from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from backend.ml.inference import inference_engine
from backend.app.schemas.prediction import PredictionResponse

router = APIRouter(tags=["Inference"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_traffic_sign(
    file: UploadFile = File(...),
    model_id: str = Query("resnet18", description="Model architecture: 'resnet18', 'custom_cnn', 'svm', 'rf', 'knn'")
):
    # Validate MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format (JPEG, PNG, PPM).")

    try:
        contents = await file.read()
        result = inference_engine.predict(contents, model_id=model_id)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
