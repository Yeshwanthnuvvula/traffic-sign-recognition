import sys
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.routes import health, predict, dataset, models

app = FastAPI(
    title="SIGNAL / VISION: Traffic Sign Recognition System",
    description="Scientific Computer Vision Research Platform for the 43-Class GTSRB Benchmark.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for Sample Thumbnails & EDA Charts
EDA_DIR = BASE_DIR / "backend" / "results" / "eda"
if not EDA_DIR.exists():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
(EDA_DIR / "samples").mkdir(parents=True, exist_ok=True)

app.mount("/api/static/eda", StaticFiles(directory=str(EDA_DIR)), name="eda_static")
app.mount("/api/static/samples", StaticFiles(directory=str(EDA_DIR / "samples")), name="samples_static")

# Include Routers under /api
app.include_router(health.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

# Serve React Production Frontend if built
DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "title": "SIGNAL / VISION API",
            "description": "Traffic Sign Recognition Using Machine Learning and Hyperparameter Optimization",
            "documentation": "/docs"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
