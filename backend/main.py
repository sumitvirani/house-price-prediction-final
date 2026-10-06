"""
House Price Prediction – FastAPI Backend
==========================================
Start:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HouseFeatures,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "model"

MODEL_PATH      = MODEL_DIR / "model.pkl"
SCALER_PATH     = MODEL_DIR / "scaler.pkl"
FEATURES_PATH   = MODEL_DIR / "feature_info.json"
METRICS_PATH    = MODEL_DIR / "metrics.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

VERSION = "1.0.0"

# ── Load artifacts at startup ──────────────────────────────────────────────────
def _load_artifact(path: Path, label: str):
    if not path.exists():
        log.error("Missing artifact: %s", path)
        raise RuntimeError(
            f"Required model artifact not found: {path}\n"
            "Please run: python model/train.py"
        )
    return joblib.load(path)


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        log.error("Missing JSON file: %s", path)
        raise RuntimeError(
            f"Required JSON file not found: {path}\n"
            "Please run: python model/train.py"
        )
    with open(path) as f:
        return json.load(f)


try:
    model       = _load_artifact(MODEL_PATH,  "model")
    scaler      = _load_artifact(SCALER_PATH, "scaler")
    feature_info = _load_json(FEATURES_PATH,  "feature_info")
    metrics_data = _load_json(METRICS_PATH,   "metrics")
    _model_loaded = True
    log.info("Model and artifacts loaded successfully.")
except RuntimeError as e:
    log.warning(str(e))
    model = scaler = feature_info = metrics_data = None
    _model_loaded = False

FEATURE_NAMES = list(feature_info.keys()) if feature_info else [
    "MedInc","HouseAge","AveRooms","AveBedrms",
    "Population","AveOccup","Latitude","Longitude",
]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predict California house prices using a GradientBoostingRegressor "
        "trained on the California Housing dataset."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    return {
        "message": "House Price Prediction API",
        "version": VERSION,
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model_loaded else "degraded – model not loaded",
        model_loaded=_model_loaded,
        version=VERSION,
    )


@app.get("/feature-info", tags=["Model"])
def get_feature_info() -> Dict[str, Any]:
    """Return metadata for every model feature (used by the frontend)."""
    if not feature_info:
        raise HTTPException(status_code=503, detail="Feature info not available. Run training first.")
    return feature_info


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
def get_metrics() -> MetricsResponse:
    """Return model evaluation metrics."""
    if not metrics_data:
        raise HTTPException(status_code=503, detail="Metrics not available. Run training first.")
    return MetricsResponse(**{k: metrics_data[k] for k in MetricsResponse.model_fields})


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    """Return model metadata."""
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return ModelInfoResponse(
        model_type   = metrics_data.get("model_type", "GradientBoostingRegressor"),
        dataset      = metrics_data.get("dataset",    "California Housing (sklearn)"),
        feature_names= FEATURE_NAMES,
        n_estimators = getattr(model, "n_estimators", None),
        version      = VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures) -> PredictionResponse:
    """
    Predict median house price for a California block group.

    Returns the predicted price in USD along with a ±15 % confidence range.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first: python model/train.py",
        )

    input_dict = features.model_dump()
    import pandas as pd
    feature_vector = pd.DataFrame([[input_dict[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    try:
        scaled = scaler.transform(feature_vector)
        prediction = float(model.predict(scaled)[0])
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Clamp to a sensible range ($50 000 – $800 000)
    prediction = max(50_000.0, min(prediction, 800_000.0))

    # ±15 % confidence interval (heuristic based on model RMSE)
    margin = prediction * 0.15
    return PredictionResponse(
        predicted_price  = round(prediction, 2),
        price_range_low  = round(max(0.0, prediction - margin), 2),
        price_range_high = round(prediction + margin, 2),
        currency         = "USD",
        model_version    = VERSION,
        input_features   = input_dict,
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
