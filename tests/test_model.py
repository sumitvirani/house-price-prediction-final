"""
Tests for model loading, prediction pipeline, and feature consistency.
Run:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DATA_DIR  = Path(__file__).resolve().parent.parent / "data"

MODEL_AVAILABLE = (MODEL_DIR / "model.pkl").exists()


# ── Model files exist ──────────────────────────────────────────────────────────
def test_model_file_exists():
    assert (MODEL_DIR / "model.pkl").exists(), "model.pkl not found – run python model/train.py"


def test_scaler_file_exists():
    assert (MODEL_DIR / "scaler.pkl").exists(), "scaler.pkl not found – run python model/train.py"


def test_feature_info_exists():
    assert (MODEL_DIR / "feature_info.json").exists(), "feature_info.json not found"


def test_metrics_file_exists():
    assert (MODEL_DIR / "metrics.json").exists(), "metrics.json not found"


def test_feature_importance_exists():
    assert (MODEL_DIR / "feature_importance.csv").exists(), "feature_importance.csv not found"


def test_dataset_exists():
    assert (DATA_DIR / "house_data.csv").exists(), "house_data.csv not found"


# ── Model loads correctly ──────────────────────────────────────────────────────
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_model_loads():
    import joblib
    model = joblib.load(MODEL_DIR / "model.pkl")
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_scaler_loads():
    import joblib
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    assert scaler is not None
    assert hasattr(scaler, "transform")


# ── Feature info consistency ───────────────────────────────────────────────────
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_feature_names_match():
    import json
    import joblib
    with open(MODEL_DIR / "feature_info.json") as f:
        feature_info = json.load(f)
    feature_names = list(feature_info.keys())
    expected = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
    assert feature_names == expected, f"Feature names mismatch: {feature_names}"


# ── Model makes predictions ────────────────────────────────────────────────────
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_model_prediction():
    import joblib
    model  = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    sample = np.array([[3.87, 28.0, 5.43, 1.10, 1425.0, 3.07, 35.63, -119.57]])
    scaled = scaler.transform(sample)
    pred   = model.predict(scaled)

    assert len(pred) == 1
    assert 50_000 < pred[0] < 700_000, f"Prediction {pred[0]} outside expected range"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_prediction_responds_to_income():
    """Higher income should predict higher price."""
    import joblib
    model  = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    base = [3.87, 28.0, 5.43, 1.10, 1425.0, 3.07, 35.63, -119.57]
    low  = scaler.transform([base[:]])
    high = scaler.transform([[12.0] + base[1:]])

    pred_low  = model.predict(low)[0]
    pred_high = model.predict(high)[0]
    assert pred_high > pred_low, "Higher income should yield higher predicted price"


# ── Metrics validity ───────────────────────────────────────────────────────────
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_metrics_values():
    import json
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)

    assert metrics["test_r2"]  > 0.70, f"Test R² too low: {metrics['test_r2']}"
    assert metrics["train_r2"] > 0.80, f"Train R² too low: {metrics['train_r2']}"
    assert metrics["test_mae"] < 60_000, f"Test MAE too high: ${metrics['test_mae']:,.0f}"


# ── Dataset validity ───────────────────────────────────────────────────────────
@pytest.mark.skipif(not (DATA_DIR / "house_data.csv").exists(), reason="Dataset not generated yet")
def test_dataset_shape():
    df = pd.read_csv(DATA_DIR / "house_data.csv")
    assert len(df) > 10_000, f"Dataset too small: {len(df)} rows"
    assert "Price" in df.columns
    for feat in ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]:
        assert feat in df.columns, f"Missing column: {feat}"


@pytest.mark.skipif(not (DATA_DIR / "house_data.csv").exists(), reason="Dataset not generated yet")
def test_dataset_no_nulls():
    df = pd.read_csv(DATA_DIR / "house_data.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"


# ── Feature importance ─────────────────────────────────────────────────────────
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet")
def test_feature_importance_shape():
    fi = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    assert len(fi) == 8
    assert "feature"    in fi.columns
    assert "importance" in fi.columns
    assert fi["importance"].sum() == pytest.approx(1.0, abs=0.01)
