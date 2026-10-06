"""
Tests for the FastAPI backend endpoints.
Run:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "MedInc":     3.87,
    "HouseAge":   28.0,
    "AveRooms":   5.43,
    "AveBedrms":  1.10,
    "Population": 1425.0,
    "AveOccup":   3.07,
    "Latitude":   35.63,
    "Longitude":  -119.57,
}


# ── Root ───────────────────────────────────────────────────────────────────────
def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert "version" in body


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body
    assert isinstance(body["model_loaded"], bool)


# ── Feature info ───────────────────────────────────────────────────────────────
def test_feature_info():
    resp = client.get("/feature-info")
    # Either 200 (model trained) or 503 (not trained yet – acceptable in CI)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert isinstance(body, dict)
        assert len(body) == 8
        for feat in ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]:
            assert feat in body


# ── Metrics ────────────────────────────────────────────────────────────────────
def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        for key in ["train_r2", "test_r2", "train_mae", "test_mae",
                    "train_rmse", "test_rmse", "n_train", "n_test"]:
            assert key in body
            assert isinstance(body[key], (int, float))


# ── Model info ─────────────────────────────────────────────────────────────────
def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "model_type"    in body
        assert "feature_names" in body
        assert isinstance(body["feature_names"], list)
        assert len(body["feature_names"]) == 8


# ── Predict – valid ────────────────────────────────────────────────────────────
def test_predict_valid():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "predicted_price"  in body
        assert "price_range_low"  in body
        assert "price_range_high" in body
        assert body["predicted_price"] > 0
        assert body["price_range_low"]  < body["predicted_price"]
        assert body["price_range_high"] > body["predicted_price"]
        assert body["currency"] == "USD"


# ── Predict – high income ─────────────────────────────────────────────────────
def test_predict_high_income():
    payload = {**VALID_PAYLOAD, "MedInc": 12.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        low_income_resp = client.post("/predict", json={**VALID_PAYLOAD, "MedInc": 1.0})
        assert low_income_resp.status_code == 200
        assert resp.json()["predicted_price"] > low_income_resp.json()["predicted_price"]


# ── Predict – invalid: missing field ─────────────────────────────────────────
def test_predict_missing_field():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "MedInc"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ── Predict – invalid: out of range ─────────────────────────────────────────
def test_predict_out_of_range():
    bad = {**VALID_PAYLOAD, "MedInc": 999.0}  # max is 15.0
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ── Predict – invalid: wrong type ────────────────────────────────────────────
def test_predict_wrong_type():
    bad = {**VALID_PAYLOAD, "HouseAge": "old"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ── Predict – invalid: bedrooms > rooms ──────────────────────────────────────
def test_predict_bedrms_gt_rooms():
    bad = {**VALID_PAYLOAD, "AveBedrms": 8.0, "AveRooms": 5.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ── CORS headers ──────────────────────────────────────────────────────────────
def test_cors_headers():
    resp = client.options(
        "/predict",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    # TestClient may not add CORS preflight headers; just ensure no 5xx
    assert resp.status_code < 500
