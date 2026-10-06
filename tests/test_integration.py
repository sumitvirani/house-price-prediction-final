"""
Integration tests – verify frontend/backend API contract.
Run:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
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


def test_predict_response_schema():
    """
    Verify that /predict returns exactly the fields that the Streamlit
    frontend reads: predicted_price, price_range_low, price_range_high,
    currency, model_version, input_features.
    """
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        required_keys = [
            "predicted_price",
            "price_range_low",
            "price_range_high",
            "currency",
            "model_version",
            "input_features",
        ]
        for key in required_keys:
            assert key in body, f"Missing key in /predict response: {key}"

        # input_features must echo all 8 input features
        for feat in VALID_PAYLOAD:
            assert feat in body["input_features"], f"Missing echo field: {feat}"


def test_feature_info_keys_match_predict_schema():
    """
    The feature keys returned by /feature-info must exactly match the fields
    accepted by /predict.
    """
    fi_resp = client.get("/feature-info")
    assert fi_resp.status_code in (200, 503)
    if fi_resp.status_code == 200:
        feature_names = list(fi_resp.json().keys())
        predict_fields = list(VALID_PAYLOAD.keys())
        assert feature_names == predict_fields, (
            f"Feature info keys {feature_names} don't match predict fields {predict_fields}"
        )


def test_health_then_predict_workflow():
    """Simulate the frontend workflow: health check → predict."""
    health = client.get("/health")
    assert health.status_code == 200
    is_loaded = health.json().get("model_loaded", False)

    if is_loaded:
        pred = client.post("/predict", json=VALID_PAYLOAD)
        assert pred.status_code == 200
        body = pred.json()
        assert body["predicted_price"] > 0


def test_price_range_is_ordered():
    """low < predicted_price < high."""
    resp = client.post("/predict", json=VALID_PAYLOAD)
    if resp.status_code == 200:
        body = resp.json()
        assert body["price_range_low"] < body["predicted_price"]
        assert body["price_range_high"] > body["predicted_price"]


def test_input_echo_matches_input():
    """input_features in the response must equal what was sent."""
    resp = client.post("/predict", json=VALID_PAYLOAD)
    if resp.status_code == 200:
        echo = resp.json()["input_features"]
        for k, v in VALID_PAYLOAD.items():
            assert echo[k] == pytest.approx(v, rel=1e-4), f"Echo mismatch for {k}"


def test_multiple_predictions_deterministic():
    """Same input should return the same prediction every time."""
    r1 = client.post("/predict", json=VALID_PAYLOAD)
    r2 = client.post("/predict", json=VALID_PAYLOAD)
    if r1.status_code == 200 and r2.status_code == 200:
        assert r1.json()["predicted_price"] == r2.json()["predicted_price"]


def test_all_endpoints_reachable():
    """Smoke-test every documented endpoint."""
    endpoints = ["/", "/health", "/feature-info", "/metrics", "/model-info"]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code in (200, 503), f"{ep} returned {resp.status_code}"


def test_predict_boundary_values():
    """Test with minimum and maximum allowed feature values."""
    min_payload = {
        "MedInc": 0.5, "HouseAge": 1.0, "AveRooms": 1.0, "AveBedrms": 1.0,
        "Population": 3.0, "AveOccup": 1.0, "Latitude": 32.5, "Longitude": -124.4,
    }
    max_payload = {
        "MedInc": 15.0, "HouseAge": 52.0, "AveRooms": 20.0, "AveBedrms": 5.0,
        "Population": 5000.0, "AveOccup": 10.0, "Latitude": 42.0, "Longitude": -114.3,
    }
    for payload in [min_payload, max_payload]:
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert resp.json()["predicted_price"] > 0
