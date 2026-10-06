"""
House Price Prediction - Model Training Script
================================================
Dataset: California Housing (sklearn built-in)
Model:   GradientBoostingRegressor
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure stdout uses UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR   = BASE_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH   = os.path.join(MODEL_DIR, "feature_info.json")
METRICS_PATH    = os.path.join(MODEL_DIR, "metrics.json")
IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.csv")
DATA_PATH       = os.path.join(DATA_DIR,  "house_data.csv")

# ── Feature metadata (used by API + frontend) ──────────────────────────────────
FEATURE_INFO = {
    "MedInc": {
        "display_name": "Median Income (10k USD)",
        "description":  "Median income in the block group (in tens of thousands of dollars)",
        "min": 0.5,  "max": 15.0, "default": 3.87, "step": 0.1,
    },
    "HouseAge": {
        "display_name": "House Age (years)",
        "description":  "Median age of houses in the block group",
        "min": 1.0,  "max": 52.0, "default": 28.0, "step": 1.0,
    },
    "AveRooms": {
        "display_name": "Average Rooms",
        "description":  "Average number of rooms per household",
        "min": 1.0,  "max": 20.0, "default": 5.43, "step": 0.1,
    },
    "AveBedrms": {
        "display_name": "Average Bedrooms",
        "description":  "Average number of bedrooms per household",
        "min": 1.0,  "max": 5.0,  "default": 1.10, "step": 0.1,
    },
    "Population": {
        "display_name": "Block Population",
        "description":  "Population of the block group",
        "min": 3.0,  "max": 5000.0, "default": 1425.0, "step": 10.0,
    },
    "AveOccup": {
        "display_name": "Average Occupancy",
        "description":  "Average number of household members",
        "min": 1.0,  "max": 10.0, "default": 3.07, "step": 0.1,
    },
    "Latitude": {
        "display_name": "Latitude",
        "description":  "Block group latitude",
        "min": 32.5, "max": 42.0, "default": 35.63, "step": 0.01,
    },
    "Longitude": {
        "display_name": "Longitude",
        "description":  "Block group longitude",
        "min": -124.4, "max": -114.3, "default": -119.57, "step": 0.01,
    },
}

FEATURE_NAMES = list(FEATURE_INFO.keys())


def load_data() -> pd.DataFrame:
    """Fetch the California Housing dataset and return as a DataFrame."""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    # Rename target column for clarity
    df.rename(columns={"MedHouseVal": "Price"}, inplace=True)
    # Price is in 100k USD – multiply by 100 000 for dollar values
    df["Price"] = df["Price"] * 100_000
    print(f"  Dataset shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    """Clean data, split, and scale features."""
    # Cap extreme outliers in target (houses capped at $500 001 in the raw data)
    df = df[df["Price"] < 500_001].copy()

    # Cap feature outliers at 99th percentile
    for col in FEATURE_NAMES:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)

    X = df[FEATURE_NAMES]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, X_train, X_test, df


def train_model(X_train, y_train):
    """Train and return a GradientBoostingRegressor."""
    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Compute evaluation metrics."""
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = {
        "train_r2":   float(r2_score(y_train, y_pred_train)),
        "test_r2":    float(r2_score(y_test,  y_pred_test)),
        "train_mae":  float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae":   float(mean_absolute_error(y_test,  y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_test,  y_pred_test))),
        "n_train":    int(len(y_train)),
        "n_test":     int(len(y_test)),
    }

    print("\n-- Evaluation Metrics --")
    print(f"  Train R2   : {metrics['train_r2']:.4f}")
    print(f"  Test  R2   : {metrics['test_r2']:.4f}")
    print(f"  Train MAE  : ${metrics['train_mae']:,.0f}")
    print(f"  Test  MAE  : ${metrics['test_mae']:,.0f}")
    print(f"  Train RMSE : ${metrics['train_rmse']:,.0f}")
    print(f"  Test  RMSE : ${metrics['test_rmse']:,.0f}")
    print("------------------------\n")
    return metrics


def save_feature_importance(model):
    """Save feature importance to CSV."""
    importance = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    print(f"  Feature importance saved -> {IMPORTANCE_PATH}")
    return importance


def main():
    # 1. Load
    df = load_data()

    # 2. Preprocess
    X_train_s, X_test_s, y_train, y_test, scaler, X_train_raw, X_test_raw, df_clean = preprocess(df)

    # 3. Train
    model = train_model(X_train_s, y_train)

    # 4. Evaluate
    metrics = evaluate_model(model, X_train_s, X_test_s, y_train, y_test)

    # 5. Save model artifacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Model  saved -> {MODEL_PATH}")
    print(f"  Scaler saved -> {SCALER_PATH}")

    # 6. Save feature info
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_INFO, f, indent=2)
    print(f"  Feature info saved -> {FEATURES_PATH}")

    # 7. Save metrics
    metrics["model_type"]     = "GradientBoostingRegressor"
    metrics["feature_names"]  = FEATURE_NAMES
    metrics["dataset"]        = "California Housing (sklearn)"
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved -> {METRICS_PATH}")

    # 8. Save feature importance
    save_feature_importance(model)

    # 9. Save dataset CSV for frontend Data Explorer
    df_clean.to_csv(DATA_PATH, index=False)
    print(f"  Dataset CSV saved -> {DATA_PATH}")

    print("\n[OK] Training complete - all artifacts saved.\n")


if __name__ == "__main__":
    main()
