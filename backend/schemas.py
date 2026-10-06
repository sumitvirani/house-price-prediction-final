"""
Pydantic schemas for the House Price Prediction API.
All feature names must stay in sync with model/train.py → FEATURE_NAMES.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List


class HouseFeatures(BaseModel):
    """Input schema for /predict endpoint."""

    MedInc: float = Field(
        ..., ge=0.5, le=15.0,
        description="Median income in block group (tens of thousands of USD)",
        json_schema_extra={"example": 3.87},
    )
    HouseAge: float = Field(
        ..., ge=1.0, le=52.0,
        description="Median house age in block group (years)",
        json_schema_extra={"example": 28.0},
    )
    AveRooms: float = Field(
        ..., ge=1.0, le=20.0,
        description="Average number of rooms per household",
        json_schema_extra={"example": 5.43},
    )
    AveBedrms: float = Field(
        ..., ge=1.0, le=5.0,
        description="Average number of bedrooms per household",
        json_schema_extra={"example": 1.10},
    )
    Population: float = Field(
        ..., ge=3.0, le=5000.0,
        description="Block group population",
        json_schema_extra={"example": 1425.0},
    )
    AveOccup: float = Field(
        ..., ge=1.0, le=10.0,
        description="Average household occupancy",
        json_schema_extra={"example": 3.07},
    )
    Latitude: float = Field(
        ..., ge=32.5, le=42.0,
        description="Block group latitude",
        json_schema_extra={"example": 35.63},
    )
    Longitude: float = Field(
        ..., ge=-124.4, le=-114.3,
        description="Block group longitude",
        json_schema_extra={"example": -119.57},
    )

    @field_validator("AveBedrms")
    @classmethod
    def bedrms_le_rooms(cls, v, info):
        rooms = info.data.get("AveRooms")
        if rooms is not None and v > rooms:
            raise ValueError("AveBedrms cannot exceed AveRooms")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc":     3.87,
                "HouseAge":   28.0,
                "AveRooms":   5.43,
                "AveBedrms":  1.10,
                "Population": 1425.0,
                "AveOccup":   3.07,
                "Latitude":   35.63,
                "Longitude":  -119.57,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    predicted_price: float          = Field(..., description="Predicted house price in USD")
    price_range_low: float          = Field(..., description="Lower bound of confidence range")
    price_range_high: float         = Field(..., description="Upper bound of confidence range")
    currency: str                   = Field(default="USD")
    model_version: str              = Field(default="1.0.0")
    input_features: Dict[str, Any]  = Field(..., description="Echo of input features")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    dataset: str
    feature_names: List[str]
    n_estimators: Optional[int]
    version: str


class MetricsResponse(BaseModel):
    train_r2:   float
    test_r2:    float
    train_mae:  float
    test_mae:   float
    train_rmse: float
    test_rmse:  float
    n_train:    int
    n_test:     int
    model_type: str
    dataset:    str
