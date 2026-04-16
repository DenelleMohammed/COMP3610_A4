from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import numpy as np
import json
import time
import uuid
import os

from fastapi import Request
from fastapi.responses import JSONResponse
from typing import List

model = None
scaler = None
feature_columns = None
metadata = None
start_time = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/taxi_tip_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "models/feature_columns.json")
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_columns, metadata, start_time

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_columns = json.load(f)

    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    start_time = time.time()
    yield

app = FastAPI(
    title="Taxi Tip Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

class TripInput(BaseModel):
    trip_distance: float = Field(..., gt=0, le=100, description="Trip distance in miles")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Day of week")
    fare_amount: float = Field(..., ge=0, le=500, description="Fare amount in USD")
    trip_duration_minutes: float = Field(..., gt=0, le=300, description="Trip duration in minutes")
    passenger_count: int = Field(..., ge=1, le=8, description="Passenger count")
    RatecodeID: int = Field(..., ge=1, le=6, description="Taxi rate code")
    trip_speed_mph: float = Field(..., ge=0, le=100, description="Estimated speed in mph")
    payment_type: int = Field(..., ge=1, le=6, description="Payment type")
    pickup_borough: str = Field(..., description="Pickup borough")
    dropoff_borough: str = Field(..., description="Dropoff borough")

class PredictionResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str

class BatchInput(BaseModel):
    records: List[TripInput] = Field(..., max_length=100)

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float

def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    df["is_weekend"] = (df["pickup_day_of_week"] >= 5).astype(int)
    df["log_trip_distance"] = np.log1p(df["trip_distance"])
    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, 1)
    df["fare_per_minute"] = df["fare_amount"] / df["trip_duration_minutes"].replace(0, 1)

    df = pd.get_dummies(df, columns=["pickup_borough", "dropoff_borough"])
    df = df.astype({col: "int" for col in df.columns if "borough" in col})

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripInput):
    input_df = pd.DataFrame([input_data.model_dump()])
    processed = preprocess_input(input_df)
    processed_scaled = scaler.transform(processed)

    prediction = model.predict(processed_scaled)[0]

    return PredictionResponse(
        prediction=round(float(prediction), 2),
        prediction_id=str(uuid.uuid4()),
        model_version=metadata["version"]
    )

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    start = time.time()
    predictions = []

    for record in batch.records:
        input_df = pd.DataFrame([record.model_dump()])
        processed = preprocess_input(input_df)
        processed_scaled = scaler.transform(processed)

        prediction = model.predict(processed_scaled)[0]

        predictions.append(
            PredictionResponse(
                prediction=round(float(prediction), 2),
                prediction_id=str(uuid.uuid4()),
                model_version=metadata["version"]
            )
        )

    elapsed = (time.time() - start) * 1000

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2)
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": metadata["version"],
        "uptime_seconds": round(time.time() - start_time, 1)
    }

@app.get("/model/info")
def model_info():
    return {
        "model_name": metadata["model_name"],
        "version": metadata["version"],
        "features": metadata["features"],
        "metrics": metadata["metrics"]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again."
        },
    )