# Taxi Tip Prediction API

This repository contains a FastAPI service that predicts taxi tips from trip details using a pre-trained scikit-learn model. The project also includes the model artifacts, a test suite, Docker support, and an MLflow tracking setup for experiment history.

## What Is In This Repo

- `app.py` exposes the API and loads the trained model, scaler, feature list, and metadata at startup.
- `test_app.py` contains API tests built with `pytest` and `TestClient`.
- `models/` stores the serialized model assets used by the service.
- `mlartifacts/` contains saved MLflow run artifacts.
- `assignment4.ipynb` is the notebook used for analysis and model development.
- `cleaned_taxi.parquet` is the processed dataset available in the workspace.
- `Dockerfile` and `docker-compose.yml` provide containerized runtime options.

## Model Summary

The API uses a taxi tip regressor with the following recorded metadata:

- Model name: `taxi-tip-regressor`
- Version: `1.0.0`
- Metrics: MAE `1.19`, RMSE `2.382`, R2 `0.623`

The request schema accepts trip-level fields such as distance, pickup time, fare amount, trip duration, passenger count, rate code, speed, payment type, and pickup/dropoff boroughs. The service also creates engineered features internally, including weekend flags, log distance, fare per mile, fare per minute, and borough one-hot columns.

## Project Structure

```text
app.py                    FastAPI application
assignment4.ipynb         Notebook for analysis and training work
cleaned_taxi.parquet      Cleaned dataset used in the project
docker-compose.yml        Multi-service Docker setup
Dockerfile                Image definition for the API service
models/                   Serialized model, scaler, and metadata files
mlartifacts/              MLflow artifact history
requirements.txt          Python dependencies
test_app.py               API tests
```

## Requirements

- Python 3.11+
- `pip`
- Optional: Docker and Docker Compose for containerized execution

## Local Setup

Install the dependencies and start the API locally:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The service reads model files from `models/` by default. You can override the paths with these environment variables if needed:

- `MODEL_PATH`
- `SCALER_PATH`
- `FEATURE_COLUMNS_PATH`
- `MODEL_METADATA_PATH`
- `MLFLOW_TRACKING_URI`

Once the server is running, the interactive docs are available at `http://localhost:8000/docs`.

## Docker Setup

Build and run the stack with Docker Compose:

```bash
docker compose up --build
```

With the Compose configuration in this repo:

- The API is exposed on `http://localhost:8001`
- MLflow is exposed on `http://localhost:5001`

The API container runs Uvicorn on port `8000` internally, and the MLflow service is used as the tracking backend referenced by the app.

To stop the stack:

```bash
docker compose down
```

## API Endpoints

### `GET /health`

Returns basic service status, whether the model is loaded, the model version, and uptime.

### `GET /model/info`

Returns model metadata such as the model name, version, feature list, and evaluation metrics.

### `POST /predict`

Returns a single prediction for one trip record.

Example request:

```json
{
	"trip_distance": 2.5,
	"pickup_hour": 14,
	"pickup_day_of_week": 2,
	"fare_amount": 12.5,
	"trip_duration_minutes": 10,
	"passenger_count": 1,
	"RatecodeID": 1,
	"trip_speed_mph": 15,
	"payment_type": 1,
	"pickup_borough": "Manhattan",
	"dropoff_borough": "Brooklyn"
}
```

Example response:

```json
{
	"prediction": 3.42,
	"prediction_id": "8d5d8d3a-9c4e-4c2c-b8c8-6cc2d65f5f85",
	"model_version": "1.0.0"
}
```

### `POST /predict/batch`

Accepts up to 100 records in a single request and returns a prediction for each one, along with the batch size and processing time.

Example request shape:

```json
{
	"records": [
		{
			"trip_distance": 2.5,
			"pickup_hour": 14,
			"pickup_day_of_week": 2,
			"fare_amount": 12.5,
			"trip_duration_minutes": 10,
			"passenger_count": 1,
			"RatecodeID": 1,
			"trip_speed_mph": 15,
			"payment_type": 1,
			"pickup_borough": "Manhattan",
			"dropoff_borough": "Brooklyn"
		}
	]
}
```

## Testing

Run the API tests with:

```bash
pytest
```

The tests cover single and batch prediction requests, validation errors, the health endpoint, and the model info endpoint.

## MLflow

The repository includes MLflow artifacts and a Compose service for tracking. If you want to inspect experiments locally, start the MLflow service and open the exposed port in your browser.

## Notes

- The app loads `models/taxi_tip_model.pkl`, `models/scaler.pkl`, `models/feature_columns.json`, and `models/model_metadata.json` on startup.
- The API returns standard FastAPI validation errors for missing fields, invalid types, and out-of-range values.
- If you update the model artifacts, make sure the feature list in `models/feature_columns.json` stays aligned with the preprocessing logic in `app.py`.