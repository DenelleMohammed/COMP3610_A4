from fastapi.testclient import TestClient
from app import app


valid_payload = {
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


def test_predict_valid():
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "prediction_id" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], (int, float))


def test_batch_prediction():
    batch_payload = {
        "records": [
            valid_payload,
            {
                "trip_distance": 5.0,
                "pickup_hour": 9,
                "pickup_day_of_week": 4,
                "fare_amount": 20.0,
                "trip_duration_minutes": 18,
                "passenger_count": 2,
                "RatecodeID": 1,
                "trip_speed_mph": 16,
                "payment_type": 1,
                "pickup_borough": "Queens",
                "dropoff_borough": "Manhattan"
            }
        ]
    }

    with TestClient(app) as client:
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert "processing_time_ms" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2


def test_predict_missing_field():
    bad_payload = valid_payload.copy()
    del bad_payload["fare_amount"]

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


def test_predict_invalid_type():
    bad_payload = valid_payload.copy()
    bad_payload["trip_distance"] = "far"

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


def test_predict_out_of_range():
    bad_payload = valid_payload.copy()
    bad_payload["pickup_hour"] = 30

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "model_version" in data


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "features" in data
        assert "metrics" in data


def test_zero_distance_edge_case():
    bad_payload = valid_payload.copy()
    bad_payload["trip_distance"] = 0

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422