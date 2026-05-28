"""
API tests for FastAPI app.

These tests validate:
- /health returns healthy when artifacts exist
- /predict returns predictions with correct schema

Note: Requires fastapi and uvicorn installed in the environment.
"""

from pathlib import Path
import json
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _artifacts_available():
    return all(
        (
            (PROJECT_ROOT / "models" / "best_logistic_model.pkl").exists(),
            (PROJECT_ROOT / "models" / "label_encoder.pkl").exists(),
            (PROJECT_ROOT / "models" / "robust_scaler.pkl").exists(),
            (PROJECT_ROOT / "models" / "ordinal_mappings.pkl").exists(),
            (PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv").exists(),
        )
    )


@pytest.mark.skipif(not _artifacts_available(), reason="Required artifacts missing")
def test_health_endpoint():
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
    except Exception as e:
        pytest.skip(f"FastAPI not available or import failed: {e}")

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "healthy"
    assert body.get("message")
    assert "n_features_expected" in body


@pytest.mark.skipif(not _artifacts_available(), reason="Required artifacts missing")
def test_predict_endpoint():
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
    except Exception as e:
        pytest.skip(f"FastAPI not available or import failed: {e}")

    client = TestClient(app)

    payload = [
        {
            "age": 45,
            "gender": "Female",
            "sleep_quality_index": 6.0,
            "brain_fog_level": 7.0,
            "physical_pain_score": 5.0,
            "stress_level": 6.0,
            "depression_phq9_score": 12.0,
            "fatigue_severity_scale_score": 7.5,
            "pem_duration_hours": 24.0,
            "hours_of_sleep_per_night": 7.0,
            "pem_present": 1,
            "work_status": "Working",
            "social_activity_level": "Low",
            "exercise_frequency": "Often",
            "meditation_or_mindfulness": "Yes",
        }
    ]

    r = client.post("/predict", content=json.dumps(payload))
    assert r.status_code == 200
    body = r.json()
    assert body.get("n_predictions") == 1
    assert isinstance(body.get("predictions"), list)
    assert isinstance(body.get("probabilities"), dict)
    assert "Depression" in body["probabilities"]
    assert "ME/CFS" in body["probabilities"]
    assert "Both" in body["probabilities"]

