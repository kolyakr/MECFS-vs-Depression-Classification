"""
Validate that /predict feature engineering matches the offline pipeline.

Strategy:
- Take a small sample (first 5 rows) from raw holdout data
- Call /predict?include_features=true with the raw records
- Independently run inference.predict on the same records
- Compare the engineered feature matrices (columns and values)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _artifacts_exist():
    return all(
        (
            (PROJECT_ROOT / "models" / "best_logistic_model.pkl").exists(),
            (PROJECT_ROOT / "models" / "label_encoder.pkl").exists(),
            (PROJECT_ROOT / "models" / "robust_scaler.pkl").exists(),
            (PROJECT_ROOT / "models" / "ordinal_mappings.pkl").exists(),
            (PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv").exists(),
            (PROJECT_ROOT / "data" / "raw" / "holdout.csv").exists(),
        )
    )


def test_predict_feature_engineering_matches_offline():
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        from src.inference_pipeline.inference import predict as offline_predict
    except Exception as e:
        import pytest
        pytest.skip(f"Dependencies not available: {e}")

    if not _artifacts_exist():
        import pytest
        pytest.skip("Required artifacts or data are missing")

    raw_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "holdout.csv").head(5)
    input_df = raw_df.drop(columns=["diagnosis"], errors="ignore").copy()

    records = input_df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None

    client = TestClient(app)
    resp = client.post("/predict", params={"include_features": "true"}, json=records)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    api_feature_cols = body.get("feature_columns")
    api_features = body.get("engineered_features")
    assert api_feature_cols and api_features, "API did not return engineered features"

    api_X = pd.DataFrame(api_features)[api_feature_cols]

    offline_out = offline_predict(raw_df.copy())
    expected_cols = api_feature_cols
    offline_X = offline_out[expected_cols]

    assert api_X.columns.tolist() == offline_X.columns.tolist()

    diff = np.abs(api_X.to_numpy(dtype=float) - offline_X.to_numpy(dtype=float))
    max_abs_diff = float(np.nanmax(diff)) if diff.size else 0.0
    assert max_abs_diff < 1e-8, f"Engineered features differ (max abs diff {max_abs_diff})"


