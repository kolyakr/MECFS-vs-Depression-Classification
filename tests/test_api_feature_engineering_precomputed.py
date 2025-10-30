"""
Compare API-engineered features with the precomputed feature_engineered_data.csv.

Process:
- Load N rows from data/raw/me_cfs_vs_depression_dataset.csv
- Call /predict?include_features=true with those raw rows
- Load same N rows from data/processed/feature_engineered_data.csv
- Align feature columns (drop diagnosis) and compare values with tolerance
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _files_exist():
    return all(
        (
            (PROJECT_ROOT / "data" / "raw" / "me_cfs_vs_depression_dataset.csv").exists(),
            (PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv").exists(),
            (PROJECT_ROOT / "models" / "best_logistic_model.pkl").exists(),
        )
    )


def test_api_engineering_matches_precomputed():
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
    except Exception as e:
        import pytest
        pytest.skip(f"Dependencies not available: {e}")

    if not _files_exist():
        import pytest
        pytest.skip("Required raw or precomputed files are missing")

    raw_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "me_cfs_vs_depression_dataset.csv")
    fe_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv")

    N = 5
    no_na_idx = raw_df.dropna().index[:N]
    if len(no_na_idx) < N:
        import pytest
        pytest.skip("Not enough non-missing rows to run comparison")
    raw_slice = raw_df.loc[no_na_idx].copy()
    fe_slice = fe_df.loc[no_na_idx].copy()

    records = raw_slice.drop(columns=["diagnosis"], errors="ignore").to_dict(orient="records")
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

    pre_cols = [c for c in fe_slice.columns if c != "diagnosis"]

    common_cols = [c for c in api_feature_cols if c in pre_cols]
    assert len(common_cols) == len(api_feature_cols), (
        f"Column mismatch. API has {len(api_feature_cols)} feature cols,"
        f" but only {len(common_cols)} are present in precomputed data."
    )

    pre_X = fe_slice[common_cols]
    api_X = api_X[common_cols]

    api_arr = api_X.to_numpy(dtype=float)
    pre_arr = pre_X.to_numpy(dtype=float)
    diff = np.abs(api_arr - pre_arr)
    max_abs_diff = float(np.nanmax(diff)) if diff.size else 0.0

    if max_abs_diff >= 1e-8:
        col_diffs = {}
        for j, col in enumerate(common_cols):
            col_max = float(np.nanmax(diff[:, j]))
            if col_max >= 1e-8:
                col_diffs[col] = col_max
        print("\n⚠️ Columns with differences (max abs diff per column):")
        for col, d in col_diffs.items():
            print(f"  {col}: {d}")
        print("\nSample row diffs (first 5 rows):")
        for i in range(min(5, len(api_X))):
            row_diff = diff[i]
            if np.any(row_diff >= 1e-8):
                print(f"Row {i}:")
                for j, col in enumerate(common_cols):
                    if row_diff[j] >= 1e-8:
                        print(f"  {col}: api={api_arr[i, j]}, pre={pre_arr[i, j]}, diff={row_diff[j]}")

    assert max_abs_diff < 1e-8, f"Engineered features differ from precomputed (max abs diff {max_abs_diff})"


