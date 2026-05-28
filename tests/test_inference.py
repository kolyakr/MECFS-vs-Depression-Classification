"""
Tests for the inference pipeline predict() function.

These tests ensure:
- No exceptions during prediction with valid inputs
- Output contains required columns and correct shapes
- Probabilities are within [0, 1] and sum to ~1
- Predicted labels are within expected range
- Feature alignment works when input one-hot names differ (spaces vs underscores)
"""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_paths():
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data" / "processed"
    return {
        "model": models_dir / "best_logistic_model.pkl",
        "label_encoder": models_dir / "label_encoder.pkl",
        "scaler": models_dir / "robust_scaler.pkl",
        "ordinal_mappings": models_dir / "ordinal_mappings.pkl",
        "train_fe": data_dir / "feature_engineered_train.csv",
    }


def test_predict_basic_success():
    paths = _get_paths()
    required = [
        paths["model"],
        paths["label_encoder"],
        paths["scaler"],
        paths["ordinal_mappings"],
        paths["train_fe"],
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        import pytest

        pytest.skip(f"Missing required artifacts: {missing}")

    from src.inference_pipeline.inference import predict

    df = pd.DataFrame(
        [
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
            },
            {
                "age": 38,
                "gender": "Male",
                "sleep_quality_index": 8.0,
                "brain_fog_level": 3.0,
                "physical_pain_score": 2.0,
                "stress_level": 3.0,
                "depression_phq9_score": 5.0,
                "fatigue_severity_scale_score": 3.0,
                "pem_duration_hours": 10.0,
                "hours_of_sleep_per_night": 8.0,
                "pem_present": 0,
                "work_status": "Partially working",
                "social_activity_level": "High",
                "exercise_frequency": "Sometimes",
                "meditation_or_mindfulness": "No",
            },
        ]
    )

    out = predict(
        df,
        model_path=str(paths["model"]),
        label_encoder_path=str(paths["label_encoder"]),
        scaler_path=str(paths["scaler"]),
        ordinal_mappings_path=str(paths["ordinal_mappings"]),
    )

    assert len(out) == 2
    assert "predicted_diagnosis" in out.columns
    assert "prediction_confidence" in out.columns

    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    assert len(prob_cols) == 3  

    probs = out[prob_cols].to_numpy(dtype=float)
    assert np.isfinite(probs).all()
    assert np.logical_and(probs >= 0.0, probs <= 1.0).all()
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    preds = out["predicted_diagnosis"].to_numpy()
    unique_preds = set(np.unique(preds))
    allowed_numeric = {0, 1, 2}
    allowed_labels = {"Depression", "ME/CFS", "Both"}
    assert unique_preds.issubset(allowed_numeric) or unique_preds.issubset(allowed_labels)


def test_predict_alignment_and_renaming():
    paths = _get_paths()
    required = [
        paths["model"],
        paths["label_encoder"],
        paths["scaler"],
        paths["ordinal_mappings"],
        paths["train_fe"],
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        import pytest

        pytest.skip(f"Missing required artifacts: {missing}")

    from src.inference_pipeline.inference import predict

    df = pd.DataFrame(
        [
            {
                "age": 52,
                "gender": "Female",  
                "sleep_quality_index": 7.0,
                "brain_fog_level": 3.0,
                "physical_pain_score": 4.0,
                "stress_level": 5.0,
                "depression_phq9_score": 16.0,
                "fatigue_severity_scale_score": 4.5,
                "pem_duration_hours": 6.0,
                "hours_of_sleep_per_night": 7.5,
                "pem_present": 0,
                "work_status": "Not working",  
                "social_activity_level": "Medium",
                "exercise_frequency": "Often",
                "meditation_or_mindfulness": "No",  
            }
        ]
    )

    out = predict(
        df,
        model_path=str(paths["model"]),
        label_encoder_path=str(paths["label_encoder"]),
        scaler_path=str(paths["scaler"]),
        ordinal_mappings_path=str(paths["ordinal_mappings"]),
    )

    assert len(out) == 1
    prob_cols = [c for c in out.columns if c.startswith("prob_")]
    assert len(prob_cols) == 3
    probs = out[prob_cols].to_numpy(dtype=float)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

