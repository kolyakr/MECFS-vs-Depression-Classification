"""
Inference pipeline for MECFS vs Depression Classification.

- Takes raw input data (same schema as original dataset)
- Applies preprocessing + feature engineering using saved encoders
- Aligns features with training schema
- Returns predictions with probabilities
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

from src.feature_pipeline.preprocess import median_imputer, mode_imputation
from src.feature_pipeline.feature_engineering import (
    label_encoder_transform, ordinal_encoder_transform, 
    one_hot_encoding, apply_robust_scaler
)

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_logistic_model.pkl"
DEFAULT_LABEL_ENCODER = PROJECT_ROOT / "models" / "label_encoder.pkl"
DEFAULT_SCALER = PROJECT_ROOT / "models" / "robust_scaler.pkl"
DEFAULT_ORDINAL_MAPPINGS = PROJECT_ROOT / "models" / "ordinal_mappings.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("Inference using project root:", PROJECT_ROOT)

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "diagnosis"]
else:
    TRAIN_FEATURE_COLUMNS = None
    print("⚠️ Warning: Training feature columns not found. Schema alignment will be skipped.")


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    label_encoder_path: Path | str = DEFAULT_LABEL_ENCODER,
    scaler_path: Path | str = DEFAULT_SCALER,
    ordinal_mappings_path: Path | str = DEFAULT_ORDINAL_MAPPINGS,
) -> pd.DataFrame:
    """
    Run inference on raw input data.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Raw input data with original schema
    model_path : Path | str
        Path to trained model
    label_encoder_path : Path | str
        Path to label encoder
    scaler_path : Path | str
        Path to robust scaler
    ordinal_mappings_path : Path | str
        Path to ordinal mappings
        
    Returns
    -------
    pd.DataFrame
        Predictions with probabilities and feature-engineered data
    """
    print("    Starting inference pipeline...")
    print(f"   Input shape: {input_df.shape}")
    
    df = input_df.copy()
    
    print("\n1. Handling missing values...")
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values != 0]

    imputer_stats = None
    try:
        from joblib import load as _load
        imputer_stats_path = PROJECT_ROOT / "models" / "imputer_stats.pkl"
        if imputer_stats_path.exists():
            imputer_stats = _load(imputer_stats_path)
    except Exception:
        imputer_stats = None

    if len(missing_columns) > 0:
        if imputer_stats is not None:
            print(f"   Found missing values in {len(missing_columns)} columns")
            for col in missing_columns.index:
                if col in imputer_stats.get("numeric_median", {}):
                    df[col] = df[col].fillna(imputer_stats["numeric_median"][col])
                    print(f"   - Filled {col} with saved median")
                elif col in imputer_stats.get("categorical_mode", {}):
                    df[col] = df[col].fillna(imputer_stats["categorical_mode"][col])
                    print(f"   - Filled {col} with saved mode")
                else:
                    if df[col].dtype.kind in "biufc":
                        median_imputer(df, col)
                        print(f"   - Filled {col} with median (fallback)")
                    else:
                        mode_imputation(df, col)
                        print(f"   - Filled {col} with mode (fallback)")
        else:
            print(f"   Found missing values in {len(missing_columns)} columns")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns
            for col in missing_columns.index:
                if col in numeric_columns:
                    median_imputer(df, col)
                    print(f"   - Filled {col} with median")
                elif col in categorical_columns:
                    mode_imputation(df, col)
                    print(f"   - Filled {col} with mode")
    else:
        print("   - No missing values found")
    
    print("\n2. Loading encoders and scaler...")
    
    if not Path(label_encoder_path).exists():
        raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not Path(ordinal_mappings_path).exists():
        raise FileNotFoundError(f"Ordinal mappings not found at {ordinal_mappings_path}")
    
    label_encoder = load(label_encoder_path)
    scaler = load(scaler_path)
    ordinal_mappings = load(ordinal_mappings_path)
    
    print("   - Encoders and scaler loaded successfully")
    
    print("\n3. Applying feature engineering...")
    
    if "diagnosis" in df.columns:
        df["diagnosis"] = label_encoder.transform(df["diagnosis"])
        print("   - Applied label encoding to diagnosis")
    
    if "exercise_frequency" in df.columns:
        ordinal_encoder_transform(df, "exercise_frequency", ordinal_mappings["exercise_frequency"])
        print("   - Applied ordinal encoding to exercise_frequency")
    
    if "social_activity_level" in df.columns:
        ordinal_encoder_transform(df, "social_activity_level", ordinal_mappings["social_activity_level"])
        print("   - Applied ordinal encoding to social_activity_level")
    
    expected_one_hot_columns = {
        "work_status": ["work_status_Partially working", "work_status_Not working"],
        "gender": ["gender_Female"],
        "meditation_or_mindfulness": ["meditation_or_mindfulness_No"]
    }
    for col, expected_cols in expected_one_hot_columns.items():
        if col in df.columns:
            one_hot_encoding(df, col, expected_columns=expected_cols)
            print(f"   - Applied one-hot encoding to {col}")
    
    if "pem_present" in df.columns:
        df["pem_present"] = df["pem_present"].astype(int)
        print("   - Converted pem_present to int")
    
    all_numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    one_hot_cols = [col for cols in expected_one_hot_columns.values() for col in cols]
    ordinal_cols = ["social_activity_level", "exercise_frequency"]
    numeric_columns = [
        col for col in all_numeric_columns
        if col != 'diagnosis' and col not in one_hot_cols and col not in ordinal_cols
    ]
    
    if numeric_columns:
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        print(f"   - Applied RobustScaler to {len(numeric_columns)} numeric columns: {numeric_columns}")
    
    y_true = None
    if "diagnosis" in df.columns:
        y_true = df["diagnosis"].tolist()
        df = df.drop(columns=["diagnosis"])
        print("   - Separated target variable for evaluation")
    
    print(f"\n4. Aligning features with training schema...")
    expected_feature_columns = None
    try:
        model_tmp = load(model_path)
        if hasattr(model_tmp, "feature_names_in_"):
            expected_feature_columns = list(model_tmp.feature_names_in_)
            print(f"   Using feature names from model: {len(expected_feature_columns)}")
    except Exception:
        expected_feature_columns = None

    if expected_feature_columns is None:
        expected_feature_columns = TRAIN_FEATURE_COLUMNS

    if expected_feature_columns is not None:
        print(f"   Expected training features: {len(expected_feature_columns)}")
        print(f"   Current features: {len(df.columns)}")

        missing_cols = set(expected_feature_columns) - set(df.columns)
        if missing_cols:
            print(f"   Adding missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = 0

        extra_cols = set(df.columns) - set(expected_feature_columns)
        if extra_cols:
            print(f"   Removing extra columns: {extra_cols}")
            df = df.drop(columns=extra_cols)

        df = df.reindex(columns=expected_feature_columns, fill_value=0)
        print(f"   Final feature count: {len(df.columns)}")
    
    print(f"\n5. Loading model and making predictions...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load(model_path)
   
    predictions = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    print(f"   - Model loaded: {type(model).__name__}")
    print(f"   - Predictions shape: {predictions.shape}")
    
    print(f"\n6. Building output...")
    print(f"   Label encoder classes (alphabetical): {label_encoder.classes_}")

    output_df = df.copy()

    model_label_to_id = {"Depression": 0, "ME/CFS": 1, "Both": 2}
    model_id_to_label = {v: k for k, v in model_label_to_id.items()}

    predictions_label = [model_id_to_label[p] for p in predictions]
    predictions_id = predictions.tolist()  

    output_df["predicted_diagnosis"] = predictions_label
    output_df["predicted_diagnosis_id"] = predictions_id

    for label, idx in model_label_to_id.items():
        output_df[f"prob_{label}"] = prediction_proba[:, idx]

    if y_true is not None:
        output_df["actual_diagnosis"] = y_true
        print("   - Added actual diagnosis for evaluation")

    output_df["prediction_confidence"] = np.max(prediction_proba, axis=1)

    print(f"   Inference complete!")
    print(f"   Output shape: {output_df.shape}")
    print(f"   Numeric predictions (Depression=0, ME/CFS=1, Both=2): {predictions_id}")
    print(f"   String predictions: {predictions_label}")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new MECFS/Depression data")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--label_encoder", type=str, default=str(DEFAULT_LABEL_ENCODER), help="Path to label encoder")
    parser.add_argument("--scaler", type=str, default=str(DEFAULT_SCALER), help="Path to robust scaler")
    parser.add_argument("--ordinal_mappings", type=str, default=str(DEFAULT_ORDINAL_MAPPINGS), help="Path to ordinal mappings")
    
    args = parser.parse_args()
    
    input_df = pd.read_csv(args.input)
    
    predictions_df = predict(
        input_df,
        model_path=args.model,
        label_encoder_path=args.label_encoder,
        scaler_path=args.scaler,
        ordinal_mappings_path=args.ordinal_mappings,
    )
    
    predictions_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
