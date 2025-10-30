"""
Feature engineering for MECFS vs Depression Classification.

- Applies categorical encoding (label, ordinal, one-hot)
- Applies RobustScaler for numeric features
- Saves encoders and scaler for inference
- Saves feature-engineered data to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def label_encoder_transform(data: pd.DataFrame, column: str, label_encoder: LabelEncoder = None) -> LabelEncoder:
    """
    Apply label encoding to a column.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    column : str
        Column to encode
    label_encoder : LabelEncoder, optional
        Fitted encoder. If None, will fit on data
        
    Returns
    -------
    LabelEncoder
        Fitted encoder
    """
    if label_encoder is None:
        label_encoder = LabelEncoder()
        data[column] = label_encoder.fit_transform(data[column])
    else:
        data[column] = label_encoder.transform(data[column])
    
    return label_encoder


def ordinal_encoder_transform(data: pd.DataFrame, column: str, mapping: dict) -> None:
    """Apply ordinal encoding using provided mapping."""
    data[column] = data[column].map(mapping)


def one_hot_encoding(data: pd.DataFrame, column: str, expected_columns: list = None) -> None:
    """
    Apply one-hot encoding to a column (k-1 dummy variables).
    Matches notebook behavior: skips first unique category (unique()[0]).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    column : str
        Column to encode
    expected_columns : list, optional
        If provided, create only these specific one-hot columns (with underscores).
        Column names should use underscores, but original data may have spaces.
    """
    if expected_columns is not None:
        for col_name in expected_columns:
            category_with_underscore = col_name.replace(f"{column}_", "")
            category_with_space = category_with_underscore.replace("_", " ")
            data[col_name] = (data[column] == category_with_space).astype(int)
    else:
        categories = data[column].unique()
        if len(categories) > 1:
            for category in categories[1:]:  
                normalized_category = category.replace(" ", "_")
                data[f"{column}_{normalized_category}"] = (data[column] == category).astype(int)
    
    data.drop(columns=[column], inplace=True)


def apply_robust_scaler(data: pd.DataFrame, numeric_columns: list, scaler: RobustScaler = None) -> RobustScaler:
    """
    Apply RobustScaler to numeric columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    numeric_columns : list
        List of numeric column names
    scaler : RobustScaler, optional
        Fitted scaler. If None, will fit on data
        
    Returns
    -------
    RobustScaler
        Fitted scaler
    """
    if scaler is None:
        scaler = RobustScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    else:
        data[numeric_columns] = scaler.transform(data[numeric_columns])
    
    return scaler


def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and save outputs + encoders.
    
    Parameters
    ----------
    in_train_path : Path | str | None
        Path to cleaned training data
    in_eval_path : Path | str | None
        Path to cleaned evaluation data
    in_holdout_path : Path | str | None
        Path to cleaned holdout data
    output_dir : Path | str
        Output directory for processed data
        
    Returns
    -------
    tuple
        (train_df, eval_df, holdout_df, label_encoder, scaler)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"
    
    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)
    
    print("Feature engineering starting...")
    print(f"Train shape: {train_df.shape}")
    print(f"Eval shape: {eval_df.shape}")
    print(f"Holdout shape: {holdout_df.shape}")
    
    train_df = train_df.copy()
    eval_df = eval_df.copy()
    holdout_df = holdout_df.copy()
    
    print("\n1. Label encoding diagnosis...")
    label_encoder = LabelEncoder()
    
    categories_by_freq = train_df['diagnosis'].value_counts().index.tolist()
    label_encoder.fit(categories_by_freq)
    
    train_df['diagnosis'] = label_encoder.transform(train_df['diagnosis'])
    eval_df['diagnosis'] = label_encoder.transform(eval_df['diagnosis'])
    holdout_df['diagnosis'] = label_encoder.transform(holdout_df['diagnosis'])
    
    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"   Diagnosis mapping: {mapping}")
    print(f"   Expected: 0=Depression, 1=ME/CFS, 2=Both")
    
    print("\n2. Ordinal encoding...")
    exercise_frequency_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Daily": 4}
    social_activity_mapping = {"Very low": 0, "Low": 1, "Medium": 2, "High": 3, "Very high": 4}
    
    for df in [train_df, eval_df, holdout_df]:
        ordinal_encoder_transform(df, "exercise_frequency", exercise_frequency_mapping)
        ordinal_encoder_transform(df, "social_activity_level", social_activity_mapping)
    
    print("   - exercise_frequency: Never(0) -> Rarely(1) -> Sometimes(2) -> Often(3) -> Daily(4)")
    print("   - social_activity_level: Very low(0) -> Low(1) -> Medium(2) -> High(3) -> Very high(4)")
    
    print("\n3. One-hot encoding...")
    expected_one_hot_columns = {
        "work_status": ["work_status_Partially working", "work_status_Not working"],
        "gender": ["gender_Female"],
        "meditation_or_mindfulness": ["meditation_or_mindfulness_No"]
    }
    
    for df in [train_df, eval_df, holdout_df]:
        for col, expected_cols in expected_one_hot_columns.items():
            one_hot_encoding(df, col, expected_columns=expected_cols)
    
    print(f"   - Applied one-hot encoding to: work_status, gender, meditation_or_mindfulness")
    print(f"   - Created columns: {[col for cols in expected_one_hot_columns.values() for col in cols]}")
    
    print("\n4. Converting pem_present to int...")
    for df in [train_df, eval_df, holdout_df]:
        df["pem_present"] = df["pem_present"].astype(int)
    
    print("\n5. Identifying numeric columns...")
    numeric_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    one_hot_cols = [col for cols in expected_one_hot_columns.values() for col in cols]
    ordinal_cols = ["social_activity_level", "exercise_frequency"]
    numeric_columns = [
        col for col in numeric_columns
        if col != 'diagnosis' and col not in one_hot_cols and col not in ordinal_cols
    ]
    
    print(f"   Numeric columns for scaling ({len(numeric_columns)}): {numeric_columns}")
    
    print("\n6. Applying RobustScaler...")
    scaler = RobustScaler()
    cleaned_combined_path = PROCESSED_DIR / "cleaning_data.csv"
    if cleaned_combined_path.exists():
        cleaned_all = pd.read_csv(cleaned_combined_path)
        scaler.fit(cleaned_all[numeric_columns])
        print("   - RobustScaler fitted on cleaning_data.csv (combined cleaned data)")
    else:
        try:
            cleaned_train = pd.read_csv(PROCESSED_DIR / "cleaning_train.csv")
            cleaned_eval = pd.read_csv(PROCESSED_DIR / "cleaning_eval.csv")
            cleaned_holdout = pd.read_csv(PROCESSED_DIR / "cleaning_holdout.csv")
            cleaned_all = pd.concat([cleaned_train, cleaned_eval, cleaned_holdout], ignore_index=True)
            scaler.fit(cleaned_all[numeric_columns])
            print("   - RobustScaler fitted on full cleaned dataset (train+eval+holdout)")
        except Exception:
            scaler.fit(train_df[numeric_columns])
            print("- RobustScaler fitted on training data (fallback)")

    train_df[numeric_columns] = scaler.transform(train_df[numeric_columns])
    eval_df[numeric_columns] = scaler.transform(eval_df[numeric_columns])
    holdout_df[numeric_columns] = scaler.transform(holdout_df[numeric_columns])
    
    print("\n7. Saving encoders and scaler...")
    dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    dump(scaler, MODELS_DIR / "robust_scaler.pkl")
    
    ordinal_mappings = {
        "exercise_frequency": exercise_frequency_mapping,
        "social_activity_level": social_activity_mapping
    }
    dump(ordinal_mappings, MODELS_DIR / "ordinal_mappings.pkl")

    print("   - Computing imputer statistics for inference consistency")
    try:
        cleaned_all = pd.read_csv(PROCESSED_DIR / "cleaning_data.csv")
    except Exception:
        cleaned_all = None

    imputer_stats = {"numeric_median": {}, "categorical_mode": {}}
    if cleaned_all is not None:
        num_cols = cleaned_all.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = cleaned_all.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in num_cols:
            imputer_stats["numeric_median"][col] = float(cleaned_all[col].median())
        for col in cat_cols:
            if cleaned_all[col].dropna().empty:
                continue
            imputer_stats["categorical_mode"][col] = str(cleaned_all[col].mode(dropna=True).iloc[0])
        dump(imputer_stats, MODELS_DIR / "imputer_stats.pkl")
        print("   - Saved imputer_stats.pkl")
    else:
        print("   - Skipped imputer stats (cleaned splits not found)")
    
    print("   - Saved label_encoder.pkl, robust_scaler.pkl, ordinal_mappings.pkl")
    
    print("\n8. Saving feature-engineered data...")
    train_df.to_csv(output_dir / "feature_engineered_train.csv", index=False)
    eval_df.to_csv(output_dir / "feature_engineered_eval.csv", index=False)
    holdout_df.to_csv(output_dir / "feature_engineered_holdout.csv", index=False)
    
    print("Feature engineering complete!")
    print(f"   Train shape: {train_df.shape}")
    print(f"   Eval shape: {eval_df.shape}")
    print(f"   Holdout shape: {holdout_df.shape}")
    print(f"   Final columns: {list(train_df.columns)}")
    
    return train_df, eval_df, holdout_df, label_encoder, scaler


if __name__ == "__main__":
    run_feature_engineering()
