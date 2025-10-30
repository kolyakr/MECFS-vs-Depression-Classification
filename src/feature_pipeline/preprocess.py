"""
Preprocessing for MECFS vs Depression Classification.

- Handles missing value imputation (median for numeric, mode for categorical)
- Saves cleaned data to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def median_imputer(data: pd.DataFrame, column: str) -> None:
    """Fill missing values in numeric column with median."""
    median = data[column].median()
    data.loc[data[column].isna(), column] = median


def mode_imputation(data: pd.DataFrame, column: str) -> None:
    """Fill missing values in categorical column with mode."""
    most_frequent_category = data[column].value_counts().idxmax()
    data.loc[data[column].isna(), column] = most_frequent_category


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)
    
    print(f"Processing {split} split: {df.shape}")
    
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values != 0]
    
    if len(missing_columns) > 0:
        print(f"Missing values found in {len(missing_columns)} columns:")
        print(missing_columns)
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        
        for col in missing_columns.index:
            if col in numeric_columns:
                median_imputer(df, col)
                print(f"  - Filled {col} with median")
            elif col in categorical_columns:
                mode_imputation(df, col)
                print(f"  - Filled {col} with mode")
    else:
        print("  - No missing values found")
    
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values still remain")
    else:
        print("  All missing values handled")
    
    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"Preprocessed {split} saved to {out_path} ({df.shape})")
    
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
):
    """Run preprocessing for all splits."""
    for split in splits:
        preprocess_split(split, raw_dir=raw_dir, processed_dir=processed_dir)


if __name__ == "__main__":
    run_preprocess()
