"""
Load and split the raw dataset for MECFS vs Depression Classification.

- Loads raw CSV data
- Splits into train/eval/holdout sets
- Saves splits to data/raw/
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_and_split_data(
    raw_path: str = "data/raw/me_cfs_vs_depression_dataset.csv",
    output_dir: Path | str = DATA_DIR,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load raw dataset and split into train/eval/holdout sets.
    
    Parameters
    ----------
    raw_path : str
        Path to raw CSV file
    output_dir : Path | str
        Directory to save split files
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from remaining after test)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (train_df, eval_df, holdout_df)
    """
    df = pd.read_csv(raw_path)
    
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['diagnosis'].value_counts()}")
    
    train_val, holdout_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['diagnosis']
    )
    
    # Second split: separate train and eval from remaining data
    adjusted_val_size = val_size / (1 - test_size)
    
    train_df, eval_df = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val['diagnosis']
    )
    
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)
    
    print(f"Data split completed (saved to {outdir})")
    print(f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")
    print(f"   Train target distribution:\n{train_df['diagnosis'].value_counts()}")
    print(f"   Eval target distribution:\n{eval_df['diagnosis'].value_counts()}")
    print(f"   Holdout target distribution:\n{holdout_df['diagnosis'].value_counts()}")
    
    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
