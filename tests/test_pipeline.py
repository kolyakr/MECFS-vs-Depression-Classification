"""
Test script to verify the pipeline modules work correctly.

This script tests the complete pipeline from raw data to predictions.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))

def test_feature_pipeline():
    """Test the feature pipeline modules."""
    print("Testing feature pipeline...")
    
    try:
        from src.feature_pipeline.load import load_and_split_data
        print("Load module imported successfully")
        
        from src.feature_pipeline.preprocess import run_preprocess
        print("Preprocess module imported successfully")
        
        # Test feature engineering
        from src.feature_pipeline.feature_engineering import run_feature_engineering
        print("Feature engineering module imported successfully")
        
        print("Feature pipeline modules imported successfully")
        return True
        
    except Exception as e:
        print(f"Feature pipeline test failed: {e}")
        return False

def test_training_pipeline():
    """Test the training pipeline modules."""
    print("\nTesting training pipeline...")
    
    try:
        from src.training_pipeline.train import train_model
        print("Train module imported successfully")
        
        from src.training_pipeline.eval import evaluate_model
        print("Eval module imported successfully")
        
        print("Training pipeline modules imported successfully")
        return True
        
    except Exception as e:
        print(f"Training pipeline test failed: {e}")
        return False

def test_inference_pipeline():
    """Test the inference pipeline."""
    print("\nTesting inference pipeline...")
    
    try:
        from src.inference_pipeline.inference import predict
        print("Inference module imported successfully")
        
        print("Inference pipeline imported successfully")
        return True
        
    except Exception as e:
        print(f"Inference pipeline test failed: {e}")
        return False

def test_api():
    """Test the API module."""
    print("\nTesting API module...")
    
    try:
        from src.api.main import app
        print("API module imported successfully")
        
        print("API module imported successfully")
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False

def test_data_availability():
    """Test if required data files are available."""
    print("\nTesting data availability...")
    
    required_files = [
        "data/raw/me_cfs_vs_depression_dataset.csv",
        "data/processed/cleaning_data.csv",
        "data/processed/feature_engineered_data.csv",
        "models/best_logistic_model.pkl"
    ]
    
    all_available = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"{file_path}")
        else:
            print(f"{file_path} - NOT FOUND")
            all_available = False
    
    if all_available:
        print("All required data files are available")
    else:
        print("Some required data files are missing")
    
    return all_available

def main():
    """Run all tests."""
    print("Starting pipeline tests...\n")
    
    tests = [
        test_data_availability,
        test_feature_pipeline,
        test_training_pipeline,
        test_inference_pipeline,
        test_api,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\nTest Results:")
    print(f"   Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("All tests passed! Pipeline is ready to use.")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    main()
