"""
Simple test to verify the pipeline modules work.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        print("Testing imports...")
        
        from src.feature_pipeline.load import load_and_split_data
        from src.feature_pipeline.preprocess import run_preprocess
        from src.feature_pipeline.feature_engineering import run_feature_engineering
        print("Feature pipeline imports successful")
        
        from src.training_pipeline.train import train_model
        from src.training_pipeline.eval import evaluate_model
        print("Training pipeline imports successful")
        
        from src.inference_pipeline.inference import predict
        print("Inference pipeline imports successful")
        
        from src.api.main import app
        print("API imports successful")
        
        print("\nAll imports successful! Pipeline is ready.")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
