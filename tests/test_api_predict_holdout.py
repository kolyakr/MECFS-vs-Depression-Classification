"""
Test the /predict endpoint using holdout data.

This test:
1. Loads raw holdout data (original format before feature engineering)
2. Sends it to /predict endpoint
3. Compares predictions with actual diagnoses
4. Calculates accuracy and per-class metrics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    if PYTEST_AVAILABLE:
        pass


def load_raw_holdout_data():
    """Load raw holdout data in original format."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "holdout.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Holdout data not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    return df


def load_label_encoder():
    """Load label encoder to decode actual diagnoses."""
    encoder_path = PROJECT_ROOT / "models" / "label_encoder.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
    
    return load(encoder_path)


def test_predict_holdout_accuracy():
    """Test /predict endpoint with holdout data and calculate accuracy."""
    if not FASTAPI_AVAILABLE:
        if PYTEST_AVAILABLE:
            pytest.skip("FastAPI not available")
        else:
            raise ImportError("FastAPI not available - install with: uv add fastapi")
    
    model_path = PROJECT_ROOT / "models" / "best_logistic_model.pkl"
    if not model_path.exists():
        if PYTEST_AVAILABLE:
            pytest.skip("Model not found")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    holdout_df = load_raw_holdout_data()
    label_encoder = load_label_encoder()
    
    input_df = holdout_df.drop(columns=["diagnosis"]).copy()
    
    input_data = input_df.to_dict("records")
    
    for record in input_data:
        for key, value in record.items():
            try:
                if pd.isna(value):
                    record[key] = None
            except (TypeError, ValueError):
                if isinstance(value, float) and np.isnan(value):
                    record[key] = None
    
    actual_diagnoses = holdout_df["diagnosis"].tolist()
    
    client = TestClient(app)
    response = client.post("/predict", json=input_data)
    
    assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
    
    result = response.json()
    
    predicted_diagnoses = result["predictions"]
    probabilities = result["probabilities"]
    confidences = result["confidence"]
    
    assert len(predicted_diagnoses) == len(actual_diagnoses), \
        f"Prediction count ({len(predicted_diagnoses)}) != actual count ({len(actual_diagnoses)})"
    
    correct = sum(1 for pred, actual in zip(predicted_diagnoses, actual_diagnoses) if pred == actual)
    accuracy = correct / len(actual_diagnoses)
    
    print(f"\nHoldout Test Results:")
    print(f"   Total samples: {len(actual_diagnoses)}")
    print(f"   Correct predictions: {correct}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    classes = ["Depression", "ME/CFS", "Both"]
    class_stats = {}
    
    for class_name in classes:
        actual_indices = [i for i, diag in enumerate(actual_diagnoses) if diag == class_name]
        
        if len(actual_indices) == 0:
            continue
            
        correct_class = sum(1 for i in actual_indices if predicted_diagnoses[i] == class_name)
        class_accuracy = correct_class / len(actual_indices)
        
        class_stats[class_name] = {
            "total": len(actual_indices),
            "correct": correct_class,
            "accuracy": class_accuracy
        }
        
        print(f"   {class_name}: {correct_class}/{len(actual_indices)} correct ({class_accuracy*100:.2f}%)")
    
    assert "Depression" in probabilities
    assert "ME/CFS" in probabilities
    assert "Both" in probabilities
    
    for i in range(len(predicted_diagnoses)):
        prob_sum = probabilities["Depression"][i] + probabilities["ME/CFS"][i] + probabilities["Both"][i]
        assert abs(prob_sum - 1.0) < 0.001, f"Probabilities don't sum to 1.0 for sample {i}: {prob_sum}"
    
    assert len(confidences) == len(predicted_diagnoses)
    assert all(0.0 <= conf <= 1.0 for conf in confidences)
    
    assert accuracy > 0.0, "Accuracy should be greater than 0"
    
    print(f"\nSample Predictions (first 5):")
    for i in range(min(5, len(actual_diagnoses))):
        print(f"   Sample {i+1}:")
        print(f"     Actual: {actual_diagnoses[i]}")
        print(f"     Predicted: {predicted_diagnoses[i]}")
        print(f"     Probabilities: Dep={probabilities['Depression'][i]:.3f}, "
              f"ME/CFS={probabilities['ME/CFS'][i]:.3f}, Both={probabilities['Both'][i]:.3f}")
        print(f"     Confidence: {confidences[i]:.3f}")
        print(f"     Correct: {'✓' if predicted_diagnoses[i] == actual_diagnoses[i] else '✗'}")
    
    return accuracy, class_stats


def test_predict_holdout_confusion_matrix():
    """Test /predict endpoint and generate confusion matrix."""
    if not FASTAPI_AVAILABLE:
        if PYTEST_AVAILABLE:
            pytest.skip("FastAPI not available")
        else:
            raise ImportError("FastAPI not available - install with: uv add fastapi")
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        if PYTEST_AVAILABLE:
            pytest.skip("scikit-learn not available")
        else:
            raise ImportError("scikit-learn not available - install with: uv add scikit-learn")
    
    holdout_df = load_raw_holdout_data()
    
    input_df = holdout_df.drop(columns=["diagnosis"]).copy()
    
    input_data = input_df.to_dict("records")
    
    for record in input_data:
        for key, value in record.items():
            try:
                if pd.isna(value):
                    record[key] = None
            except (TypeError, ValueError):
                if isinstance(value, float) and np.isnan(value):
                    record[key] = None
    actual_diagnoses = holdout_df["diagnosis"].tolist()
    
    client = TestClient(app)
    response = client.post("/predict", json=input_data)
    
    assert response.status_code == 200
    result = response.json()
    predicted_diagnoses = result["predictions"]
    
    classes = ["Depression", "ME/CFS", "Both"]
    cm = confusion_matrix(actual_diagnoses, predicted_diagnoses, labels=classes)
    
    print(f"\nConfusion Matrix:")
    print("                  Predicted")
    print("               ", "  ".join(f"{c:>10}" for c in classes))
    for i, class_name in enumerate(classes):
        print(f"Actual {class_name:>10}  ", "  ".join(f"{val:>10}" for val in cm[i]))
    
    report = classification_report(actual_diagnoses, predicted_diagnoses, 
                                   labels=classes, output_dict=True)
    
    print(f"\nClassification Report:")
    for class_name in classes:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name}:")
            print(f"     Precision: {metrics['precision']:.4f}")
            print(f"     Recall: {metrics['recall']:.4f}")
            print(f"     F1-score: {metrics['f1-score']:.4f}")
    
    print(f"\n   Overall:")
    print(f"     Accuracy: {report['accuracy']:.4f}")
    print(f"     Macro avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"     Weighted avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    assert report['accuracy'] > 0.0


if __name__ == "__main__":
    print("Running holdout prediction tests...\n")
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available!")
        print("   Install with: uv add fastapi")
        sys.exit(1)
    
    try:
        accuracy, class_stats = test_predict_holdout_accuracy()
        print("\n" + "="*50 + "\n")
        test_predict_holdout_confusion_matrix()
        print(f"\nAll tests passed!")
        print(f"   Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except FileNotFoundError as e:
        print(f"\nMissing required file: {e}")
        print("   Make sure you've run the feature engineering pipeline first!")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

