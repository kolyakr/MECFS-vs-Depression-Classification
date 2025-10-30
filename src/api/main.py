"""
FastAPI application for MECFS vs Depression Classification.

- Provides REST API endpoints for model predictions
- Health checks and model status
- Batch prediction capabilities
"""

from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

try:
    from src.db.models import Base, Prediction, InferenceInput
    from src.db.session import get_engine, get_session
except Exception:
    Base = None
    Prediction = None
    InferenceInput = None
    def get_engine():
        return create_engine(
            "postgresql+psycopg2://postgres:some_creative_password@127.0.0.1:5433/me_cfs_vs_depression"
        )
    def get_session():
        return OrmSession(bind=get_engine())

from src.inference_pipeline.inference import predict

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_logistic_model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "robust_scaler.pkl"
ORDINAL_MAPPINGS_PATH = PROJECT_ROOT / "models" / "ordinal_mappings.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"

# Load expected training features for alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "diagnosis"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="MECFS vs Depression Classification API",
    description="API for predicting ME/CFS vs Depression classification",
    version="1.0.0"
)


@app.on_event("startup")
def on_startup():
    """Initialize database tables if not present."""
    try:
        engine = get_engine()
        if Base is not None:
            Base.metadata.create_all(bind=engine)

        try:
            from sqlalchemy import text
            from sqlalchemy import inspect as _inspect
            insp = _inspect(engine)
            has_features = insp.has_table("features")
            needs_load = True
            if has_features:
                with engine.connect() as conn:
                    res = conn.execute(text("SELECT COUNT(*) FROM features"))
                    count = res.scalar() or 0
                    needs_load = count == 0

            if needs_load:
                features_path = PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv"
                if not features_path.exists():
                    features_path = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
                if features_path.exists():
                    df_init = pd.read_csv(features_path)
                    df_init["created_at"] = random_timestamp_2025(len(df_init))
                    with engine.begin() as conn:
                        df_init.to_sql("features", con=conn, if_exists="append", index=False)
        except Exception:
            pass
    except Exception:
        pass

# ----------------------------
# Helper functions
# ----------------------------
def check_model_availability() -> Dict[str, Any]:
    """Check if all required model files are available."""
    status = {
        "model_available": MODEL_PATH.exists(),
        "label_encoder_available": LABEL_ENCODER_PATH.exists(),
        "scaler_available": SCALER_PATH.exists(),
        "ordinal_mappings_available": ORDINAL_MAPPINGS_PATH.exists(),
        "train_features_available": TRAIN_FE_PATH.exists(),
    }
    
    status["all_available"] = all(status.values())
    return status


def random_timestamp_2025(n: int):
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31, 23, 59, 59)
    total_seconds = int((end - start).total_seconds())
    return [start + timedelta(seconds=int(np.random.randint(0, total_seconds))) for _ in range(n)]

def validate_input_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Validate and convert input data to DataFrame."""
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
    
    try:
        df = pd.DataFrame(data)
        
        expected_columns = [
            "age", "gender", "sleep_quality_index", "brain_fog_level", 
            "physical_pain_score", "stress_level", "depression_phq9_score",
            "fatigue_severity_scale_score", "pem_duration_hours", 
            "hours_of_sleep_per_night", "pem_present", "work_status",
            "social_activity_level", "exercise_frequency", 
            "meditation_or_mindfulness"
        ]
        
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {list(missing_columns)}"
            )
        
        return df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "MECFS vs Depression Classification API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "train_model": "/train-model",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    status = check_model_availability()
    
    if not status["all_available"]:
        return {
            "status": "unhealthy",
            "message": "Some required model files are missing",
            "details": status
        }
    
    return {
        "status": "healthy",
        "message": "All systems operational",
        "model_path": str(MODEL_PATH),
        "n_features_expected": len(TRAIN_FEATURE_COLUMNS) if TRAIN_FEATURE_COLUMNS else "unknown",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict_batch(
    data: List[Dict[str, Any]],
    include_features: bool = Query(False, description="Include engineered feature matrix in response"),
):
    """
    Batch prediction endpoint.
    
    Accepts a list of patient records and returns predictions for all.
    """
    status = check_model_availability()
    if not status["all_available"]:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check /health endpoint."
        )
    
    df = validate_input_data(data)
    
    try:
        predictions_df = predict(
            df,
            model_path=MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            scaler_path=SCALER_PATH,
            ordinal_mappings_path=ORDINAL_MAPPINGS_PATH,
        )
        
        response = {
            "predictions": predictions_df["predicted_diagnosis"].tolist(),
            "probabilities": {
                "Depression": predictions_df["prob_Depression"].tolist(),
                "ME/CFS": predictions_df["prob_ME/CFS"].tolist(),
                "Both": predictions_df["prob_Both"].tolist(),
            },
            "confidence": predictions_df["prediction_confidence"].tolist(),
            "n_predictions": len(predictions_df),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            db: OrmSession = get_session()
            # Save input payload as one record per call
            if InferenceInput is not None:
                payload_record = InferenceInput(
                    payload={"data": data, "include_features": include_features},
                )
                db.add(payload_record)
                db.flush()  

            if Prediction is not None:
                now_ts = datetime.now()
                records = []
                for i in range(len(predictions_df)):
                    rec = Prediction(
                        source="inference",
                        predicted_label=response["predictions"][i],
                        prob_depression=response["probabilities"]["Depression"][i],
                        prob_me_cfs=response["probabilities"]["ME/CFS"][i],
                        prob_both=response["probabilities"]["Both"][i],
                        confidence=response["confidence"][i],
                        created_at=now_ts,
                    )
                    records.append(rec)
                db.add_all(records)
            db.commit()
        except SQLAlchemyError:
            if 'db' in locals():
                db.rollback()
        finally:
            if 'db' in locals():
                db.close()

        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/train-model")
def train_model_from_db(test_size: float = 0.1, random_state: int = 42):
    """Train LogisticRegression on features from DB (90:10 split) and log predictions with source='train'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from joblib import dump

    engine = get_engine()
    try:
        features_df = pd.read_sql_table("features", con=engine)
        if "diagnosis" not in features_df.columns:
            raise HTTPException(status_code=400, detail="'features' table must include 'diagnosis' column")

        y = features_df["diagnosis"]
        X = features_df.drop(columns=["diagnosis"])
        # Drop non-feature columns if present
        drop_cols = [c for c in ["created_at", "id"] if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        model = LogisticRegression(
            C=1,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(model, MODEL_PATH)

        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)

        try:
            db: OrmSession = get_session()
            if Prediction is not None:
                now_ts = datetime.now()
                records = []
                for i in range(len(y_pred)):
                    rec = Prediction(
                        source="train",
                        predicted_label=str(y_pred[i]),
                        predicted_id=int(y_pred[i]),
                        prob_depression=float(y_proba[i, 0]),
                        prob_me_cfs=float(y_proba[i, 1]),
                        prob_both=float(y_proba[i, 2]),
                        confidence=float(np.max(y_proba[i, :])),
                        created_at=now_ts,
                    )
                    records.append(rec)
                db.add_all(records)
                db.commit()
        except SQLAlchemyError:
            if 'db' in locals():
                db.rollback()
        finally:
            if 'db' in locals():
                db.close()

        return {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "n_train_logged": int(len(y_pred)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model_info")
def model_info():
    """Get information about the trained model."""
    status = check_model_availability()
    
    if not status["all_available"]:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        from joblib import load
        model = load(MODEL_PATH)
        
        return {
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "n_classes": len(model.classes_) if hasattr(model, 'classes_') else "unknown",
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else "unknown",
            "n_features": len(TRAIN_FEATURE_COLUMNS) if TRAIN_FEATURE_COLUMNS else "unknown",
            "feature_columns": TRAIN_FEATURE_COLUMNS,
            "model_path": str(MODEL_PATH),
            "last_updated": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat()
        }
    except Exception as e:
        return {
            "error": f"Could not load model info: {str(e)}",
            "model_path": str(MODEL_PATH)
        }

# ----------------------------
# Error handlers
# ----------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/", "/health", "/predict", "/predict_single", "/model_info", "/docs"]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check the logs for more details"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
