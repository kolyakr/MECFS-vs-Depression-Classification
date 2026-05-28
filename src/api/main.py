"""
FastAPI application for MECFS vs Depression Classification.

- Provides REST API endpoints for model predictions
- Health checks and model status
- Batch prediction capabilities
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

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
from src.monitoring.report import generate_dashboard

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
            "model_info": "/model_info",
            "monitor": "/monitor",
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
        
        if include_features:
            prediction_cols = {
                "predicted_diagnosis", "predicted_diagnosis_id",
                "prob_Depression", "prob_ME/CFS", "prob_Both",
                "prediction_confidence", "actual_diagnosis"
            }
            feat_cols = [c for c in predictions_df.columns if c not in prediction_cols]
            response["feature_columns"] = feat_cols
            response["engineered_features"] = predictions_df[feat_cols].to_dict(orient="records")
        
        try:
            db: OrmSession = get_session()
            # Save input rows: one InferenceInput per item with explicit columns
            request_id = str(uuid.uuid4())
            if InferenceInput is not None:
                input_rows = []
                for item in data:
                    row = InferenceInput(
                        request_id=request_id,
                        include_features=bool(include_features),
                        age=item.get("age"),
                        gender=item.get("gender"),
                        pem_present=item.get("pem_present"),
                        work_status=item.get("work_status"),
                        stress_level=item.get("stress_level"),
                        brain_fog_level=item.get("brain_fog_level"),
                        exercise_frequency=item.get("exercise_frequency"),
                        pem_duration_hours=item.get("pem_duration_hours"),
                        physical_pain_score=item.get("physical_pain_score"),
                        sleep_quality_index=item.get("sleep_quality_index"),
                        depression_phq9_score=item.get("depression_phq9_score"),
                        social_activity_level=item.get("social_activity_level"),
                        hours_of_sleep_per_night=item.get("hours_of_sleep_per_night"),
                        meditation_or_mindfulness=item.get("meditation_or_mindfulness"),
                        fatigue_severity_scale_score=item.get("fatigue_severity_scale_score"),
                    )
                    input_rows.append(row)
                if input_rows:
                    db.add_all(input_rows)
                db.flush()

            if Prediction is not None:
                now_ts = datetime.now()
                records = []
                for i in range(len(predictions_df)):
                    rec = Prediction(
                        request_id=request_id,
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
            # Append engineered features into features with source='inference'
            try:
                engine = get_engine()
                fe = predictions_df.copy()
                # remove prediction-only columns if present
                drop_cols = [c for c in [
                    "predicted_diagnosis", "predicted_diagnosis_id",
                    "prob_Depression", "prob_ME/CFS", "prob_Both",
                    "prediction_confidence", "actual_diagnosis"
                ] if c in fe.columns]
                if drop_cols:
                    fe = fe.drop(columns=drop_cols)
                # persist numeric diagnosis for inference rows from predicted ids
                if "predicted_diagnosis_id" in predictions_df.columns and "diagnosis" not in fe.columns:
                    fe["diagnosis"] = predictions_df["predicted_diagnosis_id"].astype(int)
                fe["created_at"] = now_ts
                fe["source"] = "inference"
                with engine.begin() as conn:
                    fe.to_sql("features", con=conn, if_exists="append", index=False)
            except Exception:
                pass
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
def train_model_from_db(
    test_size: float = 0.1,
    random_state: int = 42,
    include_inference: bool = Query(False, description="If true, use both source='train' and source='inference' for training"),
):
    """Train LogisticRegression on features from DB (90:10 split) and log predictions with source='train'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from joblib import dump

    engine = get_engine()
    try:
        features_df = pd.read_sql_table("features", con=engine)
        if "source" in features_df.columns and not include_inference:
            features_df = features_df[features_df["source"] == "train"]
        if "diagnosis" not in features_df.columns:
            raise HTTPException(status_code=400, detail="'features' table must include 'diagnosis' column")

        y = features_df["diagnosis"]
        X = features_df.drop(columns=["diagnosis"])
        drop_cols = [c for c in ["created_at", "id", "source"] if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        X = X.select_dtypes(include=[np.number])

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

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        try:
            db: OrmSession = get_session()
            from sqlalchemy import text as _text
            n_rows = int(len(features_df))
            n_train = int((features_df["source"] == "train").sum()) if "source" in features_df.columns else n_rows
            n_inference = int((features_df["source"] == "inference").sum()) if "source" in features_df.columns else 0
            insert_sql = _text(
                "INSERT INTO metrics (run_type, accuracy, precision, recall, f1, n_rows, n_train, n_inference) "
                "VALUES (:run_type, :accuracy, :precision, :recall, :f1, :n_rows, :n_train, :n_inference)"
            )
            db.execute(insert_sql, {
                "run_type": "train",
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_rows": n_rows,
                "n_train": n_train,
                "n_inference": n_inference,
            })
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

@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    """Generate and return Evidently dashboard HTML."""
    try:
        out_path = PROJECT_ROOT / "reports" / "monitoring_report.html"
        path = generate_dashboard(out_path)
        
        if not path.exists():
            raise HTTPException(status_code=500, detail="Report file was not generated")
        
        html_content = path.read_text(encoding="utf-8")
        
        return HTMLResponse(
            content=html_content,
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitor generation failed: {str(e)}")

# ----------------------------
# Error handlers
# ----------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/", "/health", "/predict", "/train-model", "/model_info", "/monitor", "/docs"]}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please check the logs for more details"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
