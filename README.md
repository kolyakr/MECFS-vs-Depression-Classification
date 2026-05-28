## MECFS vs Depression Classification API

[English](README.md) | [Українська](README_UA.md)

A production-ready FastAPI service for classifying patient records into Depression, ME/CFS, or Both. It includes a full pipeline: preprocessing, feature engineering, model training (Logistic Regression), inference, and a PostgreSQL-backed audit trail for predictions and inputs.

### Highlights

- Endpoints for health, prediction, training from database, and model info
- Logged predictions (train and inference) with probabilities and confidence
- Logged raw HTTP inputs for traceability

---

## 1) Project Structure

```
.
├─ src/
│  ├─ api/main.py                    # FastAPI app and endpoints
│  ├─ inference_pipeline/inference.py # Inference pipeline
│  ├─ feature_pipeline/               # Preprocessing & feature-engineering utilities
│  └─ db/
│     ├─ models.py                   # SQLAlchemy ORM models
│     └─ session.py                  # Engine + session factory
├─ data/processed/feature_engineered_data.csv
├─ models/                           # Saved model and artifacts
└─ notebooks/                        # Training & EDA notebooks
```

---

## 2) Requirements

- Python 3.11+
- PostgreSQL reachable at:
  - `postgresql+psycopg2://postgres:password@127.0.0.1:5433/me_cfs_vs_depression`
  - Change this in `src/db/session.py` if needed.

Install dependencies (recommended: `uv`):

- With `uv` (honors `pyproject.toml` and `uv.lock`):

```powershell
uv sync
```

- Or install key runtime deps with pip (if you prefer manual):

```powershell
pip install fastapi uvicorn[standard] sqlalchemy psycopg2-binary pandas numpy scikit-learn joblib
```

---

## 3) Database Setup

Ensure PostgreSQL is running and the database exists.

Create DB (example with `psql`):

```powershell
psql -h 127.0.0.1 -p 5433 -U postgres -c "CREATE DATABASE me_cfs_vs_depression;"
```

Tables the app uses:

- `predictions`: stores predictions from training and inference with `source` = "train" | "inference"
- `inference_inputs`: stores raw HTTP payloads for each predict request
- `features`: main feature matrix used to train the best model; auto-created by CSV load

Startup behavior:

- On app start, ORM tables are created if missing.
- If `features` table is missing or empty, the app auto-loads `data/processed/feature_engineered_data.csv` (falls back to `feature_engineered_train.csv`) and adds a 2025 `created_at` timestamp to each row.

## 4) Run the API

```powershell
uvicorn src.api.main:app --reload
```

Open docs: `http://localhost:8000/docs`

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/health
```

---

## 4.5) Run the Frontend (Streamlit UI)

To launch the interactive dashboard, run:

```powershell
streamlit run src/app.py
```

Open in browser: `http://localhost:8501`

---

## 5) Endpoints Overview

- `GET /` – API info
- `GET /health` – model and artifact availability
- `POST /predict` – batch prediction (logs inputs and predictions to DB)
- `GET /model_info` – model type, params, features
- `POST /train-model` – train Logistic Regression on DB `features`, save model, log train predictions
- `GET /monitor` – generate and return Evidently monitoring dashboard (HTML) comparing reference (train) vs current (inference) data

---

## 6) Training

Train the model from the `features` table (no JSON body required):

![alt text](./images/image-1.png)

```powershell
# Default split 90:10, seed=42
Invoke-RestMethod -Method Post -Uri http://localhost:8000/train-model

# Custom split and seed
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/train-model?test_size=0.15&random_state=123"
```

Effects:

- Saves model to `models/best_logistic_model.pkl`
- Logs all train-split predictions into `predictions` with `source="train"`

---

## 7) Inference

Batch predictions (provide a list of records using the original schema fields like `age`, `gender`, `sleep_quality_index`, etc.):

![alt text](./images/image.png)

```powershell
$body = @(
  @{ age = 45; gender = "Female"; sleep_quality_index = 12; brain_fog_level = 7;
     physical_pain_score = 6; stress_level = 5; depression_phq9_score = 8;
     fatigue_severity_scale_score = 40; pem_duration_hours = 24; hours_of_sleep_per_night = 6;
     pem_present = 1; work_status = "Partially working"; social_activity_level = "Low";
     exercise_frequency = "Rarely"; meditation_or_mindfulness = "No" }
)
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body ($body | ConvertTo-Json) -ContentType 'application/json'
```

Behavior:

- Saves input payload to `inference_inputs`
- Saves each prediction to `predictions` with `source="inference"`

---

## 7.5) Monitoring Dashboard

Access the Evidently monitoring dashboard to compare reference (training) data with current (inference) data:

```powershell
# Open in browser
Start-Process "http://localhost:8000/monitor"

# Or use curl/Invoke-RestMethod to save HTML
Invoke-WebRequest -Uri http://localhost:8000/monitor -OutFile monitoring_report.html
```

The dashboard includes:

- **Data Quality Report**: Missing values, unique values, data types
- **Data Drift Report**: Detects changes in feature distributions
- **Target Drift Report**: Detects changes in diagnosis distribution (if available)
- **Metrics Trend**: Training metrics over time (accuracy, precision, recall, F1)

The report is generated on-demand and compares:

- **Reference data**: Records from `features` table where `source='train'`
- **Current data**: Records from `features` table where `source='inference'`

You can also generate the report locally:

```powershell
python -m src.monitoring.report
# Output saved to: reports/monitoring_report.html
```

---

## 8) Model Artifacts

Expected in `models/`:

- `best_logistic_model.pkl`
- `label_encoder.pkl`
- `robust_scaler.pkl`
- `ordinal_mappings.pkl`
- `imputer_stats.pkl`

`/health` and `/model_info` summarize availability/details.

---

## 9) Notebooks

Refer to `notebooks/` (e.g., `02_feature_eng_encoding.ipynb`, `04_modeling.ipynb`) for details on how feature engineering and Logistic Regression parameters were derived. The service mirrors these steps via `src/feature_pipeline` and `src/inference_pipeline`.

---

## 10) Configuration

- Database URL is defined in `src/db/session.py`. Update it to match your environment if needed.
- On startup, `features` is auto-seeded if empty. You can re-seed via `POST /load-features`.

---

## 11) Troubleshooting

- If the API starts but DB logging fails, predictions still work; errors are caught and do not break responses.
- Ensure PostgreSQL is reachable and credentials are correct.
- If `feature_engineered_data.csv` is absent, the app falls back to `feature_engineered_train.csv` when seeding.
