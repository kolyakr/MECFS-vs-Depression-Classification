from __future__ import annotations
from pathlib import Path
from datetime import datetime

import pandas as pd
from src.db.session import get_engine
from src.monitoring.metrics_section import append_metrics_section

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fetch_reference_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch reference (train) and current (inference) data from features table."""
    engine = get_engine()
    features = pd.read_sql_table("features", con=engine)
    
    if "source" in features.columns:
        ref = features[features["source"] == "train"].copy()
        cur = features[features["source"] == "inference"].copy()
    else:
        ref = features.copy()
        cur = features.iloc[0:0].copy()

    for df_ in (ref, cur):
        if "created_at" not in df_.columns:
            df_["created_at"] = pd.Timestamp(datetime.utcnow())

    return ref, cur


def generate_dashboard(output_html: Path) -> Path:
    """Generate Evidently data quality + drift report comparing reference and current data."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from evidently import Report, Dataset
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    from evidently.metrics import ValueDrift

    ref, cur = fetch_reference_current()

    common = sorted(list(set(ref.columns) & set(cur.columns)))
    logger.info(f"Reference data shape: {ref.shape}")
    logger.info(f"Current data shape: {cur.shape}")

    drop_like = {"created_at", "id", "source"}
    feature_cols = [c for c in common if c not in drop_like]
    ref_use = ref[feature_cols].copy()
    cur_use = cur[feature_cols].copy()

    metrics = [
        DataSummaryPreset(),
        DataDriftPreset(),
    ]
    logger.info("Using data quality metrics and data drift")

    if "diagnosis" in ref_use.columns:
        metrics.append(ValueDrift(column="diagnosis"))
        logger.info("Added Prediction Drift Metric for 'diagnosis' column")

    report = Report(metrics=metrics)
    try:
        logger.info("Running Evidently report...")
        current_dataset = Dataset.from_pandas(cur_use)
        reference_dataset = Dataset.from_pandas(ref_use)
        snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)
        logger.info("Evidently report generated successfully")
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise

    output_html.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving report to: {output_html}")
    snapshot.save_html(str(output_html))

    append_metrics_section(output_html)

    logger.info("Dashboard generation completed")
    logger.info(f"Output file: {output_html}")

    return output_html


if __name__ == "__main__":
    out = PROJECT_ROOT / "reports" / "monitoring_report.html"
    path = generate_dashboard(out)
    print(f"Dashboard saved to {path}")
