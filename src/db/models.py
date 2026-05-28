from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    source = Column(String(32), nullable=False)  
    request_id = Column(String(36), nullable=True)

    predicted_label = Column(String(64), nullable=True)
    predicted_id = Column(Integer, nullable=True)
    prob_depression = Column(Float, nullable=True)
    prob_me_cfs = Column(Float, nullable=True)
    prob_both = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)


class InferenceInput(Base):
    __tablename__ = "inference_inputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    request_id = Column(String(36), nullable=True)
    include_features = Column(Boolean, nullable=True)

    age = Column(Integer, nullable=True)
    gender = Column(String(16), nullable=True)
    pem_present = Column(Integer, nullable=True)
    work_status = Column(String(32), nullable=True)
    stress_level = Column(Float, nullable=True)
    brain_fog_level = Column(Float, nullable=True)
    exercise_frequency = Column(String(16), nullable=True)
    pem_duration_hours = Column(Float, nullable=True)
    physical_pain_score = Column(Float, nullable=True)
    sleep_quality_index = Column(Float, nullable=True)
    depression_phq9_score = Column(Float, nullable=True)
    social_activity_level = Column(String(16), nullable=True)
    hours_of_sleep_per_night = Column(Float, nullable=True)
    meditation_or_mindfulness = Column(String(8), nullable=True)
    fatigue_severity_scale_score = Column(Float, nullable=True)


