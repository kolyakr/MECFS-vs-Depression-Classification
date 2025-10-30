from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB


Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    source = Column(String(32), nullable=False)  # "train" or "inference"

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
    payload = Column(JSONB, nullable=False)


