from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DATABASE_URL = "postgresql+psycopg2://postgres:some_creative_password@127.0.0.1:5433/me_cfs_vs_depression"


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def get_session():
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()


