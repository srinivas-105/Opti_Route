"""SQLAlchemy database setup — SQLite by default, PostgreSQL in production."""
import os
from datetime import datetime
from typing import Generator

try:
    from sqlalchemy import (
        Column, DateTime, Float, Integer, String, Text, create_engine
    )
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

    try:
        from .config import get_settings
        _db_url = get_settings().database_url
    except Exception:
        _db_url = os.getenv("DATABASE_URL", "sqlite:///./route_prediction.db")

    engine = create_engine(
        _db_url,
        connect_args={"check_same_thread": False} if "sqlite" in _db_url else {},
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    class Base(DeclarativeBase):
        pass

    class PredictionLog(Base):
        __tablename__ = "prediction_logs"
        id = Column(Integer, primary_key=True, index=True)
        driver_id = Column(String(50), index=True)
        prediction_type = Column(String(20))          # "daily" | "weekly"
        input_locations = Column(Text)                # JSON
        recommended_route = Column(Text)              # JSON
        total_distance_km = Column(Float, nullable=True)
        travel_time_min = Column(Float, nullable=True)
        predicted_time_hours = Column(Float, nullable=True)
        confidence = Column(Float, nullable=True)
        route_score = Column(Float, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

    class TrainingRun(Base):
        __tablename__ = "training_runs"
        id = Column(Integer, primary_key=True, index=True)
        triggered_by = Column(String(50), default="api")
        status = Column(String(20), default="running")  # running|success|failed
        error_message = Column(Text, nullable=True)
        started_at = Column(DateTime, default=datetime.utcnow)
        finished_at = Column(DateTime, nullable=True)

    def init_db():
        Base.metadata.create_all(bind=engine)

    def get_db() -> Generator[Session, None, None]:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

except ImportError:
    # SQLAlchemy not installed — provide no-op stubs
    PredictionLog = None       # type: ignore
    TrainingRun = None         # type: ignore
    SessionLocal = None        # type: ignore

    def init_db():             # type: ignore
        pass

    def get_db():              # type: ignore
        yield None
