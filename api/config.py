"""Application settings — reads from .env file."""
import os
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
    from pydantic import ConfigDict

    class Settings(BaseSettings):
        model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

        google_maps_api_key: str = ""
        database_url: str = "sqlite:///./route_prediction.db"
        model_path: str = "model/route_model.pkl"
        scaler_path: str = "model/scaler.pkl"
        encoder_path: str = "model/encoders.pkl"
        data_dir: str = "data"
        model_dir: str = "model"
        app_port: int = 8000
        cache_ttl_seconds: int = 86400

except ImportError:
    class Settings:  # type: ignore
        def __init__(self):
            self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
            self.database_url = os.getenv("DATABASE_URL", "sqlite:///./route_prediction.db")
            self.model_path = os.getenv("MODEL_PATH", "model/route_model.pkl")
            self.scaler_path = os.getenv("SCALER_PATH", "model/scaler.pkl")
            self.encoder_path = os.getenv("ENCODER_PATH", "model/encoders.pkl")
            self.data_dir = os.getenv("DATA_DIR", "data")
            self.model_dir = os.getenv("MODEL_DIR", "model")
            self.app_port = int(os.getenv("APP_PORT", "8000"))
            self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "86400"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
