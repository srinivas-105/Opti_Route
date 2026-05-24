"""
FastAPI application — Opti_Route.

SECURITY:
  - GOOGLE_MAPS_API_KEY is read from .env on the server only.
  - The /dashboard endpoint injects the Maps JS key server-side into the HTML.
  - No JSON endpoint ever returns any API key.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel as _PBM

class _PolylineRequest(_PBM):
    coordinates: List[Dict]

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from . import prediction_engine as engine
from .google_maps_service import is_available, haversine_km

try:
    from .config import get_settings
except Exception:
    def get_settings():
        class _S:
            google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        return _S()

try:
    from .database import PredictionLog, TrainingRun, get_db, init_db
    from sqlalchemy.orm import Session
except Exception:
    PredictionLog = TrainingRun = None
    Session = None
    def init_db(): pass
    def get_db():
        yield None

from .schemas import (
    DailyPredictionRequest, DailyPredictionResponse,
    HealthResponse, RetrainResponse,
    WeeklyPredictionRequest, WeeklyPredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    engine.load_artifacts()
    yield


app = FastAPI(
    title="OptiRoute — AI Route Prediction",
    version="2.0.0",
    description="ML-powered sales route optimisation with real Google Maps road rendering.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_model_loaded(),
        data_loaded=engine.is_data_loaded(),
        google_maps_available=is_available(),
        google_maps_mode="live" if is_available() else "mock",
        timestamp=datetime.utcnow().isoformat(),
        last_training=engine._last_training,
    )


# ── Data endpoints ─────────────────────────────────────────────

@app.get("/stores")
def list_stores():
    try:
        return engine.get_stores()
    except Exception:
        raise HTTPException(500, "Could not load stores")


@app.get("/drivers")
def list_drivers():
    try:
        return engine.get_drivers()
    except Exception:
        raise HTTPException(500, "Could not load drivers")


# ── Predictions ────────────────────────────────────────────────

@app.post("/predict/daily", response_model=DailyPredictionResponse)
def predict_daily(req: DailyPredictionRequest, db=Depends(get_db)):
    try:
        result = engine.predict_daily(req.driver_id, req.date, req.locations)
    except Exception as exc:
        logger.exception("Daily prediction failed")
        raise HTTPException(500, str(exc))

    if PredictionLog is not None and db is not None:
        try:
            log = PredictionLog(
                driver_id=req.driver_id,
                prediction_type="daily",
                input_locations=json.dumps(req.locations),
                recommended_route=json.dumps(result.recommended_route),
                total_distance_km=result.total_distance_km,
                travel_time_min=result.travel_time_min,
                predicted_time_hours=result.travel_time_min / 60,
                confidence=result.confidence,
                route_score=result.route_score,
            )
            db.add(log)
            db.commit()
        except Exception:
            pass

    return result


@app.post("/predict/weekly", response_model=WeeklyPredictionResponse)
def predict_weekly(req: WeeklyPredictionRequest, db=Depends(get_db)):
    try:
        result = engine.predict_weekly(req.driver_id, req.week)
    except Exception as exc:
        logger.exception("Weekly prediction failed")
        raise HTTPException(500, str(exc))

    if PredictionLog is not None and db is not None:
        try:
            all_stops = (result.monday + result.tuesday + result.wednesday
                         + result.thursday + result.friday)
            log = PredictionLog(
                driver_id=req.driver_id,
                prediction_type="weekly",
                input_locations=json.dumps([]),
                recommended_route=json.dumps(all_stops),
                total_distance_km=result.weekly_distance_km,
                confidence=result.avg_confidence,
            )
            db.add(log)
            db.commit()
        except Exception:
            pass

    return result


# ── Retrain ────────────────────────────────────────────────────

def _retrain_task():
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "scripts/setup.py", "--retrain-only"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0:
            engine._last_training = datetime.utcnow().isoformat()
            engine.load_artifacts()
            logger.info("Retrain succeeded")
        else:
            logger.error("Retrain failed: %s", r.stderr[-300:])
    except Exception as e:
        logger.error("Retrain error: %s", e)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_task)
    return RetrainResponse(
        status="accepted",
        message="Model retraining started in background. Check /health for status.",
    )


@app.post('/route/polyline')
def route_polyline(payload: _PolylineRequest):
    """Fetch real road polyline for arbitrary coordinates."""
    coords = payload.coordinates
    if not coords or len(coords) < 2:
        raise HTTPException(status_code=400, detail='Need at least 2 coordinates')
    from .google_maps_service import _get_route_polyline
    pts = [(float(p['lat']), float(p['lng'])) for p in coords]
    return _get_route_polyline(pts)


# ── Dashboard — Maps JS key injected server-side ───────────────

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """
    Serves dashboard HTML with the Maps JS key injected server-side.
    The key is NEVER exposed via any JSON endpoint.
    """
    html_path = os.path.join(
        os.path.dirname(__file__), "..", "dashboard", "index.html"
    )
    if not os.path.exists(html_path):
        raise HTTPException(404, "Dashboard not found")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    key = settings.google_maps_api_key
    if key and key.strip() and key not in ("YOUR_GOOGLE_MAPS_API_KEY_HERE",):
        html = html.replace("__GMAPS_KEY__", key)
    else:
        # No key — remove the Maps script tag; dashboard shows text-only fallback
        html = html.replace(
            '<script src="https://maps.googleapis.com/maps/api/js?key=__GMAPS_KEY__&callback=_mapsLoaded&libraries=geometry&loading=async" async defer></script>',
            "<script>window._mapsLoadFailed=true;</script>",
        )
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/dashboard">')
