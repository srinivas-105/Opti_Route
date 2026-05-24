"""Core prediction logic: load model, score stops, optimise route."""
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import google_maps_service as gms

try:
    from .config import get_settings
    settings = get_settings()
except Exception:
    class _S:
        model_path = os.getenv("MODEL_PATH", "model/route_model.pkl")
        scaler_path = os.getenv("SCALER_PATH", "model/scaler.pkl")
        encoder_path = os.getenv("ENCODER_PATH", "model/encoders.pkl")
        data_dir = os.getenv("DATA_DIR", "data")
    settings = _S()

from .schemas import (
    DailyPredictionResponse, StopDetail,
    WeeklyPredictionResponse, DayPlan,
)

logger = logging.getLogger(__name__)

# ── Module state ───────────────────────────────────────────────
_model = None
_scaler = None
_encoders = None
_locations_df: Optional[pd.DataFrame] = None
_drivers_df:   Optional[pd.DataFrame] = None
_trips_df:     Optional[pd.DataFrame] = None
_last_training: Optional[str] = None


def load_artifacts():
    global _model, _scaler, _encoders, _locations_df, _drivers_df, _trips_df
    if os.path.exists(settings.model_path):
        with open(settings.model_path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model loaded from %s", settings.model_path)
    if os.path.exists(settings.scaler_path):
        with open(settings.scaler_path, "rb") as f:
            _scaler = pickle.load(f)
    if os.path.exists(settings.encoder_path):
        with open(settings.encoder_path, "rb") as f:
            _encoders = pickle.load(f)

    loc = os.path.join(settings.data_dir, "locations.csv")
    drv = os.path.join(settings.data_dir, "drivers.csv")
    trp = os.path.join(settings.data_dir, "trips.csv")

    if os.path.exists(loc):
        _locations_df = pd.read_csv(loc)
        logger.info("Loaded %d locations", len(_locations_df))
    if os.path.exists(drv):
        _drivers_df = pd.read_csv(drv)
    if os.path.exists(trp):
        _trips_df = pd.read_csv(trp)


def is_model_loaded(): return _model is not None
def is_data_loaded():  return _locations_df is not None


def get_stores() -> List[Dict]:
    if _locations_df is None:
        return []
    return _locations_df.to_dict(orient="records")


def get_drivers() -> List[Dict]:
    if _drivers_df is None:
        return []
    return _drivers_df.to_dict(orient="records")


# ── Location lookup ────────────────────────────────────────────

def _get_location(store_id: str) -> Optional[Dict]:
    if _locations_df is None:
        return None
    row = _locations_df[_locations_df["store_id"] == store_id]
    return row.iloc[0].to_dict() if not row.empty else None


# ── Feature builder ────────────────────────────────────────────

FEATURE_COLS = [
    "day_of_week", "week_number", "visit_hour",
    "is_weekend", "is_morning", "is_afternoon", "is_rush_hour",
    "stop_sequence", "stop_position_norm", "daily_stop_count",
    "daily_total_distance_km", "area_density",
    "driver_avg_speed_inferred", "driver_avg_daily_visits", "driver_avg_efficiency",
    "store_visit_count", "store_avg_duration", "priority_encoded",
    "latitude", "longitude",
    "traffic_category", "traffic_multiplier",
]


def _build_features(
    store_id: str,
    driver_id: str,
    date_str: str,
    stop_seq: int,
    daily_count: int,
) -> Optional[np.ndarray]:
    loc = _get_location(store_id)
    if loc is None:
        return None

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    hour = min(9 + stop_seq, 17)

    # Driver stats
    spd, visits, eff = 40.0, 5.0, 0.75
    if _drivers_df is not None:
        row = _drivers_df[_drivers_df["driver_id"] == driver_id]
        if not row.empty:
            spd    = float(row.iloc[0].get("avg_speed_kmh", 40.0))
            visits = float(row.iloc[0].get("avg_daily_visits", 5.0))
            eff    = float(row.iloc[0].get("avg_efficiency", 0.75))

    # Store stats
    visit_count, avg_dur = 10, 30.0
    if _trips_df is not None:
        col = "store_id" if "store_id" in _trips_df.columns else (
              "stop_id"  if "stop_id"  in _trips_df.columns else None)
        if col:
            mask = _trips_df[col] == store_id
            visit_count = int(mask.sum())
            if visit_count > 0 and "visit_duration_min" in _trips_df.columns:
                avg_dur = float(_trips_df.loc[mask, "visit_duration_min"].mean())

    priority = {"high": 3, "medium": 2, "low": 1}.get(
        str(loc.get("priority", "medium")).lower(), 2)

    traffic_cat = 1
    if dt.weekday() in (0, 4): traffic_cat = 2
    elif dt.weekday() >= 5:    traffic_cat = 0

    is_rush    = 1 if hour in (8, 9, 17, 18) else 0
    is_morning = 1 if hour < 12 else 0

    feats = np.array([
        dt.weekday(),
        dt.isocalendar()[1],
        hour,
        int(dt.weekday() >= 5),
        is_morning,
        1 - is_morning,
        is_rush,
        stop_seq,
        stop_seq / max(daily_count - 1, 1),
        daily_count,
        0.0,
        1.0,
        spd, visits, eff,
        visit_count, avg_dur,
        priority,
        float(loc.get("latitude", 17.4)),
        float(loc.get("longitude", 78.4)),
        traffic_cat,
        1.0 + 0.2 * traffic_cat,
    ], dtype=np.float32)
    return feats


def _score_stops(ids: List[str], driver_id: str, date_str: str) -> Dict[str, float]:
    scores = {}
    for i, sid in enumerate(ids):
        feat = _build_features(sid, driver_id, date_str, i, len(ids))
        if feat is None:
            scores[sid] = 0.5
            continue
        X = feat.reshape(1, -1)
        if _scaler is not None:
            try: X = _scaler.transform(X)
            except Exception: pass
        if _model is not None:
            try: scores[sid] = float(_model.predict(X)[0])
            except Exception: scores[sid] = 0.5
        else:
            scores[sid] = 0.5
    return scores


# ── Daily prediction ───────────────────────────────────────────

def predict_daily(
    driver_id: str,
    date_str: str,
    location_ids: List[str],
) -> DailyPredictionResponse:
    scores = _score_stops(location_ids, driver_id, date_str)

    coords: List[Tuple[float, float]] = []
    valid_ids: List[str] = []
    detail_by_id: Dict[str, StopDetail] = {}

    for sid in location_ids:
        loc = _get_location(sid)
        if loc is not None:
            lat = float(loc.get("latitude", 17.4065))
            lng = float(loc.get("longitude", 78.4772))
            coords.append((lat, lng))
            valid_ids.append(sid)
            detail_by_id[sid] = StopDetail(
                store_id=sid,
                name=str(loc.get("store_name", sid)),
                address=str(loc.get("address", "")),
                lat=lat, lng=lng,
                score=round(scores.get(sid, 0.5), 4),
                visit_order=0,
            )

    # Fallback if no location data found
    if not coords:
        coords = [(17.4065 + i * 0.01, 78.4772 + i * 0.01) for i in range(len(location_ids))]
        valid_ids = location_ids
        for i, sid in enumerate(location_ids):
            detail_by_id[sid] = StopDetail(
                store_id=sid, name=sid, address="",
                lat=coords[i][0], lng=coords[i][1],
                score=scores.get(sid, 0.5), visit_order=i+1,
            )

    # ── ML-weighted TSP ────────────────────────────────────────
    # ML scores bias the nearest-neighbour TSP: high-scored stops
    # are preferred at equal distance, ensuring the model output
    # actually shapes the visit order (not just distance alone).
    order_idx, total_km, total_min = gms.optimise_route_order_ml(
        coords, [scores.get(sid, 0.5) for sid in valid_ids]
    )
    ordered_ids    = [valid_ids[i]   for i in order_idx]
    ordered_coords = [coords[i]      for i in order_idx]

    for rank, sid in enumerate(ordered_ids):
        if sid in detail_by_id:
            detail_by_id[sid].visit_order = rank + 1

    # Per-leg ETA from distance matrix
    leg_mins: List[Optional[float]] = [None]
    if gms.is_available() and len(ordered_coords) > 1:
        for i in range(1, len(ordered_coords)):
            pair = [ordered_coords[i-1], ordered_coords[i]]
            mat = gms.get_distance_matrix([ordered_coords[i-1]], [ordered_coords[i]])
            if mat:
                dur, _ = gms.parse_distance_matrix(mat)
                leg_mins.append(round(dur[0][0], 1))
            else:
                leg_mins.append(round(gms.haversine_km(*ordered_coords[i-1], *ordered_coords[i]) / 40 * 60, 1))
    else:
        for i in range(1, len(ordered_coords)):
            leg_mins.append(round(gms.haversine_km(*ordered_coords[i-1], *ordered_coords[i]) / 40 * 60, 1))

    for i, sid in enumerate(ordered_ids):
        if sid in detail_by_id:
            detail_by_id[sid].eta_min = leg_mins[i]

    ordered_stops = [detail_by_id[sid] for sid in ordered_ids if sid in detail_by_id]

    # Directions API for encoded polyline (server-side — key stays on server)
    polyline_encoded = None
    if gms.is_available() and len(ordered_coords) >= 2:
        dir_data = gms.get_directions(
            ordered_coords[0],
            ordered_coords[-1],
            ordered_coords[1:-1] if len(ordered_coords) > 2 else None,
        )
        if dir_data and dir_data.get("routes"):
            route = dir_data["routes"][0]
            polyline_encoded = route.get("overview_polyline", {}).get("points")
            legs = route.get("legs", [])
            if legs:
                total_km  = round(sum(l["distance"]["value"] for l in legs) / 1000, 2)
                total_min = round(sum(
                    l.get("duration_in_traffic", l["duration"])["value"] for l in legs
                ) / 60, 1)

    # Build route_polyline payload for response
    # This is GEOMETRY ONLY — no API key is in this dict
    if polyline_encoded:
        pts = gms.decode_polyline(polyline_encoded)
        route_polyline: Dict = {
            "polyline_points": pts,
            "source": "directions_api",
            "total_distance_km": total_km,
            "total_time_min": total_min,
        }
    else:
        # Straight-line fallback points — dashboard DirectionsService
        # will STILL draw real roads using these as waypoints
        pts = [{"lat": c[0], "lng": c[1]} for c in ordered_coords]
        legs = []
        for i in range(len(ordered_coords) - 1):
            km = gms.haversine_km(*ordered_coords[i], *ordered_coords[i+1])
            legs.append({"distance_km": round(km, 2), "duration_min": round(km / 40 * 60, 1)})
        route_polyline = {
            "polyline_points": pts,
            "legs": legs,
            "source": "mock_haversine",
            "total_distance_km": total_km,
            "total_time_min": total_min,
        }

    conf        = float(np.mean(list(scores.values())))
    route_score = float(np.median(list(scores.values())))
    maps_link   = gms.build_google_maps_link(ordered_coords)

    return DailyPredictionResponse(
        driver_id=driver_id,
        date=date_str,
        recommended_route=ordered_ids,
        stops=ordered_stops,
        stop_details=ordered_stops,
        predicted_time=f"{total_min / 60:.1f} hours",
        total_distance_km=total_km,
        travel_time_min=total_min,
        confidence=round(conf, 4),
        route_score=round(route_score, 4),
        stop_scores={k: round(v, 4) for k, v in scores.items()},
        google_maps_link=maps_link,
        google_maps_url=maps_link,
        polyline_encoded=polyline_encoded,
        route_polyline=route_polyline,
    )


# ── Weekly prediction ──────────────────────────────────────────

def predict_weekly(driver_id: str, week_str: str) -> WeeklyPredictionResponse:
    year, wk = week_str.split("-W")
    monday = datetime.fromisocalendar(int(year), int(wk), 1)

    store_ids = (list(_locations_df["store_id"].unique())
                 if _locations_df is not None
                 else [f"S{i:03d}" for i in range(1, 13)])

    n = len(store_ids)
    per_day = max(1, n // 5)
    day_chunks = [store_ids[d * per_day:(d + 1) * per_day if d < 4 else n] for d in range(5)]
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    day_plans: List[DayPlan] = []
    result_days: Dict[str, List[str]] = {}
    total_km = 0.0
    confs: List[float] = []

    for d, (name, stores) in enumerate(zip(day_names, day_chunks)):
        if not stores:
            result_days[name] = []
            continue
        date_str = (monday + timedelta(days=d)).strftime("%Y-%m-%d")
        resp = predict_daily(driver_id, date_str, stores)
        result_days[name] = resp.recommended_route
        total_km += resp.total_distance_km
        confs.append(resp.confidence)
        day_plans.append(DayPlan(
            date=date_str,
            stops=resp.recommended_route,
            distance_km=resp.total_distance_km,
            travel_time_min=resp.travel_time_min,
            confidence=resp.confidence,
        ))

    avg_conf = float(np.mean(confs)) if confs else 0.0

    return WeeklyPredictionResponse(
        driver_id=driver_id,
        week=week_str,
        monday=result_days.get("monday", []),
        tuesday=result_days.get("tuesday", []),
        wednesday=result_days.get("wednesday", []),
        thursday=result_days.get("thursday", []),
        friday=result_days.get("friday", []),
        day_plans=day_plans,
        weekly_distance=f"{total_km:.1f} km",
        weekly_distance_km=round(total_km, 2),
        schedule=result_days,
        avg_confidence=round(avg_conf, 4),
    )


# ── Compat wrapper for tests ───────────────────────────────────

class PredictionEngine:
    def __init__(self, model=None, scaler=None, encoders=None,
                 locations_df=None, trips_df=None, maps_service=None):
        self._model = model
    def is_model_loaded(self): return self._model is not None
    def is_data_loaded(self): return _locations_df is not None
    def predict_daily(self, *a): return predict_daily(*a)
    def predict_weekly(self, *a): return predict_weekly(*a)