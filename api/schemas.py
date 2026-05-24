"""Pydantic request / response schemas."""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator


# ── Requests ──────────────────────────────────────────────────

class DailyPredictionRequest(BaseModel):
    driver_id: str
    date: str          # "YYYY-MM-DD"
    locations: List[str]

    @field_validator("locations")
    @classmethod
    def at_least_two(cls, v: List[str]):
        if not v or len(v) < 2:
            raise ValueError("locations must contain at least two store IDs")
        return v


class WeeklyPredictionRequest(BaseModel):
    driver_id: str
    week: str          # "YYYY-Www"


# ── Sub-models ─────────────────────────────────────────────────

class StopDetail(BaseModel):
    store_id: str
    name: str
    address: str
    lat: float
    lng: float
    score: float
    visit_order: int
    eta_min: Optional[float] = None   # minutes from previous stop


class DayPlan(BaseModel):
    date: str
    stops: List[str]
    distance_km: float
    travel_time_min: float
    confidence: float


# ── Responses ──────────────────────────────────────────────────

class DailyPredictionResponse(BaseModel):
    driver_id: str
    date: str
    recommended_route: List[str]
    stops: List[StopDetail]
    stop_details: Optional[List[StopDetail]] = None
    predicted_time: str
    total_distance_km: float
    travel_time_min: float
    confidence: float
    route_score: float
    stop_scores: Dict[str, float]
    google_maps_link: str
    google_maps_url: Optional[str] = None
    polyline_encoded: Optional[str] = None
    route_polyline: Optional[Dict[str, Any]] = None
    waypoint_order: Optional[List[int]] = None


class WeeklyPredictionResponse(BaseModel):
    driver_id: str
    week: str
    monday: Optional[List[str]] = []
    tuesday: Optional[List[str]] = []
    wednesday: Optional[List[str]] = []
    thursday: Optional[List[str]] = []
    friday: Optional[List[str]] = []
    day_plans: List[DayPlan] = []
    weekly_distance: str
    weekly_distance_km: float
    schedule: Dict[str, List[str]]
    avg_confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    google_maps_available: bool
    google_maps_mode: str
    timestamp: str
    last_training: Optional[str] = None


class RetrainResponse(BaseModel):
    status: str
    message: str
