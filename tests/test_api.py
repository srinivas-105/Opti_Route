"""
Unit tests for Opti_Route API.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([0.8, 0.75, 0.85, 0.7, 0.9])
    return m


def make_mock_locations():
    return pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(1, 61)],
        "store_name": [f"Store {i}" for i in range(1, 61)],
        "address": [f"Address {i}, Hyderabad" for i in range(1, 61)],
        "area": ["Banjara Hills"] * 60,
        "latitude": [17.38 + i * 0.002 for i in range(60)],
        "longitude": [78.48 + i * 0.002 for i in range(60)],
        "priority": ["medium"] * 60,
    })


def make_mock_trips():
    return pd.DataFrame({
        "trip_id": ["T00001"],
        "driver_id": ["D01"],
        "date": ["2026-05-01"],
        "stop_id": ["S001"],
        "route_efficiency_score": [0.8],
        "visit_duration_min": [20],
        "speed_kmh": [40.0],
    })


@pytest.fixture
def client():
    from api import main as app_module

    mock_locs  = make_mock_locations()
    mock_trips = make_mock_trips()
    mock_model = make_mock_model()
    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    from api.google_maps_service import GoogleMapsService
    from api.prediction_engine import PredictionEngine

    maps = GoogleMapsService(api_key="")   # mock mode
    engine = PredictionEngine(mock_model, mock_scaler, {}, mock_locs, mock_trips, maps)

    app_module._model   = mock_model
    app_module._scaler  = mock_scaler
    app_module._locations = mock_locs
    app_module._trips   = mock_trips
    app_module._maps    = maps
    app_module._engine  = engine

    return TestClient(app_module.app)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "model_loaded" in d
        assert "google_maps_available" in d
        assert "google_maps_mode" in d

    def test_health_fields(self, client):
        r = client.get("/health")
        d = r.json()
        for field in ["status","model_loaded","data_loaded","timestamp","google_maps_mode"]:
            assert field in d


class TestDailyPredict:
    def test_basic_prediction(self, client):
        r = client.post("/predict/daily", json={
            "driver_id": "D01",
            "date": "2026-05-20",
            "locations": ["S001","S002","S003","S004","S005"],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["driver_id"] == "D01"
        assert len(d["recommended_route"]) == 5
        assert "total_distance_km" in d
        assert "route_polyline" in d
        assert "polyline_points" in d["route_polyline"]
        assert "legs" in d["route_polyline"]

    def test_stop_details_present(self, client):
        r = client.post("/predict/daily", json={
            "driver_id": "D02",
            "date": "2026-05-20",
            "locations": ["S001","S002","S003"],
        })
        assert r.status_code == 200
        d = r.json()
        assert len(d["stop_details"]) == 3
        for sd in d["stop_details"]:
            assert "lat" in sd and "lng" in sd
            assert "visit_order" in sd

    def test_confidence_range(self, client):
        r = client.post("/predict/daily", json={
            "driver_id": "D01",
            "date": "2026-05-20",
            "locations": ["S001","S002","S003"],
        })
        d = r.json()
        assert 0 <= d["confidence"] <= 1

    def test_too_few_locations(self, client):
        r = client.post("/predict/daily", json={
            "driver_id": "D01",
            "date": "2026-05-20",
            "locations": ["S001"],
        })
        assert r.status_code == 422

    def test_google_maps_url_present(self, client):
        r = client.post("/predict/daily", json={
            "driver_id": "D01",
            "date": "2026-05-20",
            "locations": ["S001","S002","S003"],
        })
        d = r.json()
        assert "google_maps_url" in d
        assert d["google_maps_url"].startswith("https://")


class TestWeeklyPredict:
    def test_basic_weekly(self, client):
        r = client.post("/predict/weekly", json={
            "driver_id": "D01",
            "week": "2026-W20",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["driver_id"] == "D01"
        assert "schedule" in d
        assert "monday" in d["schedule"]

    def test_five_days(self, client):
        r = client.post("/predict/weekly", json={"driver_id":"D01","week":"2026-W20"})
        d = r.json()
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            assert day in d["schedule"]

    def test_weekly_distance(self, client):
        r = client.post("/predict/weekly", json={"driver_id":"D01","week":"2026-W20"})
        d = r.json()
        assert d["weekly_distance_km"] >= 0


class TestPolylineEndpoint:
    def test_polyline_endpoint(self, client):
        r = client.post("/route/polyline", json={
            "coordinates": [
                {"lat": 17.38, "lng": 78.49},
                {"lat": 17.40, "lng": 78.50},
                {"lat": 17.42, "lng": 78.51},
            ]
        })
        assert r.status_code == 200
        d = r.json()
        assert "polyline_points" in d
        assert "total_distance_km" in d
        assert "source" in d


class TestRetrain:
    def test_retrain_response(self, client):
        r = client.post("/retrain")
        assert r.status_code == 200
        d = r.json()
        assert "status" in d
        assert "message" in d


class TestGoogleMapsService:
    def test_haversine_mock(self):
        from api.google_maps_service import GoogleMapsService
        svc = GoogleMapsService(api_key="")
        assert not svc.available
        result = svc.get_route_polyline([(17.38, 78.49), (17.40, 78.50)])
        assert result["source"] == "mock_haversine"
        assert len(result["polyline_points"]) >= 2
        assert result["total_distance_km"] > 0

    def test_optimise_route_single_stop(self):
        from api.google_maps_service import GoogleMapsService
        svc = GoogleMapsService(api_key="")
        route, km, mins = svc.optimise_route_order([(17.38, 78.49)])
        assert route == [0]
        assert km == 0.0

    def test_decode_polyline(self):
        from api.google_maps_service import _decode_polyline
        # Simple known encoding for [(0,0), (1,1)]
        encoded = "_ibE_ibE_ibE_ibE"
        pts = _decode_polyline(encoded)
        assert isinstance(pts, list)
        assert all("lat" in p and "lng" in p for p in pts)
