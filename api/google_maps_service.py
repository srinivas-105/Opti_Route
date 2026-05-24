"""
Google Maps service wrapper.

SECURITY: The API key is read from server-side env ONLY via config.py.
It is NEVER returned in any API response, never logged, never sent to JS.
The dashboard gets the Maps JS key injected server-side via /dashboard endpoint.
"""
import hashlib
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import requests

try:
    import diskcache
    _cache_backend = "diskcache"
except ImportError:
    diskcache = None
    _cache_backend = "memory"

try:
    from .config import get_settings
except Exception:
    def get_settings():
        class _S:
            google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
            cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
        return _S()

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Cache setup ────────────────────────────────────────────────
if diskcache:
    _cache = diskcache.Cache("./google_api_cache", size_limit=200_000_000)
else:
    class _MemCache:
        def __init__(self): self._d = {}
        def __contains__(self, k): return k in self._d
        def __getitem__(self, k): return self._d[k]
        def set(self, k, v, expire=None): self._d[k] = v
    _cache = _MemCache()


def _key(*args) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def _api_key() -> str:
    """Server-side only. NEVER include in any response body."""
    return settings.google_maps_api_key


def is_available() -> bool:
    k = _api_key()
    return bool(k) and k not in ("", "YOUR_GOOGLE_MAPS_API_KEY_HERE", "your_key_here")


# ── Haversine fallback ─────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Polyline decoder ───────────────────────────────────────────

def decode_polyline(enc: str) -> List[Dict[str, float]]:
    """Decode Google encoded polyline → list of {lat, lng}."""
    result = []
    index = lat = lng = 0
    while index < len(enc):
        for is_lat in (True, False):
            val = shift = 0
            while True:
                b = ord(enc[index]) - 63
                index += 1
                val |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(val >> 1) if val & 1 else val >> 1
            if is_lat:
                lat += delta
            else:
                lng += delta
        result.append({"lat": lat * 1e-5, "lng": lng * 1e-5})
    return result


# ── Geocoding ──────────────────────────────────────────────────

def geocode(address: str) -> Optional[Tuple[float, float]]:
    if not is_available():
        return None
    k = _key("geocode", address)
    if k in _cache:
        return _cache[k]
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": _api_key()},
            timeout=10,
        )
        data = r.json()
        if data.get("status") == "OK":
            loc = data["results"][0]["geometry"]["location"]
            res = (loc["lat"], loc["lng"])
            _cache.set(k, res, expire=settings.cache_ttl_seconds)
            return res
        logger.warning("Geocode failed '%s': %s", address, data.get("status"))
    except Exception as e:
        logger.error("Geocode error: %s", e)
    return None


# ── Distance Matrix ────────────────────────────────────────────

def get_distance_matrix(
    origins: List[Tuple[float, float]],
    destinations: List[Tuple[float, float]],
) -> Optional[Dict]:
    if not is_available():
        return None
    k = _key("dm", origins, destinations)
    if k in _cache:
        return _cache[k]
    try:
        def fmt(pts): return "|".join(f"{la},{lo}" for la, lo in pts)
        r = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": fmt(origins),
                "destinations": fmt(destinations),
                "key": _api_key(),
                "departure_time": "now",
                "traffic_model": "best_guess",
                "mode": "driving",
            },
            timeout=15,
        )
        data = r.json()
        if data.get("status") == "OK":
            _cache.set(k, data, expire=settings.cache_ttl_seconds)
            return data
        logger.warning("Distance Matrix failed: %s", data.get("status"))
    except Exception as e:
        logger.error("Distance Matrix error: %s", e)
    return None


def parse_distance_matrix(matrix: Dict) -> Tuple[List[List[float]], List[List[float]]]:
    durations, distances = [], []
    for row in matrix.get("rows", []):
        dr, di = [], []
        for el in row.get("elements", []):
            if el.get("status") == "OK":
                dur = el.get("duration_in_traffic", el["duration"])
                dr.append(dur["value"] / 60)
                di.append(el["distance"]["value"] / 1000)
            else:
                dr.append(9999.0)
                di.append(9999.0)
        durations.append(dr)
        distances.append(di)
    return durations, distances


# ── Route optimisation (nearest-neighbour TSP) ─────────────────

def optimise_route_order(
    locations: List[Tuple[float, float]],
) -> Tuple[List[int], float, float]:
    """Returns (ordered_indices, total_km, total_min)."""
    n = len(locations)
    if n <= 1:
        return list(range(n)), 0.0, 0.0

    matrix_data = get_distance_matrix(locations, locations)
    if matrix_data:
        durations, distances = parse_distance_matrix(matrix_data)
    else:
        distances = [
            [haversine_km(*locations[i], *locations[j]) for j in range(n)]
            for i in range(n)
        ]
        durations = [[d / 40 * 60 for d in row] for row in distances]

    visited = [False] * n
    order = [0]
    visited[0] = True
    total_km = total_min = 0.0

    for _ in range(n - 1):
        cur = order[-1]
        best_j, best_d = -1, float("inf")
        for j in range(n):
            if not visited[j] and distances[cur][j] < best_d:
                best_j, best_d = j, distances[cur][j]
        order.append(best_j)
        visited[best_j] = True
        total_km += distances[cur][best_j]
        total_min += durations[cur][best_j]

    return order, round(total_km, 2), round(total_min, 1)


def optimise_route_order_ml(
    locations: List[Tuple[float, float]],
    scores: List[float],
) -> Tuple[List[int], float, float]:
    """
    ML-weighted nearest-neighbour TSP.

    At each step we pick the next unvisited stop using a combined cost:
        cost(i→j) = distance_km(i,j) / score_weight(j)

    where score_weight = 0.5 + score  (so scores in [0,1] → weights in [0.5, 1.5])
    A stop with score=1.0 appears 3× cheaper to visit than score=0.0,
    meaning the ML model genuinely influences visit order, not just distance.
    """
    n = len(locations)
    if n <= 1:
        return list(range(n)), 0.0, 0.0

    # Clamp scores to [0, 1]
    safe_scores = [max(0.0, min(1.0, s)) for s in (scores or [0.5] * n)]
    # Pad if lengths mismatch
    while len(safe_scores) < n:
        safe_scores.append(0.5)

    matrix_data = get_distance_matrix(locations, locations)
    if matrix_data:
        durations, distances = parse_distance_matrix(matrix_data)
    else:
        distances = [
            [haversine_km(*locations[i], *locations[j]) for j in range(n)]
            for i in range(n)
        ]
        durations = [[d / 40 * 60 for d in row] for row in distances]

    visited = [False] * n
    order = [0]
    visited[0] = True
    total_km = total_min = 0.0

    for _ in range(n - 1):
        cur = order[-1]
        best_j, best_cost = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                weight = 0.5 + safe_scores[j]   # range [0.5, 1.5]
                cost   = distances[cur][j] / weight
                if cost < best_cost:
                    best_j, best_cost = j, cost
        order.append(best_j)
        visited[best_j] = True
        total_km  += distances[cur][best_j]
        total_min += durations[cur][best_j]

    return order, round(total_km, 2), round(total_min, 1)


# ── Directions API ─────────────────────────────────────────────

def get_directions(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    waypoints: Optional[List[Tuple[float, float]]] = None,
) -> Optional[Dict]:
    """
    Calls Google Directions API server-side.
    Returns full response dict (cached). API key never leaves server.
    """
    if not is_available():
        return None
    k = _key("dir", origin, destination, waypoints)
    if k in _cache:
        return _cache[k]
    try:
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "key": _api_key(),
            "mode": "driving",
            "departure_time": "now",
            "traffic_model": "best_guess",
        }
        if waypoints:
            params["waypoints"] = "|".join(f"{la},{lo}" for la, lo in waypoints)
        r = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params,
            timeout=15,
        )
        data = r.json()
        if data.get("status") == "OK":
            _cache.set(k, data, expire=settings.cache_ttl_seconds)
            return data
        logger.warning("Directions failed: %s — %s", data.get("status"), data.get("error_message", ""))
    except Exception as e:
        logger.error("Directions error: %s", e)
    return None


# ── Places API ─────────────────────────────────────────────────

def nearby_places(
    center: Tuple[float, float],
    radius: int = 1000,
    place_type: str = "store",
) -> List[Dict]:
    if not is_available():
        return []
    k = _key("places", center, radius, place_type)
    if k in _cache:
        return _cache[k]
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{center[0]},{center[1]}",
                "radius": radius,
                "type": place_type,
                "key": _api_key(),
            },
            timeout=10,
        )
        data = r.json()
        if data.get("status") in ("OK", "ZERO_RESULTS"):
            res = data.get("results", [])
            _cache.set(k, res, expire=settings.cache_ttl_seconds)
            return res
    except Exception as e:
        logger.error("Places error: %s", e)
    return []


# ── Google Maps URL builder ────────────────────────────────────

def build_google_maps_link(locations: List[Tuple[float, float]]) -> str:
    if not locations:
        return ""
    def fmt(c): return f"{c[0]},{c[1]}"
    if len(locations) == 1:
        return f"https://www.google.com/maps/search/?api=1&query={fmt(locations[0])}"
    url = (f"https://www.google.com/maps/dir/?api=1"
           f"&origin={fmt(locations[0])}&destination={fmt(locations[-1])}")
    if len(locations) > 2:
        url += "&waypoints=" + "|".join(fmt(c) for c in locations[1:-1])
    url += "&travelmode=driving"
    return url


# ── Instance wrapper (used by tests) ──────────────────────────

class GoogleMapsService:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.available = is_available()

    def optimise_route_order(self, locs): return optimise_route_order(locs)
    def get_route_polyline(self, locs): return _get_route_polyline(locs)


def _get_route_polyline(locations: List[Tuple[float, float]]) -> Dict:
    """
    Fetch road geometry from Directions API (server-side, key never exposed).
    Falls back to raw lat/lng points — but the DASHBOARD always uses
    DirectionsService client-side, so even the fallback gets real roads drawn.
    """
    if len(locations) < 2:
        return {"polyline_points": [{"lat": l[0], "lng": l[1]} for l in locations],
                "total_distance_km": 0.0, "total_time_min": 0.0, "source": "mock"}

    origin = locations[0]
    dest = locations[-1]
    wps = locations[1:-1] if len(locations) > 2 else None

    data = get_directions(origin, dest, wps)
    if data and data.get("routes"):
        route = data["routes"][0]
        enc = route.get("overview_polyline", {}).get("points", "")
        pts = decode_polyline(enc) if enc else [{"lat": l[0], "lng": l[1]} for l in locations]
        legs = route.get("legs", [])
        total_km = sum(leg["distance"]["value"] for leg in legs) / 1000 if legs else 0.0
        total_min = sum(
            leg.get("duration_in_traffic", leg.get("duration", {})).get("value", 0)
            for leg in legs
        ) / 60 if legs else 0.0
        per_leg = []
        for leg in legs:
            per_leg.append({
                "distance_km": round(leg["distance"]["value"] / 1000, 2),
                "duration_min": round(
                    leg.get("duration_in_traffic", leg["duration"])["value"] / 60, 1),
            })
        return {
            "polyline_points": pts,
            "legs": per_leg,
            "total_distance_km": round(total_km, 2),
            "total_time_min": round(total_min, 1),
            "source": "directions_api",
        }

    # Fallback — raw coords; dashboard DirectionsService will still draw real roads
    pts = [{"lat": l[0], "lng": l[1]} for l in locations]
    legs = []
    total_km = 0.0
    for i in range(len(locations) - 1):
        km = haversine_km(*locations[i], *locations[i+1])
        total_km += km
        legs.append({"distance_km": round(km, 2), "duration_min": round(km / 40 * 60, 1)})
    return {
        "polyline_points": pts,
        "legs": legs,
        "total_distance_km": round(total_km, 2),
        "total_time_min": round(total_km / 40 * 60, 1),
        "source": "mock_haversine",
    }

# Alias for test imports
_decode_polyline = decode_polyline