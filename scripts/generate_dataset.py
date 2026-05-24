"""
Synthetic dataset generator.
Creates: data/locations.csv (60 stores), data/drivers.csv (12 drivers),
         data/trips.csv (1400+ records)
"""

import os, random, math, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Hyderabad area bounding box ──────────────────────────────────────────────
LAT_CENTER, LNG_CENTER = 17.3850, 78.4867
RADIUS_DEG = 0.15          # ~16 km radius

AREAS = [
    ("Banjara Hills",    17.4156, 78.4347),
    ("Jubilee Hills",    17.4318, 78.4071),
    ("Hitech City",      17.4471, 78.3738),
    ("Kondapur",         17.4603, 78.3596),
    ("Gachibowli",       17.4400, 78.3489),
    ("Madhapur",         17.4483, 78.3915),
    ("Kukatpally",       17.4947, 78.3996),
    ("SR Nagar",         17.4418, 78.4512),
    ("Ameerpet",         17.4374, 78.4487),
    ("Begumpet",         17.4418, 78.4691),
    ("Secunderabad",     17.4399, 78.4983),
    ("LB Nagar",         17.3492, 78.5514),
    ("Mehdipatnam",      17.3938, 78.4368),
    ("Tolichowki",       17.4033, 78.4197),
    ("Manikonda",        17.4020, 78.3900),
    ("Nanakramguda",     17.4254, 78.3530),
    ("Shamshabad",       17.2543, 78.4288),
    ("Uppal",            17.4053, 78.5598),
    ("Dilsukhnagar",     17.3688, 78.5241),
    ("Attapur",          17.3614, 78.4328),
]

STORE_TYPES = ["Pharmacy", "Grocery", "Electronics", "Apparel", "Hardware",
               "Medical Equipment", "FMCG Distributor", "Auto Parts", "Bakery", "Restaurant"]


def _rand_near(lat, lng, spread=0.02):
    return (
        lat + random.uniform(-spread, spread),
        lng + random.uniform(-spread, spread),
    )


# ─── Locations ───────────────────────────────────────────────────────────────
def generate_locations(n=60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        area = AREAS[i % len(AREAS)]
        lat, lng = _rand_near(area[1], area[2])
        rows.append({
            "store_id": f"S{i+1:03d}",
            "store_name": f"{area[0]} {STORE_TYPES[i % len(STORE_TYPES)]} {i+1}",
            "address": f"Plot {random.randint(1,200)}, {area[0]}, Hyderabad",
            "area": area[0],
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "priority": random.choice(["high", "medium", "low"]),
            "avg_visit_duration_min": random.randint(10, 45),
            "store_type": STORE_TYPES[i % len(STORE_TYPES)],
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/locations.csv", index=False)
    print(f"  Locations: {len(df)} rows → {DATA_DIR}/locations.csv")
    return df


# ─── Drivers ─────────────────────────────────────────────────────────────────
def generate_drivers(n=12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "driver_id": f"D{i+1:02d}",
            "name": f"Driver {i+1}",
            "avg_speed_kmh": random.randint(30, 55),
            "experience_years": random.randint(1, 15),
            "vehicle_type": random.choice(["bike", "car", "van"]),
            "home_lat": round(LAT_CENTER + random.uniform(-0.1, 0.1), 6),
            "home_lng": round(LNG_CENTER + random.uniform(-0.1, 0.1), 6),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/drivers.csv", index=False)
    print(f"  Drivers:   {len(df)} rows → {DATA_DIR}/drivers.csv")
    return df


# ─── Trips ───────────────────────────────────────────────────────────────────
def generate_trips(locations: pd.DataFrame, drivers: pd.DataFrame,
                   weeks=10) -> pd.DataFrame:
    store_ids = locations["store_id"].tolist()
    rows = []
    start_date = datetime(2025, 9, 1)

    for week in range(weeks):
        for driver_row in drivers.itertuples():
            for day in range(5):   # Mon–Fri
                date = start_date + timedelta(weeks=week, days=day)
                n_stops = random.randint(4, 8)
                day_stores = random.sample(store_ids, n_stops)
                visit_hour = random.randint(8, 17)

                for seq, store_id in enumerate(day_stores):
                    loc = locations[locations["store_id"] == store_id].iloc[0]
                    is_rush = int(visit_hour in range(8, 10) or visit_hour in range(17, 19))
                    traffic = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
                    traffic_mult = [1.0, 1.3, 1.8][traffic]
                    duration = random.randint(10, 45)
                    speed_kmh = driver_row.avg_speed_kmh * (1 - 0.1 * traffic) + random.gauss(0, 3)
                    score = max(0, min(1,
                        0.4 * (1 - seq / n_stops)
                        + 0.2 * (1 / traffic_mult)
                        + 0.2 * random.random()
                        + 0.1 * int(loc["priority"] == "high")
                        + 0.1 * (1 - is_rush * 0.5)
                    ))
                    rows.append({
                        "trip_id": f"T{len(rows)+1:05d}",
                        "driver_id": driver_row.driver_id,
                        "date": date.strftime("%Y-%m-%d"),
                        "day_of_week": date.weekday(),
                        "week_number": date.isocalendar()[1],
                        "stop_id": store_id,
                        "stop_sequence": seq,
                        "daily_stop_count": n_stops,
                        "visit_hour": visit_hour,
                        "visit_duration_min": duration,
                        "latitude": loc["latitude"],
                        "longitude": loc["longitude"],
                        "area": loc["area"],
                        "priority": loc["priority"],
                        "traffic_category": traffic,
                        "traffic_multiplier": round(traffic_mult, 2),
                        "is_rush_hour": is_rush,
                        "speed_kmh": round(speed_kmh, 1),
                        "route_efficiency_score": round(score, 4),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/trips.csv", index=False)
    print(f"  Trips:     {len(df)} rows → {DATA_DIR}/trips.csv")

    # Sample preview
    df.head(20).to_json(f"{DATA_DIR}/sample_trips.json", orient="records", indent=2)
    return df


if __name__ == "__main__":
    print("Generating synthetic dataset…")
    loc = generate_locations(60)
    drv = generate_drivers(12)
    trp = generate_trips(loc, drv, weeks=10)
    print(f"Done. Total trips: {len(trp)}")
