"""Feature engineering pipeline — reads trips.csv, writes features.csv"""

import os
import math
import pandas as pd
import numpy as np

DATA_DIR = os.getenv("DATA_DIR", "data")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_features(trips_path: str = None, out_path: str = None) -> pd.DataFrame:
    trips_path = trips_path or f"{DATA_DIR}/trips.csv"
    out_path   = out_path   or f"{DATA_DIR}/features.csv"

    df = pd.read_csv(trips_path, parse_dates=["date"])
    print(f"  Loaded {len(df)} trips")

    # ── Time features ────────────────────────────────────────────────────────
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_morning"]    = ((df["visit_hour"] >= 8) & (df["visit_hour"] < 12)).astype(int)
    df["is_afternoon"]  = ((df["visit_hour"] >= 12) & (df["visit_hour"] < 17)).astype(int)
    df["is_rush_hour"]  = df.get("is_rush_hour", ((df["visit_hour"].isin(range(8,10))) | (df["visit_hour"].isin(range(17,19)))).astype(int))

    # ── Route features ────────────────────────────────────────────────────────
    df["stop_position_norm"] = df["stop_sequence"] / df["daily_stop_count"].clip(lower=1)

    grp = df.groupby(["driver_id", "date"])
    df["daily_total_distance_km"] = grp["latitude"].transform(
        lambda x: sum(
            haversine_km(x.iloc[i], df.loc[x.index[i], "longitude"],
                         x.iloc[i+1], df.loc[x.index[i+1], "longitude"])
            for i in range(len(x) - 1)
        ) if len(x) > 1 else 0
    )

    # Area density: how many stores in same area visited same day
    area_cnt = df.groupby(["driver_id", "date", "area"])["stop_id"].transform("count")
    df["area_density"] = area_cnt

    # ── Driver features ───────────────────────────────────────────────────────
    driver_stats = df.groupby("driver_id").agg(
        driver_avg_speed_inferred=("speed_kmh", "mean"),
        driver_avg_daily_visits=("daily_stop_count", "mean"),
        driver_avg_efficiency=("route_efficiency_score", "mean"),
    ).reset_index()
    df = df.merge(driver_stats, on="driver_id", how="left")

    # ── Location features ─────────────────────────────────────────────────────
    store_stats = df.groupby("stop_id").agg(
        store_visit_count=("trip_id", "count"),
        store_avg_duration=("visit_duration_min", "mean"),
    ).reset_index()
    df = df.merge(store_stats, on="stop_id", how="left")

    priority_map = {"high": 3, "medium": 2, "low": 1}
    df["priority_encoded"] = df["priority"].map(priority_map).fillna(2).astype(int)

    # ── Final columns ─────────────────────────────────────────────────────────
    feature_cols = [
        "trip_id", "driver_id", "stop_id", "date",
        # time
        "day_of_week", "week_number", "visit_hour",
        "is_weekend", "is_morning", "is_afternoon", "is_rush_hour",
        # route
        "stop_sequence", "stop_position_norm", "daily_stop_count",
        "daily_total_distance_km", "area_density",
        # driver
        "driver_avg_speed_inferred", "driver_avg_daily_visits", "driver_avg_efficiency",
        # location
        "store_visit_count", "store_avg_duration", "priority_encoded",
        "latitude", "longitude",
        # traffic
        "traffic_category", "traffic_multiplier",
        # target
        "route_efficiency_score",
    ]
    out = df[[c for c in feature_cols if c in df.columns]]
    out.to_csv(out_path, index=False)
    print(f"  Features:  {len(out)} rows, {len(out.columns)} cols → {out_path}")
    return out


if __name__ == "__main__":
    build_features()
