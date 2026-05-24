"""Train XGBoost model — saves model/route_model.pkl, scaler.pkl, encoders.pkl, metrics.json"""

import os, json, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR  = os.getenv("DATA_DIR",  "data")
MODEL_DIR = os.getenv("MODEL_DIR", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

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
TARGET = "route_efficiency_score"


def train():
    print("Loading features…")
    df = pd.read_csv(f"{DATA_DIR}/features.csv")
    df = df.dropna(subset=FEATURE_COLS + [TARGET])

    X = df[FEATURE_COLS].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("Training XGBoost…")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1])[:15])

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "top_features": importance,
    }

    # Save
    with open(f"{MODEL_DIR}/route_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/encoders.pkl", "wb") as f:
        pickle.dump({}, f)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved model → {MODEL_DIR}/route_model.pkl")
    print(f"  Metrics     → {MODEL_DIR}/metrics.json")
    return metrics


if __name__ == "__main__":
    train()
