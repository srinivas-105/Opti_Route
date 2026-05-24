"""One-shot setup: generate data → features → train model"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("=" * 50)
print("Opti_Route Setup")
print("=" * 50)

print("\n[1/3] Generating synthetic dataset…")
from scripts.generate_dataset import generate_locations, generate_drivers, generate_trips
loc = generate_locations(60)
drv = generate_drivers(12)
trp = generate_trips(loc, drv, weeks=10)

print("\n[2/3] Running feature engineering…")
from scripts.feature_engineering import build_features
build_features()

print("\n[3/3] Training XGBoost model…")
from scripts.train_model import train
metrics = train()

print("\n" + "=" * 50)
print("Setup complete!")
print(f"  Model R²: {metrics['r2']}")
print(f"  CV R²:    {metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")
print("\nStart API:  uvicorn api.main:app --reload")
print("Dashboard:  http://localhost:8000/dashboard")
print("API Docs:   http://localhost:8000/docs")
print("=" * 50)
