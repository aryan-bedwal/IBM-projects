"""
House Price Prediction — Model Training Script
================================================
Loads the California Housing dataset, engineers features, trains several
regression models, evaluates them, and saves the best model + metadata
for use by dashboard.py.

Run:
    python train_model.py
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic housing dataset locally (no download required).

    Feature relationships are modeled after well-known housing datasets
    (Ames Housing / California Housing): square footage, quality, location,
    and age drive price, with realistic noise and some non-linear effects.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    sqft_living = rng.normal(1800, 700, n_samples).clip(400, 6000)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = (rng.integers(1, 5, n_samples) + rng.choice([0, 0.5], n_samples)).clip(1, 5)
    lot_size = rng.normal(8000, 4000, n_samples).clip(1000, 40000)
    year_built = rng.integers(1920, 2023, n_samples)
    house_age = 2024 - year_built
    overall_quality = rng.integers(1, 11, n_samples)  # 1-10 scale
    garage_cars = rng.integers(0, 4, n_samples)
    stories = rng.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.35, 0.1, 0.35, 0.1, 0.1])
    distance_to_city_km = rng.exponential(15, n_samples).clip(0.5, 80)
    school_rating = rng.integers(1, 11, n_samples)  # 1-10 scale
    crime_index = rng.normal(50, 20, n_samples).clip(0, 100)  # lower is safer
    has_renovation = rng.choice([0, 1], n_samples, p=[0.75, 0.25])

    # Price model: additive effects + interaction + noise
    price = (
        50_000
        + sqft_living * 150
        + bedrooms * 8_000
        + bathrooms * 12_000
        + lot_size * 2.5
        + overall_quality * 18_000
        + garage_cars * 9_000
        + school_rating * 7_000
        - house_age * 900
        - distance_to_city_km * 1_800
        - crime_index * 500
        + has_renovation * 15_000
        + (sqft_living * overall_quality) * 5  # quality amplifies size effect
    )
    price += rng.normal(0, 25_000, n_samples)  # noise
    price = price.clip(50_000, None)

    df = pd.DataFrame(
        {
            "SqFtLiving": sqft_living.round(0),
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "LotSize": lot_size.round(0),
            "YearBuilt": year_built,
            "HouseAge": house_age,
            "OverallQuality": overall_quality,
            "GarageCars": garage_cars,
            "Stories": stories,
            "DistanceToCityKm": distance_to_city_km.round(2),
            "SchoolRating": school_rating,
            "CrimeIndex": crime_index.round(1),
            "HasRenovation": has_renovation,
            "MedHouseVal": price.round(0),
        }
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features that typically help house price models."""
    df = df.copy()
    df["SqFtPerBedroom"] = df["SqFtLiving"] / df["Bedrooms"].replace(0, np.nan)
    df["BathBedRatio"] = df["Bathrooms"] / df["Bedrooms"].replace(0, np.nan)
    df["LotToLivingRatio"] = df["LotSize"] / df["SqFtLiving"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


def build_candidate_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }


def main():
    print("Loading data...")
    df = load_data()
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != "MedHouseVal"]
    X = df[feature_cols]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    models = build_candidate_models()
    results = {}
    trained = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"  RMSE=${metrics['RMSE']:,.0f}  MAE=${metrics['MAE']:,.0f}  R2={metrics['R2']:.4f}")

    # Pick best model by RMSE
    best_name = min(results, key=lambda n: results[n]["RMSE"])
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (RMSE=${results[best_name]['RMSE']:,.0f})")

    # Feature importance (if available) for the best model
    importance = None
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        importance = dict(zip(feature_cols, np.abs(best_model.coef_).tolist()))

    # Save artifacts
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    df.to_csv("housing_data.csv", index=False)
    X_test.assign(MedHouseVal=y_test).to_csv("test_data.csv", index=False)

    metadata = {
        "best_model": best_name,
        "results": results,
        "feature_cols": feature_cols,
        "feature_importance": importance,
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved: best_model.pkl, feature_cols.pkl, housing_data.csv, test_data.csv, model_metadata.json")


if __name__ == "__main__":
    main()
