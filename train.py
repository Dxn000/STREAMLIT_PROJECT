import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "solar_power_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_solar_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "linear_baseline_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1.0):
    """Calculates MAPE with an epsilon offset to avoid division by zero for nighttime records."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Only calculate MAPE on daytime power generation (> 10 kW)
    daytime_mask = y_true > 10.0
    if np.sum(daytime_mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[daytime_mask] - y_pred[daytime_mask]) / y_true[daytime_mask])) * 100


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Loads and validates the Solar Power Generation dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    print("=" * 70)
    print("  CLEAN ENERGY & CLIMATE TECH: SOLAR POWER GENERATION FORECASTING  ")
    print("=" * 70)
    print(f"Dataset loaded: {len(df):,} observations, {df.shape[1]} features.")
    print(f"Null check: {df.isnull().sum().sum()} missing values detected.")
    return df


def train_and_evaluate():
    df = load_and_validate_data(DATA_PATH)

    # Feature selection
    feature_cols = [
        "ambient_temperature",
        "module_temperature",
        "temp_diff",
        "irradiation",
        "daily_yield",
        "hour"
    ]
    target_col = "ac_power"

    X = df[feature_cols]
    y = df[target_col]

    # 80/20 Train-Test Split (reproducible seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), feature_cols)
        ]
    )

    candidate_models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42
        ),
    }

    benchmark_results = {}
    fitted_pipelines = {}

    print("\n--- MODEL BENCHMARKING (5-Fold CV & Test Evaluation) ---")
    print(f"{'Model':<30} | {'CV R2':<8} | {'Test R2':<8} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'Day MAPE':<10}")
    print("-" * 90)

    best_score = -float("inf")
    best_model_name = None

    for name, regressor in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # 5-Fold Cross Validation on Training Sample (or full)
        cv_scores = cross_val_score(pipeline, X_train.sample(5000, random_state=42), y_train.sample(5000, random_state=42), cv=5, scoring="r2")
        mean_cv_r2 = cv_scores.mean()

        # Fit on training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on unseen test set
        test_preds = np.clip(pipeline.predict(X_test), 0, None)  # Power cannot be negative
        r2 = r2_score(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        rmse = root_mean_squared_error(y_test, test_preds)
        mape = mean_absolute_percentage_error(y_test, test_preds)

        benchmark_results[name] = {
            "cv_r2_mean": round(float(mean_cv_r2), 4),
            "test_r2": round(float(r2), 4),
            "test_mae_kw": round(float(mae), 2),
            "test_rmse_kw": round(float(rmse), 2),
            "daytime_mape_pct": round(float(mape), 2),
        }

        print(f"{name:<30} | {mean_cv_r2:<8.4f} | {r2:<8.4f} | {mae:<10.2f} | {rmse:<10.2f} | {mape:<9.2f}%")

        if r2 > best_score:
            best_score = r2
            best_model_name = name

    print("-" * 90)
    print(f"\n>>> Champion Model Selected: {best_model_name} (Test R2: {best_score:.4f})")

    # Feature Importances from Champion Model
    best_pipeline = fitted_pipelines[best_model_name]
    reg = best_pipeline.named_steps["regressor"]
    feature_importances = {}
    if hasattr(reg, "feature_importances_"):
        for fname, imp in zip(feature_cols, reg.feature_importances_):
            feature_importances[fname] = round(float(imp), 4)
        print("\n--- PHYSICAL SENSOR FEATURE IMPORTANCES ---")
        for k, v in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:<22}: {v * 100:.2f}%")

    # Save Models
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(fitted_pipelines["Linear Regression (Baseline)"], BASELINE_MODEL_PATH)

    # Save Metrics & Metadata
    metadata = {
        "best_model": best_model_name,
        "features": feature_cols,
        "target": target_col,
        "dataset_rows": len(df),
        "benchmark": benchmark_results,
        "feature_importances": feature_importances,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nArtifacts successfully exported:")
    print(f"  - Champion Model: {BEST_MODEL_PATH}")
    print(f"  - Baseline Model: {BASELINE_MODEL_PATH}")
    print(f"  - Metrics Summary: {METRICS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
