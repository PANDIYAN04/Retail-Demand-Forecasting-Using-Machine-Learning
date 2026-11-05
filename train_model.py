"""Train, compare, and save the retail demand forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from modeling import (
    DATA_PATH,
    DATE_COLUMN,
    INPUT_FEATURES,
    MODEL_PATH,
    TARGET,
    evaluate_regression,
    load_dataset,
    make_pipeline,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"


def chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Split by distinct dates: 60% train, 20% validation, 20% final test."""
    dates = data[DATE_COLUMN].drop_duplicates().sort_values().to_numpy()
    if len(dates) < 5:
        raise ValueError("At least five distinct dates are required for time splits.")
    train_end = int(len(dates) * 0.60)
    validation_end = int(len(dates) * 0.80)
    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]

    train = data[data[DATE_COLUMN].isin(train_dates)].copy()
    validation = data[data[DATE_COLUMN].isin(validation_dates)].copy()
    test = data[data[DATE_COLUMN].isin(test_dates)].copy()
    return train, validation, test


def get_model_candidates() -> dict[str, list]:
    return {
        "Linear Regression": [LinearRegression()],
        "Random Forest": [
            RandomForestRegressor(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            RandomForestRegressor(
                n_estimators=160,
                max_depth=None,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42,
            ),
        ],
        "XGBoost": [
            XGBRegressor(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                n_jobs=4,
                random_state=42,
            ),
            XGBRegressor(
                n_estimators=350,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                n_jobs=4,
                random_state=42,
            ),
        ],
    }


def get_feature_importance(pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessing"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    return importance.sort_values("Importance", ascending=False).reset_index(drop=True)


def train() -> None:
    data = load_dataset(DATA_PATH)
    train_data, validation_data, test_data = chronological_split(data)
    x_train = train_data[INPUT_FEATURES]
    y_train = train_data[TARGET]
    x_validation = validation_data[INPUT_FEATURES]
    y_validation = validation_data[TARGET]

    comparison_rows = []
    selected_models = {}
    for model_name, candidates in get_model_candidates().items():
        best_pipeline = None
        best_metrics = None
        best_mae = float("inf")
        for candidate in candidates:
            pipeline = make_pipeline(candidate)
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_validation)
            metrics = evaluate_regression(y_validation, predictions)
            if metrics["MAE"] < best_mae:
                best_pipeline = pipeline
                best_metrics = metrics
                best_mae = metrics["MAE"]
        if best_pipeline is None or best_metrics is None:
            raise RuntimeError(f"No candidate model was trained for {model_name}.")
        selected_models[model_name] = best_pipeline
        comparison_rows.append({"Model": model_name, **best_metrics})

    comparison = pd.DataFrame(comparison_rows).sort_values("MAE").reset_index(drop=True)
    best_name = str(comparison.iloc[0]["Model"])
    final_pipeline = selected_models[best_name]

    # Keep the final test period untouched until model selection is complete.
    refit_x = pd.concat([x_train, x_validation], ignore_index=True)
    refit_y = pd.concat([y_train, y_validation], ignore_index=True)
    final_pipeline.fit(refit_x, refit_y)
    test_predictions = final_pipeline.predict(test_data[INPUT_FEATURES])
    test_metrics = evaluate_regression(test_data[TARGET], test_predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipeline,
        "selected_model": best_name,
        "input_features": INPUT_FEATURES,
        "categories": sorted(data["Category"].dropna().astype(str).unique().tolist()),
        "validation_metrics": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "test_dates": {
            "start": test_data[DATE_COLUMN].min().strftime("%Y-%m-%d"),
            "end": test_data[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        },
        "dataset_rows": int(len(data)),
        "dataset_date_range": {
            "start": data[DATE_COLUMN].min().strftime("%Y-%m-%d"),
            "end": data[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    comparison.to_csv(MODEL_DIR / "validation_model_comparison.csv", index=False)
    predictions_frame = pd.DataFrame(
        {
            "Date": test_data[DATE_COLUMN].to_numpy(),
            "Actual Demand": test_data[TARGET].to_numpy(),
            "Predicted Demand": test_predictions,
        }
    )
    predictions_frame.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    get_feature_importance(final_pipeline).to_csv(
        MODEL_DIR / "feature_importance.csv", index=False
    )
    summary = {
        "selected_model": best_name,
        "validation_metrics": bundle["validation_metrics"],
        "test_metrics": test_metrics,
        "test_dates": bundle["test_dates"],
        "dataset_rows": len(data),
        "dataset_date_range": bundle["dataset_date_range"],
    }
    (MODEL_DIR / "training_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Dataset: {len(data):,} usable rows")
    print(f"Validation period: {validation_data[DATE_COLUMN].min().date()} to "
          f"{validation_data[DATE_COLUMN].max().date()}")
    print(f"Final test period: {test_data[DATE_COLUMN].min().date()} to "
          f"{test_data[DATE_COLUMN].max().date()}")
    print("\nValidation model comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected model: {best_name}")
    print(f"Untouched final test metrics: {test_metrics}")
    print(f"Saved pipeline: {MODEL_PATH}")


if __name__ == "__main__":
    train()
