"""Shared preprocessing and evaluation helpers for demand forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

DATA_PATH = Path(__file__).resolve().parent / "demand_forecasting.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "demand_pipeline.joblib"

TARGET = "Demand"
DATE_COLUMN = "Date"
CATEGORICAL_FEATURES = ["Category"]
NUMERIC_FEATURES = [
    "Price",
    "Discount",
    "Inventory Level",
    "Promotion",
    "Competitor Pricing",
    "Discounted Price",
    "Price Difference",
]
INPUT_FEATURES = [
    "Price",
    "Discount",
    "Inventory Level",
    "Promotion",
    "Competitor Pricing",
    "Category",
]
REQUIRED_TRAINING_COLUMNS = [DATE_COLUMN, *INPUT_FEATURES, TARGET]


class DemandFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive pricing features from the inputs available in the dashboard."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DemandFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.loc[:, INPUT_FEATURES].copy()
        features["Discounted Price"] = features["Price"] * (
            1 - features["Discount"] / 100
        )
        features["Price Difference"] = (
            features["Price"] - features["Competitor Pricing"]
        )
        return features


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate source data without changing the original CSV."""
    data = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_TRAINING_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.loc[:, REQUIRED_TRAINING_COLUMNS].copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    for column in NUMERIC_FEATURES[:5] + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[DATE_COLUMN, TARGET])
    data = data.drop_duplicates()
    if data.empty:
        raise ValueError("No usable rows remain after validating dates and demand.")
    if data["Category"].dropna().empty:
        raise ValueError("No usable category values were found in the dataset.")
    return data.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(model: BaseEstimator) -> Pipeline:
    """Build the same complete preprocessing-and-model pipeline for all models."""
    return Pipeline(
        steps=[
            ("feature_engineering", DemandFeatureEngineer()),
            ("preprocessing", make_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load a trusted, locally generated joblib bundle with useful errors."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model pipeline not found at {path}. Run `python train_model.py` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"The model bundle at {path} has an unexpected format.")
    return bundle


def validate_prediction_inputs(data: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Validate prediction columns and values, then return a clean copy."""
    missing_columns = sorted(set(INPUT_FEATURES) - set(data.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    clean = data.loc[:, INPUT_FEATURES].copy()
    numeric_columns = [column for column in INPUT_FEATURES if column != "Category"]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[numeric_columns].isna().any().any():
        raise ValueError("Numeric inputs must contain valid, non-empty numbers.")
    if clean["Category"].isna().any():
        raise ValueError("Category values cannot be blank.")
    clean["Category"] = clean["Category"].astype(str).str.strip()

    if not np.isfinite(clean[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Numeric inputs must be finite values.")
    invalid_categories = sorted(set(clean["Category"]) - set(categories))
    if invalid_categories:
        raise ValueError(f"Unknown category values: {invalid_categories}")
    if (clean[["Price", "Inventory Level", "Competitor Pricing"]] < 0).any().any():
        raise ValueError("Price, inventory, and competitor pricing cannot be negative.")
    if not clean["Discount"].between(0, 100).all():
        raise ValueError("Discount must be between 0 and 100.")
    if not np.equal(clean["Inventory Level"] % 1, 0).all():
        raise ValueError("Inventory level must be a whole number.")
    if not clean["Promotion"].isin([0, 1]).all():
        raise ValueError("Promotion must be 0 (no) or 1 (yes).")
    return clean
