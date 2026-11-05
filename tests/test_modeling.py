import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import (
    INPUT_FEATURES,
    DemandFeatureEngineer,
    load_model_bundle,
    validate_prediction_inputs,
)
from train_model import chronological_split


class DemandModelingTests(unittest.TestCase):
    def setUp(self):
        self.row = pd.DataFrame(
            [
                {
                    "Price": 100.0,
                    "Discount": 10.0,
                    "Inventory Level": 50,
                    "Promotion": 1,
                    "Competitor Pricing": 90.0,
                    "Category": "Clothing",
                }
            ]
        )

    def test_feature_engineering_uses_supplied_values(self):
        engineered = DemandFeatureEngineer().fit_transform(self.row)
        self.assertAlmostEqual(engineered.loc[0, "Discounted Price"], 90.0)
        self.assertAlmostEqual(engineered.loc[0, "Price Difference"], 10.0)

    def test_input_validation_preserves_required_model_columns(self):
        validated = validate_prediction_inputs(self.row, ["Clothing"])
        self.assertEqual(validated.columns.tolist(), INPUT_FEATURES)
        self.assertEqual(validated.loc[0, "Category"], "Clothing")

    def test_input_validation_rejects_unknown_category(self):
        invalid = self.row.assign(Category="Unknown")
        with self.assertRaisesRegex(ValueError, "Unknown category"):
            validate_prediction_inputs(invalid, ["Clothing"])

    def test_input_validation_rejects_invalid_promotion(self):
        invalid = self.row.assign(Promotion=2)
        with self.assertRaisesRegex(ValueError, "Promotion"):
            validate_prediction_inputs(invalid, ["Clothing"])

    def test_input_validation_rejects_infinite_price(self):
        invalid = self.row.assign(Price=float("inf"))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_inputs(invalid, ["Clothing"])

    def test_bulk_csv_prediction_returns_finite_values(self):
        bundle = load_model_bundle()
        csv_path = Path(__file__).parent / "fixtures" / "sample_bulk_predictions.csv"
        csv_data = pd.read_csv(csv_path)
        validated = validate_prediction_inputs(csv_data, bundle["categories"])
        predictions = bundle["pipeline"].predict(validated)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.isfinite(predictions).all())

    def test_chronological_split_keeps_dates_separate(self):
        dates = pd.date_range("2024-01-01", periods=10)
        data = pd.DataFrame(
            {
                "Date": dates.repeat(2),
                "Demand": range(20),
                "Price": 10.0,
            }
        )
        train, validation, test = chronological_split(data)
        self.assertLess(train["Date"].max(), validation["Date"].min())
        self.assertLess(validation["Date"].max(), test["Date"].min())
        self.assertEqual(len(train) + len(validation) + len(test), len(data))

    def test_saved_pipeline_predicts_one_row(self):
        bundle = load_model_bundle()
        prediction = bundle["pipeline"].predict(self.row)
        self.assertEqual(len(prediction), 1)


if __name__ == "__main__":
    unittest.main()
