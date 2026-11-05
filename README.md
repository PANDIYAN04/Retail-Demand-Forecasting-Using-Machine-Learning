# Retail Demand Forecasting Using Machine Learning

A retail analytics and demand estimation project built with
Python, scikit-learn, XGBoost, Plotly, and Streamlit. The dashboard explores
historical sales patterns, compares three regression models, and predicts
demand from product category, price, discount, inventory, promotion, and
competitor pricing.

> **Scope:** This is a demand-estimation demonstration built on the supplied
> historical dataset. It is not a guarantee of future sales or a full
> store/product-level forecasting system.

## Project objectives

- Explore a retail dataset and summarize demand trends.
- Build a reusable preprocessing-and-regression pipeline.
- Compare Linear Regression, Random Forest, and XGBoost.
- Use date-ordered training, validation, and final test periods.
- Provide interactive single-row and CSV-based predictions.

## Problem statement

Retailers need to estimate product demand while considering pricing, discount,
inventory, promotions, and category. This project learns relationships between
those available product features and the dataset's `Demand` field, then exposes
the estimate through an interactive dashboard.

## Dataset

The original `demand_forecasting.csv` is preserved unchanged. It contains
**76,000 records** and 16 columns, covering **2022-01-01 to 2024-01-30**:
date, store and product identifiers, category, region, inventory, units sold
and ordered, price, discount, weather, promotion, competitor pricing,
seasonality, epidemic indicator, and demand. The observed source file has no
missing cells or duplicate rows.
The categories are Clothing, Electronics, Furniture, Groceries, and Toys.

The prediction model uses the six fields available in its input form:

| Input | Meaning |
| --- | --- |
| `Category` | Product category |
| `Price` | Product price |
| `Discount` | Discount percentage |
| `Inventory Level` | Available inventory |
| `Promotion` | Promotion flag: `0` or `1` |
| `Competitor Pricing` | Competitor's price |

`Discounted Price` and `Price Difference` (own price minus competitor price)
are derived inside the shared pipeline. Date is used to order the data split,
not as a user-supplied predictor. Other source fields are not used by this
model and are not represented as prediction inputs.

## Technologies

- Python
- pandas and NumPy
- scikit-learn (`Pipeline`, imputation, scaling, one-hot encoding, regression)
- XGBoost
- Plotly
- Streamlit
- joblib for the complete model pipeline artifact

## Machine learning workflow

1. Load and validate the source file; parse dates, coerce numeric columns,
   discard rows without a valid date or target, and remove duplicate rows.
2. Sort chronologically and split by **distinct dates**: first 60% for model
   fitting, next 20% for validation/model selection, and final 20% held out for
   the final test result. Rows from one date stay in the same split.
3. Use a single scikit-learn pipeline for each model: derive pricing features,
   impute numeric values with training medians, standardize numeric values,
   impute category values, one-hot encode categories, and fit the regressor.
   The one-hot encoder ignores unknown categories safely; dashboard inputs
   validate categories against the model's training categories.
4. Compare Linear Regression with small, documented Random Forest and XGBoost
   candidate configurations. Select each algorithm's candidate and the overall
   model using validation MAE.
5. Refit the selected pipeline on training plus validation data, evaluate once
   on the untouched final test period, and save the complete pipeline along
   with metrics, test predictions, and feature importance under `models/`.
6. Use the same serialized pipeline for interactive and bulk predictions.

### Model evaluation results

The exact model and metrics are generated from the included dataset by running
`python train_model.py`. The validation comparison and final holdout result
below are the values from the saved training run (not fabricated estimates).

**Selected model:** `Random Forest`

Validation period: `2023-04-02` to `2023-08-31` · Final test period:
`2023-09-01` to `2024-01-30`

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 32.70 | 42.13 | 0.315 |
| Random Forest | 28.87 | 38.71 | 0.421 |
| XGBoost | 29.13 | 38.71 | 0.421 |

Selected model, final test period:

| Test MAE | Test RMSE | Test R² |
| ---: | ---: | ---: |
| 26.18 | 34.22 | 0.398 |

The numerical metrics are in demand units except for R². Lower MAE/RMSE and
higher R² are generally preferred; model selection in this project uses the
lowest validation MAE. Rerunning training can update the saved results.

## Application features

- **Home Dashboard:** dataset size, average and total demand, category count,
  date coverage, and monthly average-demand trend.
- **Data Analysis:** filtered demand distribution, category comparison, price
  versus demand, promotion comparison, and discount analysis.
- **Demand Prediction:** validated product inputs and a demand estimate in
  units.
- **Model Performance:** validation comparison, final test MAE/RMSE/R²,
  actual-versus-predicted chart, and feature importance.
- **Bulk Prediction:** downloadable CSV template, input validation, results
  table, and downloadable prediction CSV.

## Project structure

```text
.
├── app.py                              # Streamlit application entry point
├── modeling.py                         # Shared features, pipeline, validation
├── train_model.py                      # Reproducible model training workflow
├── requirements.txt                    # Runtime and training dependencies
├── demand_forecasting.csv              # Original source dataset
├── analysis.ipynb                      # Original exploratory notebook
├── machine_learning.ipynb              # Original model exploration notebook
├── preprocessed_demand_forecasting_data.csv  # Original derived export
├── models/
│   ├── demand_pipeline.joblib          # Complete fitted pipeline and metadata
│   ├── feature_importance.csv
│   ├── test_predictions.csv
│   ├── training_results.json
│   └── validation_model_comparison.csv
└── tests/
    ├── fixtures/
    │   └── sample_bulk_predictions.csv
    └── test_modeling.py
```

The notebooks and legacy model files are retained for reference. The
application uses `models/demand_pipeline.joblib`, not the legacy encoder/model
pickle pair. You do not need to run a notebook to train or launch the app.

## Install and run locally

Use Python 3.11 or newer. From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies and train the artifacts:

```bash
python -m pip install -r requirements.txt
python train_model.py
```

The training command creates/replaces the files under `models/`. Launch the
dashboard from the project root:

```bash
streamlit run app.py
```

Run the focused checks after training:

```bash
python -m unittest discover -s tests -v
```

## Screenshots

<img width="1917" height="967" alt="Screenshot 2026-11-05 175316" src="https://github.com/user-attachments/assets/fd8d9ad6-16e9-4b01-b06e-97ec3858d86d" />

Add screenshots of the Home Dashboard, Model Performance, and prediction
sections here after running the application.

## Future improvements

- Add a date-aware forecasting interface if users can provide a forecast date.
- Evaluate rolling-origin backtests and demand error by category.
- Add unit tests for more malformed CSV cases and app interactions.
- Track additional known-at-prediction-time product and store context.
- Add data drift and retraining checks only when the project needs them.

## Notes and limitations

- The dataset has 100 records per date; the chronological split keeps all
  records from any one date together.
- The model does not use store ID, product ID, region, weather, seasonality,
  units sold, or units ordered. Bulk prediction accepts only the same six
  inputs used by the single prediction form.
- Validation metrics are used for model selection. The final test scores are
  reported separately to avoid selecting on the final test period.
- Feature importance indicates model reliance, not causal effects.
- Only load the bundled joblib artifact from a trusted source; joblib files,
  like pickle files, can execute code when loaded.
