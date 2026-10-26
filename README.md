# Residential Gas Consumption Prediction

**Notebook:** `notebooks/02_model_training.ipynb`  
**Task:** Regression: *How much gas will this home use in 2024?*

## Objective

Predict annual residential gas consumption in 2024 from building characteristics and historical gas consumption.

## Features and target

* **Target:** `Gcons2024` (kWh)
* **Building characteristics:** `PROP_TYPE`, `PROP_AGE_BAND`, `FLOOR_AREA_BAND`, `COUNCIL_TAX_BAND`, `REGION`, `MAIN_HEAT_FUEL`
* **Historical gas consumption:** `Gcons2023`, `Gcons2022`, `Gcons2021`

The target-year flag `GasValFlag2024` is excluded from the input features to avoid target leakage.

## Methodology

1. **Split:** 80/20 train/test split (`random_state=42`): 31,601 training and 7,901 test homes.
2. **Preprocessing:**
   * Numerical features: median imputation and standardisation.
   * Categorical features: most-frequent imputation and one-hot encoding.
3. **Model comparison:** five-fold cross-validation, with Mean Absolute Error (MAE) as the main metric. Lower is better.
4. **Hyperparameter tuning:** `GridSearchCV` on Random Forest (three-fold) and Histogram Gradient Boosting (five-fold).

## Cross-validation results

| Model | Mean CV MAE (kWh) |
|---|---:|
| Baseline (median) | 4,901.07 |
| Decision Tree | 2,642.68 |
| Random Forest (default) | 1,849.93 |
| Linear Regression | 1,782.46 |
| Ridge Regression | 1,782.43 |
| Gradient Boosting (tuned) | 1,756.12 |
| **Random Forest (tuned)** | **1,752.97** |

The selected model is a **Random Forest** with `max_depth=10` and `n_estimators=100`.

## Test results

| Metric | Result |
|---|---:|
| MAE | 1,777.18 kWh |
| RMSE | 3,129.35 kWh |
| R² | 0.7811 |

## Key findings

* `Gcons2023` dominates the model's feature importance (0.90), followed by `Gcons2022` (0.03) and `Gcons2021` (0.03). Last year's consumption is the strongest predictor of this year's.
* Gas consumption in 2023 and 2024 is highly correlated (Pearson r = 0.89).
* A simple **previous-year baseline** (predict 2024 = 2023) achieved a **lower MAE** (1,684.92 vs 1,730.62 kWh) on the 7,794 test homes with a 2023 reading. The Random Forest achieved a **lower RMSE** (3,012.57 vs 3,117.82 kWh), so it makes fewer very large errors.
* Prediction errors grow for homes with higher consumption.

## Dataset

The notebook uses the cleaned dataset prepared on Day 1: `data/anon2026_50k_cleaned.csv`. It contains 49,998 homes x 96 columns. The model uses the **39,502 homes** that have a 2024 gas reading.

## Visualisations

All plots are saved in `figures/`.

### Actual vs predicted gas consumption

Points close to the diagonal are accurate predictions. The spread grows for homes with high consumption.

![Actual vs predicted 2024 gas consumption](figures/actual_vs_predicted.png)

### Feature importance

![Random Forest feature importance](/figures/feature_importance.png)

### 2023 vs 2024 gas consumption

![2023 vs 2024 gas consumption](figures/gas_consumption_correlation.png)

### Residuals vs predicted consumption

Positive residuals mean the model under-predicted, negative ones that it over-predicted.

![Residuals vs predicted gas consumption](figures/residuals_vs_predicted.png)

## Saved Model

`models/final_gas_consumption_model.joblib`: the full Random Forest pipeline (preprocessing + model). Load it with `joblib.load()` to predict the 2024 gas consumption of a new building. See `models/README.md` for the model card.

Model files are ignored by `.gitignore` (`models/*.joblib`). Running the notebook recreates the file.

## How to Run

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies: `pip install -r requirements.txt`
4. Check that `data/anon2026_50k_cleaned.csv` is in place.
5. Open `notebooks/02_model_training.ipynb` and run the cells in order (**Run All**).

## Limitations

* Historical gas consumption dominates the model's feature importance.
* The previous-year baseline performs better on MAE for the evaluated subset.
* Prediction errors are larger for high-consumption properties.
* The test set was also used for baseline comparisons, so it is not a completely untouched final benchmark.
* Feature importance describes what the model relies on. It does not establish causation.
* Results may differ on other datasets, regions or future years.

## AI Use Disclosure

AI tools were used to support aspects of coding, debugging, explanation, and documentation.

## Authors

IAAC AI Methods — Group 4
