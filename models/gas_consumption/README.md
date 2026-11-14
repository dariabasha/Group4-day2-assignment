---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - random-forest
  - energy
  - housing
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - mae
  - rmse
  - r2
---

# Residential gas consumption: predicting a home's 2024 gas use from its building data and past consumption

This model predicts the **annual gas consumption in 2024 (kWh)** of a home in England or Wales, from six building characteristics and its gas consumption in 2021, 2022 and 2023. It is a scikit-learn pipeline (preprocessing + Random Forest) trained on 31,601 homes from the UK NEED anonymised dataset. We built it as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/Group4-day2-assignment
- **Team:** Group 4 – Fawad Saleem, Hetashri Kalyani, Daria Basha, Luis Wulf

## Intended use

- **Intended users and use:** energy suppliers or local energy-planning teams that want a **first estimate of next year's gas demand** of homes for which they know the building data and the last three years of meter readings. Best used on groups of homes, for planning and comparison.
- **Out-of-scope uses:** homes outside England and Wales; homes without past gas readings; years far from 2021–2024; billing or any decision about an individual household; judging how energy-efficient a home is (the model mostly repeats last year's consumption, it does not measure efficiency).

## Model details

| | |
|---|---|
| **Problem type** | Regression |
| **Algorithm** | `RandomForestRegressor` |
| **Key hyperparameters** | `n_estimators=100`, `max_depth=10`, `random_state=42` (chosen with `GridSearchCV`) |
| **Preprocessing** | Numeric columns: median imputation + `StandardScaler`. Categorical columns: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`) |
| **File** | `model.joblib`: the whole fitted scikit-learn `Pipeline` |
| **Library versions** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0 |

## Input features

The model expects a `pandas.DataFrame` with exactly these 9 columns:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `PROP_TYPE` | category (text) | – | `Flat`, `Mid terrace`, `End terrace`, `Semi detached`, `Detached`, `Bungalow` | one-hot |
| `PROP_AGE_BAND` | category (int) | – | Construction age band, 1 (oldest) to 4 (newest) | one-hot |
| `FLOOR_AREA_BAND` | category (int) | – | 1 = up to 50 m², 2 = 51–100, 3 = 101–150, 4 = 151–200, 5 = over 200 m² | one-hot |
| `COUNCIL_TAX_BAND` | category (text) | – | `A` to `I` | one-hot |
| `REGION` | category (text) | – | ONS region code, e.g. `E12000007` = London, `W92000004` = Wales | one-hot |
| `MAIN_HEAT_FUEL` | category (int) | – | 1 = gas, 2 = electricity | one-hot |
| `Gcons2023` | numeric | kWh/year | Gas consumption in 2023 | median imputation, scaling |
| `Gcons2022` | numeric | kWh/year | Gas consumption in 2022 | median imputation, scaling |
| `Gcons2021` | numeric | kWh/year | Gas consumption in 2021 | median imputation, scaling |

⚠️ Use the **same types** as in the training data: integers for the bands and `MAIN_HEAT_FUEL` (not text like `"1"`), text for the other categories. A value the model has not seen is silently ignored by the one-hot encoder.

## Output

`.predict()` returns the predicted **2024 gas consumption in kWh per year** (no log transform).

## Training data

| | |
|---|---|
| **Dataset** | National Energy Efficiency Data-Framework (NEED), anonymised 50k sample |
| **Source** | [UK Department for Energy Security and Net Zero](https://www.gov.uk/government/collections/national-energy-efficiency-data-framework-need) |
| **Licence** | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
| **One row is** | one home in England or Wales |
| **Size used** | 39,502 homes with a 2024 gas reading (train 31,601; test 7,901) |
| **Coverage** | England and Wales; gas readings 2021–2024 |

**Cleaning:** exact duplicates removed (49,998 of 50,000 homes kept) and column types checked, see our Day 1 repository: https://github.com/iaac-maai/Group4-day1-assignment. For this model we kept only homes with a 2024 gas reading. The flag `GasValFlag2024` was excluded from the inputs, because it is only known together with the target (leakage).

## Training procedure

- **Split:** 80 / 20 train/test, `random_state=42`
- **Validation:** 5-fold cross-validation (`KFold`, shuffled, `random_state=42`) on the training set, MAE as the main metric
- **Models compared:**

| Model | CV MAE in kWh (mean ± std) |
|---|---:|
| Baseline (`DummyRegressor`, median) | 4,901 ± 48 |
| Decision Tree | 2,643 ± 14 |
| Random Forest (default) | 1,850 ± 10 |
| Linear Regression | 1,782 ± 17 |
| Ridge Regression | 1,782 ± 17 |
| Histogram Gradient Boosting | 1,759 ± 18 |
| **Random Forest (tuned)** | **1,753 ± 11** |

- **Tuning:** `GridSearchCV` on Random Forest (`n_estimators` 50/100, `max_depth` 10/20, 3-fold) and on Gradient Boosting (`learning_rate`, `max_iter`, `max_leaf_nodes`, 5-fold).
- **Why this model:** it has the lowest cross-validated MAE. The differences between the tuned Random Forest, Gradient Boosting and the linear models are small (about 30 kWh, close to the fold-to-fold noise), so a linear model would be a reasonable, faster and more interpretable alternative.

## Evaluation

Results on the held-out **test set** (7,901 homes):

| Metric | Test | Cross-validation |
|---|---:|---:|
| MAE | 1,777 kWh | 1,753 ± 11 kWh |
| RMSE | 3,129 kWh | – |
| R² | 0.78 | – |

**Comparison with a simple rule:** on the 7,794 test homes with a 2023 reading, predicting "2024 = 2023" gives MAE 1,685 kWh and RMSE 3,118 kWh. The model has a **higher MAE** (1,731 kWh) but a **lower RMSE** (3,013 kWh): it is not better for an average home, but it makes fewer very large errors.

**Where it performs worst:** large homes. The median absolute error grows from about 900 kWh for floor area band 1 to about 2,100 kWh for band 5 (over 200 m²), and there are few large homes in the data.

## Limitations and biases

- `Gcons2023` gets about 90% of the feature importance: the model mainly carries last year's consumption forward. It cannot predict the effect of a change in the home (new insulation, new occupants).
- Errors grow for high-consumption and large homes.
- The test set was also used for the previous-year comparison above, so it is not a completely untouched benchmark.
- Consumption depends on occupants, heating habits and weather, which are not in the data. NEED rounds readings to 100 kWh.
- Feature importance shows what the model relies on, not what causes high gas use.

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day2-group4-gas-consumption", filename="model.joblib")
model = joblib.load(path)

home = pd.DataFrame([{
    "PROP_TYPE": "Detached",
    "PROP_AGE_BAND": 1,
    "FLOOR_AREA_BAND": 5,
    "COUNCIL_TAX_BAND": "D",
    "REGION": "E12000007",
    "MAIN_HEAT_FUEL": 1,
    "Gcons2023": 12500,
    "Gcons2022": 11800,
    "Gcons2021": 11000,
}])
print(f"Predicted 2024 gas use: {model.predict(home)[0]:,.0f} kWh")
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions of scikit-learn.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/Group4-day2-assignment. Run `notebook/02_model_training.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Fawad Saleem, Hetashri Kalyani, Daria Basha, Luis Wulf (Group 4). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is the NEED anonymised dataset (UK Department for Energy Security and Net Zero), published under the Open Government Licence v3.0.
