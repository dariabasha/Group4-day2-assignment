---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - sklearn
  - tabular-classification
  - gradient-boosting
  - energy
  - housing
  - epc
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - f1
  - accuracy
---

# EPC band: estimating a home's energy rating from its gas and electricity use

This model estimates the **EPC band** (A/B, C, D, E or F/G) of a gas-heated home in England or Wales, from its 2024 gas and electricity consumption, its size, age, type and region, and three efficiency flags (loft insulation, cavity-wall insulation, solar PV). It is a scikit-learn pipeline (preprocessing + Histogram Gradient Boosting) trained on 21,492 homes from the UK NEED anonymised dataset. We built it as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/Group4-day2-assignment
- **Team:** Group 4 – Fawad Saleem, Hetashri Kalyani, Daria Basha, Luis Wulf

## Intended use

- **Intended users and use:** councils, retrofit programmes and students who want a **rough first estimate** of the energy rating of homes without an EPC, or who want to explore how consumption relates to the rating ("if this is the gas use, which band is likely?").
- **Out-of-scope uses:** replacing a real EPC assessment; any decision about an individual household (sale, rent, tariff, grant); **finding the least efficient homes** (the model almost never predicts E or F/G); homes not heated by mains gas; homes outside England and Wales.

## Model details

| | |
|---|---|
| **Problem type** | Classification, 5 ordered classes |
| **Algorithm** | `HistGradientBoostingClassifier` |
| **Key hyperparameters** | `learning_rate=0.1`, `max_leaf_nodes=31`, `class_weight=None`, `random_state=42` |
| **Preprocessing** | Numeric columns and ordered bands: median imputation (no scaling for the tree model). Categories without order: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`) |
| **File** | `epc_band_gradient_boosting.joblib`: the whole fitted scikit-learn `Pipeline` |
| **Library versions** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0 |

## Input features

The model expects a `pandas.DataFrame` with exactly these 9 columns:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `Gcons2024` | numeric | kWh/year | Gas consumption in 2024 | median imputation |
| `Econs2024` | numeric | kWh/year | Electricity consumption in 2024 | median imputation |
| `FLOOR_AREA_BAND` | ordered (int) | – | 1 = up to 50 m², 2 = 51–100, 3 = 101–150, 4 = 151–200, 5 = over 200 m² | kept as a number |
| `PROP_AGE_BAND` | ordered (int) | – | Construction age band, 1 (oldest) to 4 (newest) | kept as a number |
| `LI_FLAG` | 0/1 | – | Loft insulation installed | kept as a number |
| `CWI_FLAG` | 0/1 | – | Cavity-wall insulation installed | kept as a number |
| `PV_FLAG` | 0/1 | – | Solar PV installed | kept as a number |
| `PROP_TYPE` | category (text) | – | `Flat`, `Mid terrace`, `End terrace`, `Semi detached`, `Detached`, `Bungalow` | one-hot |
| `REGION` | category (text) | – | ONS region code, e.g. `E12000007` = London, `W92000004` = Wales | one-hot |

## Output

`.predict()` returns one of 5 EPC bands as text: `A/B` (most efficient), `C`, `D`, `E`, `F/G` (least efficient). The NEED data groups A with B and F with G. `.predict_proba()` returns the probability of each band, in the order of `model.classes_`.

## Training data

| | |
|---|---|
| **Dataset** | National Energy Efficiency Data-Framework (NEED), anonymised 50k sample |
| **Source** | [UK Department for Energy Security and Net Zero](https://www.gov.uk/government/collections/national-energy-efficiency-data-framework-need) |
| **Licence** | [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) |
| **One row is** | one home in England or Wales |
| **Size used** | 26,866 homes with a 2024 gas reading and a known EPC band (train 21,492; test 5,374) |
| **Coverage** | England and Wales; consumption in 2024 |

**Cleaning:** exact duplicates removed (49,998 of 50,000 homes kept) and column types checked, see our Day 1 repository: https://github.com/iaac-maai/Group4-day1-assignment. For this model we kept homes with a 2024 gas reading and set aside the 12,636 homes with `No EPC`. `MAIN_HEAT_FUEL` was dropped because it has the same value (mains gas) for every home in this subset.

**Class balance:** A/B 12.9%, C 35.5%, D 40.9%, E 9.6%, F/G 1.2%.

## Training procedure

- **Split:** 80 / 20 train/test, **stratified** by EPC band, `random_state=42`
- **Validation:** stratified 5-fold cross-validation on the training set, **macro F1** as the main metric (every band counts equally, so the rare bands matter)
- **Models compared:**

| Model | CV macro F1 (mean ± std) | CV accuracy |
|---|---:|---:|
| Baseline (`DummyClassifier`, most frequent) | 0.116 ± 0.000 | 0.409 |
| KNN (k=25) | 0.381 ± 0.003 | 0.577 |
| Random Forest | 0.387 ± 0.004 | 0.594 |
| Decision Tree (`max_depth=8`) | 0.390 ± 0.006 | 0.594 |
| Logistic Regression | 0.395 ± 0.002 | 0.602 |
| **Histogram Gradient Boosting** | **0.397 ± 0.008** | **0.599** |

- **Which inputs matter:** gas only reaches macro F1 0.20; adding electricity adds nothing; adding building information doubles it to 0.40.
- **Tuning:** `GridSearchCV` on `learning_rate` (0.05/0.1), `max_leaf_nodes` (15/31) and `class_weight` (none/balanced), 5-fold.
- **Why this model:** the grid search ranked `class_weight="balanced"` slightly higher (macro F1 0.410 vs 0.397), but that model is much less accurate (48% vs 60%) and only about 4% of its F/G predictions are correct. We chose the unweighted model. The models are all within about 0.02 macro F1, so the choice among them matters much less than the inputs.

## Evaluation

Results on the held-out **test set** (5,374 homes):

| Metric | Test | Cross-validation |
|---|---:|---:|
| Macro F1 | 0.40 | 0.397 ± 0.008 |
| Accuracy (exact band) | 0.59 | 0.599 ± 0.004 |
| Within ±1 band | 0.98 | – |

Per band (test set):

| Band | Precision | Recall | F1 | Homes |
|---|---:|---:|---:|---:|
| A/B | 0.68 | 0.81 | 0.74 | 695 |
| C | 0.61 | 0.45 | 0.51 | 1,906 |
| D | 0.57 | 0.80 | 0.66 | 2,197 |
| E | 0.39 | 0.03 | 0.06 | 514 |
| F/G | 0.00 | 0.00 | 0.00 | 62 |

**Where it performs worst:** the inefficient bands. E and F/G are almost always predicted as D, because they are rare and their consumption is close to that of D homes. Many C homes are also predicted as D.

## Limitations and biases

- The model is **biased towards D** when it is unsure. For the homes without an EPC it predicts D for 79% of them, against 41% D among homes with an EPC. This is the model's habit, not a finding about these homes.
- An EPC is calculated mainly from the building fabric, not from measured consumption. Consumption says more about **how big** a home is than about **how efficient** it is. Property age is the strongest predictor.
- Consumption also depends on occupants, heating habits and weather, which are not in the data.
- Only mains-gas homes in England and Wales with a 2024 gas reading; homes without an EPC may differ from those with one.
- Feature importance (permutation importance on the test set) shows what the model relies on, not causes.

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/day2-group4-epc-band", filename="epc_band_gradient_boosting.joblib")
model = joblib.load(path)

home = pd.DataFrame([{
    "Gcons2024": 10000, "Econs2024": 2500,
    "FLOOR_AREA_BAND": 2, "PROP_AGE_BAND": 1,
    "LI_FLAG": 0, "CWI_FLAG": 0, "PV_FLAG": 0,
    "PROP_TYPE": "Semi detached", "REGION": "E12000008",
}])
print("EPC band:", model.predict(home)[0])
print(dict(zip(model.classes_, model.predict_proba(home)[0].round(2))))
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions of scikit-learn.

## Reproducibility

The full pipeline is in the GitHub repository: https://github.com/iaac-maai/Group4-day2-assignment. Run `notebook/03_epc_band_prediction.ipynb` with the packages in `requirements.txt`. A detailed write-up is in `README_03_epc_band_prediction.md`.

## Authors and licence

Fawad Saleem, Hetashri Kalyani, Daria Basha, Luis Wulf (Group 4). Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is the NEED anonymised dataset (UK Department for Energy Security and Net Zero), published under the Open Government Licence v3.0.
