# Models

This repository has **two models**. Each one has its own model card, which is also the README of its Hugging Face repository:

| Model | Task | Model card | Hugging Face |
|---|---|---|---|
| Gas consumption (Random Forest) | Regression: 2024 gas use in kWh | [gas_consumption/README.md](gas_consumption/README.md) | https://huggingface.co/iaac-maai/Day2-team4-residential-gas-consumption-predictor |
| EPC band (Gradient Boosting) | Classification: EPC band A/B–F/G | [epc_band/README.md](epc_band/README.md) | https://huggingface.co/iaac-maai/Day2-Team4-epc-band-predictor |

The trained `.joblib` files are **not** committed to GitHub (see `.gitignore`). Running the notebooks saves them here:

- `notebook/02_model_training.ipynb` → `models/final_gas_consumption_model.joblib`
- `notebook/03_epc_band_prediction.ipynb` → `models/epc_band_gradient_boosting.joblib`

On Hugging Face, each `.joblib` file is stored next to its model card, with the same file name.
