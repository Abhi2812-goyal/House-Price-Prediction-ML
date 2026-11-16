# Ames HomeValue AI — House Price Prediction

An end-to-end regression system that estimates residential sale prices from property characteristics using the Kaggle **House Prices: Advanced Regression Techniques** dataset.

## Dataset

Source: [Kaggle competition data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Download `train.csv` from Kaggle and place it at `data/train.csv`. The training file contains 1,460 rows, 80 explanatory columns and the `SalePrice` target. This project uses eight documented input features. The data describes historical house sales in Ames, Iowa; predictions are educational estimates, not current Indian property valuations.

## Data contract

| Feature | Type | Allowed range | Why it matters |
|---|---|---:|---|
| OverallQual | integer | 1–10 | Overall material and finish quality |
| GrLivArea | integer | 300–6000 sq ft | Above-ground living space |
| GarageCars | integer | 0–4 | Garage capacity |
| TotalBsmtSF | integer | 0–3000 sq ft | Basement size |
| FullBath | integer | 0–4 | Full bathrooms above grade |
| BedroomAbvGr | integer | 0–8 | Bedrooms above grade |
| YearBuilt | integer | 1872–2010 | Construction year in this dataset |
| LotArea | integer | 1000–200000 sq ft | Land area |
| SalePrice (target) | float | > 0 | Historical sale price in USD |

Every API input range is enforced by Pydantic. Invalid requests return HTTP 422.

## Local setup (Windows PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put Kaggle's `train.csv` in `data/`, then run:

```powershell
python train.py --n_estimators 200 --max_depth 18
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open MLflow at `http://127.0.0.1:5000`. In another terminal:

```powershell
uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000` for the UI and `http://127.0.0.1:8000/docs` for Swagger.

## Experiments

`train.py` runs Linear Regression, Random Forest and Extra Trees on the same 80/20 split. It tracks three MLflow runs with RMSE, MAE and R², compares validation RMSE, and saves the best pipeline to `model/model.joblib`. Run it at least once before starting the API.

Add the actual output from `model_comparison.json` here after training. Do not invent or estimate model scores.

## API

- `GET /health` — health and model-loaded status
- `GET /` — interactive prediction UI
- `POST /predict` — validated prediction and up to three rule-based tips

Prediction inputs are appended to `prediction_logs.jsonl` for basic monitoring.

## Tests

```powershell
pytest -v --tb=short
```

The suite covers health, UI, positive predictions, a domain-logic comparison and HTTP 422 validation.

## Docker

Train the model first, then:

```powershell
docker build -t houseprice-ai .
docker run -d -p 8000:8000 --restart unless-stopped houseprice-ai
```

The serving image uses `python:3.11-slim` and `requirements-serve.txt`.

## GitHub Actions

Push to `main` to run the workflow in `.github/workflows/ci.yml`. It installs dependencies, trains a baseline model and runs pytest.

## Limitations and next steps

The source data is historical Ames, Iowa data from 2006–2010. It is not representative of current prices in India or other markets. Further work could use a current local dataset, geographic variables, explainability (SHAP), drift monitoring and periodic retraining.

## Team

Add team member names, roll numbers, section and individual contributions before submission.
