import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "FullBath", "BedroomAbvGr", "YearBuilt", "LotArea"
]
TARGET = "SalePrice"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=18)
    args = parser.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(
            "data/train.csv is missing. Download train.csv from Kaggle and place it in data/."
        )

    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required].copy()
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), FEATURES)],
        remainder="drop"
    )

    candidates = [
        ("LinearRegression", LinearRegression(), {"algorithm": "LinearRegression"}),
        ("RandomForest", RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ), {"algorithm": "RandomForest", "n_estimators": args.n_estimators,
            "max_depth": args.max_depth}),
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=250, max_depth=args.max_depth,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ), {"algorithm": "ExtraTrees", "n_estimators": 250,
            "max_depth": args.max_depth})
    ]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("House-Price-Prediction")
    results = []
    best_rmse = float("inf")
    best_pipeline = None
    best_name = None

    for name, estimator, params in candidates:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_valid)
            rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
            mae = float(mean_absolute_error(y_valid, pred))
            r2 = float(r2_score(y_valid, pred))
            mlflow.log_params(params)
            mlflow.log_param("features", ",".join(FEATURES))
            mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            results.append({
                "model": name, "RMSE": round(rmse, 2),
                "MAE": round(mae, 2), "R2": round(r2, 4)
            })
            if rmse < best_rmse:
                best_rmse, best_pipeline, best_name = rmse, pipeline, name

    joblib.dump(best_pipeline, MODEL_DIR / "model.joblib")
    report = {
        "best_model": best_name,
        "validation_rows": len(y_valid),
        "training_rows": len(y_train),
        "features": FEATURES,
        "results": sorted(results, key=lambda x: x["RMSE"])
    }
    (ROOT / "model_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nBest model saved to model/model.joblib")


if __name__ == "__main__":
    main()
