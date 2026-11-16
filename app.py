import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.joblib"
FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "FullBath", "BedroomAbvGr", "YearBuilt", "LotArea"
]

if not MODEL_PATH.exists():
    raise RuntimeError("Model not found. Run: python train.py")

# Load once at application startup, not once per prediction.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Ames HomeValue AI",
    description="House sale-price prediction API using a trained regression pipeline.",
    version="1.0.0"
)


class HouseFeatures(BaseModel):
    OverallQual: int = Field(ge=1, le=10, description="Overall house quality (1-10)")
    GrLivArea: int = Field(ge=300, le=6000, description="Above-ground living area in sq ft")
    GarageCars: int = Field(ge=0, le=4, description="Garage capacity in cars")
    TotalBsmtSF: int = Field(ge=0, le=3000, description="Total basement area in sq ft")
    FullBath: int = Field(ge=0, le=4, description="Number of full bathrooms")
    BedroomAbvGr: int = Field(ge=0, le=8, description="Bedrooms above ground")
    YearBuilt: int = Field(ge=1872, le=2010, description="Construction year")
    LotArea: int = Field(ge=1000, le=200000, description="Plot area in sq ft")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/predict")
def predict(house: HouseFeatures):
    row = pd.DataFrame([house.model_dump()], columns=FEATURES)
    price = max(0.0, float(model.predict(row)[0]))

    tips = []
    if house.OverallQual <= 5:
        tips.append("Improving overall construction and finish quality may increase estimated value.")
    if house.GrLivArea < 1200:
        tips.append("Larger above-ground living space is often associated with higher sale prices.")
    if house.GarageCars < 2:
        tips.append("Additional garage capacity may improve the property's appeal.")
    if house.YearBuilt < 1980:
        tips.append("Renovation and maintenance history can materially affect the actual sale price.")
    if not tips:
        tips.append("The property has several strong features; compare nearby recent sales before pricing.")

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": house.model_dump(),
        "predicted_price": round(price, 2)
    }
    with (ROOT / "prediction_logs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    return {
        "predicted_price": round(price, 2),
        "currency": "USD",
        "tips": tips[:3],
        "disclaimer": "This is an educational estimate based on Ames, Iowa historical data, not a professional appraisal."
    }
