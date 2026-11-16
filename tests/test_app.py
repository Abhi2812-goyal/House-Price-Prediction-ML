
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post("/predict", json={
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "YearBuilt": 2003,
        "LotArea": 8450
    })
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_stronger_house_costs_more():
    weak = {
        "OverallQual": 3,
        "GrLivArea": 900,
        "GarageCars": 0,
        "TotalBsmtSF": 500,
        "FullBath": 1,
        "BedroomAbvGr": 2,
        "YearBuilt": 1950,
        "LotArea": 5000
    }
    strong = {
        "OverallQual": 10,
        "GrLivArea": 3000,
        "GarageCars": 3,
        "TotalBsmtSF": 2000,
        "FullBath": 3,
        "BedroomAbvGr": 4,
        "YearBuilt": 2010,
        "LotArea": 15000
    }

    weak_price = client.post("/predict", json=weak).json()["predicted_price"]
    strong_price = client.post("/predict", json=strong).json()["predicted_price"]

    assert strong_price > weak_price


def test_invalid_prediction():
    response = client.post("/predict", json={
        "OverallQual": 15,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "BedroomAbvGr": 3,
        "YearBuilt": 2003,
        "LotArea": 8450
    })
    assert response.status_code == 422
