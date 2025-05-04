from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np

# Définition de l'énumération pour ocean_proximity
class OceanProximity(str, Enum):
    NEAR_BAY = "NEAR BAY"
    LESS_1H_OCEAN = "LESS THAN ONE HOUR TO OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"

# Modèle de données d'entrée
class CaliforniaHousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float = Field(..., ge=0)
    total_rooms: float = Field(..., ge=0)
    total_bedrooms: float = Field(..., ge=0)
    population: float = Field(..., ge=0)
    households: float = Field(..., ge=0)
    median_income: float = Field(..., ge=0)
    ocean_proximity: OceanProximity

app = FastAPI()

# Chargement du modèle
model = joblib.load("california_housing.pkl")

# Encodage simple de ocean_proximity
ocean_mapping = {
    "NEAR BAY": 0,
    "LESS THAN ONE HOUR TO OCEAN": 1,
    "INLAND": 2,
    "NEAR OCEAN": 3,
    "ISLAND": 4
}

@app.get("/")
def main():
    return {"message": "Welcome to California Housing API"}

@app.post("/predict")
def predict_house_price(data: CaliforniaHousingData):
    try:
        if data.ocean_proximity.value not in ocean_mapping:
            raise HTTPException(status_code=400, detail="Invalid ocean proximity value")

        features = [
            data.longitude,
            data.latitude,
            data.housing_median_age,
            data.total_rooms,
            data.total_bedrooms,
            data.population,
            data.households,
            data.median_income,
            ocean_mapping[data.ocean_proximity.value]
        ]

        # Prédiction
        prediction = model.predict([features])[0]

        return {
            "status": "success",
            "prediction": round(prediction, 2),
            "score": "81.69%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
