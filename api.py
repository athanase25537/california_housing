from fastapi import FastAPI, HTTPException
import requests
import joblib
from pydantic import BaseModel, Field
from enum import Enum
import os
import gdown
import joblib
from io import BytesIO

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
# URL corrigée pour le téléchargement direct
file_id = "1XJLP_fiK2J9KDC21f5vKiVYimoydIYuj"
url = f"https://drive.google.com/uc?id={file_id}"

# Méthode 1 - Via gdown (recommandé)
try:
    output = "california_housing.pkl"
    gdown.download(url, output, quiet=False)
    model = joblib.load(output)
except Exception as e:
    print(f"Erreur avec gdown: {e}")
    
    # Méthode 2 - Fallback avec requests
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Gestion de la confirmation Google
    token = None
    for key, value in response.cookies.items():
        if 'download_warning' in key:
            token = value
    
    if token:
        url = f"{url}&confirm={token}"
        response = session.get(url)
    
    # Sauvegarde temporaire pour inspection
    with open("temp_download.pkl", "wb") as f:
        f.write(response.content)
    
    try:
        model = joblib.load("temp_download.pkl")
    except Exception as e:
        print(f"Le fichier semble corrompu. Erreur: {e}")
        print("Veuillez vérifier que le fichier est un modèle joblib valide.")

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
            "score": "95.69%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
