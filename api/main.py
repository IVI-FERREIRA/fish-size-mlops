# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import predict


app = FastAPI(title="Fish Size Prediction API")


class FishFeatures(BaseModel):
    Length1: float
    Length2: float
    Length3: float
    Height: float
    Width: float


@app.post("/predict")
def predict_fish_weight(features: FishFeatures):
    """
    Endpoint de inferência.
    Recebe as medidas do peixe e retorna o peso estimado.
    """
    prediction = predict(features.dict())
    return {"estimated_weight_g": prediction}
