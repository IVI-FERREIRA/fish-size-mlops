# api/main.py
from fastapi import UploadFile, File
from src.vision import extract_features_from_image
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

@app.post("/predict-image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Recebe uma imagem do peixe, extrai medidas simples
    e estima o peso usando o modelo treinado.
    """
    image_bytes = await file.read()

    features = extract_features_from_image(image_bytes)
    prediction = predict(features)

    return {
        "features_extracted": features,
        "estimated_weight_g": prediction
    }


@app.post("/predict")
def predict_fish_weight(features: FishFeatures):
    """
    Endpoint de inferência.
    Recebe as medidas do peixe e retorna o peso estimado.
    """
    prediction = predict(features.dict())
    return {"estimated_weight_g": prediction}
