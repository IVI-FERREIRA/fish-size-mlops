# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.vision import extract_features_from_image
from src.inference import predict

app = FastAPI(title="Fish Size Prediction API")


def validate_features(f):
    # todas as medidas precisam ser > 0
    if any(v <= 0 for v in f.values()):
        raise HTTPException(status_code=400, detail="Medidas devem ser > 0")

    # comprimentos devem ser crescentes
    if not (f["Length1"] <= f["Length2"] <= f["Length3"]):
        raise HTTPException(status_code=400, detail="Comprimentos inválidos")

    return f


class FishFeatures(BaseModel):
    Length1: float
    Length2: float
    Length3: float
    Height: float
    Width: float


@app.post("/predict-image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Recebe uma imagem do peixe, extrai medidas
    e estima o peso usando o modelo treinado.
    """
    image_bytes = await file.read()

    # 1) extrai features da imagem
    features = extract_features_from_image(image_bytes)

    # 2) valida features
    features = validate_features(features)

    # 3) faz predição
    prediction = predict(features)

    return {
        "features_extracted": features,
        "estimated_weight_g": round(prediction, 2)
    }


@app.post("/predict")
def predict_fish_weight(features: FishFeatures):
    """
    Endpoint de inferência via JSON.
    Recebe as medidas do peixe e retorna o peso estimado.
    """
    data = validate_features(features.dict())
    prediction = predict(data)
    return {"estimated_weight_g": prediction}
