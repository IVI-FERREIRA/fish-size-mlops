# src/inference.py
from typing import Dict
import pandas as pd
import joblib

MODEL_PATH = "/app/model.pkl"

_model = None  # cache do modelo em memória


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: Dict[str, float]) -> float:
    model = load_model()
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return float(prediction[0])


if __name__ == "__main__":
    sample_input = {
        "Length1": 23.2,
        "Length2": 25.4,
        "Length3": 30.1,
        "Height": 11.5,
        "Width": 4.2,
    }

    result = predict(sample_input)
    print(f"Peso estimado: {result:.2f} g")
