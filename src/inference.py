# src/inference.py

from typing import Dict
import mlflow.pyfunc
import pandas as pd


MODEL_NAME = "fish_size_model"
MODEL_STAGE = "None"

_model = None  # cache do modelo em memória


def load_model():
    """
    Carrega o modelo do MLflow apenas uma vez e mantém em cache.
    """
    global _model
    if _model is None:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        _model = mlflow.pyfunc.load_model(model_uri)
    return _model


def predict(features: Dict[str, float]) -> float:
    """
    Realiza a inferência a partir das features de entrada.
    """
    model = load_model()

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)

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
