# src/inference.py

from typing import Dict, List
import mlflow.pyfunc
import pandas as pd


# Nome do modelo registrado no MLflow
MODEL_NAME = "fish_size_model"
MODEL_STAGE = "None"  # usando a última versão registrada


def load_model():
    """
    Carrega o modelo registrado no MLflow.
    Em produção, isso poderia apontar para um Model Registry remoto.
    """
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(features: Dict[str, float]) -> float:
    """
    Realiza a inferência a partir das features de entrada.

    Espera um dicionário com:
    Length1, Length2, Length3, Height, Width
    """

    model = load_model()

    # Converte entrada para DataFrame (formato esperado pelo modelo)
    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)

    # Retorna valor escalar
    return float(prediction[0])


if __name__ == "__main__":
    # Exemplo simples de uso local
    sample_input = {
        "Length1": 23.2,
        "Length2": 25.4,
        "Length3": 30.1,
        "Height": 11.5,
        "Width": 4.2,
    }

    result = predict(sample_input)
    print(f"Peso estimado: {result:.2f} g")
