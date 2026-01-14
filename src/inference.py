# src/inference.py
from typing import Dict
import pandas as pd
import joblib
from pathlib import Path

# Resolve o caminho do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminho do modelo (funciona local e no Docker)
MODEL_PATH = BASE_DIR / "model.pkl"

_model = None

# Carrega o modelo apenas uma vez 
def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

# /predict - Numeros
def predict(features: Dict[str, float]) -> float:
    model = load_model()
# Forçar ordem correta
    EXPECTED_COLUMNS = ["Length1", "Length2", "Length3", "Height", "Width"]
    df = pd.DataFrame(
        [[features[c] for c in EXPECTED_COLUMNS]],
        columns=EXPECTED_COLUMNS
    )

    prediction = model.predict(df)
    return float(prediction[0])
