# src/data_prep.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Caminhos do projeto
RAW_DATA_PATH = Path("data/raw/fish.csv")
PROCESSED_DATA_DIR = Path("data/processed")


def prepare_data(test_size: float = 0.2, random_state: int = 42) -> None:

    """
    Prepara os dados para treino e teste.

    - Lê o dataset bruto
    - Seleciona features e target
    - Divide em treino e teste
    - Salva os arquivos processados
    """

    # Cria diretório de dados processados se não existir
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega dados brutos
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Remove registros inválidos (peso zero ou negativo)
    df = df[df["Weight"] > 0]

    # Features e target
    features = ["Length1", "Length2", "Length3", "Height", "Width"]
    target = "Weight"

    X = df[features]
    y = df[target]

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Salva dados processados
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


if __name__ == "__main__":
    prepare_data()
