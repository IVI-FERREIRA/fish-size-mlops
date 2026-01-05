# src/train.py

from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn



from sklearn.metrics import mean_absolute_error, r2_score

from src.model import get_model


# Caminhos
PROCESSED_DATA_DIR = Path("data/processed")


def load_data():
    """Carrega dados de treino e teste já processados."""
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train():
    # MLflow local
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fish-size-regression")

    X_train, X_test, y_train, y_test = load_data()
    model = get_model()

    with mlflow.start_run():
        # Treinamento
        model.fit(X_train, y_train)

        # Avaliação
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log de métricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log do modelo
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="fish_size_model"
        )

        print(f"Treinamento finalizado | MAE: {mae:.2f} | R2: {r2:.2f}")


if __name__ == "__main__":
    train()
