# src/model.py

from sklearn.ensemble import RandomForestRegressor


def get_model(random_state: int = 42) -> RandomForestRegressor:
    """
    Cria e retorna o modelo de regressão.

    Centralizar a criação do modelo facilita:
    - troca de algoritmo
    - tuning futuro
    - reutilização em treino e inferência
    """

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    return model
