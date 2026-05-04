"""
src/triaje_ia/ml/inferencia.py
────────────────────────────────
Carga del modelo entrenado y prediccion sobre el vector de 88 features.
"""

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from triaje_ia.config import MODELS_DIR
from triaje_ia.data.features import TODAS_FEATURES

NOMBRE_MODELO_DEFAULT = "lgbm_ganador"

_GENDER_MAP = {"M": 0, "F": 1}


def cargar_modelo(nombre: str = NOMBRE_MODELO_DEFAULT):
    """
    Carga el modelo .joblib desde models/.

    Args:
        nombre: nombre del fichero sin extension (p.ej. 'lgbm_ganador').

    Returns:
        Modelo sklearn/lightgbm cargado.

    Raises:
        FileNotFoundError: si el fichero no existe en models/.
    """
    ruta = MODELS_DIR / f"{nombre}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {ruta}\n"
            "Asegurate de que el fichero .joblib esta en models/."
        )
    logger.info(f"Cargando modelo: {ruta.name}")
    return joblib.load(ruta)


def predecir_proba(modelo, X: pd.DataFrame) -> np.ndarray:
    """
    Predice probabilidades por clase de acuity sobre un DataFrame de features.

    Aplica el encoding de gender (M->0, F->1) antes de la inferencia.
    El modelo espera las columnas en el orden exacto de TODAS_FEATURES.

    Args:
        modelo: modelo cargado con cargar_modelo().
        X: DataFrame con al menos las columnas de TODAS_FEATURES.

    Returns:
        np.ndarray shape (n_filas, 5) con P(acuity=1..5) por fila.
    """
    X_inf = X[TODAS_FEATURES].copy()

    if "gender" in X_inf.columns:
        X_inf["gender"] = X_inf["gender"].map(_GENDER_MAP).fillna(0).astype("int8")

    probas = modelo.predict_proba(X_inf)
    return probas
