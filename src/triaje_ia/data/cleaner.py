"""
src/triaje_ia/data/cleaner.py

Limpieza estructural del dataset base producido por loader.py.

Responsabilidades:
  - Eliminar filas sin target (acuity es nulo o no es un número entero entre 1 y 5)
  - Eliminar columnas no útiles para el modelo
  - Convertir vitales erróneas a NaN (errores de registro)
  - Parsear y normalizar pain (campo texto libre → numérico)
  - Agrupar race en 6 categorías
  - Normalizar tipos de datos

NO responsabilidades (van en el Pipeline de sklearn):
  - Imputación de NaN (debe fitear solo sobre X_train)
  - Escalado / normalización de features
  - Encoding de categóricas
"""

import numpy as np
import pandas as pd
from loguru import logger

from triaje_ia.config import DATA_INTERIM

# Rangos fisiológicos aceptables
# Valores fuera de rango → NaN (errores de registro, no outliers clínicos)
RANGOS_VITALES: dict[str, tuple[float, float]] = {
    "temperature": (95.0, 107.0),   # °F
    "heartrate":   (20.0, 300.0),
    "resprate":    (4.0,  60.0),
    "o2sat":       (50.0, 100.0),
    "sbp":         (40.0, 300.0),
    "dbp":         (10.0, 200.0),
}

# Agrupación de race
# race se mantiene como variable de auditoría de sesgos, no entra al modelo
RACE_MAP: dict[str, str] = {
    "WHITE":                                      "WHITE",
    "WHITE - OTHER EUROPEAN":                     "WHITE",
    "WHITE - RUSSIAN":                            "WHITE",
    "WHITE - BRAZILIAN":                          "WHITE",
    "WHITE - EASTERN EUROPEAN":                   "WHITE",
    "PORTUGUESE":                                 "WHITE",
    "BLACK/AFRICAN AMERICAN":                     "BLACK",
    "BLACK/CAPE VERDEAN":                         "BLACK",
    "BLACK/AFRICAN":                              "BLACK",
    "BLACK/CARIBBEAN ISLAND":                     "BLACK",
    "HISPANIC/LATINO - PUERTO RICAN":             "HISPANIC",
    "HISPANIC/LATINO - DOMINICAN":                "HISPANIC",
    "HISPANIC OR LATINO":                         "HISPANIC",
    "HISPANIC/LATINO - GUATEMALAN":               "HISPANIC",
    "HISPANIC/LATINO - SALVADORAN":               "HISPANIC",
    "HISPANIC/LATINO - COLUMBIAN":                "HISPANIC",
    "HISPANIC/LATINO - MEXICAN":                  "HISPANIC",
    "HISPANIC/LATINO - HONDURAN":                 "HISPANIC",
    "HISPANIC/LATINO - CUBAN":                    "HISPANIC",
    "HISPANIC/LATINO - CENTRAL AMERICAN":         "HISPANIC",
    "SOUTH AMERICAN":                             "HISPANIC",
    "ASIAN":                                      "ASIAN",
    "ASIAN - CHINESE":                            "ASIAN",
    "ASIAN - ASIAN INDIAN":                       "ASIAN",
    "ASIAN - SOUTH EAST ASIAN":                   "ASIAN",
    "ASIAN - KOREAN":                             "ASIAN",
    "OTHER":                                      "OTHER",
    "AMERICAN INDIAN/ALASKA NATIVE":              "OTHER",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER":  "OTHER",
    "MULTIPLE RACE/ETHNICITY":                    "OTHER",
    "UNKNOWN":                                    "UNKNOWN",
    "PATIENT DECLINED TO ANSWER":                 "UNKNOWN",
    "UNABLE TO OBTAIN":                           "UNKNOWN",
}

# Funciones privadas

def _parsear_pain(val) -> float:
    """Convierte pain a numérico 0-10. Cualquier texto libre → NaN."""
    if pd.isna(val):
        return np.nan
    try:
        num = float(str(val).strip().strip('"').strip("'"))
        if 0 <= num <= 10:
            return round(num)
        return np.nan
    except ValueError:
        return np.nan


def _limpiar_vitales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a NaN los valores fuera de rango fisiológico.
    No imputa — eso va en el Pipeline de sklearn.
    """
    df = df.copy()
    for col, (vmin, vmax) in RANGOS_VITALES.items():
        if col not in df.columns:
            continue
        fuera = ((df[col] < vmin) | (df[col] > vmax))
        n = fuera.sum()
        if n > 0:
            df.loc[fuera, col] = np.nan
            logger.info(f"{col}: {n:,} valores fuera de [{vmin}, {vmax}] → NaN")
    return df


def _agrupar_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las 33 categorías de race en 6 grupos canónicos.
    Sobrescribe la columna ``race`` in-place; no crea columnas nuevas.
    """
    df = df.copy()
    df["race"] = df["race"].map(RACE_MAP).fillna("UNKNOWN")
    return df

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estructural del dataset base.

    Operaciones (en orden):
    1. Eliminar columnas no útiles para el modelo
    2. Eliminar filas sin acuity (target principal no imputable)
    3. Limpiar vitales fuera de rango fisiológico → NaN
    4. Parsear pain → numérico 0-10, resto → NaN
    5. Agrupar race en 6 categorías
    6. Ajustar tipos de datos

    Args:
        df: Dataset base producido por loader.py

    Returns:
        DataFrame limpio listo para el Pipeline de sklearn.
    """
    logger.info(f"Limpieza: entrada {len(df):,} filas | {df.shape[1]} columnas")
    df = df.copy()

    # 1. Eliminar columnas no útiles
    cols_eliminar = [
        "hadm_id",        # ID administrativo, no es feature
        "disposition",    # sustituida por label_disposition
        # outtime: se conserva para feature engineering temporal
    ]
    df = df.drop(columns=[c for c in cols_eliminar if c in df.columns])
    logger.info(f"Columnas eliminadas: {cols_eliminar}")

    # 2. Eliminar filas sin acuity — target no imputable
    n_antes = len(df)
    df = df.dropna(subset=["acuity"])
    n_perdidas = n_antes - len(df)
    logger.info(f"Filas sin acuity eliminadas: {n_perdidas:,} ({n_perdidas/n_antes*100:.2f}%)")

    # 3. Vitales fuera de rango fisiológico → NaN
    df = _limpiar_vitales(df)

    # 4. Pain → numérico
    df["pain"] = df["pain"].apply(_parsear_pain)

    # 5. Race → 6 categorías
    df = _agrupar_race(df)

    # 6. Sanear medicacion_raw: NaN → "" (red de seguridad; loader.py ya hace fillna)
    if "medicacion_raw" in df.columns:
        df["medicacion_raw"] = df["medicacion_raw"].fillna("")

    # 7. Tipos de datos
    df["acuity"] = df["acuity"].astype("int8")
    df["pain"]   = df["pain"].astype("Int8")
    df["age"]    = df["age"].astype("Int16")

    logger.success(
        f"Limpieza completada: {len(df):,} filas | {df.shape[1]} columnas | "
        f"perdidas: {n_antes - len(df):,} ({(n_antes - len(df))/n_antes*100:.2f}%)"
    )
    return df


def cargar_dataset_limpio(forzar: bool = False) -> pd.DataFrame:
    """
    Devuelve el dataset limpio, usando caché si existe.

    Args:
        forzar: Si True, recalcula aunque exista caché.

    Returns:
        DataFrame limpio listo para feature engineering.
    """
    from triaje_ia.data.loader import cargar_dataset_base

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    cache = DATA_INTERIM / "dataset_clean.parquet"

    if cache.exists() and not forzar:
        logger.warning(
            f"Cargando desde caché: {cache}. "
            "Si cambiaste el pipeline, usa forzar=True."
        )
        df = pd.read_parquet(cache)
        logger.success(f"Cargado: {len(df):,} filas | {df.shape[1]} columnas")
        return df

    df_raw = cargar_dataset_base(forzar=False)
    df     = limpiar_dataset(df_raw)

    df.to_parquet(cache, index=False)
    logger.success(f"Guardado en {cache}")
    return df
