"""
src/triaje_ia/inference/adapter.py
────────────────────────────────────
Convierte un VectorClinico (salida del LLM, unidades clinicas) en un
DataFrame con las 88 features MIMIC-IV listas para el modelo.

Mapeo de campos:
  VectorClinico            -> MIMIC-IV-ED
  ─────────────────────────────────────────────────────────────────
  edad                     -> age
  sexo (M/F/Otro)          -> gender (M/F)  [Otro -> M, decision conservadora]
  presion_sistolica        -> sbp
  presion_diastolica       -> dbp
  frecuencia_cardiaca      -> heartrate
  frecuencia_respiratoria  -> resprate
  saturacion_oxigeno       -> o2sat
  temperatura (Celsius)    -> temperature (Fahrenheit)
  nivel_dolor              -> pain
  sintomas_presentes       -> chiefcomplaint (join con coma)
  medicacion_habitual      -> medicacion_raw (join con coma, lowercase)
  len(medicacion_habitual) -> n_medicamentos

Defaults para campos sin equivalente en VectorClinico:
  race               = UNKNOWN  (sin datos demograficos de raza)
  arrival_transport  = UNKNOWN  (modo de llegada desconocido)
  Bloque 6 (historial ED): todas las features a 0/NaN
    - primera_visita  = 1 (asumimos primera visita)
    - n_visitas_previas, visitas_ultimo_mes, visitas_ultimo_ano = 0
    - dias_desde_ultima_visita = NaN
    - frecuentador = 0
    - hx_* = 0 (sin comorbilidades registradas)

NOTA: el bloque 6 no se calcula desde edstays en produccion porque requiere
el historial completo del paciente. Los defaults son conservadores y estan
documentados en MODEL_CARD.md.
"""

import numpy as np
import pandas as pd

from triaje_ia.data.features import (
    CCS_MACROFLAGS,
    calcular_features_bloque1,
    calcular_features_bloque2,
    calcular_features_bloque3,
    calcular_features_bloque4,
    calcular_features_bloque5,
    TODAS_FEATURES,
)
from triaje_ia.llm.schemas import VectorClinico

_SEXO_MAP = {"M": "M", "F": "F", "Otro": "M"}


def _celsius_a_fahrenheit(celsius: float | None) -> float | None:
    if celsius is None:
        return None
    return round(celsius * 9 / 5 + 32, 1)


def _mapear_sexo(sexo: str) -> str:
    return _SEXO_MAP.get(sexo, "M")


def vectorclinico_a_features(v: VectorClinico) -> pd.DataFrame:
    """
    Convierte un VectorClinico extraido por el LLM en un DataFrame de 88 features.

    Args:
        v: VectorClinico validado por Pydantic.

    Returns:
        DataFrame de 1 fila con exactamente las columnas de TODAS_FEATURES.
    """
    bloque6_defaults = {
        "n_visitas_previas":         0,
        "primera_visita":            1,
        "dias_desde_ultima_visita":  np.nan,
        "visitas_ultimo_mes":        0,
        "visitas_ultimo_año":        0,
        "frecuentador":              0,
        **{k: 0 for k in CCS_MACROFLAGS},
    }

    row = {
        "age":               v.edad,
        "gender":            _mapear_sexo(v.sexo),
        "race":              "UNKNOWN",
        "arrival_transport": "UNKNOWN",
        "heartrate":         v.frecuencia_cardiaca,
        "resprate":          v.frecuencia_respiratoria,
        "o2sat":             v.saturacion_oxigeno,
        "sbp":               v.presion_sistolica,
        "dbp":               v.presion_diastolica,
        "temperature":       _celsius_a_fahrenheit(v.temperatura),
        "pain":              v.nivel_dolor,
        "chiefcomplaint":    ", ".join(v.sintomas_presentes) if v.sintomas_presentes else "",
        "n_medicamentos":    len(v.medicacion_habitual),
        "medicacion_raw":    ", ".join(v.medicacion_habitual).lower(),
        **bloque6_defaults,
    }

    df = pd.DataFrame([row])

    # Coercionar vitales a float para que np.isnan funcione con valores None
    _cols_float = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain", "age"]
    for col in _cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = calcular_features_bloque1(df)
    df = calcular_features_bloque2(df)
    df = calcular_features_bloque3(df)
    df = calcular_features_bloque4(df)
    df = calcular_features_bloque5(df)

    faltantes = [f for f in TODAS_FEATURES if f not in df.columns]
    if faltantes:
        raise ValueError(
            f"El adapter no produjo todas las features esperadas. Faltan: {faltantes}"
        )

    return df[TODAS_FEATURES]
