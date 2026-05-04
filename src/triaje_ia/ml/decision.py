"""
src/triaje_ia/ml/decision.py
──────────────────────────────
Politicas de decision sobre el vector de probabilidades del modelo.

Politica oficial para demo y TFG: threshold_a1
  Si P(acuity=1) >= 0.20 -> asignar acuity 1, independientemente del argmax.
  Justificacion: el coste clinico de un falso negativo en paciente critico
  supera ampliamente el de un falso positivo.
"""

import numpy as np

UMBRAL_A1_DEFAULT: float = 0.20


def argmax(probas: np.ndarray) -> int:
    """
    Devuelve la clase con mayor probabilidad.
    acuity 1-5: el indice 0 del array corresponde a acuity=1.
    """
    return int(np.argmax(probas)) + 1


def threshold_a1(probas: np.ndarray, umbral: float = UMBRAL_A1_DEFAULT) -> int:
    """
    Politica de seguridad clinica para acuity=1 (critico).

    Si P(acuity=1) >= umbral, devuelve 1 aunque el argmax sea otra clase.
    En caso contrario devuelve el argmax normal.

    Args:
        probas: array de probabilidades shape (5,), orden [P(1), P(2), ..., P(5)].
        umbral: umbral minimo para activar acuity=1. Por defecto 0.20.

    Returns:
        acuity predicho (1-5).
    """
    if probas[0] >= umbral:
        return 1
    return argmax(probas)


def multi_threshold(probas: np.ndarray, umbrales: dict[int, float]) -> int:
    """
    Politica experimental con umbral independiente por clase.

    Recorre acuity 1..5 en orden de severidad. Devuelve el primero cuya
    probabilidad supere el umbral definido. Si ninguno supera su umbral,
    devuelve el argmax.

    Args:
        probas: array de probabilidades shape (5,).
        umbrales: dict {acuity: umbral_minimo}, p.ej. {1: 0.20, 2: 0.35}.

    Returns:
        acuity predicho (1-5).
    """
    for acuity in [1, 2, 3, 4, 5]:
        if probas[acuity - 1] >= umbrales.get(acuity, 1.1):
            return acuity
    return argmax(probas)
