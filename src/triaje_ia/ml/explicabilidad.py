"""
src/triaje_ia/ml/explicabilidad.py
────────────────────────────────────
Módulo de explicabilidad SHAP para inferencia en producción.

Genera explicaciones individuales usando TreeExplainer (O(TLD))
para modelos tree-based (LightGBM, Random Forest, XGBoost).

Uso:
    from triaje_ia.ml.explicabilidad import crear_explainer, explicar_prediccion

    explainer = crear_explainer(modelo)
    resultado = explicar_prediccion(explainer, X_fila, clase_predicha=2)

    # resultado["top_positivas"]  → features que suben P(clase)
    # resultado["top_negativas"]  → features que bajan P(clase)
    # resultado["shap_values"]    → array completo para waterfall plot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger

# ────────────────────────────────────────────────────────────
# Tipos de retorno
# ────────────────────────────────────────────────────────────

@dataclass
class FeatureContribucion:
    """Una feature con su contribución SHAP a la probabilidad de una clase."""
    nombre: str
    valor: Any           # valor de la feature en esta observación
    shap_value: float    # contribución SHAP (puede ser negativa)

    @property
    def impacto_abs(self) -> float:
        return abs(self.shap_value)


@dataclass
class ExplicacionSHAP:
    """Resultado completo de la explicación SHAP para una predicción."""
    clase_explicada: int                          # acuity 1-5
    base_value: float                              # E[f(x)] para la clase
    shap_values: np.ndarray                        # shape (n_features,)
    feature_names: list[str]                       # nombres de las 88 features
    feature_values: np.ndarray                     # valores de las features
    top_positivas: list[FeatureContribucion] = field(default_factory=list)
    top_negativas: list[FeatureContribucion] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialización para logging / API."""
        return {
            "clase_explicada": self.clase_explicada,
            "base_value": round(float(self.base_value), 5),
            "top_positivas": [
                {"feature": f.nombre, "valor": _safe_value(f.valor),
                 "shap": round(float(f.shap_value), 5)}
                for f in self.top_positivas
            ],
            "top_negativas": [
                {"feature": f.nombre, "valor": _safe_value(f.valor),
                 "shap": round(float(f.shap_value), 5)}
                for f in self.top_negativas
            ],
        }


def _safe_value(v: Any) -> Any:
    """Convierte valores numpy a tipos nativos de Python para serialización."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if pd.isna(v):
        return None
    return v


# ────────────────────────────────────────────────────────────
# Explainer
# ────────────────────────────────────────────────────────────

def crear_explainer(modelo) -> shap.TreeExplainer:
    """
    Crea un TreeExplainer para el modelo dado.

    Compatible con:
    - LightGBM (Booster o LGBMClassifier)
    - RandomForest (sklearn)
    - XGBoost (Booster o XGBClassifier)

    Args:
        modelo: modelo entrenado tree-based.

    Returns:
        shap.TreeExplainer listo para .shap_values()
    """
    logger.info(f"Creando TreeExplainer para {type(modelo).__name__}")
    explainer = shap.TreeExplainer(modelo)
    logger.success("TreeExplainer creado correctamente")
    return explainer


def explicar_prediccion(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    clase_predicha: int,
    top_n: int = 5,
) -> ExplicacionSHAP:
    """
    Genera una explicación SHAP para una predicción individual.

    Args:
        explainer: TreeExplainer creado con crear_explainer().
        X: DataFrame de 1 fila con las 88 features (orden TODAS_FEATURES).
        clase_predicha: clase predicha (acuity 1-5). Se usa como índice
                        de clase para extraer los SHAP values relevantes.
        top_n: número de features top positivas/negativas a devolver.

    Returns:
        ExplicacionSHAP con todos los datos para visualización.
    """
    if len(X) != 1:
        raise ValueError(f"Se espera exactamente 1 fila, recibidas {len(X)}")

    X_shap = X

    idx_clase = clase_predicha - 1  # acuity 1-5 → índice 0-4

    # Calcular SHAP values — TreeExplainer devuelve lista de arrays por clase
    sv_raw = explainer.shap_values(X_shap)

    # sv_raw puede ser:
    #   - lista de n_clases arrays shape (1, n_features)  [RF, XGB]
    #   - np.ndarray shape (1, n_features, n_clases)      [LGBM]
    #   - np.ndarray shape (1, n_features)                [binario]
    if isinstance(sv_raw, list):
        shap_clase = sv_raw[idx_clase][0]  # shape (n_features,)
        shap_all = np.stack([arr[0] for arr in sv_raw], axis=-1)
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        # LGBM: shape (1, n_features, n_clases)
        shap_clase = sv_raw[0, :, idx_clase]
        shap_all = sv_raw[0]  # (n_features, n_clases)
    else:
        # Binario o formato inesperado
        shap_clase = sv_raw[0] if sv_raw.ndim == 2 else sv_raw
        shap_all = None

    # Base value
    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        base_value = float(bv[idx_clase])
    else:
        base_value = float(bv)

    # Nombres y valores de features
    feature_names = list(X_shap.columns) if hasattr(X_shap, 'columns') else list(X.columns)
    feature_values = X_shap.iloc[0].values if hasattr(X_shap, 'iloc') else X.iloc[0].values

    # Top contribuciones positivas (suben probabilidad de la clase)
    indices_pos = np.argsort(-shap_clase)
    top_pos = []
    for i in indices_pos:
        if shap_clase[i] > 0 and len(top_pos) < top_n:
            top_pos.append(FeatureContribucion(
                nombre=feature_names[i],
                valor=feature_values[i],
                shap_value=float(shap_clase[i]),
            ))

    # Top contribuciones negativas (bajan probabilidad de la clase)
    indices_neg = np.argsort(shap_clase)
    top_neg = []
    for i in indices_neg:
        if shap_clase[i] < 0 and len(top_neg) < top_n:
            top_neg.append(FeatureContribucion(
                nombre=feature_names[i],
                valor=feature_values[i],
                shap_value=float(shap_clase[i]),
            ))

    resultado = ExplicacionSHAP(
        clase_explicada=clase_predicha,
        base_value=base_value,
        shap_values=shap_clase,
        feature_names=feature_names,
        feature_values=feature_values,
        top_positivas=top_pos,
        top_negativas=top_neg,
    )
    # Adjuntar array completo multi-clase para uso avanzado
    if shap_all is not None:
        resultado._shap_all = shap_all

    logger.info(
        f"Explicación SHAP para acuity={clase_predicha}: "
        f"top+ {[f.nombre for f in top_pos[:3]]}, "
        f"top- {[f.nombre for f in top_neg[:3]]}"
    )
    return resultado
