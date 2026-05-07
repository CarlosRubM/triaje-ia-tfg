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

from triaje_ia.data.features import TODAS_FEATURES


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

    @property
    def shap_values_todas_clases(self) -> np.ndarray | None:
        """Shape (n_features, n_clases) si se almacenó el array completo."""
        return getattr(self, "_shap_all", None)

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

def _es_ordinal_frank_hall(modelo) -> bool:
    """Detecta si el modelo es un OrdinalFrankHall."""
    return hasattr(modelo, "classifiers_") and hasattr(modelo, "n_classes_")


def _extraer_clasificador_binario(modelo, clase: int):
    """
    Extrae el clasificador binario relevante de un OrdinalFrankHall.

    Para clase k (1-indexed), el corte más informativo es:
    - clase 1: h1 (P(Y≤1) vs P(Y>1))
    - clase k (2..K-1): h_{k-1} (P(Y≤k-1) vs P(Y>k-1))
    - clase K: h_{K-1} (P(Y≤K-1) vs P(Y>K-1))

    Si el clasificador binario es un Pipeline (preprocesador + clasificador),
    se extrae el estimador final para que TreeExplainer funcione.
    """
    idx = min(clase - 1, len(modelo.classifiers_) - 1)
    clf = modelo.classifiers_[idx]

    # Si es un Pipeline sklearn, extraer el step final (el clasificador)
    from sklearn.pipeline import Pipeline as SkPipeline
    if isinstance(clf, SkPipeline):
        clf = clf.named_steps["clasificador"]

    return clf, idx


def crear_explainer(modelo, clase: int | None = None) -> shap.TreeExplainer:
    """
    Crea un TreeExplainer para el modelo dado.

    Compatible con:
    - LightGBM (Booster o LGBMClassifier)
    - RandomForest (sklearn)
    - XGBoost (Booster o XGBClassifier)
    - OrdinalFrankHall (usa el clasificador binario del corte relevante)

    Para OrdinalFrankHall, se debe pasar `clase` para seleccionar el
    clasificador binario correcto. El explainer explica P(Y>threshold_k).

    Args:
        modelo: modelo entrenado tree-based o OrdinalFrankHall.
        clase: acuity 1-5, requerido para OrdinalFrankHall.

    Returns:
        shap.TreeExplainer listo para .shap_values()
    """
    if _es_ordinal_frank_hall(modelo):
        if clase is None:
            clase = 1  # default: explicar el corte más crítico
        clf_binario, idx = _extraer_clasificador_binario(modelo, clase)
        logger.info(
            f"OrdinalFrankHall detectado: usando h{idx+1} "
            f"(corte ordinal para clase {clase})"
        )
        explainer = shap.TreeExplainer(clf_binario)
        logger.success(f"TreeExplainer creado para clasificador binario h{idx+1}")
        return explainer

    logger.info(f"Creando TreeExplainer para {type(modelo).__name__}")
    explainer = shap.TreeExplainer(modelo)
    logger.success("TreeExplainer creado correctamente")
    return explainer


def _preprocesar_para_shap(
    modelo,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocesa X a través del pipeline del modelo si es OrdinalFrankHall.

    OrdinalFrankHall contiene pipelines internos con un preprocesador.
    SHAP TreeExplainer opera sobre el clasificador base, no el pipeline,
    así que debemos preprocesar X primero.
    """
    if _es_ordinal_frank_hall(modelo):
        from sklearn.pipeline import Pipeline as SkPipeline
        clf_pipeline = modelo.classifiers_[0]  # todos comparten el mismo preprocesador
        if isinstance(clf_pipeline, SkPipeline):
            prepro = clf_pipeline.named_steps.get("preprocesador")
            if prepro is not None:
                return prepro.transform(X)
    return X


def explicar_prediccion(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    clase_predicha: int,
    top_n: int = 5,
    modelo=None,
) -> ExplicacionSHAP:
    """
    Genera una explicación SHAP para una predicción individual.

    Args:
        explainer: TreeExplainer creado con crear_explainer().
        X: DataFrame de 1 fila con las 88 features (orden TODAS_FEATURES).
        clase_predicha: clase predicha (acuity 1-5). Se usa como índice
                        de clase para extraer los SHAP values relevantes.
        top_n: número de features top positivas/negativas a devolver.
        modelo: modelo original (necesario para OrdinalFrankHall para
                preprocesar X antes de SHAP).

    Returns:
        ExplicacionSHAP con todos los datos para visualización.
    """
    if len(X) != 1:
        raise ValueError(f"Se espera exactamente 1 fila, recibidas {len(X)}")

    # Para OrdinalFrankHall, preprocesar X a través del pipeline interno
    if modelo is not None and _es_ordinal_frank_hall(modelo):
        X_shap = _preprocesar_para_shap(modelo, X)
    else:
        X_shap = X

    idx_clase = clase_predicha - 1  # acuity 1-5 → índice 0-4

    # Calcular SHAP values — TreeExplainer devuelve lista de arrays por clase
    sv_raw = explainer.shap_values(X_shap)

    # sv_raw puede ser:
    #   - lista de 2 arrays shape (1, n_features) [binario de OrdinalFrankHall]
    #   - lista de n_clases arrays shape (1, n_features)  [RF, XGB]
    #   - np.ndarray shape (1, n_features, n_clases)      [LGBM]
    #   - np.ndarray shape (1, n_features)                [binario]
    if isinstance(sv_raw, list):
        if len(sv_raw) == 2:
            # Clasificador binario (OrdinalFrankHall): clase positiva = index 1
            shap_clase = sv_raw[1][0]  # P(Y > threshold)
            shap_all = None
        else:
            # Lista de arrays por clase (RF, XGBoost multiclase)
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
        if len(bv) == 2:
            base_value = float(bv[1])  # binario: clase positiva
        else:
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


def generar_shap_explanation_object(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    clase: int,
    modelo=None,
) -> shap.Explanation:
    """
    Genera un objeto shap.Explanation compatible con shap.plots.waterfall()
    y streamlit-shap.

    Útil para renderizar directamente con:
        st_shap(shap.plots.waterfall(explanation))

    Para OrdinalFrankHall, pasa `modelo` para preprocesar X correctamente.

    Args:
        explainer: TreeExplainer.
        X: DataFrame de 1 fila.
        clase: acuity 1-5.
        modelo: modelo original (necesario para OrdinalFrankHall).

    Returns:
        shap.Explanation para una sola observación y clase.
    """
    # Preprocesar si es OrdinalFrankHall
    if modelo is not None and _es_ordinal_frank_hall(modelo):
        X_shap = _preprocesar_para_shap(modelo, X)
    else:
        X_shap = X

    sv_raw = explainer.shap_values(X_shap)

    # Para clasificador binario (OrdinalFrankHall), usar clase positiva
    if isinstance(sv_raw, list) and len(sv_raw) == 2:
        values = sv_raw[1][0]
    elif isinstance(sv_raw, list):
        idx = clase - 1
        values = sv_raw[idx][0]
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        idx = clase - 1
        values = sv_raw[0, :, idx]
    else:
        values = sv_raw[0]

    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        base_value = float(bv[1]) if len(bv) == 2 else float(bv[clase - 1])
    else:
        base_value = float(bv)

    feature_names = list(X_shap.columns) if hasattr(X_shap, 'columns') else list(X.columns)
    feature_values = X_shap.iloc[0].values if hasattr(X_shap, 'iloc') else X.iloc[0].values

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names,
    )
