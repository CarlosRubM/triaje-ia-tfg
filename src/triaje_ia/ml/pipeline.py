"""
src/triaje_ia/ml/pipeline.py
─────────────────────────────
Preprocesador sklearn para las 88 features del dataset MIMIC-IV-ED.

Responsabilidades:
  - Definir el ColumnTransformer de preprocesamiento (imputacion + encoding).
  - Exponer construir_pipeline(clasificador) para entrenamiento reproducible.

Notas:
  - LightGBM maneja NaN nativamente; el imputador es necesario para RF y XGBoost.
  - gender (M/F) se codifica como 0/1 con OrdinalEncoder.
  - Las features binarias no necesitan imputacion (son flags 0/1 calculados).
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from triaje_ia.data.features import FEATURES_CONTINUAS, FEATURES_CATEGORICAS


def construir_preprocesador() -> ColumnTransformer:
    """
    ColumnTransformer para las 88 features.

    - Continuas: imputacion por mediana.
    - Categoricas (gender): OrdinalEncoder con M=0, F=1.
    - Binarias: passthrough (sin transformacion).
    """
    continuas_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categoricas_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["M", "F"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("continuas",   continuas_transformer,   FEATURES_CONTINUAS),
            ("categoricas", categoricas_transformer,  FEATURES_CATEGORICAS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def construir_pipeline(clasificador) -> Pipeline:
    """
    Pipeline completo: preprocesador + clasificador.

    Args:
        clasificador: estimador sklearn compatible con fit/predict_proba.

    Returns:
        Pipeline listo para fit(X, y).
    """
    return Pipeline([
        ("preprocesador", construir_preprocesador()),
        ("clasificador",  clasificador),
    ])


def construir_preprocesador_lgbm() -> ColumnTransformer:
    """
    ColumnTransformer para LGBM: sin imputación de continuas.

    LightGBM maneja NaN nativamente — imputar con mediana viola la
    naturaleza NMAR de los vitales (ver memoria TFG §features.py).
    Solo se procesan las categóricas (gender: OrdinalEncoder).
    Binarias y continuas pasan sin transformación.
    """
    categoricas_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["M", "F"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("categoricas", categoricas_transformer, FEATURES_CATEGORICAS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def construir_pipeline_lgbm(clasificador) -> Pipeline:
    """
    Pipeline para LightGBM: sin imputar continuas (NaN nativo).

    Usa set_output(transform='pandas') para preservar nombres de features,
    necesario para SHAP TreeExplainer.

    Args:
        clasificador: estimador sklearn compatible con fit/predict_proba.

    Returns:
        Pipeline listo para fit(X, y).
    """
    return Pipeline([
        ("preprocesador", construir_preprocesador_lgbm()),
        ("clasificador",  clasificador),
    ])
