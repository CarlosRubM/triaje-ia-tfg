"""
tests/test_pipeline.py
───────────────────────
Tests para pipeline.py — verifica que construir_pipeline_lgbm no imputa NaN.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from triaje_ia.data.features import (
    FEATURES_CATEGORICAS,
    FEATURES_CONTINUAS,
    FEATURES_BINARIAS,
    TODAS_FEATURES,
)
from triaje_ia.ml.pipeline import (
    construir_pipeline,
    construir_pipeline_lgbm,
    construir_preprocesador,
    construir_preprocesador_lgbm,
)


@pytest.fixture
def df_dummy():
    """DataFrame de 5 filas con NaN en features continuas."""
    np.random.seed(42)
    n = 5
    data = {}
    for f in FEATURES_CONTINUAS:
        vals = np.random.rand(n).astype(float)
        vals[0] = np.nan  # al menos un NaN por feature
        data[f] = vals
    for f in FEATURES_BINARIAS:
        data[f] = np.random.randint(0, 2, size=n).astype("int8")
    for f in FEATURES_CATEGORICAS:
        data[f] = np.random.choice(["M", "F"], size=n)
    return pd.DataFrame(data)[TODAS_FEATURES]


class TestPreprocesadorLGBM:
    """construir_preprocesador_lgbm() no debe imputar continuas."""

    def test_preserva_nan_en_continuas(self, df_dummy):
        prepro = construir_preprocesador_lgbm()
        result = prepro.fit_transform(df_dummy)

        # result es un DataFrame (set_output=pandas)
        assert isinstance(result, pd.DataFrame)

        # Verificar que los NaN originales siguen ahí
        for col in FEATURES_CONTINUAS:
            if col in result.columns:
                assert result[col].isna().sum() > 0, (
                    f"NaN en {col} fue imputado — LGBM necesita NaN nativos"
                )

    def test_encodea_gender(self, df_dummy):
        prepro = construir_preprocesador_lgbm()
        result = prepro.fit_transform(df_dummy)
        # Gender debe ser numérico (0 o 1) después del encoding
        assert result["gender"].dtype in (np.float64, np.int64, np.int8, np.float32)


class TestPreprocesadorOriginal:
    """construir_preprocesador() SÍ imputa continuas (para modelos lineales)."""

    def test_imputa_nan_en_continuas(self, df_dummy):
        prepro = construir_preprocesador()
        result = prepro.fit_transform(df_dummy)
        # El resultado es un ndarray, no DataFrame
        # No debería tener NaN en las primeras len(FEATURES_CONTINUAS) columnas
        n_continuas = len(FEATURES_CONTINUAS)
        assert not np.isnan(result[:, :n_continuas]).any(), (
            "El preprocesador original debería imputar NaN en continuas"
        )


class TestPipelineLGBM:
    """construir_pipeline_lgbm() produce un pipeline funcional."""

    def test_fit_predict(self, df_dummy):
        y = np.array([1, 2, 3, 4, 5])
        clf = DecisionTreeClassifier(random_state=42)
        pipe = construir_pipeline_lgbm(clf)
        pipe.fit(df_dummy, y)
        preds = pipe.predict(df_dummy)
        assert len(preds) == len(df_dummy)

    def test_pipeline_tiene_dos_steps(self, df_dummy):
        clf = DecisionTreeClassifier()
        pipe = construir_pipeline_lgbm(clf)
        assert "preprocesador" in pipe.named_steps
        assert "clasificador" in pipe.named_steps
