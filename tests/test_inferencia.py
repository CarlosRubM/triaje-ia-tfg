"""
tests/test_inferencia.py
─────────────────────────
Tests para inferencia.py — carga de modelo y predicción.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from triaje_ia.data.features import TODAS_FEATURES
from triaje_ia.ml.inferencia import predecir_proba


@pytest.fixture
def df_dummy():
    """DataFrame de 3 filas con las 88 features."""
    np.random.seed(42)
    n = 3
    data = {}
    for f in TODAS_FEATURES:
        if f == "gender":
            data[f] = np.random.choice(["M", "F"], size=n)
        else:
            data[f] = np.random.rand(n)
    return pd.DataFrame(data)


@pytest.fixture
def mock_modelo():
    """Modelo mock que devuelve predict_proba con shape (n, 5)."""
    modelo = MagicMock()
    modelo.predict_proba = MagicMock(
        side_effect=lambda X: np.random.dirichlet([1, 1, 1, 1, 1], size=len(X))
    )
    return modelo


class TestPredecirProba:
    def test_shape_output(self, mock_modelo, df_dummy):
        """predict_proba devuelve shape (n, 5)."""
        probas = predecir_proba(mock_modelo, df_dummy)
        assert probas.shape == (3, 5)

    def test_probas_suman_uno(self, mock_modelo, df_dummy):
        """Las probabilidades por fila suman ~1."""
        probas = predecir_proba(mock_modelo, df_dummy)
        sumas = probas.sum(axis=1)
        np.testing.assert_allclose(sumas, 1.0, atol=1e-6)

    def test_no_modifica_df_original(self, mock_modelo, df_dummy):
        """predecir_proba no muta el DataFrame de entrada."""
        df_copia = df_dummy.copy()
        predecir_proba(mock_modelo, df_dummy)
        pd.testing.assert_frame_equal(df_dummy, df_copia)

    def test_gender_no_se_encodea_manualmente(self, mock_modelo, df_dummy):
        """Gender se pasa tal cual (M/F) — el pipeline del modelo lo encodea."""
        predecir_proba(mock_modelo, df_dummy)
        # Verificar que predict_proba recibió gender como string, no int
        X_recibido = mock_modelo.predict_proba.call_args[0][0]
        assert X_recibido["gender"].dtype == object

    def test_selecciona_solo_todas_features(self, mock_modelo, df_dummy):
        """Solo pasa las columnas de TODAS_FEATURES al modelo."""
        df_extra = df_dummy.copy()
        df_extra["columna_extra"] = 999
        predecir_proba(mock_modelo, df_extra)
        X_recibido = mock_modelo.predict_proba.call_args[0][0]
        assert "columna_extra" not in X_recibido.columns
        assert list(X_recibido.columns) == TODAS_FEATURES
