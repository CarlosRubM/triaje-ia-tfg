"""
tests/test_explicabilidad.py
──────────────────────────────
Tests para explicabilidad.py — SHAP con modelos estándar y OrdinalFrankHall.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from triaje_ia.ml.explicabilidad import (
    _es_ordinal_frank_hall,
    _extraer_clasificador_binario,
    ExplicacionSHAP,
    FeatureContribucion,
)


@pytest.fixture
def mock_ordinal_model():
    """Mock de OrdinalFrankHall con 4 clasificadores binarios."""
    modelo = MagicMock()
    modelo.n_classes_ = 5
    modelo.classes_ = np.array([1, 2, 3, 4, 5])

    # Cada clasificador interno es un Pipeline mock
    clfs = []
    for i in range(4):
        pipeline_mock = MagicMock()
        pipeline_mock.__class__.__name__ = "Pipeline"

        clf_interno = MagicMock()
        clf_interno.__class__.__name__ = "LGBMClassifier"

        # Para isinstance check en sklearn Pipeline
        from sklearn.pipeline import Pipeline as SkPipeline
        pipeline_mock.__class__ = SkPipeline
        pipeline_mock.named_steps = {"preprocesador": MagicMock(), "clasificador": clf_interno}

        clfs.append(pipeline_mock)

    modelo.classifiers_ = clfs
    return modelo


class TestDeteccionOrdinal:
    def test_detecta_ordinal(self, mock_ordinal_model):
        assert _es_ordinal_frank_hall(mock_ordinal_model)

    def test_no_detecta_modelo_normal(self):
        modelo = MagicMock(spec=["predict", "predict_proba"])
        assert not _es_ordinal_frank_hall(modelo)


class TestExtraccionClasificadorBinario:
    def test_extrae_clf_correcto_clase1(self, mock_ordinal_model):
        clf, idx = _extraer_clasificador_binario(mock_ordinal_model, clase=1)
        assert idx == 0
        # Debe ser el clasificador interno, no el pipeline
        assert clf.__class__.__name__ == "LGBMClassifier"

    def test_extrae_clf_correcto_clase3(self, mock_ordinal_model):
        clf, idx = _extraer_clasificador_binario(mock_ordinal_model, clase=3)
        assert idx == 2

    def test_clase5_usa_ultimo_clf(self, mock_ordinal_model):
        clf, idx = _extraer_clasificador_binario(mock_ordinal_model, clase=5)
        assert idx == 3  # max index = n_classes - 2


class TestExplicacionSHAP:
    def test_to_dict(self):
        resultado = ExplicacionSHAP(
            clase_explicada=2,
            base_value=0.3,
            shap_values=np.array([0.1, -0.2, 0.05]),
            feature_names=["f1", "f2", "f3"],
            feature_values=np.array([1.0, 2.0, 3.0]),
            top_positivas=[
                FeatureContribucion("f1", 1.0, 0.1),
            ],
            top_negativas=[
                FeatureContribucion("f2", 2.0, -0.2),
            ],
        )
        d = resultado.to_dict()
        assert d["clase_explicada"] == 2
        assert len(d["top_positivas"]) == 1
        assert len(d["top_negativas"]) == 1
        assert d["top_positivas"][0]["feature"] == "f1"
        assert d["top_negativas"][0]["shap"] < 0


class TestFeatureContribucion:
    def test_impacto_abs(self):
        fc = FeatureContribucion("test", 1.0, -0.5)
        assert fc.impacto_abs == 0.5

    def test_impacto_abs_positivo(self):
        fc = FeatureContribucion("test", 1.0, 0.3)
        assert fc.impacto_abs == 0.3
