"""
tests/test_explicabilidad.py
──────────────────────────────
Tests para los tipos de retorno de explicabilidad.py.
"""

import numpy as np
from triaje_ia.ml.explicabilidad import (
    ExplicacionSHAP,
    FeatureContribucion,
)


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
