import numpy as np
import pandas as pd

from triaje_ia.inference.predictor import TriajePredictor


class ModeloFake:
    def __init__(self, probas: np.ndarray) -> None:
        self._probas = probas

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self._probas


def crear_predictor_fake(
    probas: np.ndarray, warning_threshold_a1: float = 0.40
) -> TriajePredictor:
    predictor = TriajePredictor.__new__(TriajePredictor)
    predictor._clf = ModeloFake(probas.reshape(1, -1))
    predictor._decision_policy = "argmax_with_a1_warning"
    predictor._warning_threshold_a1 = warning_threshold_a1
    predictor._class_weights = None
    predictor._assumptions = {}
    predictor._feature_names = ["f1"]
    predictor._build_features = lambda vector, narrativa: pd.DataFrame({"f1": [0.0]})
    return predictor


def test_predictor_mantiene_argmax_y_activa_alerta_a1():
    probas = np.array([0.41, 0.44, 0.10, 0.03, 0.02])

    result = crear_predictor_fake(probas).predict(vector=None, narrativa="")

    assert result.clase_predicha == 2
    assert result.alerta_a1_activada is True


def test_predictor_no_activa_alerta_a1_si_esta_por_debajo_del_umbral():
    probas = np.array([0.39, 0.44, 0.10, 0.04, 0.03])

    result = crear_predictor_fake(probas).predict(vector=None, narrativa="")

    assert result.clase_predicha == 2
    assert result.alerta_a1_activada is False


def test_predictor_no_muestra_alerta_a1_si_argmax_ya_es_a1():
    probas = np.array([0.55, 0.30, 0.10, 0.03, 0.02])

    result = crear_predictor_fake(probas).predict(vector=None, narrativa="")

    assert result.clase_predicha == 1
    assert result.alerta_a1_activada is False
