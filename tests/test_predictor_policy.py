import numpy as np
import pandas as pd

from triaje_ia.inference.predictor import TriajePredictor


class ModeloFake:
    def __init__(self, probas: np.ndarray) -> None:
        self._probas = probas

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probas


def crear_predictor_fake(probas: np.ndarray, thresholds: list[float]) -> TriajePredictor:
    predictor = TriajePredictor.__new__(TriajePredictor)
    predictor._clf = ModeloFake(probas.reshape(1, -1))
    predictor._thresholds = thresholds
    predictor._assumptions = {}
    predictor._feature_names = ["f1"]
    predictor._build_features = lambda vector, narrativa: pd.DataFrame({"f1": [0.0]})
    return predictor


def test_predictor_aplica_threshold_a1_antes_de_ponderar_thresholds():
    probas = np.array([0.21, 0.40, 0.30, 0.06, 0.03])
    thresholds = [1.0, 3.0, 3.0, 1.0, 1.0]

    result = crear_predictor_fake(probas, thresholds).predict(vector=None, narrativa="")

    assert result.clase_predicha == 1
    assert result.threshold_a1_activado is True


def test_predictor_no_fuerza_a1_si_esta_por_debajo_del_umbral():
    probas = np.array([0.19, 0.40, 0.30, 0.06, 0.05])
    thresholds = [1.0, 3.0, 3.0, 1.0, 1.0]

    result = crear_predictor_fake(probas, thresholds).predict(vector=None, narrativa="")

    assert result.clase_predicha == 2
    assert result.threshold_a1_activado is False
