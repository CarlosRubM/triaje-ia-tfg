import numpy as np
import pytest
from triaje_ia.ml.decision import argmax, threshold_a1, UMBRAL_A1_DEFAULT


def test_argmax_clase_mas_probable():
    probas = np.array([0.05, 0.10, 0.60, 0.20, 0.05])
    assert argmax(probas) == 3


def test_argmax_acuity1():
    probas = np.array([0.70, 0.10, 0.10, 0.05, 0.05])
    assert argmax(probas) == 1


def test_threshold_a1_activa_con_prob_alta():
    probas = np.array([0.25, 0.30, 0.25, 0.10, 0.10])
    assert threshold_a1(probas) == 1


def test_threshold_a1_no_activa_con_prob_baja():
    probas = np.array([0.10, 0.15, 0.60, 0.10, 0.05])
    assert threshold_a1(probas) == 3


def test_threshold_a1_exactamente_en_umbral():
    probas = np.array([UMBRAL_A1_DEFAULT, 0.20, 0.30, 0.20, 0.10])
    assert threshold_a1(probas) == 1


def test_threshold_a1_justo_por_debajo_umbral():
    probas = np.array([UMBRAL_A1_DEFAULT - 0.01, 0.20, 0.40, 0.20, 0.11])
    assert threshold_a1(probas) == 3


def test_threshold_a1_umbral_personalizado():
    probas = np.array([0.15, 0.20, 0.45, 0.15, 0.05])
    assert threshold_a1(probas, umbral=0.10) == 1
    assert threshold_a1(probas, umbral=0.20) == 3
