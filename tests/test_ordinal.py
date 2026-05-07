"""Tests para el clasificador ordinal Frank & Hall."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from triaje_ia.ml.ordinal import OrdinalFrankHall


@pytest.fixture
def datos_sinteticos():
    """Dataset sintético con 5 clases ordinales."""
    rng = np.random.RandomState(42)
    n = 500
    X = rng.randn(n, 5)
    # Target correlacionado con primera feature
    y = np.digitize(X[:, 0], bins=[-1.5, -0.5, 0.5, 1.5]) + 1  # 1-5
    return X, y


def test_fit_crea_k_menos_1_clasificadores(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    assert len(clf.classifiers_) == 4  # 5 clases → 4 binarios


def test_predict_proba_shape(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert probas.shape == (len(X), 5)


def test_predict_proba_suman_uno(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_no_negativas(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert (probas >= 0).all()


def test_predict_devuelve_clases_validas(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({1, 2, 3, 4, 5})


def test_classes_attribute(datos_sinteticos):
    X, y = datos_sinteticos
    clf = OrdinalFrankHall(DecisionTreeClassifier(max_depth=3))
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, [1, 2, 3, 4, 5])
