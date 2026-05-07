"""
src/triaje_ia/ml/ordinal.py
─────────────────────────────
Clasificador ordinal Frank & Hall para acuity 1-5.

Descompone el problema ordinal en K-1 clasificadores binarios:
  - h1: P(acuity ≤ 1) vs P(acuity > 1)
  - h2: P(acuity ≤ 2) vs P(acuity > 2)
  - h3: P(acuity ≤ 3) vs P(acuity > 3)
  - h4: P(acuity ≤ 4) vs P(acuity > 4)

La predicción final reconstruye P(acuity=k) a partir de las
probabilidades acumulativas:
  P(Y=1) = P(Y≤1)
  P(Y=k) = P(Y≤k) - P(Y≤k-1)  para k=2..K-1
  P(Y=K) = 1 - P(Y≤K-1)

Referencia:
  Frank, E. & Hall, M. (2001). "A Simple Approach to Ordinal
  Classification". ECML.

Compatible con cualquier clasificador sklearn con predict_proba().
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from loguru import logger


class OrdinalFrankHall(BaseEstimator, ClassifierMixin):
    """
    Clasificador ordinal via descomposición Frank & Hall.

    Args:
        base_estimator: clasificador sklearn con predict_proba().
            Se clona K-1 veces (una por corte ordinal).

    Ejemplo:
        from lightgbm import LGBMClassifier
        clf = OrdinalFrankHall(LGBMClassifier(n_estimators=500))
        clf.fit(X_train, y_train)  # y con valores 1,2,3,4,5
        probas = clf.predict_proba(X_test)  # shape (n, 5)
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, **fit_params):
        """
        Entrena K-1 clasificadores binarios.

        Args:
            X: features, shape (n_samples, n_features).
            y: target ordinal con valores enteros (ej: 1,2,3,4,5).
        """
        self.classes_ = np.sort(np.unique(y))
        self.n_classes_ = len(self.classes_)
        self.classifiers_ = []

        for k in range(self.n_classes_ - 1):
            threshold = self.classes_[k]
            y_bin = (y > threshold).astype(int)

            clf = clone(self.base_estimator)
            clf.fit(X, y_bin, **fit_params)
            self.classifiers_.append(clf)

            n_pos = y_bin.sum()
            n_neg = len(y_bin) - n_pos
            logger.info(
                f"  h{k+1}: Y≤{threshold} vs Y>{threshold} "
                f"(neg={n_neg:,} pos={n_pos:,})"
            )

        logger.success(
            f"OrdinalFrankHall: {len(self.classifiers_)} "
            "clasificadores binarios entrenados"
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Calcula P(Y=k) para cada clase k a partir de las
        probabilidades acumulativas.

        Returns:
            np.ndarray shape (n_samples, n_classes).
        """
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        K = self.n_classes_

        # P(Y > threshold_k) para cada corte
        cum_probs = np.zeros((n, K - 1))
        for i, clf in enumerate(self.classifiers_):
            cum_probs[:, i] = clf.predict_proba(X)[:, 1]

        # P(Y ≤ k) = 1 - P(Y > k)
        cum_le = 1.0 - cum_probs  # shape (n, K-1)

        # Reconstruir P(Y=k)
        probas = np.zeros((n, K))
        probas[:, 0] = cum_le[:, 0]
        for k in range(1, K - 1):
            probas[:, k] = cum_le[:, k] - cum_le[:, k - 1]
        probas[:, K - 1] = 1.0 - cum_le[:, K - 2]

        # Clip y renormalizar (las restas pueden dar negativos)
        probas = np.clip(probas, 0, 1)
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probas = probas / row_sums

        return probas

    def predict(self, X) -> np.ndarray:
        """Devuelve la clase con mayor probabilidad."""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
