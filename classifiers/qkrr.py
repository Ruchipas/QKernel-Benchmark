"""
Quantum Kernel Ridge Regression (QKRR).

Wraps scikit-learn's KernelRidge with kernel='precomputed'.
Useful for regression tasks on top of quantum kernel matrices.
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge


class QKRR:
    """Quantum Kernel Ridge Regression with a precomputed kernel matrix."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = KernelRidge(kernel="precomputed", alpha=alpha)

    def fit(self, K_train: np.ndarray, y: np.ndarray) -> "QKRR":
        self._model.fit(K_train, y)
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        return self._model.predict(K_test)

    def score(self, K_test: np.ndarray, y: np.ndarray) -> float:
        return self._model.score(K_test, y)
