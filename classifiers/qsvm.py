"""
Checked by Bank
2026-03-08
"""

"""
Quantum SVM (QSVM).

Wraps scikit-learn's SVC with kernel='precomputed'. The kernel matrix
is computed by the quantum kernel before fitting/prediction.
"""

import numpy as np
from sklearn.svm import SVC


class QSVM:
    """Quantum SVM: SVC with a precomputed quantum kernel matrix."""

    def __init__(self, C: float = 1.0, **svc_kwargs):
        self.C = C
        self._clf = SVC(kernel="precomputed", C=C, probability=True, **svc_kwargs)
        self._X_train: np.ndarray | None = None

    def fit(self, K_train: np.ndarray, y: np.ndarray) -> "QSVM":
        """Fit the SVM given a precomputed training kernel matrix."""
        self._clf.fit(K_train, y)
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """Predict labels given the test kernel matrix K[test × train]."""
        return self._clf.predict(K_test)

    def predict_proba(self, K_test: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        return self._clf.predict_proba(K_test)

    def score(self, K_test: np.ndarray, y: np.ndarray) -> float:
        return self._clf.score(K_test, y)

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    from datasets.loader import load_dataset
    from kernels.fidelity_kernel import FidelityKernel

    N_QUBITS = 2

    X_train, X_test, y_train, y_test = load_dataset(
        name="breast_cancer",
        n_samples=100,
        n_features=N_QUBITS,
        test_size=0.25,
        random_state=42,
    )

    # PCA to reduce to n_qubits if needed (same as runner)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    if X_train.shape[1] > N_QUBITS:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        pca = PCA(n_components=N_QUBITS, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
    scaler_pi = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler_pi.fit_transform(X_train)
    X_test  = scaler_pi.transform(X_test)

    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=1024, seed=42)

    K_train = kernel.build_kernel_matrix(X_train)
    K_test = kernel.build_kernel_matrix(X_test, X_train)

    print("K_train shape:", K_train.shape)
    print("K_test shape :", K_test.shape)

    # Basic sanity checks
    print("K_train symmetric:", np.allclose(K_train, K_train.T, atol=1e-8))
    print("K_train diagonal :", np.diag(K_train))

    model = QSVM(C=1.0)
    model.fit(K_train, y_train)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("y_pred:", y_pred)
    print("y_test:", y_test)
    print(f"accuracy:  {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall:    {rec:.3f}")
    print(f"f1 score:  {f1:.3f}")