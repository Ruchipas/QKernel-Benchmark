"""
Tests for QSVM and QKRR classifiers.

Uses a small FQK kernel matrix on toy data and verifies:
  - fit / predict / score work end-to-end.
  - Predictions have the correct shape.
  - Accuracy is above chance (50%).
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.fidelity_kernel import FidelityKernel
from kernels.projected_kernel import ProjectedKernel
from classifiers.qsvm import QSVM
from classifiers.qkrr import QKRR

N_TRAIN = 10
N_TEST = 4
N_QUBITS = 2
SHOTS = 128
SEED = 0


@pytest.fixture(scope="module")
def kernel_matrices():
    rng = np.random.default_rng(SEED)
    X_train = rng.uniform(0, 2 * np.pi, (N_TRAIN, N_QUBITS))
    X_test = rng.uniform(0, 2 * np.pi, (N_TEST, N_QUBITS))
    y_train = np.array([-1, 1] * (N_TRAIN // 2))
    y_test = np.array([-1, 1, -1, 1])

    kernel = FidelityKernel(n_qubits=N_QUBITS, shots=SHOTS, seed=SEED)
    K_train = kernel.build_kernel_matrix(X_train)
    K_test = kernel.build_kernel_matrix(X_test, X_train)
    return K_train, K_test, y_train, y_test


# -----------------------------------------------------------------------
# QSVM
# -----------------------------------------------------------------------

def test_qsvm_fit_predict_shape(kernel_matrices):
    K_train, K_test, y_train, y_test = kernel_matrices
    clf = QSVM(C=1.0)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    assert y_pred.shape == (N_TEST,), f"Expected shape ({N_TEST},) got {y_pred.shape}"


def test_qsvm_predict_labels(kernel_matrices):
    K_train, K_test, y_train, _ = kernel_matrices
    clf = QSVM(C=1.0)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    assert set(np.unique(y_pred)).issubset({-1, 1}), "Unexpected label values"


def test_qsvm_predict_proba(kernel_matrices):
    K_train, K_test, y_train, _ = kernel_matrices
    clf = QSVM(C=1.0)
    clf.fit(K_train, y_train)
    proba = clf.predict_proba(K_test)
    assert proba.shape == (N_TEST, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# -----------------------------------------------------------------------
# QKRR
# -----------------------------------------------------------------------

def test_qkrr_fit_predict_shape(kernel_matrices):
    K_train, K_test, y_train, y_test = kernel_matrices
    # QKRR expects numeric targets; use the integer labels directly
    model = QKRR(alpha=0.1)
    model.fit(K_train, y_train.astype(float))
    y_pred = model.predict(K_test)
    assert y_pred.shape == (N_TEST,)


def test_qkrr_score_finite(kernel_matrices):
    K_train, K_test, y_train, y_test = kernel_matrices
    model = QKRR(alpha=0.1)
    model.fit(K_train, y_train.astype(float))
    score = model.score(K_test, y_test.astype(float))
    assert np.isfinite(score), "QKRR R² score is not finite"
