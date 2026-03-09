"""
Trainable Quantum Kernel via Kernel-Target Alignment (QKTA).

Uses a parameterized feature map U(x; θ) where θ is optimized to
maximize the Kernel-Target Alignment (KTA) with the training labels:

    A(K_θ, YYᵀ) = Tr(K_θ · YYᵀ) / sqrt(Tr(K_θ²) · Tr((YYᵀ)²))

θ is optimized via COBYLA (gradient-free), then the kernel is evaluated
as a standard FQK using the learned parameters.

Reference:
    Hubregtsen, T. et al. (2022). Training Quantum Embedding Kernels on
    Near-Term Quantum Computers. Physical Review A, 106, 042431.
    https://arxiv.org/abs/2105.02276

    Glick, J. R. et al. (2024). Covariant Quantum Kernels for Data with
    Group Structure. Nature Physics.
    https://arxiv.org/abs/2105.02426

    Qiskit QuantumKernelTrainer:
    https://qiskit-community.github.io/qiskit-machine-learning/tutorials/08_quantum_kernel_trainer.html
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from .base import QuantumKernel
from benchmark.metrics import analyze_circuit_resources


def _build_trainable_circuit(
    x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int = 1
) -> QuantumCircuit:
    """Build a parameterized feature map circuit with data x and parameters theta."""
    qc = QuantumCircuit(n_qubits)
    n_params_per_rep = n_qubits
    for r in range(reps):
        for i in range(n_qubits):
            angle = x[i % len(x)] + theta[r * n_params_per_rep + i % n_params_per_rep]
            qc.ry(angle, i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


class TrainableKernel(QuantumKernel):
    """Trainable quantum kernel: optimizes feature-map parameters via KTA."""

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        shots: int = 1024,
        seed: int = 42,
        max_iter: int = 50,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.reps = reps
        self.max_iter = max_iter
        self._n_params = n_qubits * reps
        self._theta = np.zeros(self._n_params)  # initial parameters
        self._backend = AerSimulator(seed_simulator=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # _overlap removed since we batch circuits in _build_K / build_kernel_matrix

    def _build_K(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Build the full symmetric kernel matrix for training set X."""
        n = len(X)
        K = np.zeros((n, n))

        circuits = []
        indices = []
        for i in range(n):
            for j in range(i, n):
                phi_x = _build_trainable_circuit(X[i], theta, self.n_qubits, self.reps)
                phi_xp = _build_trainable_circuit(X[j], theta, self.n_qubits, self.reps)
                qc = QuantumCircuit(self.n_qubits, self.n_qubits)
                qc.compose(phi_x, inplace=True)
                qc.compose(phi_xp.inverse(), inplace=True)
                qc.measure(range(self.n_qubits), range(self.n_qubits))
                circuits.append(qc)
                indices.append((i, j))

        if len(circuits) > 0:
            t_circuits = transpile(circuits, self._backend, optimization_level=0)
            job = self._backend.run(t_circuits, shots=self.shots)
            counts_list = job.result().get_counts()
            if not isinstance(counts_list, list):
                counts_list = [counts_list]

            zero_key = "0" * self.n_qubits
            for count, (i, j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[i, j] = val
                K[j, i] = val

        return K

    @staticmethod
    def _kta(K: np.ndarray, y: np.ndarray) -> float:
        """Kernel-Target Alignment score (higher = better)."""
        T = np.outer(y, y).astype(float)
        num = np.trace(K @ T)
        denom = np.sqrt(np.trace(K @ K) * np.trace(T @ T))
        return num / (denom + 1e-10)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableKernel":
        """Optimize θ to maximize KTA on the training set."""
        rng = np.random.default_rng(self.seed)
        theta0 = rng.uniform(-np.pi, np.pi, self._n_params)
        n_train = len(X)
        n_pairs = n_train * (n_train + 1) // 2

        with tqdm(
            total=self.max_iter,
            desc="  QKTA fit",
            unit="iter",
            ncols=88,
            leave=False,
        ) as pbar:
            def neg_kta(theta: np.ndarray) -> float:
                K = self._build_K(X, theta)
                kta = self._kta(K, y)
                pbar.update(1)
                pbar.set_postfix(kta=f"{kta:.4f}", refresh=True)
                return -kta

            result = minimize(
                neg_kta,
                theta0,
                method="COBYLA",
                options={"maxiter": self.max_iter, "rhobeg": 0.5},
            )
        self._theta = result.x
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        self._reset_stats()
        symmetric = Y is None
        Y = X if Y is None else Y
        n, m = len(X), len(Y)
        K = np.zeros((n, m))

        circuits = []
        indices = []
        for i in range(n):
            start_j = i if symmetric else 0
            for j in range(start_j, m):
                phi_x = _build_trainable_circuit(X[i], self._theta, self.n_qubits, self.reps)
                phi_xp = _build_trainable_circuit(Y[j], self._theta, self.n_qubits, self.reps)
                qc = QuantumCircuit(self.n_qubits, self.n_qubits)
                qc.compose(phi_x, inplace=True)
                qc.compose(phi_xp.inverse(), inplace=True)
                qc.measure(range(self.n_qubits), range(self.n_qubits))
                circuits.append(qc)
                indices.append((i, j))

        if len(circuits) > 0:
            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)
            
            example_qc = _build_trainable_circuit(X[0], self._theta, self.n_qubits, self.reps)
            res = analyze_circuit_resources(example_qc)
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

            t_circuits = transpile(circuits, self._backend, optimization_level=0)
            job = self._backend.run(t_circuits, shots=self.shots)
            counts_list = job.result().get_counts()
            
            if not isinstance(counts_list, list):
                counts_list = [counts_list]
                
            zero_key = "0" * self.n_qubits
            for count, (i, j) in zip(counts_list, indices):
                val = count.get(zero_key, 0) / self.shots
                K[i, j] = val
                if symmetric and i != j:
                    K[j, i] = val

        return K
