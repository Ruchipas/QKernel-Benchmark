"""
Fidelity Quantum Kernel (FQK).

Computes k(x, x') = |<ϕ(x)|ϕ(x')>|² using the adjoint (overlap) circuit:
    U†(x') · U(x)|0⟩  →  measure probability of all-zeros outcome.

Reference:
    Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567, 209–212.
    https://doi.org/10.1038/s41586-019-0980-2

    Qiskit tutorial:
    https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html

    GitHub (Qiskit Machine Learning):
    https://github.com/qiskit-community/qiskit-machine-learning
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from feature_maps.base import FeatureMap
from feature_maps.zz_map import ZZMap
from .base import QuantumKernel
from benchmark.metrics import analyze_circuit_resources


class FidelityKernel(QuantumKernel):
    """Fidelity Quantum Kernel via adjoint circuit overlap test."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        shots: int = 1024,
        seed: int = 42,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self._backend = AerSimulator(seed_simulator=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _overlap_circuit(self, x: np.ndarray, x_prime: np.ndarray) -> QuantumCircuit:
        """Build the adjoint overlap circuit U†(x') · U(x)."""
        n = self.n_qubits
        phi_x = self.feature_map.build(x)
        phi_xp = self.feature_map.build(x_prime)
        qc = QuantumCircuit(n, n)
        qc.compose(phi_x, inplace=True)
        qc.compose(phi_xp.inverse(), inplace=True)
        qc.measure(range(n), range(n))
        return qc

    # removed _run_circuit as it is batched inside build_kernel_matrix

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
                qc = self._overlap_circuit(X[i], Y[j])
                circuits.append(qc)
                indices.append((i, j))

        if len(circuits) > 0:
            example_qc = self.feature_map.build(X[0])
            res = analyze_circuit_resources(example_qc)
            self.stats.total_depth = res["total_depth"]
            self.stats.two_qubit_depth = res["two_qubit_depth"]
            self.stats.total_gates = res["total_gates"]
            self.stats.two_qubit_count = res["two_qubit_count"]
            self.stats.one_qubit_count = res["one_qubit_count"]
            self.stats.gate_breakdown = res["gate_breakdown"]

            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)

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
