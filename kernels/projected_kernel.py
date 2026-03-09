"""
Projected Quantum Kernel (PQK).

Instead of computing full state overlaps, PQK projects the quantum state
onto local 1-qubit reduced density matrices (1-RDMs) measured via Pauli
expectations ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit. A classical RBF kernel is then
applied to the resulting feature vectors.

Formula:
    k^PQ(x, x') = exp(-γ · Σ_k ||ρ_k(x) - ρ_k(x')||²)
    where ρ_k is the Bloch-vector of qubit k.

Reference:
    Huang, H.-Y. et al. (2021). Power of Data in Quantum Machine Learning.
    Nature Communications, 12, 2631.
    https://doi.org/10.1038/s41467-021-22539-9

    TensorFlow Quantum tutorial:
    https://www.tensorflow.org/quantum/tutorials/quantum_data
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator

from feature_maps.base import FeatureMap
from feature_maps.zz_map import ZZMap
from .base import QuantumKernel
from benchmark.metrics import analyze_circuit_resources


class ProjectedKernel(QuantumKernel):
    """Projected Quantum Kernel using a Gaussian kernel over local Pauli expectations."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        gamma: float = 1.0,
        shots: int = 1024,
        seed: int = 42,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.feature_map = feature_map or ZZMap(n_qubits=n_qubits, reps=2)
        self.gamma = gamma
        self._backend = AerSimulator(seed_simulator=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bloch_vector(self, x: np.ndarray) -> np.ndarray:
        """Return the 3n-dimensional Bloch vector [⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit]."""
        qc = self.feature_map.build(x)
        sv = Statevector(qc)
        bloch = []
        for qubit in range(self.n_qubits):
            # Trace out all qubits except 'qubit' to get 1-qubit reduced density matrix.
            # partial_trace takes qubits TO TRACE OUT (Qiskit 1.x module-level function).
            qubits_to_trace = [q for q in range(self.n_qubits) if q != qubit]
            rho = partial_trace(sv, qubits_to_trace)
            # Pauli expectations from 2×2 density matrix:
            # ⟨X⟩ = 2·Re(ρ[0,1]),  ⟨Y⟩ = 2·Im(ρ[1,0]),  ⟨Z⟩ = ρ[0,0] - ρ[1,1]
            data = rho.data
            ex = 2.0 * data[0, 1].real
            ey = 2.0 * data[1, 0].imag
            ez = (data[0, 0] - data[1, 1]).real
            bloch.extend([ex, ey, ez])
        return np.array(bloch, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray | None = None
    ) -> np.ndarray:
        self._reset_stats()
        symmetric = Y is None
        Y = X if Y is None else Y

        # Compute Bloch vectors for all samples
        X_bloch = np.array([self._bloch_vector(x) for x in X])
        Y_bloch = np.array([self._bloch_vector(y) for y in Y])

        self.stats.n_evaluations = len(X) + (0 if symmetric else len(Y))
        # No shots used — statevector simulation (exact)
        self.stats.total_shots = 0

        # Track detailed circuit resources
        example_qc = self.feature_map.build(X[0])
        res = analyze_circuit_resources(example_qc)
        self.stats.total_depth = res["total_depth"]
        self.stats.two_qubit_depth = res["two_qubit_depth"]
        self.stats.total_gates = res["total_gates"]
        self.stats.two_qubit_count = res["two_qubit_count"]
        self.stats.one_qubit_count = res["one_qubit_count"]
        self.stats.gate_breakdown = res["gate_breakdown"]

        # RBF kernel over Bloch vectors
        n, m = len(X), len(Y)
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                diff = X_bloch[i] - Y_bloch[j]
                K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))

        return K
