"""
Q-FLAIR: Feature-Map Architecture Learning via Kernel-Target Alignment.

Q-FLAIR greedily grows a quantum circuit architecture by iteratively
selecting the gate (from a candidate pool) that maximizes the improvement
in Kernel-Target Alignment (KTA). The final circuit is evaluated as a
standard Fidelity Quantum Kernel (FQK).

Algorithm (simplified from paper):
    1. Start with an empty circuit.
    2. For each candidate gate g in {Rx, Ry, Rz} × {each qubit}:
       - Build a trial circuit with g appended.
       - Evaluate KTA of the resulting FQK on training data.
    3. Keep the gate that gives the highest KTA gain.
    4. Repeat for n_layers steps.
    5. Evaluate final architecture as FQK for inference.

Reference:
    Barbosa, A. et al. (2023). Q-FLAIR: Fast Learning of Architectures
    for Quantum Kernel-based Models.
    https://arxiv.org/abs/2311.04965

    Shaydulin, R. & Wild, S. M. (2022). Importance of Kernel Bandwidth in
    Quantum Machine Learning. Physical Review A.
    https://arxiv.org/abs/2206.04754
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .base import QuantumKernel
from benchmark.metrics import analyze_circuit_resources


# -----------------------------------------------------------------------
# Gate factory helpers
# -----------------------------------------------------------------------

def _gate_name(gate_type: str, qubit: int) -> str:
    return f"{gate_type}(x[{qubit}])@q{qubit}"


def _apply_gate(qc: QuantumCircuit, gate_type: str, qubit: int, angle: float) -> None:
    if gate_type == "Rx":
        qc.rx(angle, qubit)
    elif gate_type == "Ry":
        qc.ry(angle, qubit)
    elif gate_type == "Rz":
        qc.rz(angle, qubit)


# -----------------------------------------------------------------------
# Q-FLAIR Kernel
# -----------------------------------------------------------------------

class QFLAIRKernel(QuantumKernel):
    """Q-FLAIR: greedy circuit architecture search via KTA maximization."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 4,
        shots: int = 1024,
        seed: int = 42,
    ):
        super().__init__(n_qubits=n_qubits, shots=shots, seed=seed)
        self.n_layers = n_layers
        self._gate_sequence: list[tuple[str, int]] = []  # learned architecture
        self._backend = AerSimulator(seed_simulator=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_circuit(
        self, x: np.ndarray, gate_sequence: list[tuple[str, int]]
    ) -> QuantumCircuit:
        """Build a data-encoding circuit from a gate sequence."""
        qc = QuantumCircuit(self.n_qubits)
        for gate_type, qubit in gate_sequence:
            angle = x[qubit % len(x)]
            _apply_gate(qc, gate_type, qubit, angle)
        return qc

    # _fqk_overlap removed since we batch circuits in _build_K / build_kernel_matrix

    def _build_K(
        self, X: np.ndarray, gate_sequence: list[tuple[str, int]]
    ) -> np.ndarray:
        n = len(X)
        K = np.zeros((n, n))

        circuits = []
        indices = []
        for i in range(n):
            for j in range(i, n):
                phi_x = self._build_circuit(X[i], gate_sequence)
                phi_xp = self._build_circuit(X[j], gate_sequence)
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
        T = np.outer(y, y).astype(float)
        num = np.trace(K @ T)
        denom = np.sqrt(np.trace(K @ K) * np.trace(T @ T))
        return num / (denom + 1e-10)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QFLAIRKernel":
        """Greedy gate selection to maximise KTA."""
        gate_types = ["Rx", "Ry", "Rz"]
        candidate_gates = [
            (gt, q) for gt in gate_types for q in range(self.n_qubits)
        ]

        current_sequence: list[tuple[str, int]] = []
        current_kta = -np.inf

        for layer in range(self.n_layers):
            best_gate = None
            best_kta = current_kta

            for gate in candidate_gates:
                trial_seq = current_sequence + [gate]
                K_trial = self._build_K(X, trial_seq)
                kta_val = self._kta(K_trial, y)
                if kta_val > best_kta:
                    best_kta = kta_val
                    best_gate = gate

            if best_gate is None:
                break  # no improvement found

            current_sequence.append(best_gate)
            current_kta = best_kta

        self._gate_sequence = current_sequence
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
                phi_x = self._build_circuit(X[i], self._gate_sequence)
                phi_xp = self._build_circuit(Y[j], self._gate_sequence)
                qc = QuantumCircuit(self.n_qubits, self.n_qubits)
                qc.compose(phi_x, inplace=True)
                qc.compose(phi_xp.inverse(), inplace=True)
                qc.measure(range(self.n_qubits), range(self.n_qubits))
                circuits.append(qc)
                indices.append((i, j))

        if len(circuits) > 0:
            self.stats.total_shots += self.shots * len(circuits)
            self.stats.n_evaluations += len(circuits)
            
            example_qc = self._build_circuit(X[0], self._gate_sequence)
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
