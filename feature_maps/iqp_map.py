"""
Checked by Bank
2026-03-07
"""

"""
IQP (Instantaneous Quantum Polynomial) feature map.

Encodes a classical vector x into a quantum state using Hadamard gates
followed by diagonal phase gates encoding x_i and pairs x_i * x_j.

Reference:
    Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567, 209–212.
    https://doi.org/10.1038/s41586-019-0980-2
"""

import numpy as np
from qiskit import QuantumCircuit

import sys
from pathlib import Path

# Allow direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .base import FeatureMap
except ImportError:
    from feature_maps.base import FeatureMap


class IQPMap(FeatureMap):
    """IQP feature map: H layer + ZZ-phase encoding for all pairs (i, j)."""

    def build(self, x: np.ndarray) -> QuantumCircuit:
        n = self._n_qubits
        qc = QuantumCircuit(n)

        for _ in range(self._reps):
            # Hadamard layer
            qc.h(range(n))
            # Single-qubit phase gates: RZ(2 * x_i)
            for i in range(n):
                qc.rz(2.0 * x[i % len(x)], i)
            # Two-qubit interactions: ZZ gate encoded as CNOT + RZ + CNOT
            for i in range(n - 1):
                for j in range(i + 1, n):
                    angle = 2.0 * x[i % len(x)] * x[j % len(x)]
                    qc.cx(i, j)
                    qc.rz(angle, j)
                    qc.cx(i, j)

        return qc

if __name__ == "__main__":
    iqp_map = IQPMap(n_qubits=2, reps=2)
    x = np.array([0.1, 0.2])
    qc = iqp_map.build(x)
    print(qc.draw())