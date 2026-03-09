"""
Checked by Bank
2026-03-07
"""

"""
Rx angle-encoding feature map.

Each qubit i receives a rotation Rx(π * x_i). Simple and fast — often
used as a baseline for quantum kernel benchmarks.

Reference:
    Schuld, M. & Killoran, N. (2019). Quantum Machine Learning in Feature
    Hilbert Spaces. PRL 122, 040504.
    https://doi.org/10.1103/PhysRevLett.122.040504
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


class RxMap(FeatureMap):
    """Simple Rx angle-encoding: Rx(π * x_i) on each qubit."""

    def build(self, x: np.ndarray) -> QuantumCircuit:
        n = self._n_qubits
        qc = QuantumCircuit(n)
        for _ in range(self._reps):
            for i in range(n):
                qc.rx(np.pi * x[i % len(x)], i)
        return qc

if __name__ == "__main__":
    rx_map = RxMap(n_qubits=2, reps=2)
    x = np.array([0.1, 0.2])
    qc = rx_map.build(x)
    print(qc.draw())
