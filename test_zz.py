from qiskit.circuit.library import ZZFeatureMap
import numpy as np
zz = ZZFeatureMap(2, reps=2)
print("Params:", zz.parameters)
print(zz.draw())
