import numpy as np
from datasets.loader import load_dataset
from kernels.fidelity_kernel import FidelityKernel
from classifiers.qsvm import QSVM
X_train, X_test, y_train, y_test = load_dataset(name="moons", n_samples=40, n_features=2, test_size=0.25, random_state=42)

# use shots=None or 0 to get exact Statevector evaluation
kernel = FidelityKernel(n_qubits=2, shots=0, seed=42)
K_train = kernel.build_kernel_matrix(X_train)
K_test = kernel.build_kernel_matrix(X_test, X_train)

model = QSVM(C=1.0)
model.fit(K_train, y_train)
y_pred = model.predict(K_test)
print("Exact accuracy:", model.score(K_test, y_test))

# Try scaling to [0, 1] instead of [0, 2*pi]?
X_train /= (2*np.pi)
X_test /= (2*np.pi)
K_train2 = kernel.build_kernel_matrix(X_train)
K_test2 = kernel.build_kernel_matrix(X_test, X_train)
model2 = QSVM(C=1.0)
model2.fit(K_train2, y_train)
print("Scaled accuracy:", model2.score(K_test2, y_test))
