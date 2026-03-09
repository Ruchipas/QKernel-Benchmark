import numpy as np
from datasets.loader import load_dataset
from kernels.fidelity_kernel import FidelityKernel

X_train, X_test, y_train, y_test = load_dataset(name="moons", n_samples=40, n_features=2, test_size=0.25, random_state=42)
kernel = FidelityKernel(n_qubits=2, shots=1024, seed=42)
K_train = kernel.build_kernel_matrix(X_train)
print("Mean off-diagonal:", np.mean(K_train[~np.eye(K_train.shape[0], dtype=bool)]))
print("Variance off-diagonal:", np.var(K_train[~np.eye(K_train.shape[0], dtype=bool)]))
print("Min off-diagonal:", np.min(K_train[~np.eye(K_train.shape[0], dtype=bool)]))
print("Max off-diagonal:", np.max(K_train[~np.eye(K_train.shape[0], dtype=bool)]))

K_test = kernel.build_kernel_matrix(X_test, X_train)
print("Mean K_test:", np.mean(K_test))
