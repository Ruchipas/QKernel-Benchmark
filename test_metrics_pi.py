import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets.loader import load_dataset
from kernels.fidelity_kernel import FidelityKernel
from classifiers.qsvm import QSVM

# Helper to manually scale since loader still does [0, 2pi]
def evaluate(scale_name, min_val, max_val):
    X_train, X_test, y_train, y_test = load_dataset(name="moons", n_samples=40, n_features=2, test_size=0.25, random_state=42)
    # Re-scale from [0, 2pi] to [min_val, max_val]
    X_train = (X_train / (2 * np.pi)) * (max_val - min_val) + min_val
    X_test = (X_test / (2 * np.pi)) * (max_val - min_val) + min_val
    
    k = FidelityKernel(n_qubits=2, shots=4096, seed=42)
    K_train = k.build_kernel_matrix(X_train)
    K_test = k.build_kernel_matrix(X_test, X_train)
    
    m = QSVM(C=1.0)
    m.fit(K_train, y_train)
    y_pred = m.predict(K_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"--- {scale_name} [{min_val:.2f}, {max_val:.2f}] ---")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

evaluate("0 to pi", 0.0, np.pi)
evaluate("-pi to pi", -np.pi, np.pi)
