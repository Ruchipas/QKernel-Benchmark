import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets.loader import load_dataset
from kernels.fidelity_kernel import FidelityKernel
from classifiers.qsvm import QSVM

# Fix random seeds for reproducibility
X_train, X_test, y_train, y_test = load_dataset(name="moons", n_samples=40, n_features=2, test_size=0.25, random_state=42)

print(f"y_test class distribution: {-1}: {sum(y_test==-1)}, {1}: {sum(y_test==1)}")

# Test different scalings applied to the [0, 2pi] normalized data from loader
# 1.0 = [0, 2pi]
# 1/(2pi) = [0, 1]
# 1/pi = [0, 2]
# 2.0 = [0, 4pi]
scales = [
    ("Default [0, 2π]", 1.0),
    ("Scaled [0, 1]", 1 / (2 * np.pi)),
    ("Scaled [0, 2]", 1 / np.pi),
    ("Scaled [-π, π] shift", "shift")
]

for label, scale in scales:
    if scale == "shift":
        Xt = X_train - np.pi
        Xv = X_test - np.pi
    else:
        Xt = X_train * scale
        Xv = X_test * scale
        
    k = FidelityKernel(n_qubits=2, shots=4096, seed=42)
    K_train = k.build_kernel_matrix(Xt)
    K_test = k.build_kernel_matrix(Xv, Xt)
    
    m = QSVM(C=1.0)
    m.fit(K_train, y_train)
    y_pred = m.predict(K_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n--- {label} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"y_pred   : {y_pred.tolist()}")
