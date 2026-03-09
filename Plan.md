# QKernel-Benchmark Plan

## Overview
A benchmarking suite for **4 quantum kernel methods** applied to classification tasks.
Built on Qiskit + Aer simulator and scikit-learn.

## Kernels Benchmarked

| Kernel | Key Idea | Reference |
|--------|----------|-----------|
| **FQK** — Fidelity Quantum Kernel | `k(x,x') = \|⟨ϕ(x)\|ϕ(x')⟩\|²` via adjoint circuit | Havlíček et al., Nature 2019 |
| **PQK** — Projected Quantum Kernel | RBF over local 1-qubit reduced density matrices | Huang et al., Nature Comms 2021 |
| **QKTA** — Trainable Kernel (Kernel-Target Alignment) | Optimize feature-map parameters to align kernel with labels | Hubregtsen et al., arXiv:2105.02276 |
| **Q-FLAIR** — Feature-Map Architecture Learning | Greedily grow circuit by KTA, then eval as FQK | Barbosa et al., arXiv:2311.04965 |

## Datasets
`ad_hoc`, `moons`, `circles`, `blobs`, `iris`, `wine`, `breast_cancer`
All PCA-reduced to `n_qubits` features and MinMax-scaled to `[0, 2π]`.

## Metrics
- **ML:** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- **Quantum Resources:** Qubit count, circuit depth, total shots, kernel evaluations, wall-clock time
- **Plots:** ROC curves (per kernel and overlaid)

## Repository Structure
```
QKernel-Benchmark/
├── feature_maps/     ← IQP, ZZ, Rx, HEA feature maps
├── kernels/          ← FQK, PQK, QKTA, Q-FLAIR
├── classifiers/      ← QSVM, QKRR
├── datasets/         ← unified dataset loader
├── benchmark/        ← metrics, ROC plots, benchmark runner
├── experiments/      ← run_single.py, run_all.py
└── tests/            ← pytest test suite
```

## Usage
```bash
# Single experiment
python experiments/run_single.py --kernel fqk --dataset moons --n_qubits 4 --shots 1024

# Full benchmark grid
python experiments/run_all.py
```
Results saved to `results/benchmark_results.csv` and `results/plots/`.
