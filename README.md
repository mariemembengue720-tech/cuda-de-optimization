# Massively Parallel Differential Evolution (DE) on GPU with CUDA

This repository contains a high-performance C++/CUDA implementation of the **Differential Evolution (DE)** algorithm, comparing sequential CPU execution against three progressive GPU-accelerated CUDA implementations.

This project is inspired by the research work of *Qin et al. (GECCO 2012)* (*"An Improved CUDA-Based Implementation of Differential Evolution on GPU"*).

---

##  Project Overview

Differential Evolution is a population-based metaheuristic for global continuous optimization. While effective, large-scale problems ($D \ge 100$) require significant computational power. This project evaluates how modern GPU architectures (NVIDIA CUDA) can accelerate DE through memory fusion, shared memory usage, and overlapping kernel execution via CUDA streams.

###  Implemented Modes

1. **`cpu`**: Sequential C++ implementation (`run_de_sequential`) serving as the baseline.
2. **`gpu`**: Basic CUDA implementation (`cuda_de`) using separate kernels for each DE operator (Mutation, Crossover, Evaluation, Selection).
3. **`gpu-fused`**: Intermediate CUDA implementation (`cuda_de_fused`) combining operators into 2 composite kernels to reduce global memory access and launch overhead.
4. **`gpu-optimized`**: Advanced CUDA implementation (`cuda_de_optimized`) utilizing:
   - Shared memory optimization per block/gene.
   - Dynamic block size configuration to maximize SM occupancy.
   - Dual **CUDA Streams** for concurrent execution of mutation index preparation and kernel evaluation.

---

##  Repository Structure

```text
.
├── kernel.h               # Shared constants, DE parameters (F, CR), and function prototypes
├── kernel.cpp             # CPU sequential implementation & objective function definitions
├── kernel.cu              # CUDA GPU implementations (basic, fused, and optimized kernels)
├── main_all_de.cpp        # CLI orchestrator & timer driver
├── benchmark_de.py        # Python automation script for 10-run statistical analysis
├── benchmark_de_results.csv # Exported experimental results (CSV format)
└── README.md              # Project documentation
```

---

##  Benchmark Functions

The evaluations follow the **CEC 2005** standard benchmark criteria ($10^4 \times D$ Max Function Evaluations):

| Index | Objective Function | Mathematical Properties |
| :---: | :--- | :--- |
| `1` | **Shifted Rastrigin** | Highly multimodal with many local minima |
| `2` | **Shifted Rosenbrock** | Non-separable, narrow parabolic valley |
| `3` | **Shifted Griewank** | Multimodal with widespread local minima |
| `4` | **Shifted Sphere** | Unimodal, baseline convergence test |

> **Note**: Function index `0` (Levy function) is guarded against `gpu-optimized` due to cross-gene thread synchronization requirements in shared memory.

---

##  Getting Started

### Prerequisites

- **NVIDIA CUDA Toolkit** (`nvcc` compiler)
- **GCC / G++ Compiler**
- **Python 3.x** with `pandas` and `numpy` (for automated benchmarks)

###  Compilation

Compile the C++/CUDA code using `nvcc`:

```bash
nvcc -O3 main_all_de.cpp kernel.cpp kernel.cu -o de_app
```

---

##  Usage

### Direct CLI Execution

Run a single benchmark test:

```bash
./de_app <mode> <dim> <pop> <obj_func> <seed>
```

#### Example:
```bash
# Run GPU-optimized DE on Rastrigin (1), Dim 50, Pop 512, Seed 42
./de_app gpu-optimized 50 512 1 42
```

### Automated Benchmark Suite

Run 10 independent runs (seeds 1–10) across all combinations of dimensions ($D \in \{10, 50, 100\}$), populations ($Pop \in \{50, 100, 500\}$), and benchmark functions:

```bash
python benchmark_de.py
```

This generates `benchmark_de_results.csv` and `benchmark_de_results.tex` (LaTeX table).

---

##  Performance Summary

Experimental speedups calculated against the sequential CPU version across $10$ independent runs:

| Dimension ($D$) | CPU | GPU (Basic) | GPU-fused | GPU-optimized |
| :---: | :---: | :---: | :---: | :---: |
| **$D = 10$** | $1.00\times$ | $0.24\times$ | $0.24\times$ | $0.30\times$ |
| **$D = 50$** | $1.00\times$ | $1.27\times$ | $1.42\times$ | **$4.51\times$** |
| **$D = 100$** | $1.00\times$ | $1.80\times$ | $1.97\times$ | **$14.25\times$** *(up to $32.5\times$)* |

### Key Observations:
- **Low Dimensions ($D=10$)**: CPU outperforms GPU due to CUDA launch overhead and memory transfer latencies.
- **High Dimensions ($D=100$)**: As problem size scales, `gpu-optimized` delivers significant speedups (up to **$32.5\times$**), proving the value of kernel fusion and CUDA streams.

---

##  References

- **Qin, A. K., Raimondo, F., Forbes, F., & Ong, Y. S. (2012)**. *An Improved CUDA-Based Implementation of Differential Evolution on GPU*. In Proceedings of the 14th annual conference on Genetic and evolutionary computation (GECCO '12), pp. 993–1000.
