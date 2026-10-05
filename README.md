# Differential Evolution (DE) - Implémentation CPU vs GPU (CUDA)

Projet d'optimisation métaheuristique comparant les performances d'exécution séquentielle (CPU) et accélérée (GPU CUDA).

## Architecture du projet
- `kernel.h` : Constantes globales, sélection des benchmarks et prototypes.
- `kernel.cpp` : Implémentation séquentielle CPU de l'algorithme DE.
- `kernel.cu` : Kernels CUDA et gestion de la mémoire VRAM GPU.
- `main.cpp` : Orchestration des tests, chronométrage et calcul du Speedup.
- `Differential_Evolution_CUDA.ipynb` : Notebook Kaggle d'expérimentation.

## Compilation et Exécution
```bash
nvcc -O3 main.cpp kernel.cpp kernel.cu -o de_app
./de_app 30 4096