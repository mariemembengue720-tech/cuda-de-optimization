# Évolution Différentielle (DE) Massivement Parallèle sur GPU avec CUDA

Ce dépôt contient une implémentation C++/CUDA haute performance de l'algorithme d'**Évolution Différentielle (DE)**, comparant l'exécution séquentielle sur CPU à trois implémentations CUDA progressives accélérées sur GPU.

Ce projet s'inspire des travaux de recherche de *Qin et al. (GECCO 2012)* (*« An Improved CUDA-Based Implementation of Differential Evolution on GPU »*).

##  Aperçu du Projet

L'Évolution Différentielle est une métaheuristique basée sur une population, utilisée pour l'optimisation continue globale. Bien qu'efficace, les problèmes à grande échelle ($D \ge 100$) nécessitent une puissance de calcul considérable. Ce projet évalue la capacité des architectures GPU modernes (NVIDIA CUDA) à accélérer la DE grâce à la fusion de kernels, l'utilisation de la mémoire partagée (*shared memory*) et le recouvrement d'exécutions via les CUDA Streams.

###  Modes Implémentés

1. **`cpu`** : Implémentation C++ séquentielle (`run_de_sequential`) servant de ligne de base (*baseline*).

2. **`gpu`** : Implémentation CUDA de base (`cuda_de`) utilisant des kernels séparés pour chaque opérateur DE (Mutation, Croisement, Évaluation, Sélection).

3. **`gpu-fused`** : Implémentation CUDA intermédiaire (`cuda_de_fused`) regroupant les opérateurs en 2 kernels composites pour réduire les accès à la mémoire globale et le surcoût de lancement des kernels.

4. **`gpu-optimized`** : Implémentation CUDA avancée (`cuda_de_optimized`) exploitant :
   * L'optimisation de la mémoire partagée par bloc/gène.
   * La configuration dynamique de la taille des blocs pour maximiser l'occupation des SM (*Streaming Multiprocessors*).
   * L'utilisation de deux **CUDA Streams** pour l'exécution simultanée de la préparation des indices de mutation et du calcul des kernels.

##  Structure du Dépôt

```
.
├── kernel.h               # Constantes partagées, paramètres DE (F, CR) et prototypes de fonctions
├── kernel.cpp             # Implémentation CPU séquentielle & définitions des fonctions objectifs
├── kernel.cu              # Implémentations CUDA GPU (kernels de base, fusionné et optimisé)
├── main_all_de.cpp        # Orchestrateur CLI & mesure des temps d'exécution
├── benchmark_de.py        # Script d'automatisation Python pour l'analyse statistique sur 10 runs
├── benchmark_de_results.csv # Résultats expérimentaux exportés (format CSV)
└── README.md              # Documentation du projet
```

##  Fonctions Benchmark

Les évaluations respectent les critères standards de la compétition **CEC 2005** ($10^4 \times D$ évaluations maximales de la fonction objectif) :

| Indice | Fonction Objectif | Propriétés Mathématiques |
| :--- | :--- | :--- |
| `1` | **Shifted Rastrigin** | Fortement multimodale avec de nombreux minima locaux |
| `2` | **Shifted Rosenbrock** | Non séparable, vallée parabolique étroite |
| `3` | **Shifted Griewank** | Multimodale avec minima locaux répartis |
| `4` | **Shifted Sphere** | Unimodale, test de convergence de base |

> **Note** : La fonction d'indice `0` (fonction de Levy) est exclue du mode `gpu-optimized` en raison des contraintes de synchronisation de threads inter-gènes en mémoire partagée.

##  Prise en Main

### Prérequis

* **NVIDIA CUDA Toolkit** (compilateur `nvcc`)
* **Compilateur GCC / G++**
* **Python 3.x** avec `pandas` et `numpy` (pour les benchmarks automatisés)

###  Compilation

Compilez le code C++/CUDA avec `nvcc` :

```bash
nvcc -O3 main_all_de.cpp kernel.cpp kernel.cu -o de_app
```

##  Utilisation

### Exécution Directe en Ligne de Commande (CLI)

Lancer un test benchmark individuel :

```bash
./de_app <mode> <dim> <pop> <obj_func> <seed>
```

#### Exemple :

```bash
# Exécution du mode GPU optimisé sur Rastrigin (1), Dim 50, Pop 512, Graine 42
./de_app gpu-optimized 50 512 1 42
```

### Suite de Benchmarks Automatisée

Exécuter 10 exécutions indépendantes (graines 1 à 10) pour toutes les combinaisons de dimensions ($D \in \{10, 50, 100\}$), de populations ($Pop \in \{50, 100, 500\}$) et de fonctions objectifs :

```bash
python benchmark_de.py
```

Cela génère les fichiers `benchmark_de_results.csv` et `benchmark_de_results.tex` (tableau LaTeX).

##  Résumé des Performances

Accélérations (*speedups*) expérimentales calculées par rapport à la version CPU séquentielle sur $10$ exécutions indépendantes :

| Dimension ($D$) | CPU | GPU (De base) | GPU-fused | GPU-optimized |
| :--- | :--- | :--- | :--- | :--- |
| $D = 10$ | $1.00\times$ | $0.24\times$ | $0.24\times$ | $0.30\times$ |
| $D = 50$ | $1.00\times$ | $1.27\times$ | $1.42\times$ | $4.51\times$ |
| $D = 100$ | $1.00\times$ | $1.80\times$ | $1.97\times$ | $14.25\times$ *(jusqu'à* $32.5\times$*)* |

### Observations Clés :

* **Dimensions Faibles ($D=10$)** : Le CPU est plus rapide que le GPU en raison du surcoût de lancement des kernels CUDA et de la latence des transferts mémoire.
* **Dimensions Élevées ($D=100$)** : À mesure que la taille du problème augmente, la version `gpu-optimized` offre des gains d'accélération très importants (jusqu'à $32.5\times$), démontrant l'intérêt de la fusion de kernels et des CUDA Streams.

##  Références

* **Qin, A. K., Raimondo, F., Forbes, F., & Ong, Y. S. (2012)**. *An Improved CUDA-Based Implementation of Differential Evolution on GPU*. In Proceedings of the 14th annual conference on Genetic and evolutionary computation (GECCO '12), pp. 993–1000.
