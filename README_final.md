# DE et PSO : versions séquentielle (CPU) et CUDA (GPU)

Projet de Master « Massively Parallel PSO and DE ». Il compare l'algorithme **Differential Evolution (DE/rand/1/bin)** en version séquentielle (CPU) et en version CUDA (GPU), avec un **PSO séquentiel** comme référence, sur 4 fonctions benchmark. Les expériences tournent sur Kaggle avec un GPU NVIDIA.

## Contenu

| Fichier | Rôle |
|---|---|
| `kernel_de_seq.h` | Déclarations du DE (CPU et GPU) |
| `kernel_de_seq.cpp` | DE séquentiel, bornes, fonctions objectif, tirages aléatoires |
| `kernel_de_seq.cu` | DE CUDA : un kernel de mutation + croisement, un kernel de sélection |
| `main_de_app.cpp` | Programme DE, mode `cpu` ou `gpu` |
| `kernel_pso_seq.h` / `.cpp` | PSO séquentiel (CPU) |
| `main_pso_seq.cpp` | Programme PSO |
| `kernel_pso_seq.cu` | Non utilisé (reste du code de base, il n'y a pas de PSO GPU) |
| `project-cuda-de-pso.ipynb` | Notebook Kaggle : préparation, expériences, résumés, graphes |
| `results_*.csv`, `summary_*.csv` | Résultats du DE et du PSO |

## Protocole

- Dimensions : 10, 50, 100. Populations : 50, 100, 500.
- 10 exécutions par configuration (seeds 1 à 10).
- Budget : 10⁴ × Dim évaluations de la fonction.
- DE : F = 0,5 ; CR = 0,3. PSO : ω = 0,5 ; c1 = c2 = 1,5. Positions bornées au domaine.
- Mesure : erreur = f − f\*. Une exécution réussit si l'erreur est inférieure à 1e-8.

| N° | Fonction | Domaine | f\* |
|---|---|---|---|
| 1 | Shifted Rastrigin | [-5, 5] | -330 |
| 2 | Shifted Rosenbrock | [-100, 100] | 390 |
| 3 | Shifted Griewank | [-600, 600] | -180 |
| 4 | Shifted Sphere | [-100, 100] | -450 |

La fonction 0 (Levy) existe dans le code mais n'est pas utilisée dans les expériences.

## Utilisation sur Kaggle

1. Activer un GPU : *Settings → Accelerator*.
2. Lancer les cellules du notebook dans l'ordre :
   1. vérification du GPU (`nvidia-smi`, `nvcc`) ;
   2. préparation des fichiers du DE, puis sweep du DE (≈ 25 min), résumé et graphes ;
   3. préparation des fichiers du PSO, puis sweep du PSO (≈ 11 min), résumé et graphes.

## Compilation et exécution à la main

```bash
# PSO (CPU)
g++ -O2 main_pso_seq.cpp kernel_pso_seq.cpp -o pso_seq
./pso_seq <dim> <pop> <func> <seed>

# DE (CPU et GPU)
g++ -O2 -c kernel_de_seq.cpp -o k_cpp.o
g++ -O2 -c main_de_app.cpp   -o main.o
nvcc -O2 -arch=native -DNUM_OF_DIMENSIONS=<dim> -DNUM_OF_PARTICLES=<pop> \
     -DSELECTED_OBJ_FUNC=<func> -c kernel_de_seq.cu -o k_cu.o
nvcc -arch=native main.o k_cpp.o k_cu.o -o de_app
./de_app <cpu|gpu> <dim> <pop> <func> <seed>
```

En mode `gpu`, `<dim>`, `<pop>` et `<func>` doivent être identiques aux valeurs `-D` de la compilation. Chaque exécution affiche la meilleure fitness, le budget d'évaluations et le temps en ms.

## Résultats

- `results_de.csv`, `results_pso.csv` : une ligne par exécution (`algo, func, dim, pop, run, seed, best, error, time_ms`).
- `summary_de.csv`, `summary_pso.csv` : une ligne par (`algo, func, dim, pop`), avec `mean_best, std_best, min_best, mean_error, std_error, success_rate, time_s`.

## Limites connues

- Les fonctions n'ont pas de vecteur de décalage : l'optimum est à l'origine et seul le biais (f\*) est appliqué.
- Les calculs sont en `float` : une erreur inférieure à environ 1,5e-5 est arrondie à 0, donc `success_rate` est approximatif.
- Le temps GPU est mesuré autour de `cuda_de` : il inclut l'initialisation de CUDA et les transferts. Le temps CPU ne mesure que la boucle principale.
- Le DE CPU remplace les individus au fil de la génération, le DE GPU met à jour toute la population d'un coup : la convergence peut différer légèrement à grande population.
- Dimension, population et fonction étant fixées à la compilation du `.cu`, le notebook recompile le GPU à chaque configuration.
