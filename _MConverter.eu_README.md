==================================================\
CUDA Differential Evolution (DE) Benchmark\
==================================================\
\
1. PRÉSENTATION DU PROJET\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
Ce projet présente une implémentation parallèle de l\'algorithme d\'Évolution Différentielle (DE) sur GPU à l\'aide de NVIDIA CUDA en C++\[cite: 1, 10\]. Il découle de la refactorisation d\'un code initial d\'optimisation par essaim de particules (PSO)\[cite: 1\].\
\
2. OBJECTIFS ET ARCHITECTURE\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
- Implémentation du schéma DE/rand/1/bin avec F = 0,5 et CR = 0,9
- Optimisation par Kernel Fusionné (Single Kernel) : toutes les étapes (mutation, croisement, évaluation et sélection) sont exécutées dans un seul kernel CUDA (\`kernelDE\`) par thread (1 thread = 1 individu)
- Réduction des accès lents à la mémoire globale du GPU en conservant les données dans les registres rapides
\
3. PROTOCOLE EXPÉRIMENTAL\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
- 4 Fonctions benchmarks : Rastrigin, Rosenbrock, Griewank, Sphere
- Dimensions (D) : 10, 50, 100
- Populations (P) : 50, 100, 500
- Itérations : MAX_ITER = D \* 10\^4
- Exécutions : 10 runs consécutifs en mémoire C++ par configuration (36 configurations au total)
\
4. RÉSULTATS ET DISCUSSION\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
- Temps d\'exécution : Stabilité autour de 136 secondes sur l\'ensemble des configurations. Cette valeur s\'explique par la latence cumulée des appels de kernels gérés par la boucle principale côté CPU, qui masque le temps GPU pur/
- Analyse stochastique : Écart-type égal à 0.0 dû au déterminisme de la graine du générateur aléatoire CUDA (\`idx + iter \* pop + 1\`)
\
5. STRUCTURE DU DÉPÔT\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
- kernel.cu : Kernel CUDA fusionné et logique GPU
- kernel.h : Constantes et prototypes
- main.cpp : Programme hôte C++ (interface CLI et boucle de runs)
- kernel.cpp : Calculs de référence et fonctions auxiliaires CPU
- report.tex : Rapport au format IEEE
\
6. COMPILATION ET EXÉCUTION\
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\
Compilation :\
nvcc -o programde main.cpp kernel.cu kernel.cpp\
\
Exemple d\'exécution (Fonction Sphere=4, D=10, P=50, 10 runs) :\
./programde 4 10 50 10
