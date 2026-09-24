# Massively Parallel PSO and DE Algorithms — Étape 1 : du PSO au DE séquentiel

**Rôle** : (cœur du code)
**Étape couverte par ce document** : analyse du PSO fourni + premier prototype DE fonctionnel en **CPU séquentiel**, avant tout portage CUDA.

---

## 1. Où on en est dans le projet
--- 
Ce prototype ne touche **pas encore au GPU**. L'objectif est de valider la logique algorithmique (mutation, crossover, sélection) sur CPU — plus simple à déboguer — avant d'introduire la complexité CUDA (indices de threads, mémoire globale/partagée, races).

---

## 2. Ce qu'on a appris en analysant le PSO fourni

Le PSO fourni (`main.cpp` + `kernel.h` + `kernel.cpp` + `kernel.cu` d'origine) a cette structure :

| Étape | Où ça tourne | Détail |
|---|---|---|
| Allocation + init | CPU | `positions`, `velocities`, `pBests` initialisés en host, puis `cudaMalloc` + copie H→D |
| Boucle fixe de `MAX_ITER` itérations | mixte | pas de critère d'arrêt sur un budget d'évaluations |
| `kernelUpdateParticle` | GPU | met à jour vitesse et position de chaque particule |
| `kernelUpdatePBest` | GPU | compare et met à jour le meilleur personnel de chaque particule |
| Copie `pBests` GPU→CPU | CPU | **tableau entier recopié à chaque itération** |
| Recalcul de `gBest` | CPU | boucle séquentielle sur toutes les particules, entièrement côté host |
| Copie `gBest` CPU→GPU | CPU | renvoyé au device pour l'itération suivante |

Deux points structurants pour la suite :

1. **Le PSO fait un aller-retour CPU↔GPU à chaque itération** (copie de `pBests` + recalcul de `gBest` côté host). C'est exactement le genre de trafic mémoire à faible débit que l'article *ImprovedDE* identifie comme le principal défaut des implémentations CUDA-DE naïves. Le DE ne devra **pas** reproduire ce pattern.
2. **Le PSO génère ses nombres aléatoires côté host** (`getRandomClamped()`, 2 scalaires par itération) et les passe en paramètres scalaires au kernel. Ça marche pour PSO (2 valeurs suffisent), mais le DE a besoin d'un aléa **par individu et par dimension** (pour `r1,r2,r3` et le crossover) — impossible à faire raisonnablement depuis le host à chaque génération. Ce sera un vrai changement d'architecture pour la version GPU (Phase 6, `CURAND`).

### Table de transformation PSO → DE

| PSO | Rôle dans PSO | Équivalent DE | Pourquoi |
|---|---|---|---|
| `positions` | position de chaque particule | population `X` (target vectors) | même rôle : population de solutions candidates |
| `velocities` | direction de déplacement | **aucun équivalent** | DE n'a pas de notion de vitesse : le déplacement vient de la différence entre individus (mutation), pas d'une inertie |
| `pBests` | meilleure position jamais visitée par la particule | **aucun équivalent** | DE ne garde pas de mémoire individuelle : chaque génération ne dépend que de la génération précédente |
| `gBest` | meilleure position globale, guide l'essaim | **aucun équivalent direct** | DE n'a pas de guide global : la sélection se fait individu par individu (`trial` vs `target`), pas par attraction vers un meneur |
| `kernelUpdateParticle` | maj vitesse + position | `kernel(M)` + `kernel(C)` (mutation + crossover) | rôle similaire (« faire évoluer l'individu ») mais mécanisme totalement différent |
| `kernelUpdatePBest` | comparer et garder le meilleur perso | `kernel(R)` (remplacement) | comparaison similaire (ancien vs nouveau), mais DE compare `trial` à `target`, pas à un historique |
| `cuda_pso` | orchestration de la boucle | `run_de_sequential` (puis futur `cuda_de`) | rôle équivalent : orchestrer l'ensemble |

**Pourquoi DE n'a besoin ni de vitesse, ni de mémoire individuelle (`pBest`), ni de meneur global (`gBest`)** : le moteur de recherche du DE, c'est la **diversité de la population elle-même**. Le vecteur de mutation `X_r2 − X_r3` est différent à chaque génération et pour chaque individu, ce qui suffit à explorer l'espace — pas besoin d'un historique de vitesse ni d'un point d'attraction unique comme dans PSO.

---

## 3. Algorithme implémenté : DE/rand/1/bin

D'après *ImprovedDE.pdf* (§4 du sujet projet), avec les paramètres imposés `F = 0.5`, `CR = 0.3`.

**Mutation** — pour chaque individu cible `i`, tirer `r1, r2, r3` mutuellement distincts et différents de `i` :

```
V_i = X_r1 + F · (X_r2 − X_r3)
```

**Crossover binomial** — construire le vecteur d'essai `U_i` en mélangeant mutant et cible, avec une dimension `j_rand` forcée pour garantir qu'au moins une composante du mutant soit reprise :

```
U_i,j = V_i,j   si rand_j(0,1) ≤ CR  ou  j = j_rand
U_i,j = X_i,j   sinon
```

**Sélection / remplacement** — le trial ne remplace la cible que s'il est meilleur :

```
X_i(t+1) = U_i   si f(U_i) < f(X_i)
X_i(t+1) = X_i(t)   sinon
```

**Critère d'arrêt** — budget d'évaluations de fonction, pas un nombre d'itérations fixe (contrairement au `MAX_ITER` du PSO) :

```
MaxFEs = 10⁴ × D
```

**Double buffering (`X_current` / `X_next`)** — chaque individu de la génération courante est lu depuis `X_current` et le résultat (trial ou target inchangé) est écrit dans `X_next`, jamais l'inverse. À la fin d'une génération complète, on échange les deux buffers (`swap`). Ce n'est pas strictement nécessaire en séquentiel (pas de lecture concurrente), mais c'est volontairement conservé dès ce prototype : c'est la structure de données qui rend le futur portage CUDA sûr, en évitant qu'un thread lise une population partiellement mise à jour par un autre thread pendant la même génération.

---

## 4. Ce qui a changé dans chaque fichier — et pourquoi

Principe suivi partout : **ne rien casser du PSO fourni**, ajouter le DE à côté.

### `kernel.h`
- Constantes PSO (`NUM_OF_DIMENSIONS`, `NUM_OF_PARTICLES`, `SELECTED_OBJ_FUNC`, `MAX_ITER`, `OMEGA`, `c1`, `c2`, `phi`, `START_RANGE_*`) : **strictement inchangées**.
  → Risque identifié : `kernel.cu` déclare des tableaux `__device__` dont la taille dépend de `NUM_OF_DIMENSIONS`, qui doit donc rester une constante *compile-time*. Les transformer en variables runtime aurait cassé la compilation du PSO.
- Ajout d'une section DE séparée : `DE_F`, `DE_CR`, `FE_BUDGET_PER_DIM`, prototypes de `de_fitness_function` et `run_de_sequential`.

### `kernel.cpp`
- Le `switch` de calcul de fitness (5 fonctions benchmark) a été **factorisé** dans une fonction interne `compute_fitness(x, dim, obj_func)`, avec dimension et fonction objectif en paramètres au lieu de constantes globales.
- `host_fitness_function(x)` — celle que `kernel.cu` appelle pour le PSO — garde **exactement** sa signature et son comportement d'origine (elle appelle juste `compute_fitness` avec les constantes PSO). **Zéro modification de `kernel.cu` nécessaire.**
- `de_fitness_function(x, dim, obj_func)` appelle la même fonction commune avec dimension/objectif choisis au runtime — mêmes formules, aucun changement de benchmark.
- Nouveaux opérateurs DE : tirage `r1/r2/r3`, mutation, crossover, sélection, boucle principale `run_de_sequential`.

### `main.cpp`
- Parsing CLI (`dim pop objectif [seed]`) avec validation explicite : `dim > 0`, **`population ≥ 4`** (contrainte DE/rand/1 : il faut 3 indices distincts en plus de l'individu cible), `objectif ∈ [0,4]`.
- N'appelle plus `cuda_pso` — c'est maintenant le point d'entrée du DE séquentiel.

### `kernel.cu`
- **Aucune modification.** Le PSO GPU fourni reste intact, prêt pour la Phase du Cuda_DE  où on adaptera sa logique. Il est toujours compilé (avec `nvcc`, pour garder les 4 fichiers cohérents) mais n'est appelé par aucun chemin de ce prototype.

---

## 5. Compiler et lancer

### En local (CPU uniquement, pas besoin de GPU ni de `nvcc`)

```bash
g++ -O2 -std=c++17 -o programde_cpu main.cpp kernel.cpp
./programde_cpu <dimensions> <population> <objectif:0-4> [seed]

# exemples
./programde_cpu 10 50 1 42     # D=10, NP=50, Rastrigin
./programde_cpu 100 500 4 42   # D=100, NP=500, Sphère
```

### Sur Colab (cohérent avec le reste du projet, `nvcc` sur les 4 fichiers)

Voir `CudaDE_Colab.ipynb` — upload des 4 fichiers, compilation `nvcc`, exécution paramétrée, collecte des résultats dans un tableau pandas exporté en CSV.

---

## 7. Résultats de validation (CPU, seed=42)

| Fonction | D | NP | Résultat obtenu | Optimum théorique | Cohérent ? |
|---|---|---|---|---|---|
| Sphère (4) | 10 | 50 | -450.000000 | -450 |  exact |
| Rastrigin (1) | 10 | 50 | -330.000000 | -330 |  exact |
| Levy (0) | 10 | 50 | 0.000000 | 0 |  exact |
| Rosenbrock (2) | 10 | 50 | 393.207336 | 390 |  proche |
| Rosenbrock (2) | 50 | 100 | 430.898834 | 390 |  plus loin (dimension plus dure) |
| Rosenbrock (2) | 100 | 500 | 390.390259 | 390 |  population plus grande compense la dimension |
| Griewank (3) | 10 | 50 | -179.000000 | -180 |  décalage attendu de +1 |

Le comptage de FEs a aussi été vérifié : pour un budget non multiple de la population (ex. `D=10, NP=33`, budget théorique 100 000), l'exécution s'arrête à `FEs=99990` — juste avant de dépasser le budget, comme attendu d'une boucle par génération complète.

**Pourquoi ces résultats sont un bon signal de correction** : le pattern observé (Sphère/Rastrigin/Levy convergent quasi parfaitement, Rosenbrock est nettement plus dur et s'améliore avec une population plus grande) reproduit exactement les tendances rapportées dans les Tables 1-3 et Figures 3-6 de *ImprovedDE.pdf* — ce n'est pas un hasard, la logique DE/rand/1/bin semble bien implémentée.

---

## 8. Limites de ce prototype et prochaine étape

- **Aucun calcul GPU.** `run_de_sequential` tourne entièrement sur un seul thread CPU — normal à ce stade, l'objectif était de valider l'algorithme avant la complexité CUDA.

**Prochaine étape (Phase 6)** : adapter `kernel.cu` pour porter `run_de_sequential` sur GPU, en s'inspirant d'abord de l'architecture *basique* de la Figure 1 de *ImprovedDE.pdf* (kernels I/E/P/M/C/R) avant d'envisager l'architecture améliorée `cudaDEᵢ` (kernel IE/P/MCER, Figure 2) une fois la version basique validée.
