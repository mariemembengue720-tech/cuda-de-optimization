# Résultats : DE et PSO, CPU et GPU

Analyse des résultats expérimentaux du projet « Massively Parallel PSO and DE » : comparaison du **DE** séquentiel (CPU), du **DE** CUDA (GPU) et du **PSO** séquentiel (CPU).

**Sommaire** : [Résumé](#résumé) · [Données et méthode](#données-et-méthode) · [Qualité (DE contre PSO)](#qualité-de-contre-pso) · [CPU contre GPU](#cpu-contre-gpu) · [Comparaison avec l'article](#comparaison-avec-larticle) · [Pistes d'amélioration](#pistes-damélioration) · [Limites](#limites)

## Résumé

- **CPU contre GPU, même algorithme** : les résultats sont statistiquement identiques dans 32 configurations sur 36.
- **DE contre PSO** : le DE est significativement meilleur dans 15 cas sur 36, le PSO dans 11, sans différence dans 10.
- **Le GPU n'est plus rapide que si la population est grande** (≳ 170 individus). En dessous, il est plus lent que le CPU.
- **Les résultats de l'article ne sont pas reproduits** sur Rastrigin et Rosenbrock en dimension 50 et 100, mais le code est cohérent avec un DE standard.

## Données et méthode

| Fichier | Contenu |
|---|---|
| `results_de.csv` | 720 runs du DE (360 CPU, 360 GPU) |
| `results_pso.csv` | 360 runs du PSO (CPU) |
| `summary_de.csv`, `summary_pso.csv` | Moyennes et écarts-types par configuration |

- 4 fonctions (Sphere, Griewank, Rastrigin, Rosenbrock), dimensions 10, 50, 100, populations 50, 100, 500, 10 runs par configuration.
- Budget de 10⁴ × Dim évaluations : le nombre de générations vaut donc 10⁴·Dim / pop.
- Erreur = écart à l'optimum de la fonction.
- Tests statistiques : Mann-Whitney (CPU contre GPU, DE contre PSO) et Kruskal-Wallis (effet de la population), au seuil de 5 %.

## Qualité (DE contre PSO)

Erreur moyenne en dimension 100 :

| Fonction | DE P50 | DE P100 | DE P500 | PSO P50 | PSO P100 | PSO P500 |
|---|---|---|---|---|---|---|
| Sphere | 0 | 0 | 0,14 | 5,05 | 1,2 | 0,0012 |
| Griewank | 0 | 0 | 0,079 | 3,01 | 0,73 | 0,086 |
| Rastrigin | 545 | 583 | 723 | 441 | 352 | 307 |
| Rosenbrock | 85 | 90 | 4,9e4 | 1910 | 158 | 154 |

**La population joue en sens inverse pour les deux algorithmes.**

- **DE** : sa convergence se fait génération par génération. Avec 500 individus il n'a que 200 (10D) à 2000 (100D) générations, ce qui n'est pas assez : Rosenbrock 100D reste à 4,9e4. La meilleure population est 50 dans 11 configurations sur 12.
- **PSO** : il converge vite, puis stagne, surtout sur les fonctions multimodales. Les itérations supplémentaires d'une petite population sont donc gaspillées, et plus de particules apportent plus de diversité. La meilleure population est 500 dans 9 configurations sur 12.

**Selon la fonction**

- **Sphere et Griewank** sont faciles : le DE atteint l'optimum avec P50 ou P100 (taux de réussite de 89 % pour Sphere et 78 % pour Griewank, contre 58 % et 14 % pour le PSO).
- **Rastrigin et Rosenbrock** sont difficiles. L'erreur du DE sur Rastrigin croît avec la dimension (≈ 3 par coordonnée en 50D, ≈ 5,5 en 100D, à P50) : il se bloque dans des minima locaux. Rosenbrock a une vallée étroite et non séparable, et avec CR = 0,3 le DE ne modifie que peu de coordonnées à la fois.
- **Taux de réussite** : personne n'atteint 1e-8 sur Rosenbrock.

## CPU contre GPU

**Qualité.** Les 4 différences significatives sont toutes à P500 et petites (Sphere 100D : 0,139 contre 0,178). Elles viennent d'un détail d'algorithme : le CPU remplace les individus au fil de la génération, le GPU met à jour toute la population d'un coup. Une simulation le reproduit (0,136 contre 0,177).

**Temps.** Le CPU suit le budget d'évaluations : 0,034 s, 0,69 s et 2,7 s pour 10, 50 et 100D (≈ D²), sans dépendre de la population. Le GPU suit ce modèle ajusté sur les mesures :

```text
temps GPU ≈ 0,3 s + itérations × (13 µs + 4,5 µs × D)
itérations = 10⁴·D / pop − 1
```

| D | CPU (µs / évaluation) | GPU (µs / itération) | Population d'équilibre |
|---|---|---|---|
| 10 | 0,34 | 53 | ≈ 155 |
| 50 | 1,38 | 249 | ≈ 181 |
| 100 | 2,71 | 461 | ≈ 170 |

Accélération CPU/GPU observée :

| Dim | P50 | P100 | P500 |
|---|---|---|---|
| 10 | 0,08 | 0,08 | 0,11 |
| 50 | 0,25 | 0,44 | 1,28 |
| 100 | 0,28 | 0,54 | 2,23 |

**Pourquoi le GPU perd à petite population**

- **Peu de threads actifs** : un thread traite un individu, donc 50 threads (2 blocs de 32) seulement, sur un GPU qui en compte des milliers.
- **Boucle séquentielle sur D** : chaque thread parcourt ses D gènes un par un, d'où les 4,5 µs par dimension, très au-dessus des quelques dizaines de nanosecondes par gène du CPU.
- **Accès mémoire non groupés** : `pop[i*D+j]` espace les threads voisins de D valeurs.
- **Deux kernels et deux synchronisations par génération.**

Une génération coûte donc à peu près la même durée sur le GPU quelle que soit la population (jusqu'à 500), alors que le CPU paie en proportion de la population. D'où le point d'équilibre vers 170 individus.

**Les 0,3 s fixes** (initialisation de CUDA, `cudaMalloc`, copies) pèsent lourd : à 10D, ils représentent au moins 70 % du temps GPU. Sans eux, l'accélération à P500 serait d'environ 2,8× en 50D et 3,0× en 100D, au lieu de 1,28 et 2,23.

## Comparaison avec l'article

Erreur moyenne, P50 / P100 / P500 :

| Cas | Article (cudaDEi) | DE CPU du projet |
|---|---|---|
| Rastrigin 50D | 18 / 2,9 / 28 | 144 / 178 / 238 |
| Rastrigin 100D | 63 / 48 / 41 | 545 / 583 / 723 |
| Rosenbrock 10D | 0,39 / 0,04 / 1,3 | 0,72 / 4,1 / 41 |
| Rosenbrock 100D | 16 / 42 / 113 | 85 / 90 / 4,9e4 |

Sphere et Griewank concordent presque partout (erreur 0, sauf à P500). Rastrigin 10D concorde pour P50 et P100.

Le code n'a pas l'air faux : un DE recodé indépendamment (double précision, avec ou sans bornes, synchrone, optimum décalé) donne les mêmes chiffres (Rastrigin 50D P100 : 175 contre 178). Sur Rastrigin, le résultat dépend fortement de CR : avec CR = 0,1 l'erreur tombe à environ 0 en 50D (P50 et P100), alors qu'à 0,3 elle reste autour de 150–180. L'article ne donne pas assez de détails pour reproduire ses chiffres.

Côté temps, le GPU ressemble aux versions à configuration fixe de l'article (cudaDE, FPDE) : il est plus lent que le CPU à petite population, et son temps baisse quand la population augmente. La version cudaDEi, plus rapide, utilise un thread par gène, fusionne les kernels et utilise des streams.

## Pistes d'amélioration

1. **GPU** : un bloc par individu avec les threads sur les gènes (réduction en mémoire partagée pour le fitness), fusion des deux kernels et suppression des synchronisations inutiles, échauffement de CUDA avant la mesure.
2. **Qualité du DE** (étape 8 du sujet) : essayer un CR plus petit pour Rastrigin et plus grand pour Rosenbrock, ou un DE adaptatif.
3. **Article** : donner les accélérations plutôt que les temps absolus (matériel différent de l'article).

## Limites

- **Pas de décalage de l'optimum** : l'optimum est à l'origine, ce qui favorise probablement le PSO, attiré vers le centre de l'essaim. Sur une autre implémentation de PSO, l'erreur de Sphere 100D passe d'environ 3 600 avec un optimum décalé à 1e-12 avec l'optimum au centre. Les erreurs PSO sont donc plutôt optimistes.
- **Calcul en `float`** : toute erreur inférieure à environ 1,5e-5 est arrondie à 0, donc les taux de réussite (seuil 1e-8) sont un peu optimistes.
- **Temps GPU** : il inclut l'initialisation de CUDA et les transferts, ce qui pénalise surtout les petites configurations.
