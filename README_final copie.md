# DE et PSO sur CPU et GPU

Ce projet étudie l’optimisation de fonctions par **Differential Evolution (DE)** et **Particle Swarm Optimization (PSO)**. Il compare la qualité des solutions et les temps de calcul des versions séquentielles CPU et parallèles CUDA.

## Organisation des quatre éléments

| Élément | Contenu et rôle |
|---|---|
| [gpu-pso.ipynb](gpu-pso.ipynb) | Notebook autonome consacré au PSO : sources C++/CUDA embarquées, compilation CPU/GPU, tests, campagne de mesures, figures et diagnostic d’échauffement. |
| [resultats_complets.zip](resultats_complets.zip) | Archive des résultats du notebook PSO : CSV CPU/GPU, synthèses, accélérations, journaux, métadonnées et diagnostics. Les fichiers sont placés directement à la racine de l’archive. |
| [notebook6a6cfc3811.ipynb](notebook6a6cfc3811.ipynb) | Notebook exécuté consacré à la comparaison commune **DE CPU, DE GPU, PSO CPU et PSO GPU**. Il contient les sources, les tests, les mesures et l’analyse statistique des quatre implémentations. |
| [de_pso_work/](de_pso_work/) | Dossier des sessions produites par le notebook à quatre implémentations. Chaque session regroupe les sources compilées, les exécutables et les résultats exportés. |

Les deux notebooks embarquent leurs sources et peuvent générer leurs fichiers sans dataset externe. `gpu-pso.ipynb` fournit les fonctions objectifs et les algorithmes PSO repris dans `notebook6a6cfc3811.ipynb`. Ce dernier réalise sa propre campagne pour mesurer les quatre implémentations dans un même environnement.

Les résultats se lisent en deux ensembles : **PSO CPU/GPU dans `resultats_complets.zip`**, et **DE/PSO CPU/GPU dans `de_pso_work/`**. Les tableaux ci-dessous indiquent explicitement leur provenance.

### Structure de `gpu-pso.ipynb`

Le notebook suit l’ordre suivant :

1. Configuration et vérification des outils Python, C++ et CUDA.
2. Génération des sources embarquées : `kernel.h`, `kernel.cpp`, `kernel.cu` et `main.cpp`.
3. Compilation des exécutables PSO CPU et GPU.
4. Tests rapides et lancement d’une configuration libre.
5. Campagne complète sur les fonctions, dimensions et populations du protocole.
6. Synthèses de qualité, temps et accélérations CPU/GPU.
7. Diagnostic répété sur Sphere 10D avec 50 particules, après échauffement intégré.

`kernel.h` définit les fonctions objectifs et les structures de paramètres. `kernel.cpp` contient les utilitaires, la génération du décalage et le PSO CPU. `kernel.cu` contient le PSO CUDA. `main.cpp` pilote les configurations, les répétitions et les exports.

### Contenu de `resultats_complets.zip`

Les noms suivants sont ceux des fichiers **à l’intérieur de l’archive** :

| Fichier | Contenu |
|---|---|
| `results_pso_cpu_reference.csv` | 360 mesures PSO CPU |
| `results_pso_gpu.csv` | 360 mesures PSO GPU |
| `results_pso_combined.csv` | Mesures CPU et GPU réunies |
| `summary_pso_by_impl.csv` | 72 synthèses par implémentation et configuration |
| `speedup_pso.csv` | 36 rapports de temps moyens CPU/GPU |
| `results_pso_cpu_reference.json`, `results_pso_gpu.json` | Paramètres, environnement, commandes, empreintes et informations d’échauffement |
| `results_pso_cpu_reference.txt`, `results_pso_gpu.txt` | Journaux des campagnes |
| `results_pso_gpu_warmup.csv` | Mesures d’échauffement de la campagne GPU |
| `fig_pso_time_vs_pop.png`, `fig_pso_speedup.png` | Figures des temps et accélérations |
| `sphere_10d_p50_warmup_diagnostic_summary.csv` | Synthèse des trois répétitions du diagnostic Sphere 10D/P50 |
| `sphere_10d_p50_warmup_diagnostic.png` | Figure du diagnostic |

L’archive contient également les fichiers de tests rapides et les mesures détaillées des répétitions du diagnostic.

### Structure de `notebook6a6cfc3811.ipynb`

Le notebook organise toute la chaîne expérimentale : configuration, génération des sources DE et PSO, compilation, tests algorithmiques, tests rapides, campagnes, contrôle des métadonnées, combinaison des quatre implémentations, statistiques, figures et archive.

Les tests précèdent les mesures. Les résultats CSV enregistrent les meilleures fitness, les erreurs, les temps, les identifiants de runs, les graines moteur et les nombres réels d’évaluations. Les JSON associent ces résultats aux sources et exécutables par SHA-256.

Le notebook fourni contient les sorties de la session **`20261026T084908Z_93351482`**, avec les validations technique et de campagne affichées.

### Structure de `de_pso_work/`

Les résultats à quatre implémentations présentés ici proviennent de [de_pso_work/20261026T084908Z_93351482/](de_pso_work/20261026T084908Z_93351482/). Dans cette session :

| Chemin relatif à la session | Rôle |
|---|---|
| `de/src/` | Sources DE, fonctions communes, pilote de campagne et tests CPU/CUDA |
| `de/build/` | Exécutables DE CPU/GPU et exécutables de tests |
| `pso_cpu/src/` | Sources PSO CPU **et GPU**, fonctions communes et pilote de campagne |
| `pso_cpu/build/` | Exécutables `pso_cpu` et `pso_gpu` |
| `results/` | Mesures, synthèses, statistiques, figures, journaux et métadonnées |

Le nom réel du sous-dossier PSO est **`pso_cpu/`**, même s’il contient aussi les sources et l’exécutable GPU.

Dans le sous-dossier `results/`, les principaux fichiers sont :

| Fichier | Contenu |
|---|---|
| `results_de.csv` | 720 mesures DE, dont 360 CPU et 360 GPU |
| `results_pso_cpu.csv`, `results_pso_gpu.csv` | 360 mesures PSO par implémentation |
| `results_de.json`, `results_pso_cpu.json`, `results_pso_gpu.json` | Métadonnées des trois campagnes |
| `results_all_impls.csv` | 1 440 mesures des quatre implémentations |
| `summary_de.csv`, `summary_pso_cpu.csv` | Synthèses des campagnes DE et PSO CPU |
| `summary_all.csv` | 144 synthèses par implémentation et configuration |
| `speedup_all.csv` | 36 comparaisons de temps, avec accélérations DE et PSO |
| `stats_mannwhitney.csv`, `stats_kruskal.csv` | Comparaisons statistiques des erreurs finales |
| `warmup_de_gpu.csv`, `warmup_pso_gpu.csv` | Mesures d’échauffement, séparées des runs mesurés |
| `runs_de.txt`, `runs_pso_cpu.txt`, `runs_pso_gpu.txt` | Journaux des campagnes |
| `algorithm_tests.txt`, `smoke_execution.json`, `validation.json` | Résultats des tests et contrôles de validation |
| `fig_success_rate.png`, `fig_speedup_de.png`, `fig_speedup_pso.png` | Figures de qualité et d’accélération |
| `fig_pso_cpu_time_vs_pop.png` | Temps du PSO CPU selon la population |

Le dossier `de_pso_work/` contient aussi des sessions de tests ou interrompues. Chaque campagne est identifiée par son dossier de session ; les fichiers de progression d’une campagne interrompue ne sont pas des résultats définitifs.

## Protocole et choix d’implémentation

### Fonctions et budget

Les expériences utilisent quatre fonctions décalées, trois dimensions **10, 50, 100**, trois populations **50, 100, 500** et **10 runs par configuration**. Cela représente 36 configurations et 360 runs par implémentation.

| Fonction | Bornes par coordonnée | Optimum théorique, ou biais |
|---|---|---:|
| Sphere | `[-100, 100]` | −450 |
| Rastrigin | `[-5, 5]` | −330 |
| Rosenbrock | `[-100, 100]` | 390 |
| Griewank | `[-600, 600]` | −180 |

Les décalages synthétiques sont générés avec `mt19937_64`, graine `20261027 + func`, dans `[0,8 × lo, 0,8 × hi]`. Cette construction permet de déplacer l’optimum de façon reproductible dans un environnement fixé. L’état des populations et le calcul de fitness utilisent la **double précision**.

Chaque run dispose de `10 000 × D` évaluations. L’initialisation coûte `P` évaluations ; le nombre de générations ou d’itérations suivantes est :

```text
n = floor((10 000 × D) / P) − 1
FEs réels = P × (1 + n)
```

Le comptage inclut donc l’initialisation. À budget fixé, une population plus grande permet moins de générations. L’erreur est `best − biais` ; un run réussit si cette erreur est strictement inférieure à `1e-8`.

### DE : mise à jour générationnelle

Le DE utilise la stratégie **DE/rand/1/bin**, avec **F = 0,5** et **CR = 0,3**. Trois donneurs distincts, excluant l’individu cible, servent à construire le mutant. Le croisement binomial conserve au moins une coordonnée du mutant et les valeurs hors domaine sont ramenées aux bornes.

CPU et GPU lisent une génération immuable pour construire tous les essais avant les remplacements. Ce choix préserve la même sémantique algorithmique des deux côtés. Le CPU utilise un tampon complet d’essais ; le GPU sépare leur construction de l’évaluation et de la sélection dans deux kernels ordonnés sur le même stream.

Un thread DE GPU traite un individu et parcourt ses coordonnées. Cette organisation facilite la lecture du code, mais limite le parallélisme disponible aux petites populations. Le DE CUDA constitue une référence ; les optimisations `cudaDEi` de fusion, mémoire partagée et streams ne sont pas implémentées.

### PSO : mise à jour synchrone

Le PSO utilise une inertie linéairement décroissante de **0,9 à 0,4**, **c1 = c2 = 2** et une vitesse maximale de **0,2 × (hi − lo)**. Le meilleur global est mis à jour après l’évaluation de toute la génération. Les coefficients aléatoires sont tirés indépendamment par particule et coordonnée.

Le CPU stocke les coordonnées par particule. Le GPU utilise un stockage par coordonnée pour regrouper les accès des threads voisins ; il répartit la mise à jour des positions et vitesses sur les éléments de l’essaim, puis évalue les particules et réduit les meilleurs scores sur le GPU.

### Générateurs aléatoires et traçabilité

Le CPU utilise `mt19937_64` et le GPU **cuRAND Philox**, avec des tirages réels en double précision. Les états GPU sont conservés entre les générations.

Dans `gpu-pso.ipynb`, les graines de mesure sont directement `1234 + run`. Dans `notebook6a6cfc3811.ipynb`, les mêmes identifiants de runs sont enregistrés dans `seed`, et une dérivation SplitMix64 propre à chaque implémentation produit `engine_seed`. Cette séparation évite de partager les flux entre les quatre implémentations. Les trajectoires CPU/GPU ne sont pas supposées identiques.

Les sources, exécutables et CSV sont identifiés par SHA-256. La reproductibilité exacte dépend aussi de l’environnement : les distributions de la bibliothèque C++ et les calculs flottants peuvent varier entre plateformes.

### Mesure des temps et échauffement

Le CPU utilise `steady_clock` et le GPU `cudaEvent`. Le périmètre porte sur l’initialisation algorithmique et les générations ; les allocations et les transferts GPU sont exclus. Le générateur CPU est construit avant le chronomètre, tandis que l’initialisation des états cuRAND appartient au kernel initial mesuré. Les accélérations décrivent les temps de calcul, pas les temps applicatifs de bout en bout.

L’échauffement précède les mesures dans le même processus GPU. Les deux conditions visées sont au moins **3 runs** et **2 secondes GPU cumulées**. Le notebook PSO fixe un plafond de 200 runs et enregistre si la cible est atteinte. Le notebook à quatre implémentations fixe un plafond de 10 000 et refuse une campagne dont la cible n’est pas atteinte.

Dans la session `20261026T084908Z_93351482`, les 36 configurations de chaque algorithme GPU atteignent les cibles : 2 554 runs d’échauffement DE GPU et 3 436 PSO GPU, exclus des 1 440 mesures.

## Résultats

### PSO CPU/GPU — `resultats_complets.zip`

Les chiffres suivants sont calculés à partir des fichiers contenus dans cette archive :

| Implémentation | Runs mesurés | Succès au seuil `1e-8` | Taux de succès |
|---|---:|---:|---:|
| PSO CPU | 360 | 48 | 13,33 % |
| PSO GPU | 360 | 43 | 11,94 % |

Dans `speedup_pso.csv`, le GPU est plus rapide dans **36 configurations sur 36**. L’accélération médiane est **3,67×**, avec des rapports compris entre **1,05× et 20,71×**.

Les métadonnées indiquent que 31 configurations GPU sur 36 ont atteint la cible d’échauffement ; cinq ont atteint le plafond de 200 runs avant les 2 secondes. Les trois répétitions du diagnostic Sphere 10D/P50 atteignent toutes la cible : leurs temps moyens mesurés sont **18,0090 ms**, **18,2413 ms** et **17,9884 ms**, avec 100 % de réussite sur ce cas.

### DE et PSO CPU/GPU — `de_pso_work/20261026T084908Z_93351482/results/`

La campagne contient 1 440 mesures, avec les taux de succès suivants :

| Implémentation | Succès / 360 | Taux de succès |
|---|---:|---:|
| DE CPU | 140 | 38,89 % |
| DE GPU | 140 | 38,89 % |
| PSO CPU | 46 | 12,78 % |
| PSO GPU | 43 | 11,94 % |

Ces taux résument toute la grille ; ils ne décrivent pas à eux seuls la qualité sur chaque problème. Sur Rastrigin, les erreurs médianes agrégées sont plus faibles pour le PSO malgré son moindre taux de réussite au seuil strict. **Aucune implémentation n’atteint `1e-8` sur Rosenbrock.**

| Algorithme | Configurations où le GPU est plus rapide | Accélération médiane | Minimum–maximum |
|---|---:|---:|---:|
| DE | 27 / 36 | 1,50× | 0,68×–9,75× |
| PSO | 36 / 36 | 4,08× | 1,17×–25,21× |

Les rapports comparent les temps moyens CPU et GPU de chaque configuration ; une valeur inférieure à 1 signifie que le GPU est plus lent. Les meilleures accélérations observées concernent P500. Les 36 rapports sont détaillés dans [speedup_all.csv](de_pso_work/20261026T084908Z_93351482/results/speedup_all.csv).

Les résultats par configuration — moyenne, écart-type, médiane, taux de succès et temps — figurent dans [summary_all.csv](de_pso_work/20261026T084908Z_93351482/results/summary_all.csv). Les figures associées sont [fig_success_rate.png](de_pso_work/20261026T084908Z_93351482/results/fig_success_rate.png), [fig_speedup_de.png](de_pso_work/20261026T084908Z_93351482/results/fig_speedup_de.png) et [fig_speedup_pso.png](de_pso_work/20261026T084908Z_93351482/results/fig_speedup_pso.png).

### Analyse statistique des quatre implémentations

Les erreurs finales sont comparées séparément par configuration, sans appariement fondé sur le numéro de graine :

- **Mann–Whitney bilatéral** : 144 comparaisons deux à deux, p-values par permutation avec 99 999 rééchantillonnages reproductibles ; correction de Holm globale sur les 144 tests.
- **Kruskal–Wallis** : 36 comparaisons des quatre implémentations, approximation asymptotique avec correction des ex æquo ; Holm global sur les 36 tests.
- Seuil : p-value corrigée strictement inférieure à `0,05`.

| Comparaison | Configurations significatives / 36 | Sens des différences détectées |
|---|---:|---|
| DE CPU / DE GPU | 0 | Aucune différence détectée après Holm |
| PSO CPU / PSO GPU | 0 | Aucune différence détectée après Holm |
| DE CPU / PSO CPU | 23 | 15 en faveur du DE, 8 en faveur du PSO |
| DE GPU / PSO GPU | 25 | 17 en faveur du DE, 8 en faveur du PSO |

Kruskal–Wallis détecte une différence dans **28 configurations sur 36** après correction. Les détails sont dans [stats_mannwhitney.csv](de_pso_work/20261026T084908Z_93351482/results/stats_mannwhitney.csv) et [stats_kruskal.csv](de_pso_work/20261026T084908Z_93351482/results/stats_kruskal.csv).

Avec 10 runs par groupe, l’absence de différence significative ne prouve pas l’équivalence CPU/GPU. Ces tests portent sur la qualité finale, pas sur la significativité des différences de temps.

## Exécution et portée des résultats

Pour exécuter le notebook à quatre implémentations, utiliser Python avec `numpy`, `pandas`, `matplotlib`, SciPy ≥ 1.13 et un compilateur C++14. Les parties CUDA nécessitent un GPU NVIDIA et `nvcc`. Sur Kaggle, activer le GPU puis lancer toutes les cellules dans l’ordre :

```python
RUN_DE_GPU = True
RUN_FULL_SWEEPS = True
WARMUP_MIN_RUNS, WARMUP_SECONDS, WARMUP_MAX_RUNS = 3, 2.0, 10000
```

`RUN_FULL_SWEEPS = False` limite l’exécution aux compilations et tests. `RUN_DE_GPU = False` permet les versions CPU, sans analyse complète des quatre implémentations.

La session documentée utilise Linux x86_64, g++ 13.3, CUDA 12.8 et l’architecture `sm_75`. Deux Tesla T4 sont visibles, mais les programmes utilisent le périphérique CUDA par défaut ; ils ne réalisent pas un calcul multi-GPU.

Les tests CPU/CUDA et les validations du notebook vérifient la cohérence de l’implémentation et des fichiers. Ils ne garantissent pas la convergence sur toutes les fonctions. Les décalages sont synthétiques, et les temps dépendent du matériel ainsi que du périmètre de mesure ; ces choix doivent être pris en compte dans toute interprétation des résultats.
