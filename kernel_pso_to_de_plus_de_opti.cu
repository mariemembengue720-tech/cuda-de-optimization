#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include <curand_kernel.h>
#include <vector>
#include <algorithm>

#include "kernel.h"

    
// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA, architecture DE BASIQUE
// (Figure 1 de ImprovedDE.pdf : kernel(I), kernel(E), kernel(P),
// kernel(M), kernel(C), kernel(R) séparés — PAS encore l'architecture
// fusionnée cudaDEi de la Figure 2, qui viendra dans une étape
// ultérieure une fois cette version de base validée).
//
// Rien au-dessus de ce bloc n'a été modifié, à l'exception des 3
// #include ajoutés en haut du fichier (curand_kernel.h pour les
// tirages aléatoires device, vector/algorithm pour l'orchestration
// hôte de cuda_de).
// ====================================================================

// Petite macro de vérification d'erreur CUDA — utile en phase de
// validation (§11 du sujet : "absence d'accès mémoire hors limites").
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "Erreur CUDA %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// ------------------------------------------------------------------
// Fitness côté device — dupliquée de compute_fitness() (kernel.cpp)
// car une fonction __device__ ne peut pas appeler une fonction host.
// Mêmes formules, MÊME bug Griewank conservé (produit initialisé à 0
// au lieu de 1, voir README §5) : on ne corrige rien sans validation
// explicite. Si les formules de benchmark changent un jour, il faudra
// modifier CETTE fonction ET compute_fitness() dans kernel.cpp — les
// deux copies ne sont pas partagées automatiquement.
// dim et obj_func sont des paramètres (pas des constantes globales)
// pour rester cohérent avec de_fitness_function() côté host.
// ------------------------------------------------------------------
__device__ float de_device_fitness(const float x[], int dim, int obj_func) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (obj_func) {
        case 0: {
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[dim-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < dim-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < dim; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        }
        case 2:
            for (int i = 0; i < dim-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < dim; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow((float)(i+1), 0.5f));
            }
            res = somme - produit + 1 - 180;
            break;
        case 4:
            for (int i = 0; i < dim; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

// ------------------------------------------------------------------
// Architecture retenue pour cette version de base : 1 thread = 1
// individu, pour TOUS les kernels (I, E, P, M, C, R). Chaque thread
// boucle sur ses `dim` dimensions en interne.
//
// Justification (pas supposée par défaut, comme demandé §17) :
// l'évaluation de fitness de nos 5 fonctions benchmark est
// intrinsèquement séquentielle sur les dimensions (Rosenbrock lit
// x[i] ET x[i+1] pour chaque terme, Griewank accumule un produit sur
// toutes les dimensions) : la décomposer sur plusieurs threads par
// individu demanderait une réduction parallèle, qui est précisément
// le genre d'optimisation que l'article réserve à la version
// improved (occupancy, shared memory). Ici, 1 thread/individu est le
// point de départ le plus simple et le plus facile à valider — c'est
// aussi celui du kernel(P)/kernel(M)/kernel(C)/kernel(R), qui
// manipulent chacun un individu entier (indices r1/r2/r3, vecteur
// mutant, crossover, remplacement).
// ------------------------------------------------------------------

// kernel(I) — Population Initialization. Initialise aussi l'état
// curand de chaque individu (réutilisé par kernel(P) et kernel(C)
// aux appels suivants : load state -> tirages -> store state, pour
// ne pas repartir de la même graine à chaque kernel).
__global__ void kernelInitPopulation(float* X, curandState* states, int dim, int pop,
                                       float rangeMin, float rangeMax, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    curandState localState;
    curand_init(seed, i, 0, &localState);

    for (int j = 0; j < dim; j++) {
        float u = curand_uniform(&localState); // (0,1]
        X[i*dim + j] = rangeMin + u * (rangeMax - rangeMin);
    }

    states[i] = localState;
}

// kernel(E) — Function Evaluation. Réutilisé deux fois par génération
// (évaluation initiale, puis évaluation des trial vectors).
__global__ void kernelEvaluate(const float* X, float* fit, int dim, int pop, int obj_func)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    fit[i] = de_device_fitness(&X[i*dim], dim, obj_func);
}

// kernel(P) — Mutation Preparation : tire r1, r2, r3 mutuellement
// distincts et != i (même contrainte que la version CPU, §4 du sujet).
__global__ void kernelMutationPrep(int* r1, int* r2, int* r3, curandState* states, int pop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    curandState localState = states[i];

    int a, b, c;
    do { a = curand(&localState) % pop; } while (a == i);
    do { b = curand(&localState) % pop; } while (b == i || b == a);
    do { c = curand(&localState) % pop; } while (c == i || c == a || c == b);

    r1[i] = a; r2[i] = b; r3[i] = c;
    states[i] = localState;
}

// kernel(M) — DE Mutation : V_i = X_r1 + F * (X_r2 - X_r3)
__global__ void kernelMutation(const float* X, const int* r1, const int* r2, const int* r3,
                                 float* V, int dim, int pop, float F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    int a = r1[i], b = r2[i], c = r3[i];
    for (int j = 0; j < dim; j++) {
        V[i*dim + j] = X[a*dim + j] + F * (X[b*dim + j] - X[c*dim + j]);
    }
}

// kernel(C) — DE Crossover binomial, avec j_rand forcé (au moins une
// composante du mutant garantie dans le trial vector, §4 du sujet).
__global__ void kernelCrossover(const float* X, const float* V, float* U, curandState* states,
                                  int dim, int pop, float CR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    curandState localState = states[i];
    int j_rand = curand(&localState) % dim;

    for (int j = 0; j < dim; j++) {
        float rand_j = curand_uniform(&localState);
        U[i*dim + j] = (rand_j <= CR || j == j_rand) ? V[i*dim + j] : X[i*dim + j];
    }

    states[i] = localState;
}

// kernel(R) — Replacement : garde le trial si meilleur, sinon la
// cible, écrit dans la population de la génération suivante
// (double buffering, §12 du sujet — chaque thread lit uniquement
// X/U, écrit uniquement X_next : aucune dépendance incorrecte entre
// threads pendant cette génération).
__global__ void kernelReplacement(const float* X, const float* U, const float* fitX, const float* fitU,
                                    float* X_next, float* fit_next, int dim, int pop)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    if (fitU[i] < fitX[i]) {
        for (int j = 0; j < dim; j++) X_next[i*dim + j] = U[i*dim + j];
        fit_next[i] = fitU[i];
    } else {
        for (int j = 0; j < dim; j++) X_next[i*dim + j] = X[i*dim + j];
        fit_next[i] = fitX[i];
    }
}

// ------------------------------------------------------------------
// Orchestration hôte — équivalent DE de cuda_pso() ci-dessus, mais
// avec une différence architecturale volontaire et importante :
//
// Le PSO fourni fait un aller-retour CPU<->GPU (copie D->H de pBests
// + recalcul de gBest côté host + copie H->D de gBest) À CHAQUE
// itération (voir README §2 / diagramme Phase 1) — c'est exactement
// le trafic mémoire à faible débit que critique ImprovedDE.pdf.
//
// Ici, X_current et X_next ne sont QUE des pointeurs device, échangés
// par un simple std::swap() de pointeurs (aucune copie mémoire) à
// chaque génération. Le seul aller-retour CPU<->GPU de toute la
// boucle est la lecture de la graine curand au tout début — aucune
// copie intermédiaire. Le résultat final n'est rapatrié qu'UNE SEULE
// fois, à la toute fin, pour y chercher le meilleur individu côté
// host (recherche linéaire simple sur `pop` éléments, pas critique).
// ------------------------------------------------------------------
extern "C" void cuda_de(int dim, int pop, int obj_func, unsigned long long seed,
                          float* out_gBest, long long* out_evaluations)
{
    size_t vecBytes = (size_t) pop * dim * sizeof(float);
    size_t popBytes = (size_t) pop * sizeof(float);
    size_t idxBytes = (size_t) pop * sizeof(int);

    float *d_X, *d_Xnext, *d_V, *d_U, *d_fitX, *d_fitU, *d_fitXnext;
    int *d_r1, *d_r2, *d_r3;
    curandState *d_states;

    CUDA_CHECK(cudaMalloc(&d_X,        vecBytes));
    CUDA_CHECK(cudaMalloc(&d_Xnext,    vecBytes));
    CUDA_CHECK(cudaMalloc(&d_V,        vecBytes));
    CUDA_CHECK(cudaMalloc(&d_U,        vecBytes));
    CUDA_CHECK(cudaMalloc(&d_fitX,     popBytes));
    CUDA_CHECK(cudaMalloc(&d_fitU,     popBytes));
    CUDA_CHECK(cudaMalloc(&d_fitXnext, popBytes));
    CUDA_CHECK(cudaMalloc(&d_r1, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_r2, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_r3, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_states, (size_t) pop * sizeof(curandState)));

    // Configuration bloc/grille simple et fixe pour cette version de
    // base (division en float pour éviter le bug de troncature entière
    // repéré dans le blocksNum du PSO fourni, ligne ~153 de ce fichier).
    // L'auto-tuning de l'occupancy (façon cudaDEi) viendra à l'étape
    // suivante, une fois cette version validée.
    int threadsPerBlock = 64;
    int blocksNum = (int) ceil((float) pop / threadsPerBlock);

    kernelInitPopulation<<<blocksNum, threadsPerBlock>>>(d_X, d_states, dim, pop,
                                                           START_RANGE_MIN, START_RANGE_MAX, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernelEvaluate<<<blocksNum, threadsPerBlock>>>(d_X, d_fitX, dim, pop, obj_func);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    long long fe = pop; // évaluation initiale de toute la population
    long long max_fes = FE_BUDGET_PER_DIM * (long long) dim;

    while (fe + pop <= max_fes) {
        kernelMutationPrep<<<blocksNum, threadsPerBlock>>>(d_r1, d_r2, d_r3, d_states, pop);
        CUDA_CHECK(cudaGetLastError());

        kernelMutation<<<blocksNum, threadsPerBlock>>>(d_X, d_r1, d_r2, d_r3, d_V, dim, pop, DE_F);
        CUDA_CHECK(cudaGetLastError());

        kernelCrossover<<<blocksNum, threadsPerBlock>>>(d_X, d_V, d_U, d_states, dim, pop, DE_CR);
        CUDA_CHECK(cudaGetLastError());

        kernelEvaluate<<<blocksNum, threadsPerBlock>>>(d_U, d_fitU, dim, pop, obj_func);
        CUDA_CHECK(cudaGetLastError());
        fe += pop;

        kernelReplacement<<<blocksNum, threadsPerBlock>>>(d_X, d_U, d_fitX, d_fitU,
                                                            d_Xnext, d_fitXnext, dim, pop);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_X, d_Xnext);       // échange de pointeurs uniquement — pas de cudaMemcpy
        std::swap(d_fitX, d_fitXnext);
    }

    // Rapatriement final UNIQUE (contrairement au PSO qui rapatrie
    // pBests à chaque itération) : on cherche le meilleur individu
    // côté host, une seule fois, sur la population finale.
    std::vector<float> h_X(pop * dim);
    std::vector<float> h_fit(pop);
    CUDA_CHECK(cudaMemcpy(h_X.data(),   d_X,    vecBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fitX, popBytes, cudaMemcpyDeviceToHost));

    int best = 0;
    for (int i = 1; i < pop; i++)
        if (h_fit[i] < h_fit[best]) best = i;

    for (int j = 0; j < dim; j++)
        out_gBest[j] = h_X[best*dim + j];

    cudaFree(d_X); cudaFree(d_Xnext); cudaFree(d_V); cudaFree(d_U);
    cudaFree(d_fitX); cudaFree(d_fitU); cudaFree(d_fitXnext);
    cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_r3);
    cudaFree(d_states);

    *out_evaluations = fe;
}

// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA avec FUSION DE KERNELS
// Architecture inspirée de la Figure 2 de ImprovedDE.pdf (cudaDEᵢ) :
// kernel(I)+kernel(E) → kernel(IE) ; kernel(M)+kernel(C)+kernel(E)+
// kernel(R) → kernel(MCER) ; kernel(P) reste séparé (comme dans
// l'article — utile pour un futur recouvrement par streams avec MCER,
// non implémenté ici).
//
// Rien au-dessus n'est modifié : cuda_de() (version de base, 6 kernels
// séparés) reste intacte pour comparaison directe. Cette section
// ajoute cuda_de_fused() à côté.
//
// IMPORTANT — ce n'est PAS encore l'implémentation complète de
// cudaDEᵢ : pas de mémoire partagée, pas de streams, pas de
// configuration bloc/grille auto-déterminée selon l'occupancy. Cette
// étape se limite à la fusion des kernels (Figure 2), qui est déjà ce
// qui réduit le plus directement le nombre de lancements de kernels —
// la mémoire partagée / streams / auto-tuning sont les optimisations
// suivantes, réservées à une étape ultérieure.
// ====================================================================

// Plafond de sécurité pour le trial vector en mémoire LOCALE au
// thread : en CUDA, un tableau local doit avoir une taille connue à
// la compilation (pas de VLA en device code). 256 est largement
// au-dessus des D=10/50/100 du protocole officiel. Dépasser cette
// valeur est vérifié explicitement côté host (voir cuda_de_fused) et
// arrête le programme proprement plutôt que de corrompre la mémoire.
#define DE_MAX_DIM 256

// kernel(IE) — Population Initialization + Function Evaluation
// fusionnés (Figure 2). Un seul lancement au lieu de deux : l'individu
// vient d'être écrit par CE thread dans le même appel de kernel, donc
// aucune raison de faire un aller-retour supplémentaire en mémoire
// globale entre les deux étapes.
__global__ void kernelInitEvaluate(float* X, float* fit, curandState* states, int dim, int pop, int obj_func,
                                     float rangeMin, float rangeMax, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    curandState localState;
    curand_init(seed, i, 0, &localState);

    for (int j = 0; j < dim; j++) {
        float u = curand_uniform(&localState);
        X[i*dim + j] = rangeMin + u * (rangeMax - rangeMin);
    }
    states[i] = localState;

    fit[i] = de_device_fitness(&X[i*dim], dim, obj_func);
}

// kernel(MCER) — Mutation + Crossover + Function Evaluation (des
// trials) + Replacement, fusionnés en un seul kernel (Figure 2).
//
// Différence importante avec la version de base : le trial vector U
// n'est plus jamais écrit en mémoire globale (buffers d_V et d_U
// supprimés) — il est construit dans un tableau LOCAL au thread
// (`trial[DE_MAX_DIM]`), utilisé pour l'évaluation, puis jeté. Seul le
// résultat final (U si meilleur, sinon X inchangé) est écrit dans
// X_next. Ça réduit à la fois le nombre de lancements de kernels (un
// seul au lieu de quatre) ET le trafic mémoire global (deux buffers
// entiers en moins, plus les allers-retours associés).
__global__ void kernelMutateCrossoverEvalReplace(const float* X, const int* r1, const int* r2, const int* r3,
                                                    curandState* states, const float* fitX,
                                                    float* X_next, float* fit_next,
                                                    int dim, int pop, int obj_func, float F, float CR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pop) return;

    curandState localState = states[i];
    int a = r1[i], b = r2[i], c = r3[i];
    int j_rand = curand(&localState) % dim;

    float trial[DE_MAX_DIM];
    for (int j = 0; j < dim; j++) {
        float mutant_j = X[a*dim + j] + F * (X[b*dim + j] - X[c*dim + j]);
        float rand_j = curand_uniform(&localState);
        trial[j] = (rand_j <= CR || j == j_rand) ? mutant_j : X[i*dim + j];
    }
    states[i] = localState;

    float f_trial = de_device_fitness(trial, dim, obj_func);

    if (f_trial < fitX[i]) {
        for (int j = 0; j < dim; j++) X_next[i*dim + j] = trial[j];
        fit_next[i] = f_trial;
    } else {
        for (int j = 0; j < dim; j++) X_next[i*dim + j] = X[i*dim + j];
        fit_next[i] = fitX[i];
    }
}

// ------------------------------------------------------------------
// Orchestration hôte — version fusionnée. Même logique globale que
// cuda_de() (double buffering par swap de pointeurs, un seul
// rapatriement final), mais 2 lancements de kernels par génération
// (kernel(P), kernel(MCER)) au lieu de 5, et 4 buffers device au lieu
// de 7 (plus de d_V, d_U, d_fitU).
// ------------------------------------------------------------------
extern "C" void cuda_de_fused(int dim, int pop, int obj_func, unsigned long long seed,
                                float* out_gBest, long long* out_evaluations)
{
    if (dim > DE_MAX_DIM) {
        fprintf(stderr,
                "Erreur : dim=%d > DE_MAX_DIM=%d pour cuda_de_fused "
                "(mémoire locale par thread bornée à la compilation).\n",
                dim, DE_MAX_DIM);
        exit(1);
    }

    size_t vecBytes = (size_t) pop * dim * sizeof(float);
    size_t popBytes = (size_t) pop * sizeof(float);
    size_t idxBytes = (size_t) pop * sizeof(int);

    float *d_X, *d_Xnext, *d_fitX, *d_fitXnext;
    int *d_r1, *d_r2, *d_r3;
    curandState *d_states;

    CUDA_CHECK(cudaMalloc(&d_X,        vecBytes));
    CUDA_CHECK(cudaMalloc(&d_Xnext,    vecBytes));
    CUDA_CHECK(cudaMalloc(&d_fitX,     popBytes));
    CUDA_CHECK(cudaMalloc(&d_fitXnext, popBytes));
    CUDA_CHECK(cudaMalloc(&d_r1, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_r2, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_r3, idxBytes));
    CUDA_CHECK(cudaMalloc(&d_states, (size_t) pop * sizeof(curandState)));

    int threadsPerBlock = 64;
    int blocksNum = (int) ceil((float) pop / threadsPerBlock);

    kernelInitEvaluate<<<blocksNum, threadsPerBlock>>>(d_X, d_fitX, d_states, dim, pop, obj_func,
                                                         START_RANGE_MIN, START_RANGE_MAX, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    long long fe = pop;
    long long max_fes = FE_BUDGET_PER_DIM * (long long) dim;

    while (fe + pop <= max_fes) {
        kernelMutationPrep<<<blocksNum, threadsPerBlock>>>(d_r1, d_r2, d_r3, d_states, pop);
        CUDA_CHECK(cudaGetLastError());

        kernelMutateCrossoverEvalReplace<<<blocksNum, threadsPerBlock>>>(
            d_X, d_r1, d_r2, d_r3, d_states, d_fitX, d_Xnext, d_fitXnext, dim, pop, obj_func, DE_F, DE_CR);
        CUDA_CHECK(cudaGetLastError());
        fe += pop;
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_X, d_Xnext);
        std::swap(d_fitX, d_fitXnext);
    }

    std::vector<float> h_X(pop * dim);
    std::vector<float> h_fit(pop);
    CUDA_CHECK(cudaMemcpy(h_X.data(),   d_X,    vecBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fitX, popBytes, cudaMemcpyDeviceToHost));

    int best = 0;
    for (int i = 1; i < pop; i++)
        if (h_fit[i] < h_fit[best]) best = i;

    for (int j = 0; j < dim; j++)
        out_gBest[j] = h_X[best*dim + j];

    cudaFree(d_X); cudaFree(d_Xnext);
    cudaFree(d_fitX); cudaFree(d_fitXnext);
    cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_r3);
    cudaFree(d_states);

    *out_evaluations = fe;
}

// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA OPTIMISÉ (mémoire partagée + streams
// + occupancy) — architecture réellement inspirée de cudaDEᵢ (Figure 2)
//
// Rien au-dessus n'est modifié : cuda_de() et cuda_de_fused() restent
// intactes pour comparaison à trois. kernelMutationPrep (déjà défini
// plus haut pour la version de base) est RÉUTILISÉ tel quel pour
// kernel(P) — sa logique ne change pas, seule sa configuration de
// lancement (threadsPerBlock) change côté host, voir plus bas.
//
// CHANGEMENT DE MAPPING DE THREADS pour kernel(IE) et kernel(MCER) :
// 1 bloc = 1 individu, 1 thread = 1 gène (blockDim.x == dim). C'est le
// mapping réellement utilisé par l'article ("each population member
// element is assigned a unique thread") — nécessaire pour que la
// mémoire partagée serve à quelque chose : avec l'ancien mapping
// (1 thread = 1 individu), r1/r2/r3 diffèrent par thread au sein d'un
// bloc, donc rien n'est réellement partageable entre threads voisins.
// Conséquence assumée : blockDim.x n'est plus un paramètre de
// performance libre pour ces deux kernels (imposé par dim) — seul
// kernel(P) garde un blockDim.x réglable, et c'est le seul kernel où
// l'auto-tuning d'occupancy (cudaOccupancyMaxPotentialBlockSize)
// s'applique dans cette version.
//
// LIMITATION EXPLICITE : cette version ne supporte PAS objectif=0
// (Levy) — ce n'est pas une des 4 fonctions du protocole officiel, et
// sa structure (dépendance à x[0] ET x[dim-1] simultanément, en plus
// du voisin x[i+1]) aurait ajouté une complexité de réduction
// significative pour zéro bénéfice vis-à-vis du sujet. cuda_de_optimized
// refuse explicitement objectif=0 avec un message clair plutôt que de
// donner un résultat faux silencieusement.
// ====================================================================

// ------------------------------------------------------------------
// Réduction de somme en mémoire partagée, pour un bloc de taille
// quelconque (pas nécessairement une puissance de 2 — nos D=10/50/100
// n'en sont pas). Appariement par paires (j, j+half) avec
// half = ceil(m/2) à chaque tour ; l'éventuel élément non apparié
// (quand m est impair) est simplement reporté tel quel au tour
// suivant, où il finira par être apparié. sdata[0..n-1] doit être
// rempli par TOUS les threads actifs avant l'appel (le premier
// __syncthreads() ci-dessous protège ce remplissage).
// ------------------------------------------------------------------
__device__ float blockReduceSum(float* sdata, int n) {
    __syncthreads();
    int m = n;
    while (m > 1) {
        int half = (m + 1) / 2;
        if (threadIdx.x < m - half) {
            sdata[threadIdx.x] += sdata[threadIdx.x + half];
        }
        __syncthreads();
        m = half;
    }
    return sdata[0];   // identique pour tous les threads du bloc
}

// Terme individuel calculé par le thread en charge de la dimension j,
// pour le vecteur s_vec (mémoire partagée). Reprend exactement les
// mêmes formules que compute_fitness()/de_device_fitness() (aucun
// changement de benchmark), réorganisées pour être calculées un
// terme à la fois par thread plutôt qu'en boucle séquentielle.
__device__ float dePerThreadTerm(const float* s_vec, int j, int dim, int obj_func) {
    switch (obj_func) {
        case 1: { // Rastrigin
            float zi = s_vec[j];
            return zi*zi - 10.0f*cosf(2.0f*phi*zi) + 10.0f;
        }
        case 2: { // Rosenbrock — a besoin du voisin j+1 (mémoire partagée)
            if (j >= dim - 1) return 0.0f;
            float zi   = s_vec[j]   + 1.0f;
            float zip1 = s_vec[j+1] + 1.0f;
            return 100.0f*(zi*zi - zip1)*(zi*zi - zip1) + (zi-1.0f)*(zi-1.0f);
        }
        case 3: { // Griewank — terme somme uniquement ; le terme produit
                   // est figé à 0 (bug connu conservé, voir README §5 —
                   // `produit` initialisé à 0 dans le code fourni, donc
                   // toujours nul quel que soit x)
            float zi = s_vec[j];
            return (zi*zi) / 4000.0f;
        }
        case 4: { // Sphere
            float zi = s_vec[j];
            return zi*zi;
        }
        default:
            return 0.0f; // objectif non supporté (Levy) — cuda_de_optimized le refuse avant d'en arriver là
    }
}

// Fitness complète d'un vecteur partagé s_vec[0..dim-1] : chaque thread
// calcule son terme, réduction en somme, puis offset final identique à
// compute_fitness(). Retourne la même valeur à tous les threads du bloc
// (blockReduceSum le garantit) — chaque thread peut donc décider
// indépendamment quoi écrire, sans re-synchronisation supplémentaire.
__device__ float blockFitness(const float* s_vec, float* s_reduce, int dim, int obj_func) {
    int j = threadIdx.x;   // invariant : blockDim.x == dim pour ces kernels
    s_reduce[j] = dePerThreadTerm(s_vec, j, dim, obj_func);
    float total = blockReduceSum(s_reduce, dim);

    switch (obj_func) {
        case 1: return total - 330.0f;
        case 2: return total + 390.0f;
        case 3: return total - 0.0f + 1.0f - 180.0f;  // -0.0f : terme produit toujours nul (bug connu)
        case 4: return total - 450.0f;
        default: return total;
    }
}

// kernel(IE) optimisé — 1 bloc = 1 individu, 1 thread = 1 gène.
// Mémoire partagée dynamique (2*dim floats) : [0,dim)=s_X (vecteur en
// cours d'init), [dim,2*dim)=s_reduce (scratch réduction). États
// curand PAR GÈNE (pas par individu) : chaque (individu,gène) a son
// propre flux aléatoire indépendant, réutilisé ensuite par
// kernel(MCER) pour le tirage du crossover par dimension.
__global__ void kernelInitEvaluateOpt(float* X, float* fit, curandState* geneStates,
                                        int dim, int pop, int obj_func,
                                        float rangeMin, float rangeMax, unsigned long long seed)
{
    extern __shared__ float smem[];
    float* s_X      = smem;
    float* s_reduce = smem + dim;

    int i = blockIdx.x;    // individu
    int j = threadIdx.x;   // gène / dimension
    int geneIdx = i*dim + j;

    curandState localState;
    curand_init(seed, geneIdx, 0, &localState);

    float u = curand_uniform(&localState);
    float val = rangeMin + u*(rangeMax - rangeMin);
    s_X[j] = val;
    X[geneIdx] = val;
    geneStates[geneIdx] = localState;

    __syncthreads();

    float f = blockFitness(s_X, s_reduce, dim, obj_func);
    if (j == 0) fit[i] = f;
}

// kernel(MCER) optimisé — 1 bloc = 1 individu, 1 thread = 1 gène.
// Mémoire partagée dynamique (3*dim floats) : s_X (vecteur courant,
// pour le "sinon" du crossover), s_trial (vecteur d'essai en
// construction, lu par les voisins pour Rosenbrock avant d'être
// écrasé par la réduction), s_reduce (scratch réduction). r1/r2/r3 et
// j_rand sont communs aux `dim` threads du bloc (même individu) :
// chargés UNE SEULE FOIS par le thread 0 puis diffusés en mémoire
// partagée statique (s_r1/s_r2/s_r3/s_jrand), au lieu d'être relus
// indépendamment par chaque thread-gène.
//
// États curand séparés en DEUX tableaux distincts (geneStates pour le
// tirage du crossover par dimension, jrandStates pour le tirage de
// j_rand par individu) : ce découpage est nécessaire pour éviter toute
// racecondition avec kernel(P), qui tourne en parallèle sur un stream
// différent et touche SON PROPRE tableau d'états (indivStates,
// disjoint de geneStates/jrandStates) — voir cuda_de_optimized.
__global__ void kernelMCEROpt(const float* X, const int* r1, const int* r2, const int* r3,
                                curandState* geneStates, curandState* jrandStates,
                                const float* fitX, float* X_next, float* fit_next,
                                int dim, int pop, int obj_func, float F, float CR)
{
    extern __shared__ float smem[];
    float* s_X      = smem;
    float* s_trial  = smem + dim;
    float* s_reduce = smem + 2*dim;

    int i = blockIdx.x;
    int j = threadIdx.x;
    int geneIdx = i*dim + j;

    __shared__ int s_r1, s_r2, s_r3, s_jrand;
    if (j == 0) {
        s_r1 = r1[i]; s_r2 = r2[i]; s_r3 = r3[i];
        curandState jState = jrandStates[i];
        s_jrand = curand(&jState) % dim;
        jrandStates[i] = jState;
    }

    s_X[j] = X[geneIdx];
    __syncthreads();   // s_r1/s_r2/s_r3/s_jrand et s_X prêts pour tout le bloc

    curandState localState = geneStates[geneIdx];
    float mutant_j = X[s_r1*dim + j] + F * (X[s_r2*dim + j] - X[s_r3*dim + j]);
    float rand_j = curand_uniform(&localState);
    geneStates[geneIdx] = localState;

    s_trial[j] = (rand_j <= CR || j == s_jrand) ? mutant_j : s_X[j];
    __syncthreads();   // s_trial complet avant qu'un voisin en lise j+1 (Rosenbrock, dans blockFitness)

    float f_trial = blockFitness(s_trial, s_reduce, dim, obj_func);   // même valeur pour tout le bloc
    bool better = (f_trial < fitX[i]);

    if (j == 0) fit_next[i] = better ? f_trial : fitX[i];
    X_next[geneIdx] = better ? s_trial[j] : s_X[j];
}

// ------------------------------------------------------------------
// Orchestration hôte — version optimisée : mémoire partagée (ci-dessus)
// + streams (recouvrement kernel(P) génération g+1 avec kernel(MCER)
// génération g, exactement comme décrit Figure 2 — kernel(P) ne lit
// jamais X, donc aucune dépendance avec kernel(MCER)) + occupancy
// réelle pour kernel(P). Double buffering des indices r1/r2/r3
// (tableaux [0]/[1]) nécessaire pour que stream1 puisse écrire les
// indices de la génération g+1 PENDANT que stream0 utilise encore ceux
// de la génération g dans kernel(MCER).
// ------------------------------------------------------------------
extern "C" void cuda_de_optimized(int dim, int pop, int obj_func, unsigned long long seed,
                                    float* out_gBest, long long* out_evaluations)
{
    if (obj_func == 0) {
        fprintf(stderr,
                "Erreur : cuda_de_optimized ne supporte pas Levy (objectif=0) — "
                "hors protocole officiel. Utilise mode=gpu ou gpu-fused pour Levy.\n");
        exit(1);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (dim > prop.maxThreadsPerBlock) {
        fprintf(stderr,
                "Erreur : dim=%d depasse maxThreadsPerBlock=%d de ce GPU "
                "(architecture 1 bloc=1 individu, 1 thread=1 gene impossible ici).\n",
                dim, prop.maxThreadsPerBlock);
        exit(1);
    }

    size_t sharedIE   = (size_t) 2 * dim * sizeof(float);
    size_t sharedMCER = (size_t) 3 * dim * sizeof(float);
    if (sharedMCER > prop.sharedMemPerBlock) {
        fprintf(stderr,
                "Erreur : memoire partagee requise (%zu octets) depasse sharedMemPerBlock=%zu de ce GPU.\n",
                sharedMCER, prop.sharedMemPerBlock);
        exit(1);
    }

    size_t vecBytes  = (size_t) pop * dim * sizeof(float);
    size_t popBytes  = (size_t) pop * sizeof(float);
    size_t idxBytes  = (size_t) pop * sizeof(int);
    size_t geneStateBytes  = (size_t) pop * dim * sizeof(curandState);
    size_t indivStateBytes = (size_t) pop * sizeof(curandState);

    float *d_X, *d_Xnext, *d_fitX, *d_fitXnext;
    int *d_r1[2], *d_r2[2], *d_r3[2];
    curandState *d_geneStates, *d_indivStates, *d_jrandStates;

    CUDA_CHECK(cudaMalloc(&d_X,        vecBytes));
    CUDA_CHECK(cudaMalloc(&d_Xnext,    vecBytes));
    CUDA_CHECK(cudaMalloc(&d_fitX,     popBytes));
    CUDA_CHECK(cudaMalloc(&d_fitXnext, popBytes));
    for (int k = 0; k < 2; k++) {
        CUDA_CHECK(cudaMalloc(&d_r1[k], idxBytes));
        CUDA_CHECK(cudaMalloc(&d_r2[k], idxBytes));
        CUDA_CHECK(cudaMalloc(&d_r3[k], idxBytes));
    }
    CUDA_CHECK(cudaMalloc(&d_geneStates,  geneStateBytes));
    CUDA_CHECK(cudaMalloc(&d_indivStates, indivStateBytes));
    CUDA_CHECK(cudaMalloc(&d_jrandStates, indivStateBytes));

    // threadsPerBlock pour kernel(P), seul kernel de cette version où
    // ce choix reste libre — calcul reel d'occupancy via l'API CUDA
    // plutot qu'une valeur fixee a la main (64 dans les versions
    // precedentes).
    int minGridSizeP, threadsPerBlockP;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSizeP, &threadsPerBlockP,
                                                    kernelMutationPrep, 0, pop));
    int blocksNumP = (int) ceil((float) pop / threadsPerBlockP);

    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    // --- kernel(IE) : population initiale + evaluation, une fois ---
    kernelInitEvaluateOpt<<<pop, dim, sharedIE>>>(d_X, d_fitX, d_geneStates, dim, pop, obj_func,
                                                    START_RANGE_MIN, START_RANGE_MAX, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Premiere preparation d'indices (generation 0) : rien a recouvrir
    // encore, lancement normal avant la boucle.
    kernelMutationPrep<<<blocksNumP, threadsPerBlockP>>>(d_r1[0], d_r2[0], d_r3[0], d_indivStates, pop);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    long long fe = pop;
    long long max_fes = FE_BUDGET_PER_DIM * (long long) dim;

    int cur = 0, nxt = 1;
    while (fe + pop <= max_fes) {

        // kernel(MCER) de la génération courante, sur stream0, avec les indices [cur]
        kernelMCEROpt<<<pop, dim, sharedMCER, stream0>>>(
            d_X, d_r1[cur], d_r2[cur], d_r3[cur], d_geneStates, d_jrandStates,
            d_fitX, d_Xnext, d_fitXnext, dim, pop, obj_func, DE_F, DE_CR);
        CUDA_CHECK(cudaGetLastError());

        // En parallèle sur stream1 : prépare les indices de la PROCHAINE
        // génération dans le buffer [nxt] — kernel(P) ne lit jamais X,
        // donc aucune dépendance avec le kernel(MCER) ci-dessus (c'est
        // exactement le recouvrement décrit Figure 2 de l'article).
        bool willContinue = (fe + 2*pop <= max_fes);
        if (willContinue) {
            kernelMutationPrep<<<blocksNumP, threadsPerBlockP, 0, stream1>>>(
                d_r1[nxt], d_r2[nxt], d_r3[nxt], d_indivStates, pop);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream0));
        fe += pop;
        std::swap(d_X, d_Xnext);
        std::swap(d_fitX, d_fitXnext);

        if (willContinue) {
            CUDA_CHECK(cudaStreamSynchronize(stream1));
        }
        std::swap(cur, nxt);
    }

    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));

    std::vector<float> h_X(pop * dim);
    std::vector<float> h_fit(pop);
    CUDA_CHECK(cudaMemcpy(h_X.data(),   d_X,    vecBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fitX, popBytes, cudaMemcpyDeviceToHost));

    int best = 0;
    for (int k = 1; k < pop; k++)
        if (h_fit[k] < h_fit[best]) best = k;
    for (int j = 0; j < dim; j++)
        out_gBest[j] = h_X[best*dim + j];

    cudaFree(d_X); cudaFree(d_Xnext); cudaFree(d_fitX); cudaFree(d_fitXnext);
    for (int k = 0; k < 2; k++) { cudaFree(d_r1[k]); cudaFree(d_r2[k]); cudaFree(d_r3[k]); }
    cudaFree(d_geneStates); cudaFree(d_indivStates); cudaFree(d_jrandStates);

    *out_evaluations = fe;
}
