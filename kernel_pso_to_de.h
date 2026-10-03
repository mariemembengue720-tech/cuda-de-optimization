#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>


// Constantes
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/

// ------------------------------------------------------------------
// Constantes PSO — INCHANGÉES par rapport au fichier fourni par le
// projet. kernel.cu déclare des tableaux __device__ dont la taille
// dépend de NUM_OF_DIMENSIONS : ce doit rester une constante
// compile-time, donc on ne la touche pas ici (risque CUDA identifié
// en Phase 1 : la transformer en variable runtime casserait la
// compilation de kernel.cu). Le DE séquentiel ci-dessous reçoit sa
// dimension/population/fonction en paramètres de fonction à la place.
// ------------------------------------------------------------------
const int SELECTED_OBJ_FUNC = 0;
const int NUM_OF_PARTICLES = 512;
const int NUM_OF_DIMENSIONS = 3;
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4);
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float phi = 3.14159265358979323846f;

// Les 3 fonctions PSO déjà présentes — inchangées, signature d'origine.
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);



// ====================================================================
// AJOUTS DE CE PROTOTYPE — Differential Evolution séquentiel (CPU)
// DE/rand/1/bin, d'après ImprovedDE.pdf : F = 0.5, CR = 0.3
// ====================================================================

// F et CR : valeurs fixées par le sujet (§5 du prompt projet). Non
// exposées en CLI pour l'instant : aucune expérimentation demandée
// dans le sujet ne fait varier F ou CR.
const float DE_F  = 0.5f;
const float DE_CR = 0.3f;

// Budget d'évaluations : MaxFEs = 10^4 * D (§5 du sujet). Remplace
// MAX_ITER (nombre d'itérations fixe) utilisé par le PSO : le DE
// s'arrête sur un budget de FEs, pas sur un nombre d'itérations fixe.
const long long FE_BUDGET_PER_DIM = 10000;

// Fitness paramétrée (dimension / fonction objectif choisies au
// runtime), utilisée par le DE dont dimension et population sont des
// paramètres d'exécution — pas des constantes compile-time comme dans
// le PSO fourni. Même calcul que host_fitness_function (factorisé en
// interne dans kernel.cpp), donc mêmes résultats à dimension/fonction
// égales : rien n'est changé dans les formules des benchmarks.
float de_fitness_function(float x[], int dim, int obj_func);

// Point d'entrée du DE séquentiel (DE/rand/1/bin, double buffering
// X_current/X_next comme demandé dans le sujet §12). Écrit le meilleur
// individu trouvé dans gBest (déjà alloué par l'appelant, taille >=
// dim) et le nombre réel de FEs consommées dans *out_evaluations.
void run_de_sequential(int dim, int pop, int obj_func,
                        float* gBest, long long* out_evaluations);


// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA, architecture DE BASIQUE (kernel.cu)
// Figure 1 de ImprovedDE.pdf : kernel(I), kernel(E), kernel(P),
// kernel(M), kernel(C), kernel(R) séparés — pas encore cudaDEi.
// ====================================================================

// Point d'entrée du DE sur GPU. La population est initialisée
// directement sur le device (kernel(I), avec curand) : pas de tableau
// host à fournir en entrée, contrairement à cuda_pso(). `seed` pilote
// curand côté device — indépendant du srand() CPU utilisé par
// run_de_sequential. Écrit le meilleur individu dans out_gBest (déjà
// alloué par l'appelant, taille >= dim) et le nombre de FEs consommées
// dans *out_evaluations.
extern "C" void cuda_de(int dim, int pop, int obj_func, unsigned long long seed,
                         float* out_gBest, long long* out_evaluations);


// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA avec FUSION DE KERNELS (kernel.cu)
// Architecture inspirée de la Figure 2 de ImprovedDE.pdf (cudaDEᵢ) :
// kernel(I)+kernel(E) → kernel(IE) ; kernel(M)+kernel(C)+kernel(E)+
// kernel(R) → kernel(MCER) ; kernel(P) reste séparé. PAS encore la
// mémoire partagée / les streams / l'auto-tuning d'occupancy de
// l'article complet — juste la fusion des kernels à ce stade.
// ====================================================================

// Même contrat que cuda_de() (signature identique), mais implémentée
// avec 2 lancements de kernels par génération au lieu de 5, et sans
// les buffers device V/U/fitU (le trial vector reste en mémoire locale
// au thread, voir DE_MAX_DIM dans kernel.cu).
extern "C" void cuda_de_fused(int dim, int pop, int obj_func, unsigned long long seed,
                               float* out_gBest, long long* out_evaluations);


// ====================================================================
// AJOUTS DE CETTE ÉTAPE — DE CUDA OPTIMISÉ (kernel.cu)
// Mémoire partagée + streams (recouvrement kernel(P)/kernel(MCER)) +
// occupancy réelle (cudaOccupancyMaxPotentialBlockSize pour kernel(P)).
// Mapping 1 bloc = 1 individu, 1 thread = 1 gène pour kernel(IE) et
// kernel(MCER) — voir les commentaires détaillés dans kernel.cu.
// ====================================================================

// Même contrat que cuda_de()/cuda_de_fused(), à une exception près :
// objectif=0 (Levy) n'est pas supporté par cette version (hors
// protocole officiel) et fait échouer le programme avec un message
// clair plutôt qu'un résultat faux.
extern "C" void cuda_de_optimized(int dim, int pop, int obj_func, unsigned long long seed,
                                   float* out_gBest, long long* out_evaluations);
