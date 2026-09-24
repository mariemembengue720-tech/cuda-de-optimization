#include "kernel.h"
#include <vector>
#include <algorithm>

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/

// ------------------------------------------------------------------
// Cœur commun du calcul de fitness — factorisé à partir du switch du
// PSO fourni (même formules, aucun changement de benchmark). Dimension
// et fonction objectif sont désormais des paramètres au lieu d'être
// lues depuis les constantes globales NUM_OF_DIMENSIONS/SELECTED_OBJ_FUNC.
// host_fitness_function() ci-dessous garde exactement sa signature et
// son comportement d'origine (c'est elle que kernel.cu appelle) : zéro
// modification de kernel.cu nécessaire pour ce prototype.
// ------------------------------------------------------------------
static float compute_fitness(float x[], int dim, int obj_func) {
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
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180;
            break;
        case 4:
            for(int i = 0; i < dim; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

float host_fitness_function(float x[]) {
    // Signature/comportement inchangés : c'est ce que kernel.cu (PSO) appelle.
    return compute_fitness(x, NUM_OF_DIMENSIONS, SELECTED_OBJ_FUNC);
}

float de_fitness_function(float x[], int dim, int obj_func) {
    return compute_fitness(x, dim, obj_func);
}

// Obtenir un random entre low et high
float getRandom(float low, float high) {
    return low + float(((high - low) + 1)*rand()/(RAND_MAX + 1.0));
}
// Obtenir un random entre 0.0f and 1.0f inclusif
float getRandomClamped() {
    return (float) rand()/(float) RAND_MAX;
}


// ====================================================================
// AJOUTS DE CE PROTOTYPE — Differential Evolution séquentiel (CPU)
// DE/rand/1/bin : mutation, crossover binomial, sélection.
// ====================================================================

// Population représentée en tableau 1D plat (individu i, dimension j)
// => index i*dim + j. Même convention mémoire que positions/velocities/
// pBests dans le PSO fourni, pour rester portable vers une version
// CUDA (Phase 6) sans changer la représentation mémoire.

static void de_init_population(std::vector<float>& X, int dim, int pop) {
    for (int idx = 0; idx < pop * dim; idx++) {
        X[idx] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
    }
}

// Tire r1, r2, r3 mutuellement distincts et différents de i — contrainte
// DE/rand/1 rappelée dans le sujet (§4, §11). Nécessite pop >= 4.
static void de_select_r1r2r3(int i, int pop, int &r1, int &r2, int &r3) {
    do { r1 = rand() % pop; } while (r1 == i);
    do { r2 = rand() % pop; } while (r2 == i || r2 == r1);
    do { r3 = rand() % pop; } while (r3 == i || r3 == r1 || r3 == r2);
}

void run_de_sequential(int dim, int pop, int obj_func,
                        float* gBest, long long* out_evaluations) {

    std::vector<float> X_current(pop * dim);
    std::vector<float> X_next(pop * dim);
    std::vector<float> fit_current(pop);
    std::vector<float> fit_next(pop);

    de_init_population(X_current, dim, pop);

    // FEs consommées par l'évaluation initiale de toute la population
    // (nécessaire pour comparer f(trial) à f(target) dès la génération 1).
    long long fe = 0;
    for (int i = 0; i < pop; i++) {
        fit_current[i] = de_fitness_function(&X_current[i*dim], dim, obj_func);
        fe++;
    }

    long long max_fes = FE_BUDGET_PER_DIM * (long long) dim;

    // Double buffering (X_current / X_next, §12 du sujet) : chaque
    // itération i lit uniquement X_current et écrit uniquement dans
    // X_next, jamais l'inverse. En séquentiel ce n'est pas nécessaire
    // pour la correction (pas de lecture concurrente), mais on le
    // garde dès ce prototype pour que la structure de données soit
    // directement portable vers une version CUDA où plusieurs threads
    // liraient/écriraient la population en parallèle : ça évite qu'un
    // thread lise une population partiellement mise à jour par un autre.
    while (fe + pop <= max_fes) {
        for (int i = 0; i < pop; i++) {
            int r1, r2, r3;
            de_select_r1r2r3(i, pop, r1, r2, r3);

            // Dimension forcée pour garantir qu'au moins une composante
            // du mutant soit reprise dans le trial vector (§4 du sujet).
            int j_rand = rand() % dim;

            std::vector<float> trial(dim);
            for (int j = 0; j < dim; j++) {
                float mutant_j = X_current[r1*dim + j] +
                                  DE_F * (X_current[r2*dim + j] - X_current[r3*dim + j]);
                float rand_j = getRandomClamped();
                trial[j] = (rand_j <= DE_CR || j == j_rand)
                             ? mutant_j
                             : X_current[i*dim + j];
            }

            float f_trial = de_fitness_function(trial.data(), dim, obj_func);
            fe++;

            if (f_trial < fit_current[i]) {
                for (int j = 0; j < dim; j++) X_next[i*dim + j] = trial[j];
                fit_next[i] = f_trial;
            } else {
                for (int j = 0; j < dim; j++) X_next[i*dim + j] = X_current[i*dim + j];
                fit_next[i] = fit_current[i];
            }
        }
        std::swap(X_current, X_next);
        std::swap(fit_current, fit_next);
    }

    int best_idx = 0;
    for (int i = 1; i < pop; i++)
        if (fit_current[i] < fit_current[best_idx]) best_idx = i;

    for (int j = 0; j < dim; j++)
        gBest[j] = X_current[best_idx*dim + j];

    *out_evaluations = fe;
}
