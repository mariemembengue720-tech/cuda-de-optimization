#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>

// Hyperparamètres DE
#define DE_F  0.5f
#define DE_CR 0.8f

// Bornes de l'espace de recherche
#define LOWER_BOUND -5.12f
#define UPPER_BOUND  5.12f

// Choix de la fonction objectif (Décommenter UNE seule fonction)
#define USE_RASTRIGIN
//#define USE_ROSENBROCK
//#define USE_SPHERE
//#define USE_GRIEWANK
//#define USE_LEVY

// Utilitaires CPU
float getRandom();
float getRandomClamped(float min, float max);
float host_fitness_function(const float* x, int dim);

// Moteur CPU
float cpu_de(float* population, float* next_pop, float* fitness, 
             int pop_size, int dim, int max_iter);

// Moteur GPU (CUDA)
extern "C" float cuda_de(float* h_population, int pop_size, int dim, int max_iter);