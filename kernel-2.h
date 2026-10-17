
extern "C" void get_search_space_limits(int selected_func, float &range_min, float &range_max);

#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

// Paramètres de l'algorithme PSO
const float OMEGA = 0.5f;
const float c1 = 1.5f;
const float c2 = 1.5f;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastrigin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/

// Fonctions utilitaires CPU
float getRandom(float low, float high);
float host_fitness_function(float x[], int obj_func, int dimensions);

// Fonction principale CUDA
extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, 
                         int num_particles, int dimensions, int max_iter, int obj_func);

#endif