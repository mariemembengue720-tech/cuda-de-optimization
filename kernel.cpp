#include "kernel.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>

float getRandom() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float getRandomClamped(float min, float max) {
    return min + getRandom() * (max - min);
}

float host_fitness_function(const float* x, int dim) {
    float res = 0.0f;

#if defined(USE_RASTRIGIN)
    for (int i = 0; i < dim; i++) {
        float zi = x[i];
        res += zi * zi - 10.0f * cosf(2.0f * M_PI * zi) + 10.0f;
    }
#elif defined(USE_ROSENBROCK)
    for (int i = 0; i < dim - 1; i++) {
        float zi = x[i];
        float zip1 = x[i + 1];
        res += 100.0f * powf(zi * zi - zip1, 2.0f) + powf(zi - 1.0f, 2.0f);
    }
#elif defined(USE_SPHERE)
    for (int i = 0; i < dim; i++) {
        res += x[i] * x[i];
    }
#elif defined(USE_GRIEWANK)
    float somme = 0.0f;
    float produit = 1.0f;
    for (int i = 0; i < dim; i++) {
        somme += (x[i] * x[i]) / 4000.0f;
        produit *= cosf(x[i] / sqrtf(i + 1));
    }
    res = somme - produit + 1.0f;
#elif defined(USE_LEVY)
    float w1 = 1.0f + (x[0] - 1.0f) / 4.0f;
    float wn = 1.0f + (x[dim - 1] - 1.0f) / 4.0f;
    res = powf(sinf(M_PI * w1), 2.0f);
    for (int i = 0; i < dim - 1; i++) {
        float wi = 1.0f + (x[i] - 1.0f) / 4.0f;
        res += powf(wi - 1.0f, 2.0f) * (1.0f + 10.0f * powf(sinf(M_PI * wi + 1.0f), 2.0f));
    }
    res += powf(wn - 1.0f, 2.0f) * (1.0f + powf(sinf(2.0f * M_PI * wn), 2.0f));
#endif

    return res;
}

float cpu_de(float* population, float* next_pop, float* fitness, 
             int pop_size, int dim, int max_iter) {
    
    for (int i = 0; i < pop_size; i++) {
        fitness[i] = host_fitness_function(&population[i * dim], dim);
    }

    float best_fitness = FLT_MAX;
    
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            int a, b, c;
            do { a = rand() % pop_size; } while (a == i);
            do { b = rand() % pop_size; } while (b == i || b == a);
            do { c = rand() % pop_size; } while (c == i || c == a || c == b);

            int j_rand = rand() % dim;
            float trial[128];

            for (int d = 0; d < dim; d++) {
                if (getRandom() < DE_CR || d == j_rand) {
                    trial[d] = population[a * dim + d] + DE_F * (population[b * dim + d] - population[c * dim + d]);
                } else {
                    trial[d] = population[i * dim + d];
                }
            }

            float f_trial = host_fitness_function(trial, dim);

            if (f_trial <= fitness[i]) {
                fitness[i] = f_trial;
                for (int d = 0; d < dim; d++) {
                    next_pop[i * dim + d] = trial[d];
                }
            } else {
                for (int d = 0; d < dim; d++) {
                    next_pop[i * dim + d] = population[i * dim + d];
                }
            }

            if (fitness[i] < best_fitness) {
                best_fitness = fitness[i];
            }
        }

        for (int k = 0; k < pop_size * dim; k++) {
            population[k] = next_pop[k];
        }
    }

    return best_fitness;
}