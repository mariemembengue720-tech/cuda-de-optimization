#include "kernel.h"
#include <cstdlib>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" float getRandom() {
    return (float)rand() / RAND_MAX;
}

// Évaluation CPU des 5 fonctions
float eval_function_cpu(const float* x, int dim, int func_id) {
    float sum = 0.0f;
    
    if (func_id == 0) { 
        float f_bias = -180.0f;
        for (int i = 0; i < dim; i++) {
            float w = 1.0f + (x[i] - 1.0f) / 4.0f;
            sum += (w - 1.0f) * (w - 1.0f) * (1.0f + 10.0f * sinf(M_PI * w + 1.0f));
        }
        return sum + f_bias;
        
    } else if (func_id == 1) { 
        float f_bias = -330.0f;
        for (int i = 0; i < dim; i++) {
            sum += (x[i] * x[i] - 10.0f * cosf(2.0f * M_PI * x[i]) + 10.0f);
        }
        return sum + f_bias;
        
    } else if (func_id == 2) { 
        float f_bias = 390.0f;
        for (int i = 0; i < dim - 1; i++) {
            float t1 = x[i + 1] - x[i] * x[i];
            float t2 = 1.0f - x[i];
            sum += 100.0f * t1 * t1 + t2 * t2;
        }
        return sum + f_bias;
        
    } else if (func_id == 3) { 
        float prod = 1.0f;
        float f_bias = -180.0f;
        for (int i = 0; i < dim; i++) {
            sum += (x[i] * x[i]) / 4000.0f;
            prod *= cosf(x[i] / sqrtf(static_cast<float>(i + 1)));
        }
        return sum - prod + 1.0f + f_bias;
        
    } else { 
        float f_bias = -450.0f;
        for (int i = 0; i < dim; i++) {
            sum += x[i] * x[i];
        }
        return sum + f_bias;
    }
}

extern "C" float cpu_de(float* h_pop, float* h_next_pop, float* h_fitness, int pop_size, int dim, int max_iter, int func_id) {
    float best_fit = FLT_MAX;
    
    // Boucle CPU complète sur les générations
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            int a = (i + 1) % pop_size;
            int b = (i + 2) % pop_size;
            int c = (i + 3) % pop_size;

            for (int d = 0; d < dim; d++) {
                h_next_pop[i * dim + d] = h_pop[a * dim + d] + 0.5f * (h_pop[b * dim + d] - h_pop[c * dim + d]);
            }

            float fit_old = eval_function_cpu(&h_pop[i * dim], dim, func_id);
            float fit_new = eval_function_cpu(&h_next_pop[i * dim], dim, func_id);

            if (fit_new < fit_old) {
                h_fitness[i] = fit_new;
                for (int d = 0; d < dim; d++) h_pop[i * dim + d] = h_next_pop[i * dim + d];
            } else {
                h_fitness[i] = fit_old;
            }

            if (h_fitness[i] < best_fit) best_fit = h_fitness[i];
        }
    }
    return best_fit;
}
