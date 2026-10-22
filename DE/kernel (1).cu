#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Évaluation GPU des 5 fonctions benchmarks avec biais (PDF Slide 6)
__device__ float eval_function_gpu(const float* x, int dim, int func_id) {
    float sum = 0.0f;
    
    if (func_id == 0) { 
        // 1. Levy (f_bias = -180.0f)
        float f_bias = -180.0f;
        for (int i = 0; i < dim; i++) {
            float w = 1.0f + (x[i] - 1.0f) / 4.0f;
            sum += (w - 1.0f) * (w - 1.0f) * (1.0f + 10.0f * sinf(M_PI * w + 1.0f));
        }
        return sum + f_bias;
        
    } else if (func_id == 1) { 
        // 2. Rastrigin (f_bias = -330.0f)
        float f_bias = -330.0f;
        for (int i = 0; i < dim; i++) {
            sum += (x[i] * x[i] - 10.0f * cosf(2.0f * M_PI * x[i]) + 10.0f);
        }
        return sum + f_bias;
        
    } else if (func_id == 2) { 
        // 3. Rosenbrock (f_bias = 390.0f)
        float f_bias = 390.0f;
        for (int i = 0; i < dim - 1; i++) {
            float t1 = x[i + 1] - x[i] * x[i];
            float t2 = 1.0f - x[i];
            sum += 100.0f * t1 * t1 + t2 * t2;
        }
        return sum + f_bias;
        
    } else if (func_id == 3) { 
        // 4. Griewank (f_bias = -180.0f)
        float prod = 1.0f;
        float f_bias = -180.0f;
        for (int i = 0; i < dim; i++) {
            sum += (x[i] * x[i]) / 4000.0f;
            prod *= cosf(x[i] / sqrtf(static_cast<float>(i + 1)));
        }
        return sum - prod + 1.0f + f_bias;
        
    } else { 
        // 5. Sphere (f_bias = -450.0f)
        float f_bias = -450.0f;
        for (int i = 0; i < dim; i++) {
            sum += x[i] * x[i];
        }
        return sum + f_bias;
    }
}

__global__ void de_kernel(float* pop, float* next_pop, float* fitness, int pop_size, int dim, int func_id, float F, float CR) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    int a = (idx + 1) % pop_size;
    int b = (idx + 2) % pop_size;
    int c = (idx + 3) % pop_size;

    for (int d = 0; d < dim; d++) {
        float mutant = pop[a * dim + d] + F * (pop[b * dim + d] - pop[c * dim + d]);
        next_pop[idx * dim + d] = mutant;
    }

    float fit_old = eval_function_gpu(&pop[idx * dim], dim, func_id);
    float fit_new = eval_function_gpu(&next_pop[idx * dim], dim, func_id);

    if (fit_new < fit_old) {
        fitness[idx] = fit_new;
        for (int d = 0; d < dim; d++) pop[idx * dim + d] = next_pop[idx * dim + d];
    } else {
        fitness[idx] = fit_old;
    }
}

extern "C" float cuda_de(float* h_pop, int pop_size, int dim, int max_iter, int func_id) {
    float *d_pop, *d_next_pop, *d_fitness;
    size_t size_pop = pop_size * dim * sizeof(float);
    size_t size_fit = pop_size * sizeof(float);

    cudaMalloc((void**)&d_pop, size_pop);
    cudaMalloc((void**)&d_next_pop, size_pop);
    cudaMalloc((void**)&d_fitness, size_fit);

    cudaMemcpy(d_pop, h_pop, size_pop, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < max_iter; iter++) {
        de_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pop, d_next_pop, d_fitness, pop_size, dim, func_id, 0.5f, 0.9f);
    }
    cudaDeviceSynchronize();

    float* h_fitness = new float[pop_size];
    cudaMemcpy(h_fitness, d_fitness, size_fit, cudaMemcpyDeviceToHost);

    float best_fit = FLT_MAX;
    for (int i = 0; i < pop_size; i++) {
        if (h_fitness[i] < best_fit) best_fit = h_fitness[i];
    }

    delete[] h_fitness;
    cudaFree(d_pop);
    cudaFree(d_next_pop);
    cudaFree(d_fitness);

    return best_fit;
}
