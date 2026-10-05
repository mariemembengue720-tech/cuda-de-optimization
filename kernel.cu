#include "kernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <iostream>

__device__ float device_fitness_function(const float* x, int dim) {
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

__global__ void setup_kernel(curandState *state, unsigned long seed, int pop_size) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < pop_size) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void kernel_de_step(float* d_population, float* d_next_pop, float* d_fitness, 
                               curandState* state, int pop_size, int dim) {
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= pop_size) return;

    curandState localState = state[id];

    int a, b, c;
    do { a = curand(&localState) % pop_size; } while (a == id);
    do { b = curand(&localState) % pop_size; } while (b == id || b == a);
    do { c = curand(&localState) % pop_size; } while (c == id || c == a || c == b);

    int j_rand = curand(&localState) % dim;
    float trial[128];

    for (int d = 0; d < dim; d++) {
        float rand_val = curand_uniform(&localState);
        if (rand_val < DE_CR || d == j_rand) {
            trial[d] = d_population[a * dim + d] + DE_F * (d_population[b * dim + d] - d_population[c * dim + d]);
        } else {
            trial[d] = d_population[id * dim + d];
        }
    }

    state[id] = localState;

    float f_trial = device_fitness_function(trial, dim);

    if (f_trial <= d_fitness[id]) {
        d_fitness[id] = f_trial;
        for (int d = 0; d < dim; d++) {
            d_next_pop[id * dim + d] = trial[d];
        }
    } else {
        for (int d = 0; d < dim; d++) {
            d_next_pop[id * dim + d] = d_population[id * dim + d];
        }
    }
}

extern "C" float cuda_de(float* h_population, int pop_size, int dim, int max_iter) {
    
    size_t pop_bytes = pop_size * dim * sizeof(float);
    size_t fit_bytes = pop_size * sizeof(float);

    float *d_population, *d_next_pop, *d_fitness;
    curandState *d_state;

    cudaMalloc(&d_population, pop_bytes);
    cudaMalloc(&d_next_pop, pop_bytes);
    cudaMalloc(&d_fitness, fit_bytes);
    cudaMalloc(&d_state, pop_size * sizeof(curandState));

    cudaMemcpy(d_population, h_population, pop_bytes, cudaMemcpyHostToDevice);

    float* h_fitness = new float[pop_size];
    for (int i = 0; i < pop_size; i++) {
        h_fitness[i] = FLT_MAX;
    }
    cudaMemcpy(d_fitness, h_fitness, fit_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, 1234, pop_size);

    for (int iter = 0; iter < max_iter; iter++) {
        kernel_de_step<<<blocksPerGrid, threadsPerBlock>>>(d_population, d_next_pop, d_fitness, d_state, pop_size, dim);
        cudaMemcpy(d_population, d_next_pop, pop_bytes, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_fitness, d_fitness, fit_bytes, cudaMemcpyDeviceToHost);

    float best_fitness = FLT_MAX;
    for (int i = 0; i < pop_size; i++) {
        if (h_fitness[i] < best_fitness) {
            best_fitness = h_fitness[i];
        }
    }

    cudaFree(d_population);
    cudaFree(d_next_pop);
    cudaFree(d_fitness);
    cudaFree(d_state);
    delete[] h_fitness;

    return best_fitness;
}