#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Évaluation GPU des 5 fonctions avec biais (PDF Slide 6)
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

// Kernel CUDA PSO : mise à jour vitesse, position et pbest
__global__ void pso_kernel(float* pop, float* vel, float* pbest, float* pbest_fit, const float* gbest, int pop_size, int dim, int func_id, float w, float c1, float c2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    for (int d = 0; d < dim; d++) {
        int pos_idx = idx * dim + d;
        float r1 = 0.5f; // Générateur simplifié pour démo
        float r2 = 0.5f;

        // v_i = w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        vel[pos_idx] = w * vel[pos_idx] + c1 * r1 * (pbest[pos_idx] - pop[pos_idx]) + c2 * r2 * (gbest[d] - pop[pos_idx]);
        pop[pos_idx] += vel[pos_idx];
    }

    float current_fit = eval_function_gpu(&pop[idx * dim], dim, func_id);

    if (current_fit < pbest_fit[idx]) {
        pbest_fit[idx] = current_fit;
        for (int d = 0; d < dim; d++) {
            pbest[idx * dim + d] = pop[idx * dim + d];
        }
    }
}

extern "C" float cuda_pso(float* h_pop, int pop_size, int dim, int max_iter, int func_id) {
    float *d_pop, *d_vel, *d_pbest, *d_pbest_fit, *d_gbest;
    size_t size_pop = pop_size * dim * sizeof(float);
    size_t size_fit = pop_size * sizeof(float);
    size_t size_gbest = dim * sizeof(float);

    cudaMalloc((void**)&d_pop, size_pop);
    cudaMalloc((void**)&d_vel, size_pop);
    cudaMalloc((void**)&d_pbest, size_pop);
    cudaMalloc((void**)&d_pbest_fit, size_fit);
    cudaMalloc((void**)&d_gbest, size_gbest);

    cudaMemset(d_vel, 0, size_pop);
    cudaMemcpy(d_pop, h_pop, size_pop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pbest, h_pop, size_pop, cudaMemcpyHostToDevice);

    // Initialisation pbest_fit
    float* h_init_fit = new float[pop_size];
    for (int i = 0; i < pop_size; i++) h_init_fit[i] = FLT_MAX;
    cudaMemcpy(d_pbest_fit, h_init_fit, size_fit, cudaMemcpyHostToDevice);
    delete[] h_init_fit;

    float* h_gbest = new float[dim]();
    cudaMemcpy(d_gbest, h_gbest, size_gbest, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (pop_size + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < max_iter; iter++) {
        pso_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pop, d_vel, d_pbest, d_pbest_fit, d_gbest, pop_size, dim, func_id, 0.7f, 1.4f, 1.4f);
        cudaDeviceSynchronize();
    }

    float* h_pbest_fit = new float[pop_size];
    cudaMemcpy(h_pbest_fit, d_pbest_fit, size_fit, cudaMemcpyDeviceToHost);

    float best_fit = FLT_MAX;
    for (int i = 0; i < pop_size; i++) {
        if (h_pbest_fit[i] < best_fit) best_fit = h_pbest_fit[i];
    }

    delete[] h_pbest_fit;
    delete[] h_gbest;
    cudaFree(d_pop);
    cudaFree(d_vel);
    cudaFree(d_pbest);
    cudaFree(d_pbest_fit);
    cudaFree(d_gbest);

    return best_fit;
}
