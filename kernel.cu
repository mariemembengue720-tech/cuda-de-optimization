#include <cuda_runtime.h>
#include <cuda.h>
#include "kernel.h"

__device__ float fitness_function(float x[], int dim, int func_id) {
    float res = 0, somme = 0, produit = 1;
    switch (func_id) {
        case 0: {
            float y1 = 1 + (x[0] - 1)/4.0f;
            float yn = 1 + (x[dim-1] - 1)/4.0f;
            res += pow(sin(phi*y1), 2);
            for (int i = 0; i < dim-1; i++) {
                float y = 1 + (x[i] - 1)/4.0f;
                float yp = 1 + (x[i+1] - 1)/4.0f;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: 
            for (int i = 0; i < dim; i++) {
                res += pow(x[i], 2) - 10*cos(2*phi*x[i]) + 10;
            }
            res -= 330;
            break;
        case 2:
            for (int i = 0; i < dim-1; i++) {
                float zi = x[i] + 1, zip1 = x[i+1] + 1;
                res += 100 * pow(pow(zi, 2) - zip1, 2) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < dim; i++) {
                somme += pow(x[i], 2)/4000.0f;
                produit *= cos(x[i]/sqrt((float)(i+1)));
            }
            res = somme - produit + 1 - 180; 
            break;
        case 4:
            for(int i = 0; i < dim; i++) {
                res += pow(x[i], 2);
            }
            res -= 450;
            break;
    }
    return res;
}

__global__ void kernelDE(float *positions, int pop, int dim, int func_id, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop) return;

    unsigned int state = idx + iter * pop + 1;
    unsigned int a = 1664525u, c = 1013904223u;
    #define NEXT_RAND() ((float)(state = state * a + c) / 4294967296.0f)

    int r1, r2, r3;
    do { r1 = (int)(NEXT_RAND() * pop) % pop; } while (r1 == idx);
    do { r2 = (int)(NEXT_RAND() * pop) % pop; } while (r2 == idx || r2 == r1);
    do { r3 = (int)(NEXT_RAND() * pop) % pop; } while (r3 == idx || r3 == r1 || r3 == r2);

    int j_rand = (int)(NEXT_RAND() * dim) % dim;

    float target[100];
    float trial[100];

    for (int d = 0; d < dim; d++) {
        target[d] = positions[idx * dim + d];
        if (NEXT_RAND() < CR || d == j_rand) {
            float xr1 = positions[r1 * dim + d];
            float xr2 = positions[r2 * dim + d];
            float xr3 = positions[r3 * dim + d];
            trial[d] = xr1 + F * (xr2 - xr3);
        } else {
            trial[d] = target[d];
        }
    }

    if (fitness_function(trial, dim, func_id) < fitness_function(target, dim, func_id)) {
        for (int d = 0; d < dim; d++) {
            positions[idx * dim + d] = trial[d];
        }
    }
    #undef NEXT_RAND
}

extern "C" void cuda_de(float *positions, float *gBest, int pop, int dim, int func_id) {
    int size = pop * dim;
    float *devPos;
    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);

    int threadsNum = 64;
    int blocksNum = (pop + threadsNum - 1) / threadsNum;

    int max_iter = dim * 10000;
    for (int iter = 0; iter < max_iter; iter++) {
        kernelDE<<<blocksNum, threadsNum>>>(devPos, pop, dim, func_id, iter);
    }

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);

    float temp[100];
    for (int i = 0; i < pop; i++) {
        for (int k = 0; k < dim; k++) {
            temp[k] = positions[i * dim + k];
        }
        if (fitness_function(temp, dim, func_id) < fitness_function(gBest, dim, func_id)) {
            for (int k = 0; k < dim; k++) {
                gBest[k] = temp[k];
            }
        }
    }

    cudaFree(devPos);
}