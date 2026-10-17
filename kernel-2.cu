#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <math_constants.h>

#include "kernel.h"

__global__ void initCurand(curandState *state, unsigned long seed, int num_particles) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_particles) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// ... (La fonction fitness_function ne change pas, copiez-la telle quelle depuis le message précédent) ...

// Kernel unifié avec ajout des bornes (range_min, range_max)
__global__ void kernelUpdateSwarm(float *positions, float *velocities, float *pBests, float *pBestFitnesses, float *gBest, 
                                  curandState *state, int num_particles, int dimensions, int obj_func,
                                  float range_min, float range_max) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_particles) return;

    float r1 = curand_uniform(&state[i]);
    float r2 = curand_uniform(&state[i]);

    for(int d = 0; d < dimensions; d++) {
        int idx = i * dimensions + d;
        
        velocities[idx] = OMEGA * velocities[idx] + 
                          c1 * r1 * (pBests[idx] - positions[idx]) + 
                          c2 * r2 * (gBest[d] - positions[idx]);
                          
        positions[idx] += velocities[idx];

        // OPTIONNEL : Clamping (empêcher la particule de sortir des limites)
        /*
        if (positions[idx] > range_max) positions[idx] = range_max;
        if (positions[idx] < range_min) positions[idx] = range_min;
        */
    }

    float current_fitness = fitness_function(positions, i, dimensions, obj_func);

    if (current_fitness < pBestFitnesses[i]) {
        pBestFitnesses[i] = current_fitness;
        for(int d = 0; d < dimensions; d++) {
            pBests[i * dimensions + d] = positions[i * dimensions + d];
        }
    }
}

// Mise à jour de la signature
extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, 
                         int num_particles, int dimensions, int max_iter, int obj_func,
                         float range_min, float range_max)
{
    // ... (Allocation mémoire, initialisation cuRAND, thrust... inchangé par rapport au précédent) ...
    // ... (Assurez-vous de bien copier la fonction cuda_pso du message précédent) ...

    for (int iter = 0; iter < max_iter; iter++)
    {     
        // On passe range_min et range_max au kernel
        kernelUpdateSwarm<<<blocksPerGrid, threadsPerBlock>>>(d_Pos, d_Vel, d_PBest, d_PBestFitnesses, d_GBest, 
                                                              d_state, num_particles, dimensions, obj_func,
                                                              range_min, range_max);
        
        thrust::device_ptr<float> min_ptr = thrust::min_element(thrust_pBestFitnesses, thrust_pBestFitnesses + num_particles);
        int best_idx = min_ptr - thrust_pBestFitnesses; 

        cudaMemcpy(d_GBest, d_PBest + (best_idx * dimensions), sizeof(float) * dimensions, cudaMemcpyDeviceToDevice);
    }
    
    // ... (Rapatriement et nettoyage inchangé) ...
}