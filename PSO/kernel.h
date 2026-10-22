#ifndef KERNEL_H
#define KERNEL_H

#define LOWER_BOUND -5.12f
#define UPPER_BOUND 5.12f

#ifdef __cplusplus
extern "C" {
#endif

float cuda_pso(float* h_pop, int pop_size, int dim, int max_iter, int func_id);
float cpu_pso(float* h_pop, float* h_vel, float* h_pbest, float* h_pbest_fit, float* h_gbest, int pop_size, int dim, int max_iter, int func_id);
float getRandom();

#ifdef __cplusplus
}
#endif

#endif
