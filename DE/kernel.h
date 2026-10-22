#ifndef KERNEL_H
#define KERNEL_H

#define LOWER_BOUND -5.12f
#define UPPER_BOUND 5.12f

#ifdef __cplusplus
extern "C" {
#endif

float cuda_de(float* h_pop, int pop_size, int dim, int max_iter, int func_id);
float cpu_de(float* h_pop, float* h_next_pop, float* h_fitness, int pop_size, int dim, int max_iter, int func_id);
float getRandom();

#ifdef __cplusplus
}
#endif

#endif
