#include "kernel.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cfloat>

int main(int argc, char* argv[]) {
    int func_id = (argc > 1) ? atoi(argv[1]) : 0;
    int dim     = (argc > 2) ? atoi(argv[2]) : 10;
    int pop_size= (argc > 3) ? atoi(argv[3]) : 50;

    int max_evals = 10000 * dim;
    int max_iter  = max_evals / pop_size;

    std::vector<float> h_pop(pop_size * dim);
    std::vector<float> h_vel(pop_size * dim, 0.0f);
    std::vector<float> h_pbest(pop_size * dim);
    std::vector<float> h_pbest_fit(pop_size, FLT_MAX);
    std::vector<float> h_gbest(dim, 0.0f);

    srand(1234);
    for (size_t i = 0; i < h_pop.size(); i++) {
        h_pop[i] = LOWER_BOUND + getRandom() * (UPPER_BOUND - LOWER_BOUND);
        h_pbest[i] = h_pop[i];
    }

    // Mesure du temps CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_fit = cpu_pso(h_pop.data(), h_vel.data(), h_pbest.data(), h_pbest_fit.data(), h_gbest.data(), pop_size, dim, max_iter, func_id);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Mesure du temps GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    float gpu_fit = cuda_pso(h_pop.data(), pop_size, dim, max_iter, func_id);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    double speedup = cpu_time / gpu_time;

    std::cout << func_id << "," << dim << "," << max_iter << "," << pop_size << ","
              << cpu_time << "," << gpu_time << "," << speedup << ","
              << cpu_fit << "," << gpu_fit << std::endl;

    return 0;
}
