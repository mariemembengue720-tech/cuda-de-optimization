#include "kernel.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char* argv[]) {
    int func_id = (argc > 1) ? atoi(argv[1]) : 0;
    int dim     = (argc > 2) ? atoi(argv[2]) : 10;
    int pop_size= (argc > 3) ? atoi(argv[3]) : 50;

    int max_evals = 10000 * dim;
    int max_iter  = max_evals / pop_size;

    std::vector<float> h_pop(pop_size * dim);
    std::vector<float> h_next_pop(pop_size * dim);
    std::vector<float> h_fitness(pop_size);

    srand(1234);
    for (size_t i = 0; i < h_pop.size(); i++) {
        h_pop[i] = LOWER_BOUND + getRandom() * (UPPER_BOUND - LOWER_BOUND);
    }

    // Mesure du temps CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float cpu_fit = cpu_de(h_pop.data(), h_next_pop.data(), h_fitness.data(), pop_size, dim, max_iter, func_id);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Mesure du temps GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    float gpu_fit = cuda_de(h_pop.data(), pop_size, dim, max_iter, func_id);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    double speedup = cpu_time / gpu_time;

    std::cout << func_id << "," << dim << "," << max_iter << "," << pop_size << ","
              << cpu_time << "," << gpu_time << "," << speedup << ","
              << cpu_fit << "," << gpu_fit << std::endl;

    return 0;
}
