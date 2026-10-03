#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>

#include "kernel.h"

int main(int argc, char* argv[]) {
    // Valeurs par défaut
    std::string mode = "gpu-fused";
    int dim = 10;
    int pop = 512;
    int obj_func = 1;
    unsigned long long seed = 42;

    if (argc > 1) mode = argv[1];
    if (argc > 2) dim = std::atoi(argv[2]);
    if (argc > 3) pop = std::atoi(argv[3]);
    if (argc > 4) obj_func = std::atoi(argv[4]);
    if (argc > 5) seed = std::stoull(argv[5]);

    srand(static_cast<unsigned int>(seed));

    std::vector<float> gBest(dim, 0.0f);
    long long evaluations = 0;

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "cpu") {
        run_de_sequential(dim, pop, obj_func, gBest.data(), &evaluations);
    } else if (mode == "gpu") {
        cuda_de(dim, pop, obj_func, seed, gBest.data(), &evaluations);
    } else if (mode == "gpu-fused") {
        cuda_de_fused(dim, pop, obj_func, seed, gBest.data(), &evaluations);
    } else if (mode == "gpu-optimized") {
        cuda_de_optimized(dim, pop, obj_func, seed, gBest.data(), &evaluations);
    } else {
        std::cerr << "Mode inconnu : " << mode 
                  << " (Modes valides : cpu, gpu, gpu-fused, gpu-optimized)" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    float best_fitness = de_fitness_function(gBest.data(), dim, obj_func);

    std::cout << "Mode: " << mode 
              << " | Dim: " << dim 
              << " | Pop: " << pop 
              << " | ObjFunc: " << obj_func 
              << " | Best Fitness: " << best_fitness 
              << " | FEs: " << evaluations 
              << " | Time: " << duration.count() << " ms" 
              << std::endl;

    return 0;
}