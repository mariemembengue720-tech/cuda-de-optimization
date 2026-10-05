#include "kernel.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

int main(int argc, char** argv) {
    int dim = 30;
    int pop_size = 256;
    int max_iter = 2000;

    if (argc >= 3) {
        dim = std::stoi(argv[1]);
        pop_size = std::stoi(argv[2]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "  Algorithme d'Evolution Differentielle (DE)" << std::endl;
    std::cout << "  Population : " << pop_size << " | Dimension : " << dim << std::endl;
    std::cout << "  Generations: " << max_iter << std::endl;
    std::cout << "==================================================" << std::endl;

    std::vector<float> population(pop_size * dim);
    std::vector<float> next_pop(pop_size * dim);
    std::vector<float> fitness(pop_size);

    srand(1234);
    for (int i = 0; i < pop_size * dim; i++) {
        population[i] = getRandomClamped(LOWER_BOUND, UPPER_BOUND);
    }

    std::vector<float> gpu_population = population;

    // --- CALCUL CPU ---
    std::cout << "\n[1] Lancement du calcul CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    float best_fitness_cpu = cpu_de(population.data(), next_pop.data(), fitness.data(), 
                                    pop_size, dim, max_iter);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;

    // --- CALCUL GPU ---
    std::cout << "[2] Lancement du calcul GPU (CUDA)..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    
    float best_fitness_gpu = cuda_de(gpu_population.data(), pop_size, dim, max_iter);
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;

    // --- RÉSULTATS ---
    double speedup = duration_cpu.count() / duration_gpu.count();

    std::cout << "\n==================================================" << std::endl;
    std::cout << std::left << std::setw(10) << "Mode" 
              << std::setw(18) << "Temps (ms)" 
              << "Meilleure Fitness" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "CPU" 
              << std::setw(18) << duration_cpu.count() 
              << best_fitness_cpu << std::endl;
    std::cout << std::left << std::setw(10) << "GPU" 
              << std::setw(18) << duration_gpu.count() 
              << best_fitness_gpu << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "   >>> ACCELERATION (SPEEDUP) : " << std::fixed << std::setprecision(2) 
              << speedup << "x <<<" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}