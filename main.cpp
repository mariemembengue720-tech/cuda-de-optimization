#include "kernel.h"
#include <vector>
#include <cmath>
#include <numeric>

int main(int argc, char** argv) {
    // Usage: ./programde <func_id> <dim> <pop> <runs>
    int func_id = (argc > 1) ? std::stoi(argv[1]) : 4;
    int dim     = (argc > 2) ? std::stoi(argv[2]) : 10;
    int pop     = (argc > 3) ? std::stoi(argv[3]) : 50;
    int runs    = (argc > 4) ? std::stoi(argv[4]) : 10;

    std::vector<float> mins;
    std::vector<double> times;
    srand((unsigned) time(NULL));

    for (int r = 0; r < runs; r++) {
        std::vector<float> positions(pop * dim);
        std::vector<float> gBest(dim);

        for (int i = 0; i < pop * dim; i++) {
            positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
        }
        for (int k = 0; k < dim; k++) {
            gBest[k] = positions[k];
        }

        clock_t begin = clock();
        cuda_de(positions.data(), gBest.data(), pop, dim, func_id);    
        clock_t end = clock();

        double elapsed = (double)(end - begin) / CLOCKS_PER_SEC;
        float best_val = host_fitness_function(gBest.data(), dim, func_id);

        mins.push_back(best_val);
        times.push_back(elapsed);
    }

    // Calcul de la moyenne et de l'écart-type
    float sum = std::accumulate(mins.begin(), mins.end(), 0.0f);
    float mean = sum / runs;
    float sq_sum = 0.0f;
    for (float v : mins) sq_sum += (v - mean) * (v - mean);
    float std_dev = std::sqrt(sq_sum / runs);

    double time_sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean_time = time_sum / runs;

    printf("RESULT | Func:%d | Dim:%d | Pop:%d | Mean_Min:%.6f | Std:%.6f | Mean_Time:%.3fs\n", 
           func_id, dim, pop, mean, std_dev, mean_time);

    return 0;
}