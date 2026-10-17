#include "kernel.h"
#include <vector>

int main(int argc, char** argv) {
    int dim = 3;
    int pop = 512;
    int selected_func = 4; // Sphere par défaut
    
    if(argc >= 3){
        dim = std::stoi(argv[1]);
        pop = std::stoi(argv[2]);
    }
    if (argc == 4) {
        selected_func = std::stoi(argv[3]);
    }

    int max_iter = dim * 10000;
    int total_elements = pop * dim;

    std::vector<float> positions(total_elements);
    std::vector<float> velocities(total_elements);
    std::vector<float> pBests(total_elements);
    std::vector<float> gBest(dim);
    
    // NOUVEAU : Récupération des bornes spécifiques à la fonction choisie
    float range_min, range_max;
    get_search_space_limits(selected_func, range_min, range_max);

    printf("--- Configuration PSO GPU ---\n");
    printf("Fonction: %d | Dimensions: %d | Particules: %d | Iterations: %d\n", selected_func, dim, pop, max_iter);
    printf("Intervalle de recherche : [%.2f, %.2f]\n\n", range_min, range_max); // Affichage
    printf("Type \t Time(s) \t Minimum\n");
    
    srand((unsigned) time(NULL));

    // Utilisation des nouvelles bornes pour l'initialisation
    for (int i = 0; i < total_elements; i++) {
        positions[i] = getRandom(range_min, range_max);
        pBests[i] = positions[i];
        velocities[i] = 0.0f; // On pourrait aussi initialiser une vitesse aléatoire entre -|range| et +|range|
    }

    for (int k = 0; k < dim; k++) {
        gBest[k] = pBests[k];
    }

    clock_t begin = clock();
    
    // On passe range_min et range_max à cuda_pso (utile si on veut implémenter un clamping (bridage) sur le GPU)
    cuda_pso(positions.data(), velocities.data(), pBests.data(), gBest.data(), 
             pop, dim, max_iter, selected_func, range_min, range_max);    
             
    clock_t end = clock();
    
    float final_fitness = host_fitness_function(gBest.data(), selected_func, dim);

    printf("GPU \t %10.3lf \t %f\n", (double)(end - begin)/CLOCKS_PER_SEC, final_fitness);

    return 0;
}