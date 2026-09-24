#include "kernel.h"

static void print_usage(const char* prog) {
    printf("Usage: %s <dimensions> <population> <objectif:0-4> [seed]\n", prog);
    printf("  objectif : 0=Levy 1=Rastrigin 2=Rosenbrock 3=Griewank 4=Sphere\n");
    printf("  contrainte DE/rand/1 : population >= 4\n");
}

int main(int argc, char** argv) {

    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    int dim = std::stoi(argv[1]);
    int pop = std::stoi(argv[2]);
    int obj = std::stoi(argv[3]);
    unsigned int seed = (argc >= 5) ? (unsigned int) std::stoul(argv[4])
                                     : (unsigned int) time(NULL);

    if (dim <= 0) {
        fprintf(stderr, "Erreur : dimension doit être > 0.\n");
        return 1;
    }
    if (pop < 4) {
        // DE/rand/1/bin a besoin de r1, r2, r3 distincts et != i => NP >= 4.
        fprintf(stderr, "Erreur : population doit être >= 4 (contrainte DE/rand/1).\n");
        return 1;
    }
    if (obj < 0 || obj > 4) {
        fprintf(stderr, "Erreur : objectif doit être entre 0 et 4.\n");
        return 1;
    }

    srand(seed);

    float* gBest = new float[dim];
    long long fes_used = 0;

    printf("Type \t Time \t\t Minimum \t\t FEs\n");

    clock_t begin = clock();
    run_de_sequential(dim, pop, obj, gBest, &fes_used);
    clock_t end = clock();

    printf("DE-CPU \t %10.3lf \t %f \t %lld\n",
           (double)(end - begin) / CLOCKS_PER_SEC,
           de_fitness_function(gBest, dim, obj),
           fes_used);

    delete[] gBest;
    return 0;
}
