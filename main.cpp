#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include "kernel.h"
#define MAX_ITERATIONS 1000
#define NUM_OF_PARTICLES 30
#define NUM_OF_DIMENSIONS 2
#define START_RANGE_MIN -5.0f
#define START_RANGE_MAX 5.0f
void cpu_de(float* positions,float* gBest){
  //float*positions: le pointeur qui vers le tableau qui contient toutes nos positions
  //float*gbest: le pointeur qui vers le tableau qui stockera la meilleures solutions
  float F = 0.5f; //F est le facteur de mutation(souvent fixé entre 0.5 et 0.9). il contrôle l'écart qu'on ajoute entre deux individus pour creer une nouvelle piste
  float CR = 0.8f;//c'est le taux de croisement. Fixé ici à 0.8 (soit 80%), il représente la probabilité qu'une valeur du vecteur mutant soit conservée dans le candidat final.
  float trial[NUM_OF_DIMENSIONS];// tableau qui sert de zone mémoire temporaire pour contruire et stocker le vecteur mutant(le nouveau candidat qu'on va tester)
  float fitness[NUM_OF_PARTICLES]; // tableau qui va contenir le score(la valeur de la fonction objectif) de chaque individu de la population
  for (int i = 0; i < NUM_OF_PARTICLES; i++) {
    //Cette boucle va passer en revue chaque individu de la population (de $0$ jusqu'à NUM_OF_PARTICLES - 1).
    fitness[i] = host_fitness_function(&positions[i * NUM_OF_DIMENSIONS]);
    //Comme les positions sont stockées à la suite dans un seul grand tableau à une dimension, cet indice permet de pointer directement vers le début du $i$-ème individu.
    //host_fitness_function(...) : C'est la fonction du projet (définie dans kernel.h) qui évalue la solution et renvoie son score. On enregistre ce résultat dans fitness[i].
    }
  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
    //MAX_ITERATIONS est le nombre de générations (ou tours de boucle) que va effectuer l'algorithme pour faire évoluer les solutions.
    for (int i = 0; i < NUM_OF_PARTICLES; i++) {
      //i est le numéro de l'individu actuel (qu'on appelle souvent la cible ou l'individu à faire évoluer).
      //Pour cet individu i, on va créer un mutant, tenter un croisement, puis décider si la nouvelle version le remplace ou non.
      //Pour appliquer la formule du mutant, il nous faut 3 individus ($a$, $b$ et $c$) choisis au hasard, mais tous différents entre eux et différents de notre individu cible $i$.
      int a, b, c;
        do { a = rand() % NUM_OF_PARTICLES; } while (a == i);
        do { b = rand() % NUM_OF_PARTICLES; } while (b == i || b == a);
        do { c = rand() % NUM_OF_PARTICLES; } while (c == i || c == a || c == b);
        //rand() % NUM_OF_PARTICLES : Génère un nombre entier aléatoire entre 0 et NUM_OF_PARTICLES - 1
        //do { ... } while (...) : Cette boucle répète le tirage au sort tant que l'indice obtenu est égal à i ou à l'un des indices déjà choisis. Cela garantit qu'on a 4 individus bien distincts (i, a, b, c).
      int j_rand = rand() % NUM_OF_DIMENSIONS;

        for (int d = 0; d < NUM_OF_DIMENSIONS; d++) {
            float r = (float)rand() / (float)RAND_MAX;
            if (r < CR || d == j_rand) {
                trial[d] = positions[a * NUM_OF_DIMENSIONS + d] + F * (positions[b * NUM_OF_DIMENSIONS + d] - positions[c * NUM_OF_DIMENSIONS + d]);
            } else {
                trial[d] = positions[i * NUM_OF_DIMENSIONS + d];
            }
        }

        float f_trial = host_fitness_function(trial);//calcule le score du candidat

        if (f_trial <= fitness[i]) {
            fitness[i] = f_trial;//Remplace l'ancien individu si le candidat est meilleur ou égal.
            for (int d = 0; d < NUM_OF_DIMENSIONS; d++) {
                positions[i * NUM_OF_DIMENSIONS + d] = trial[d];
            }
            if (f_trial < *gBest) {
                *gBest = f_trial;
            }
        }
    }
  }
}
int main(int argc, char** argv) {

    if(argc == 3){
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }
    

    float positions[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    //float velocities[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    //float pBests[NUM_OF_PARTICLES*NUM_OF_DIMENSIONS];
    float gBest;
    
    printf("Type \t Time \t  \t Minimum\n");
    
        // Initialisation du random
        srand((unsigned) time(NULL));

        for (int i = 0; i < NUM_OF_PARTICLES*NUM_OF_DIMENSIONS; i++) {
            positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
            //pBests[i] = positions[i];
            //velocities[i] = 0;
        }

        //for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            //gBest[k] = positions[k];
          //on remplace la boucle par une simple affectation
        gBest = host_fitness_function(positions);
        clock_t begin = clock();
        //cuda_pso(positions, velocities, pBests, gBest);  
        cpu_de(positions,&gBest);  
        clock_t end = clock();
        printf("CPU \t ");
        printf("%10.3lf \t", (double)(end - begin)/CLOCKS_PER_SEC);
        
        printf(" %f\n", gBest);

    return 0;
}
