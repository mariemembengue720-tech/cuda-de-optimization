#include "kernel.h"

// NOUVEAU : Définition des intervalles selon la fonction
void get_search_space_limits(int obj_func, float &min_val, float &max_val) {
    switch (obj_func) {
        case 0: // Levy
            min_val = -10.0f;
            max_val = 10.0f;
            break;
        case 1: // Rastrigin
            min_val = -5.12f;
            max_val = 5.12f;
            break;
        case 2: // Rosenbrock
            min_val = -5.0f; // Souvent entre -5 et 10, -30 et 30, ou -2.048 et 2.048
            max_val = 10.0f;
            break;
        case 3: // Griewank
            min_val = -600.0f;
            max_val = 600.0f;
            break;
        case 4: // Sphere
            min_val = -100.0f; // Peut aussi être -5.12 à 5.12, je mets large par défaut
            max_val = 100.0f;
            break;
        default: // Par défaut
            min_val = -5.12f;
            max_val = 5.12f;
            break;
    }
}

// Evaluation sur CPU 
float host_fitness_function(float x[], int obj_func, int dimensions) {
    float res = 0;
    float somme = 0;
    float produit = 1.0f; 

    switch (obj_func)  {
        case 0: {
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[dimensions-1] - 1)/4;

            res += pow(sin(M_PI*y1), 2);

            for (int i = 0; i < dimensions-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(M_PI*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < dimensions; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*M_PI*zi) + 10;
            }
            res -= 330;
            break;
        }
        case 2:
            for (int i = 0; i < dimensions-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < dimensions; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180; 
            break;
        case 4:
            for(int i = 0; i < dimensions; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }
    return res;
}

float getRandom(float low, float high) {
    return low + float(((high - low) + 1)*rand()/(RAND_MAX + 1.0));
}