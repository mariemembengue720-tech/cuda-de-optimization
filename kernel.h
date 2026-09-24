#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float F = 0.5f;   
const float CR = 0.9f;  
const float phi = 3.1415f;

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[], int dim, int func_id);

// 🚀 cuda_de accepte maintenant la population, la dimension et la fonction
extern "C" void cuda_de(float *positions, float *gBest, int pop, int dim, int func_id);

#endif