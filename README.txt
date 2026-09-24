{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 ==================================================\
  CUDA Differential Evolution (DE) Benchmark\
==================================================\
\
1. PR\'c9SENTATION DU PROJET\
-------------------------\
Ce projet pr\'e9sente une impl\'e9mentation parall\'e8le de l'algorithme d'\'c9volution Diff\'e9rentielle (DE) sur GPU \'e0 l'aide de NVIDIA CUDA en C++[cite: 1, 10]. Il d\'e9coule de la refactorisation d'un code initial d'optimisation par essaim de particules (PSO)[cite: 1].\
\
2. OBJECTIFS ET ARCHITECTURE\
----------------------------\
- Impl\'e9mentation du sch\'e9ma DE/rand/1/bin avec F = 0,5 et CR = 0,9[cite: 1, 10].\
- Optimisation par Kernel Fusionn\'e9 (Single Kernel) : toutes les \'e9tapes (mutation, croisement, \'e9valuation et s\'e9lection) sont ex\'e9cut\'e9es dans un seul kernel CUDA (`kernelDE`) par thread (1 thread = 1 individu)[cite: 1, 7, 9, 10, 11].\
- R\'e9duction des acc\'e8s lents \'e0 la m\'e9moire globale du GPU en conservant les donn\'e9es dans les registres rapides[cite: 8, 9, 10].\
\
3. PROTOCOLE EXP\'c9RIMENTAL\
-------------------------\
- 4 Fonctions benchmarks : Rastrigin, Rosenbrock, Griewank, Sphere[cite: 1, 11].\
- Dimensions (D) : 10, 50, 100[cite: 1].\
- Populations (P) : 50, 100, 500[cite: 1].\
- It\'e9rations : MAX_ITER = D * 10^4[cite: 1].\
- Ex\'e9cutions : 10 runs cons\'e9cutifs en m\'e9moire C++ par configuration (36 configurations au total)[cite: 1].\
\
4. R\'c9SULTATS ET DISCUSSION\
--------------------------\
- Temps d'ex\'e9cution : Stabilit\'e9 autour de 136 secondes sur l'ensemble des configurations[cite: 4, 5]. Cette valeur s'explique par la latence cumul\'e9e des appels de kernels g\'e9r\'e9s par la boucle principale c\'f4t\'e9 CPU, qui masque le temps GPU pur[cite: 11].\
- Analyse stochastique : \'c9cart-type \'e9gal \'e0 0.0 d\'fb au d\'e9terminisme de la graine du g\'e9n\'e9rateur al\'e9atoire CUDA (`idx + iter * pop + 1`)[cite: 4, 5].\
\
5. STRUCTURE DU D\'c9P\'d4T\
---------------------\
- kernel.cu   : Kernel CUDA fusionn\'e9 et logique GPU[cite: 1].\
- kernel.h    : Constantes et prototypes[cite: 1].\
- main.cpp    : Programme h\'f4te C++ (interface CLI et boucle de runs)[cite: 1].\
- kernel.cpp  : Calculs de r\'e9f\'e9rence et fonctions auxiliaires CPU[cite: 11].\
- report.tex  : Rapport au format IEEE[cite: 2].\
\
6. COMPILATION ET EX\'c9CUTION\
---------------------------\
Compilation :\
  nvcc -o programde main.cpp kernel.cu kernel.cpp\
\
Exemple d'ex\'e9cution (Fonction Sphere=4, D=10, P=50, 10 runs) :\
  ./programde 4 10 50 10}