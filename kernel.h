#pragma once
// ============================================================================
//  M2 project - Massively parallel PSO and DE (CUDA)
//  Unified project, 4 files : kernel.h  kernel.cpp  kernel.cu  main.cpp
//
//  5 architectures (same benchmark protocol, same budget, same shift vectors):
//    PSO-CPU      sequential PSO                      (kernel.cpp)
//    PSO-GPU      CUDA PSO, 4 kernels                 (kernel.cu)
//    DE-CPU       sequential DE/rand/1/bin            (kernel.cpp)
//    DE-GPU       CUDA DE, basic 6 kernels (Fig. 1)   (kernel.cu)
//    DE-GPU-OPT   CUDA DE improved "cudaDEi" (Fig. 2) (kernel.cu)
//                 fused kernels IE + MCER, kernel P in a 2nd stream,
//                 block size chosen with the occupancy API, shared memory
// ============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __CUDACC__
  #define HD __host__ __device__
#else
  #define HD
#endif

// Precision of the populations. Fitness is ALWAYS accumulated in double
// (errors of 1e-8 around biases such as -450 are impossible in float).
#ifdef USE_FLOAT
typedef float  real;
#else
typedef double real;
#endif

// ---------------------------------------------------------------------------
// Objective functions (numbering of the original DE code; Levy removed)
//   1 Shifted Rastrigin   [-5, 5]^D      f* = -330
//   2 Shifted Rosenbrock  [-100, 100]^D  f* =  390   (z = x - o + 1)
//   3 Shifted Griewank    [-600, 600]^D  f* = -180
//   4 Shifted Sphere      [-100, 100]^D  f* = -450
// All use z = x - o, optimum x* = o.
// ---------------------------------------------------------------------------
enum ObjFunc { F_RASTRIGIN = 1, F_ROSENBROCK = 2, F_GRIEWANK = 3, F_SPHERE = 4 };

enum Algo { A_PSO_CPU = 0, A_PSO_GPU, A_DE_CPU, A_DE_GPU, A_DE_OPT, A_COUNT };

HD inline double f_bias(int func)
{
    switch (func) {
        case F_RASTRIGIN:  return -330.0;
        case F_ROSENBROCK: return  390.0;
        case F_GRIEWANK:   return -180.0;
        default:           return -450.0;   // F_SPHERE
    }
}

// Fitness is a sum (a) and, for Griewank only, a product (b) of per-coordinate
// terms.  This decomposition is what lets the GPU evaluate ONE individual with
// D threads + a reduction, while the CPU sums sequentially: single source of truth.
HD inline void fit_term(int func, int j, int D, double xj, double xj1,
                        double oj, double oj1, double& a, double& b)
{
    const double PI = 3.14159265358979323846;
    a = 0.0; b = 1.0;
    switch (func) {
    case F_RASTRIGIN: { double z = xj - oj; a = z * z - 10.0 * cos(2.0 * PI * z) + 10.0; break; }
    case F_ROSENBROCK:
        if (j < D - 1) {
            double zi = xj - oj + 1.0, zi1 = xj1 - oj1 + 1.0;
            double t = zi * zi - zi1, u = zi - 1.0;
            a = 100.0 * t * t + u * u;
        }
        break;
    case F_GRIEWANK:  { double z = xj - oj; a = z * z / 4000.0; b = cos(z / sqrt((double)(j + 1))); break; }
    default:          { double z = xj - oj; a = z * z; break; }          // F_SPHERE
    }
}
HD inline double fit_final(int func, double sum, double prod)
{
    return (func == F_GRIEWANK ? sum - prod + 1.0 : sum) + f_bias(func);
}

// x[k*stride] = k-th coordinate (stride 1 on host, P on GPU PSO), o = shift.
HD inline double fitness(const real* x, int stride, int D, int func, const real* o)
{
    double sum = 0.0, prod = 1.0;
    for (int j = 0; j < D; j++) {
        double a, b;
        double xj1 = (j + 1 < D) ? (double)x[(j + 1) * stride] : 0.0;
        double oj1 = (j + 1 < D) ? (double)o[j + 1] : 0.0;
        fit_term(func, j, D, (double)x[j * stride], xj1, (double)o[j], oj1, a, b);
        sum += a; prod *= b;
    }
    return fit_final(func, sum, prod);
}

// ---------------------------------------------------------------------------
// Host helpers (kernel.cpp)
// ---------------------------------------------------------------------------
struct BenchInfo { const char* name; double lo, hi, bias; };
BenchInfo bench_info(int func);
const char* algo_name(int algo);

// mode : "cec" (CEC2005 vectors, dim <= 100) | "zero" | "rand" | "file:<path>"
bool   make_shift(int func, int dim, const std::string& mode, std::vector<real>& o);
double host_fitness(const std::vector<real>& x, int func, const std::vector<real>& o);

// ---------------------------------------------------------------------------
// Common parameters / result for the 5 architectures
// ---------------------------------------------------------------------------
struct OptParams {
    int       func   = F_SPHERE;
    int       dim    = 10;
    int       pop    = 50;                 // PSO 2..1024, DE >= 4
    long long maxFE  = 100000;             // protocol : 10^4 * dim
    unsigned long long seed = 1234ULL;
    // PSO
    double wMax = 0.9, wMin = 0.4, c1 = 2.0, c2 = 2.0, vmaxFrac = 0.2;
    // DE/rand/1/bin  (values of the article: CR = 0.3, F = 0.5)
    double F = 0.5, CR = 0.3;
};

struct OptResult {
    double            bestFit = 0.0;
    double            timeMs  = 0.0;       // optimisation time (allocations excluded)
    long long         fe      = 0;         // function evaluations really used
    int               iters   = 0;         // generations
    int               popPerBlock = 0;     // DE-GPU-OPT only : members per block chosen
    std::vector<real> gBest;
};

OptResult seq_pso    (const OptParams& prm, const std::vector<real>& shift);  // kernel.cpp
OptResult seq_de     (const OptParams& prm, const std::vector<real>& shift);  // kernel.cpp
OptResult cuda_pso   (const OptParams& prm, const std::vector<real>& shift);  // kernel.cu
OptResult cuda_de    (const OptParams& prm, const std::vector<real>& shift);  // kernel.cu
OptResult cuda_de_opt(const OptParams& prm, const std::vector<real>& shift);  // kernel.cu
