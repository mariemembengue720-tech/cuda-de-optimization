#include "kernel.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>

BenchInfo bench_info(int func)
{
    switch (func) {
        case F_RASTRIGIN:  return { "Rastrigin",   -5.0,    5.0, -330.0 };
        case F_ROSENBROCK: return { "Rosenbrock", -100.0,  100.0,  390.0 };
        case F_GRIEWANK:   return { "Griewank",   -600.0,  600.0, -180.0 };
        case F_SPHERE:     return { "Sphere",     -100.0,  100.0, -450.0 };
        default:           return { "?", 0, 0, 0 };
    }
}

const char* algo_name(int a)
{
    static const char* n[A_COUNT] = { "PSO-CPU", "PSO-GPU", "DE-CPU", "DE-GPU", "DE-GPU-OPT" };
    return (a >= 0 && a < A_COUNT) ? n[a] : "?";
}

// ---------------------------------------------------------------------------
// CEC2005 shift vectors (100 components) - copied from the DE code
// ---------------------------------------------------------------------------
#define SHIFT_MAX_DIM 100
static const double SH_RASTRIGIN[SHIFT_MAX_DIM] = {
    1.9005, -1.5644, -0.9788, -2.2536, 2.499,
    -3.2853, 0.9759, -3.6661, 0.0985, -3.2465,
    3.806, -2.6834, -1.3701, 4.1821, 2.4856,
    -4.2237, 3.3653, 2.1532, -3.0929, 4.3105,
    -2.9861, 3.4936, -2.7289, -4.1266, -2.59,
    1.3124, -1.799, -1.189, -0.1053, -3.1074,
    -3.9641, -4.3387, 3.0705, 3.3205, -3.8178,
    -1.498, -4.3807, 2.711, -3.7956, -2.3627,
    4.0086, -1.3728, -4.4362, -2.9183, -2.2457,
    0.3065, -0.8924, -3.2364, -1.2521, 2.7198,
    -3.9787, 3.0678, -4.24, -3.958, 3.9479,
    0.2203, -2.6124, 2.6498, -2.3256, -1.5383,
    3.476, 2.4462, 2.4575, 3.7409, -2.4887,
    3.8555, -1.1426, 1.3389, 2.2323, 2.3137,
    -4.337, 3.926, 3.5905, -1.2858, -2.0113,
    2.9087, 3.9278, 1.0812, -0.7461, 3.474,
    2.3036, -3.3781, -0.4491, 0.9402, -3.0583,
    -4.2165, -2.3604, 0.8364, 1.4773, -2.7292,
    -1.5904, 0.7696, 0.5164, 2.6576, -0.5427,
    1.0358, 0.6926, -4.2775, -1.5911, -3.5775 };
static const double SH_ROSENBROCK[SHIFT_MAX_DIM] = {
    81.0232, -48.395, 19.2316, -2.5231, 70.4338,
    47.1774, -7.8358, -86.6693, 57.8532, -9.9533,
    20.7778, 52.5486, 75.9263, 42.8773, -58.272,
    -16.9728, 78.3845, 75.0427, -16.1513, 70.8569,
    -79.5795, -26.4837, 56.3699, -88.2249, -64.9996,
    -53.5022, -54.23, 18.6826, -41.0061, -54.2134,
    -87.2506, 44.4214, -9.8826, 77.7266, -6.121,
    -14.643, 62.3198, 4.5274, -53.5234, 30.9847,
    60.8613, -86.4648, 32.6298, -21.6934, 59.7232,
    0.5063, 37.7048, -12.7993, -35.1688, -55.8623,
    -55.1823, 32.8001, -35.5024, 7.5012, -62.8428,
    35.6217, -21.8928, 64.802, 63.6579, 16.8413,
    -0.6205, 71.9584, 57.8932, 26.0838, 57.2353,
    28.8409, -28.4452, -37.8493, -28.5851, 6.1342,
    40.8803, -34.3277, 60.9292, 12.253, -23.3255,
    36.4931, 8.3828, -9.9215, 35.0221, 21.8358,
    53.0677, 82.2318, 4.0662, 68.4255, -58.8678,
    86.3544, -41.1394, -44.5807, 67.6335, 42.715,
    -65.4266, -87.8837, 70.9016, -54.1551, -36.2298,
    29.0596, -38.8064, -5.5396, -78.3393, 87.9002 };
static const double SH_GRIEWANK[SHIFT_MAX_DIM] = {
    -276.2684, -11.911, -578.7884, -287.6486, -84.3858,
    -228.6753, -458.1516, -202.2145, -105.8642, -96.4898,
    -395.7468, -572.9498, -270.3641, -566.8543, -152.4204,
    -588.3819, -282.8892, -488.8865, -346.9817, -453.0447,
    -506.5857, -475.9987, -362.0492, -233.2367, -491.9864,
    -544.0898, -73.4456, -526.9011, -502.2561, -537.2353,
    -263.1487, -205.926, -465.8232, -9.7565, -16.914,
    -452.9397, -430.6107, -162.3137, -69.0204, -302.4082,
    -527.8932, -489.4274, -74.2562, -6.8285, -546.3631,
    -415.188, -322.1499, -145.3822, -527.0019, -422.1779,
    -254.2542, -495.7289, -286.1023, -196.1644, -104.9383,
    -388.449, -57.9322, -53.9958, -114.3882, -50.9434,
    -508.0961, -77.0278, -242.9642, -11.5865, -506.949,
    -222.1662, -303.8325, -359.4041, -156.0715, -252.9711,
    -345.7062, -133.0654, -30.1874, -306.6162, -160.6673,
    -210.5113, -186.2518, -550.3127, -519.0305, -320.776,
    -471.1067, -159.1724, -44.0751, -84.1251, -538.0389,
    -444.1993, -590.9338, -216.8007, -523.2731, -317.2,
    -393.6671, -548.4339, -416.0403, -380.3826, -109.542,
    -167.761, -345.5127, -544.8631, -357.8218, -598.2117 };
static const double SH_SPHERE[SHIFT_MAX_DIM] = {
    -39.3119, 58.8999, -46.3224, -74.6515, -16.7997,
    -80.5441, -10.5935, 24.9694, 89.8384, 9.1119,
    -10.7443, -27.8558, -12.5806, 7.593, 74.8127,
    68.4959, -53.4293, 78.8544, -68.5957, 63.7432,
    31.347, -37.5016, 33.8929, -88.8045, -78.7719,
    -66.4944, 44.1972, 18.3836, 26.5212, 84.4723,
    39.1769, -61.4863, -25.6038, -81.1829, 58.6958,
    -30.8386, -72.6725, 89.9257, -15.1934, -4.3337,
    5.343, 10.5603, -77.7268, 52.0859, 40.3944,
    88.3328, -55.8306, 1.3181, 36.025, -69.9271,
    -8.6279, -56.8944, 85.1296, 17.6736, 6.1529,
    -17.6957, -58.9537, 30.3564, 15.9207, -18.0082,
    80.6411, -42.3912, 76.2776, -50.1652, -73.5736,
    28.3369, -57.9905, -22.7327, 52.0269, 39.2599,
    10.8679, 77.8207, 66.0395, -50.0667, 55.7063,
    73.7141, 38.5296, -56.7865, -89.6477, 37.9576,
    29.472, -35.4641, -31.7868, 77.3235, 54.7906,
    -48.2794, 74.2714, 72.6103, 62.964, -14.1446,
    20.4923, 46.5897, -83.6021, -46.4809, 83.7373,
    -79.6611, 24.3479, -17.2303, 72.3404, -36.4022 };

// Shift vector o :  z = x - o, optimum x* = o, f(x*) = bias.
//  "cec"        : CEC2005 vectors above (default; dim <= 100) -> identical for all 5 algorithms
//  "zero"       : o = 0
//  "rand"       : reproducible o_i in [0.8*lo, 0.8*hi]
//  "file:<path>": first `dim` numbers of a text file
bool make_shift(int func, int dim, const std::string& mode, std::vector<real>& o)
{
    o.assign(dim, (real)0);
    if (mode == "zero") return true;
    BenchInfo b = bench_info(func);

    if (mode == "cec") {
        if (dim > SHIFT_MAX_DIM) { fprintf(stderr, "cec shift: dim must be <= %d\n", SHIFT_MAX_DIM); return false; }
        const double* s = (func == F_RASTRIGIN) ? SH_RASTRIGIN : (func == F_ROSENBROCK) ? SH_ROSENBROCK
                        : (func == F_GRIEWANK)  ? SH_GRIEWANK  : SH_SPHERE;
        for (int i = 0; i < dim; i++) o[i] = (real)s[i];
        return true;
    }
    if (mode == "rand") {
        std::mt19937_64 gen(20261027ULL + (unsigned)func);
        std::uniform_real_distribution<double> U(0.8 * b.lo, 0.8 * b.hi);
        for (int i = 0; i < dim; i++) o[i] = (real)U(gen);
        return true;
    }
    if (mode.rfind("file:", 0) == 0) {
        std::ifstream f(mode.substr(5));
        if (!f) { fprintf(stderr, "cannot open shift file %s\n", mode.c_str() + 5); return false; }
        double v;
        for (int i = 0; i < dim; i++) {
            if (!(f >> v)) { fprintf(stderr, "shift file has fewer than %d values\n", dim); return false; }
            o[i] = (real)v;
        }
        return true;
    }
    fprintf(stderr, "unknown shift mode '%s'\n", mode.c_str());
    return false;
}

double host_fitness(const std::vector<real>& x, int func, const std::vector<real>& o)
{
    return fitness(x.data(), 1, (int)x.size(), func, o.data());
}

// ============================================================================
//  PSO-CPU : sequential synchronous PSO (1 thread)
//   gBest updated once per generation, r1/r2 per particle AND per dimension,
//   inertia wMax -> wMin linear, velocity and position clamping,
//   FEs = P * (1 + nIter) <= MaxFEs.
// ============================================================================
OptResult seq_pso(const OptParams& prm, const std::vector<real>& shift)
{
    const int P = prm.pop, D = prm.dim;
    if (P < 2 || D < 1 || (int)shift.size() < D) { fprintf(stderr, "seq_pso: invalid parameters\n"); exit(1); }
    const BenchInfo bi = bench_info(prm.func);
    const size_t n = (size_t)P * D;
    const real lo = (real)bi.lo, hi = (real)bi.hi;
    const real vmax = (real)(prm.vmaxFrac * (bi.hi - bi.lo));
    const real c1 = (real)prm.c1, c2 = (real)prm.c2;

    long long nIter = prm.maxFE / P - 1;
    if (nIter < 0) nIter = 0;

    std::mt19937_64 gen(prm.seed);
    std::uniform_real_distribution<real> U01((real)0, (real)1);

    std::vector<real>   pos(n), vel(n, (real)0), pbest(n), gbest(D, (real)0);
    std::vector<double> pfit(P, 1.0e300);
    double gfit = 1.0e300;

    auto evaluate = [&]() {
        for (int p = 0; p < P; p++) {
            double f = fitness(&pos[(size_t)p * D], 1, D, prm.func, shift.data());
            if (f < pfit[p]) {
                pfit[p] = f;
                for (int d = 0; d < D; d++) pbest[(size_t)p * D + d] = pos[(size_t)p * D + d];
            }
        }
    };
    auto reduce_gbest = [&]() {
        int b = 0;
        for (int p = 1; p < P; p++) if (pfit[p] < pfit[b]) b = p;
        if (pfit[b] < gfit) {
            gfit = pfit[b];
            for (int d = 0; d < D; d++) gbest[d] = pbest[(size_t)b * D + d];
        }
    };

    auto t0 = std::chrono::steady_clock::now();
    for (size_t e = 0; e < n; e++) { pos[e] = lo + (hi - lo) * U01(gen); pbest[e] = pos[e]; }
    evaluate();
    reduce_gbest();

    for (long long it = 0; it < nIter; it++) {
        const real w = (real)((nIter > 1) ? prm.wMax - (prm.wMax - prm.wMin) * (double)it / (double)(nIter - 1)
                                          : prm.wMax);
        for (int p = 0; p < P; p++) {
            for (int d = 0; d < D; d++) {
                const size_t e = (size_t)p * D + d;
                const real r1 = U01(gen), r2 = U01(gen);
                real x = pos[e];
                real v = w * vel[e] + c1 * r1 * (pbest[e] - x) + c2 * r2 * (gbest[d] - x);
                if (v >  vmax) v =  vmax;
                if (v < -vmax) v = -vmax;
                x += v;
                if (x < lo)      { x = lo; v = (real)0; }
                else if (x > hi) { x = hi; v = (real)0; }
                vel[e] = v; pos[e] = x;
            }
        }
        evaluate();
        reduce_gbest();
    }
    auto t1 = std::chrono::steady_clock::now();

    OptResult r;
    r.bestFit = gfit; r.gBest = gbest;
    r.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.iters = (int)nIter; r.fe = (long long)P * (1 + nIter);
    return r;
}

// ============================================================================
//  DE-CPU : sequential DE/rand/1/bin, synchronous (double buffer), same
//  semantics as the two CUDA versions.
//   V = X_r1 + F (X_r2 - X_r3), r1,r2,r3,i mutually distinct
//   U_j = V_j if rand_j <= CR or j == jrand, else X_ij
//   greedy replacement.   FEs = P * (1 + G) <= MaxFEs.
//  (no per-individual allocation, x*x instead of pow: fair baseline)
// ============================================================================
OptResult seq_de(const OptParams& prm, const std::vector<real>& shift)
{
    const int P = prm.pop, D = prm.dim;
    if (P < 4 || D < 1 || (int)shift.size() < D) { fprintf(stderr, "seq_de: invalid parameters (pop >= 4)\n"); exit(1); }
    const BenchInfo bi = bench_info(prm.func);
    const size_t n = (size_t)P * D;
    const real lo = (real)bi.lo, hi = (real)bi.hi;
    const real F = (real)prm.F, CR = (real)prm.CR;

    long long G = prm.maxFE / P - 1;
    if (G < 0) G = 0;

    std::mt19937_64 gen(prm.seed);
    std::uniform_real_distribution<real> U01((real)0, (real)1);
    std::uniform_int_distribution<int> UP(0, P - 1), UD(0, D - 1);

    std::vector<real>   X(n), Xn(n), trial(D);
    std::vector<double> fit(P), fitn(P);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t e = 0; e < n; e++) X[e] = lo + (hi - lo) * U01(gen);
    for (int i = 0; i < P; i++) fit[i] = fitness(&X[(size_t)i * D], 1, D, prm.func, shift.data());

    for (long long g = 0; g < G; g++) {
        for (int i = 0; i < P; i++) {
            int r1, r2, r3;
            do { r1 = UP(gen); } while (r1 == i);
            do { r2 = UP(gen); } while (r2 == i || r2 == r1);
            do { r3 = UP(gen); } while (r3 == i || r3 == r1 || r3 == r2);
            const int jr = UD(gen);
            const real* xi = &X[(size_t)i * D];
            for (int j = 0; j < D; j++) {
                real v = X[(size_t)r1 * D + j] + F * (X[(size_t)r2 * D + j] - X[(size_t)r3 * D + j]);
                trial[j] = (U01(gen) <= CR || j == jr) ? v : xi[j];
            }
            double ft = fitness(trial.data(), 1, D, prm.func, shift.data());
            real* xo = &Xn[(size_t)i * D];
            if (ft < fit[i]) { for (int j = 0; j < D; j++) xo[j] = trial[j]; fitn[i] = ft; }
            else             { for (int j = 0; j < D; j++) xo[j] = xi[j];    fitn[i] = fit[i]; }
        }
        std::swap(X, Xn);
        std::swap(fit, fitn);
    }
    auto t1 = std::chrono::steady_clock::now();

    int b = 0;
    for (int i = 1; i < P; i++) if (fit[i] < fit[b]) b = i;
    OptResult r;
    r.bestFit = fit[b];
    r.gBest.assign(X.begin() + (size_t)b * D, X.begin() + (size_t)(b + 1) * D);
    r.timeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.iters = (int)G; r.fe = (long long)P * (1 + G);
    return r;
}

// ---------------------------------------------------------------------------
// Build without CUDA (-DNO_CUDA) : CPU algorithms only
// ---------------------------------------------------------------------------
#ifdef NO_CUDA
static OptResult no_cuda() { fprintf(stderr, "built with -DNO_CUDA: GPU algorithms unavailable\n"); exit(1); }
OptResult cuda_pso   (const OptParams&, const std::vector<real>&) { return no_cuda(); }
OptResult cuda_de    (const OptParams&, const std::vector<real>&) { return no_cuda(); }
OptResult cuda_de_opt(const OptParams&, const std::vector<real>&) { return no_cuda(); }
#endif
