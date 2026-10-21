// ============================================================================
//  kernel.cu - the 3 GPU architectures
//   PSO-GPU    : kernels I, E, G, U            (coordinate-major layout, coalesced)
//   DE-GPU     : basic cudaDE of the article, Fig. 1 : kernels I, E, P, M, C, E, R
//   DE-GPU-OPT : cudaDEi of the article, Fig. 2 : kernels IE, P, MCER
//                 - I+E and M+C+E+R fused (trial vector lives in shared memory)
//                 - one thread per population-member element
//                 - members per block chosen with the occupancy API
//                 - kernel P runs in a 2nd stream, concurrently with MCER
//  All : synchronous update (double buffering), cuRAND Philox, fitness in double.
//  Timing : wall-clock bracketed by cudaDeviceSynchronize(), allocations and
//  CUDA context creation excluded (warm-up in main).
// ============================================================================
#include "kernel.h"
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

typedef curandStatePhilox4_32_10_t RNG;
static const int TB = 256;                       // threads/block of the element-wise kernels
static inline int nblk(long long n) { return (int)((n + TB - 1) / TB); }
static inline int next_pow2(int v) { int p = 1; while (p < v) p <<= 1; return p; }
static const unsigned long long SEED_X = 0x9E3779B97F4A7C15ULL;   // decorrelates RNG families

template <typename T> static T* dmalloc(size_t n) { T* p; CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); return p; }

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ===========================================================================
//  PSO-GPU
//  position/velocity/pBest stored coordinate-major : element e = d*P + p
// ===========================================================================
__global__ void pso_I(real* pos, real* vel, real* pbest, double* pfit, RNG* st,
                      int n, int P, real lo, real hi, unsigned long long seed)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    RNG s; curand_init(seed, e, 0, &s);
    real x = lo + (hi - lo) * (real)curand_uniform_double(&s);
    pos[e] = x; pbest[e] = x; vel[e] = (real)0; st[e] = s;
    if (e < P) pfit[e] = 1.0e300;
}

// one thread per particle (fitness read with stride P) + pBest update
__global__ void pso_E(const real* pos, real* pbest, double* pfit, int P, int D, int func, const real* o)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    double f = fitness(pos + p, P, D, func, o);
    if (f < pfit[p]) {
        pfit[p] = f;
        for (int d = 0; d < D; d++) pbest[(size_t)d * P + p] = pos[(size_t)d * P + p];
    }
}

// single block (blockDim = power of two >= P, <= 1024) : argmin of pBest fitness
__global__ void pso_G(const double* pfit, const real* pbest, real* gbest, double* gfit, int P, int D)
{
    __shared__ double sv[1024];
    __shared__ int    si[1024];
    int t = threadIdx.x;
    double v = 1.0e300; int id = 0;
    for (int p = t; p < P; p += blockDim.x) if (pfit[p] < v) { v = pfit[p]; id = p; }
    sv[t] = v; si[t] = id;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s && (sv[t + s] < sv[t] || (sv[t + s] == sv[t] && si[t + s] < si[t]))) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
        __syncthreads();
    }
    bool better = sv[0] < *gfit;
    __syncthreads();
    if (better) {
        for (int d = t; d < D; d += blockDim.x) gbest[d] = pbest[(size_t)d * P + si[0]];
        if (t == 0) *gfit = sv[0];
    }
}

__global__ void pso_U(real* pos, real* vel, const real* pbest, const real* gbest, RNG* st,
                      int n, int P, real w, real c1, real c2, real vmax, real lo, real hi)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int d = e / P;
    RNG s = st[e];
    real r1 = (real)curand_uniform_double(&s), r2 = (real)curand_uniform_double(&s);
    st[e] = s;
    real x = pos[e];
    real v = w * vel[e] + c1 * r1 * (pbest[e] - x) + c2 * r2 * (gbest[d] - x);
    if (v >  vmax) v =  vmax;
    if (v < -vmax) v = -vmax;
    x += v;
    if (x < lo)      { x = lo; v = (real)0; }
    else if (x > hi) { x = hi; v = (real)0; }
    vel[e] = v; pos[e] = x;
}

OptResult cuda_pso(const OptParams& prm, const std::vector<real>& shift)
{
    const int P = prm.pop, D = prm.dim;
    if (P < 2 || P > 1024 || D < 1 || (int)shift.size() < D) { fprintf(stderr, "cuda_pso: invalid parameters (2 <= pop <= 1024)\n"); exit(1); }
    const BenchInfo bi = bench_info(prm.func);
    const int n = P * D;
    long long nIter = prm.maxFE / P - 1; if (nIter < 0) nIter = 0;
    const real lo = (real)bi.lo, hi = (real)bi.hi, vmax = (real)(prm.vmaxFrac * (bi.hi - bi.lo));

    real *pos = dmalloc<real>(n), *vel = dmalloc<real>(n), *pbest = dmalloc<real>(n);
    real *gbest = dmalloc<real>(D), *o = dmalloc<real>(D);
    double *pfit = dmalloc<double>(P), *gfit = dmalloc<double>(1);
    RNG* st = dmalloc<RNG>(n);
    CUDA_CHECK(cudaMemcpy(o, shift.data(), D * sizeof(real), cudaMemcpyHostToDevice));
    double inf = 1.0e300;
    CUDA_CHECK(cudaMemcpy(gfit, &inf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(gbest, 0, D * sizeof(real)));
    const int gB = next_pow2(P);

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    pso_I<<<nblk(n), TB>>>(pos, vel, pbest, pfit, st, n, P, lo, hi, prm.seed);
    pso_E<<<nblk(P), TB>>>(pos, pbest, pfit, P, D, prm.func, o);
    pso_G<<<1, gB>>>(pfit, pbest, gbest, gfit, P, D);
    for (long long it = 0; it < nIter; it++) {
        real w = (real)((nIter > 1) ? prm.wMax - (prm.wMax - prm.wMin) * (double)it / (double)(nIter - 1) : prm.wMax);
        pso_U<<<nblk(n), TB>>>(pos, vel, pbest, gbest, st, n, P, w, (real)prm.c1, (real)prm.c2, vmax, lo, hi);
        pso_E<<<nblk(P), TB>>>(pos, pbest, pfit, P, D, prm.func, o);
        pso_G<<<1, gB>>>(pfit, pbest, gbest, gfit, P, D);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_ms();

    OptResult r;
    r.gBest.resize(D);
    CUDA_CHECK(cudaMemcpy(r.gBest.data(), gbest, D * sizeof(real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&r.bestFit, gfit, sizeof(double), cudaMemcpyDeviceToHost));
    r.timeMs = t1 - t0; r.iters = (int)nIter; r.fe = (long long)P * (1 + nIter);
    cudaFree(pos); cudaFree(vel); cudaFree(pbest); cudaFree(gbest); cudaFree(o);
    cudaFree(pfit); cudaFree(gfit); cudaFree(st);
    return r;
}

// ===========================================================================
//  DE common : population stored individual-major  X[i*D + j]
// ===========================================================================

// ---- basic DE (Fig. 1) -----------------------------------------------------
// kernel(I) : population init + RNG states (crossover : 1 per element, indices : 1 per individual)
__global__ void de_I(real* X, RNG* stE, RNG* stP, int n, int P, real lo, real hi, unsigned long long seed)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    RNG s; curand_init(seed, e, 0, &s);
    X[e] = lo + (hi - lo) * (real)curand_uniform_double(&s);
    stE[e] = s;
    if (e < P) { RNG q; curand_init(seed ^ SEED_X, e, 0, &q); stP[e] = q; }
}

// kernel(E) : ONE BLOCK PER INDIVIDUAL, D threads (padded to a power of two) + tree reduction
__global__ void de_E(const real* X, double* fit, int D, int func, const real* o)
{
    extern __shared__ double sh[];
    double* sa = sh; double* sb = sh + blockDim.x;
    const int i = blockIdx.x, j = threadIdx.x;
    double a = 0.0, b = 1.0;
    if (j < D) {
        const real* x = X + (size_t)i * D;
        fit_term(func, j, D, (double)x[j], (j + 1 < D) ? (double)x[j + 1] : 0.0,
                 (double)o[j], (j + 1 < D) ? (double)o[j + 1] : 0.0, a, b);
    }
    sa[j] = a; sb[j] = b;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (j < s) { sa[j] += sa[j + s]; sb[j] *= sb[j + s]; }
        __syncthreads();
    }
    if (j == 0) fit[i] = fit_final(func, sa[0], sb[0]);
}

// kernel(P) : r1,r2,r3 mutually distinct and != i, plus j_rand  -> idx[4*i .. 4*i+3]
__global__ void de_P(RNG* stP, int* idx, int P, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    RNG s = stP[i];
    int r1, r2, r3;
    do { r1 = curand(&s) % P; } while (r1 == i);
    do { r2 = curand(&s) % P; } while (r2 == i || r2 == r1);
    do { r3 = curand(&s) % P; } while (r3 == i || r3 == r1 || r3 == r2);
    int jr = curand(&s) % D;
    stP[i] = s;
    reinterpret_cast<int4*>(idx)[i] = make_int4(r1, r2, r3, jr);
}

// kernel(M) : V = X_r1 + F (X_r2 - X_r3)
__global__ void de_M(const real* X, real* V, const int* idx, int n, int D, real F)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int i = e / D, j = e - i * D;
    int4 id = reinterpret_cast<const int4*>(idx)[i];
    V[e] = X[(size_t)id.x * D + j] + F * (X[(size_t)id.y * D + j] - X[(size_t)id.z * D + j]);
}

// kernel(C) : binomial crossover
__global__ void de_C(const real* X, const real* V, real* U, const int* idx, RNG* stE, int n, int D, real CR)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int i = e / D, j = e - i * D;
    RNG s = stE[e];
    real u = (real)curand_uniform_double(&s);
    stE[e] = s;
    int jr = idx[4 * i + 3];
    U[e] = (u <= CR || j == jr) ? V[e] : X[e];
}

// kernel(R) : greedy replacement (X in place : only its own element is touched)
__global__ void de_R(real* X, const real* U, const double* fit, const double* fitU, double* fitNext, int n, int D)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int i = e / D;
    bool acc = fitU[i] < fit[i];
    if (acc) X[e] = U[e];
    if (e - i * D == 0) fitNext[i] = acc ? fitU[i] : fit[i];
}

static OptResult de_collect(const real* X, const double* fit, int P, int D, double t0, double t1, long long G)
{
    OptResult r;
    std::vector<double> hf(P);
    CUDA_CHECK(cudaMemcpy(hf.data(), fit, P * sizeof(double), cudaMemcpyDeviceToHost));
    int b = 0;
    for (int i = 1; i < P; i++) if (hf[i] < hf[b]) b = i;
    r.bestFit = hf[b];
    r.gBest.resize(D);
    CUDA_CHECK(cudaMemcpy(r.gBest.data(), X + (size_t)b * D, D * sizeof(real), cudaMemcpyDeviceToHost));
    r.timeMs = t1 - t0; r.iters = (int)G; r.fe = (long long)P * (1 + G);
    return r;
}

OptResult cuda_de(const OptParams& prm, const std::vector<real>& shift)
{
    const int P = prm.pop, D = prm.dim;
    if (P < 4 || D < 1 || (int)shift.size() < D) { fprintf(stderr, "cuda_de: invalid parameters (pop >= 4)\n"); exit(1); }
    const BenchInfo bi = bench_info(prm.func);
    const int n = P * D;
    long long G = prm.maxFE / P - 1; if (G < 0) G = 0;
    const real lo = (real)bi.lo, hi = (real)bi.hi;

    real *X = dmalloc<real>(n), *V = dmalloc<real>(n), *U = dmalloc<real>(n), *o = dmalloc<real>(D);
    double *fit = dmalloc<double>(P), *fit2 = dmalloc<double>(P), *fitU = dmalloc<double>(P);
    int* idx = dmalloc<int>(4 * P);
    RNG *stE = dmalloc<RNG>(n), *stP = dmalloc<RNG>(P);
    CUDA_CHECK(cudaMemcpy(o, shift.data(), D * sizeof(real), cudaMemcpyHostToDevice));
    const int eB = std::max(32, next_pow2(D));
    const size_t eSmem = 2 * eB * sizeof(double);

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    de_I<<<nblk(n), TB>>>(X, stE, stP, n, P, lo, hi, prm.seed);
    de_E<<<P, eB, eSmem>>>(X, fit, D, prm.func, o);
    for (long long g = 0; g < G; g++) {
        de_P<<<nblk(P), TB>>>(stP, idx, P, D);
        de_M<<<nblk(n), TB>>>(X, V, idx, n, D, (real)prm.F);
        de_C<<<nblk(n), TB>>>(X, V, U, idx, stE, n, D, (real)prm.CR);
        de_E<<<P, eB, eSmem>>>(U, fitU, D, prm.func, o);
        de_R<<<nblk(n), TB>>>(X, U, fit, fitU, fit2, n, D);
        std::swap(fit, fit2);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_ms();

    OptResult r = de_collect(X, fit, P, D, t0, t1, G);
    cudaFree(X); cudaFree(V); cudaFree(U); cudaFree(o); cudaFree(fit); cudaFree(fit2); cudaFree(fitU);
    cudaFree(idx); cudaFree(stE); cudaFree(stP);
    return r;
}

// ===========================================================================
//  DE-GPU-OPT  (cudaDEi)
//  block = m population members x D threads (thread t : member mi = t / D, coord j = t % D)
//  dynamic shared memory : trial[m*D], partial sums[m*D], partial products[m*D]
// ===========================================================================
__device__ __forceinline__ void seg_reduce(double* sa, double* sb, int t, int j, int D, int p2)
{
    for (int s = p2 >> 1; s > 0; s >>= 1) {
        if (j < s && j + s < D) { sa[t] += sa[t + s]; sb[t] *= sb[t + s]; }
        __syncthreads();
    }
}

// kernel(IE) : init + RNG states + evaluation, fused
__global__ void de_IE(real* X, double* fit, RNG* stE, RNG* stP, int P, int D, int m,
                      real lo, real hi, unsigned long long seed, int func, const real* o)
{
    extern __shared__ double sh[];
    const int T = m * D, t = threadIdx.x, mi = t / D, j = t - mi * D;
    const int i = blockIdx.x * m + mi;
    const bool valid = i < P;
    double *xs = sh, *sa = sh + T, *sb = sh + 2 * T;
    double x = 0.0;
    if (valid) {
        const int e = i * D + j;
        RNG s; curand_init(seed, e, 0, &s);
        x = lo + (hi - lo) * (real)curand_uniform_double(&s);
        X[e] = (real)x; stE[e] = s;
        if (j == 0) {
            RNG q0, q1;
            curand_init(seed ^ SEED_X, i, 0, &q0);
            curand_init((seed ^ SEED_X) + 1ULL, i, 0, &q1);
            stP[i] = q0; stP[P + i] = q1;                    // 2 families : one per stream
        }
    }
    xs[t] = x;
    __syncthreads();
    double a = 0.0, b = 1.0;
    if (valid) fit_term(func, j, D, xs[t], (j + 1 < D) ? xs[t + 1] : 0.0,
                        (double)o[j], (j + 1 < D) ? (double)o[j + 1] : 0.0, a, b);
    sa[t] = a; sb[t] = b;
    __syncthreads();
    int p2 = 1; while (p2 < D) p2 <<= 1;
    seg_reduce(sa, sb, t, j, D, p2);
    if (valid && j == 0) fit[i] = fit_final(func, sa[mi * D], sb[mi * D]);
}

// kernel(MCER) : mutation + crossover + evaluation + replacement, fused.
// Reads population Xin / fitIn, writes Xout / fitOut (synchronous DE, no race).
__global__ void de_MCER(const real* Xin, real* Xout, const double* fitIn, double* fitOut,
                        const int* idx, RNG* stE, int P, int D, int m, int func, const real* o,
                        real F, real CR)
{
    extern __shared__ double sh[];
    const int T = m * D, t = threadIdx.x, mi = t / D, j = t - mi * D;
    const int i = blockIdx.x * m + mi;
    const bool valid = i < P;
    double *tr = sh, *sa = sh + T, *sb = sh + 2 * T;
    const int e = i * D + j;
    double xi = 0.0, trial = 0.0, fcur = 0.0;
    if (valid) {
        int4 id = reinterpret_cast<const int4*>(idx)[i];
        xi = Xin[e];
        double v = Xin[(size_t)id.x * D + j] + (double)F * (Xin[(size_t)id.y * D + j] - Xin[(size_t)id.z * D + j]);
        RNG s = stE[e];
        real u = (real)curand_uniform_double(&s);
        stE[e] = s;
        trial = (u <= CR || j == id.w) ? v : xi;
        fcur = fitIn[i];
    }
    tr[t] = trial;
    __syncthreads();
    double a = 0.0, b = 1.0;
    if (valid) fit_term(func, j, D, tr[t], (j + 1 < D) ? tr[t + 1] : 0.0,
                        (double)o[j], (j + 1 < D) ? (double)o[j + 1] : 0.0, a, b);
    sa[t] = a; sb[t] = b;
    __syncthreads();
    int p2 = 1; while (p2 < D) p2 <<= 1;
    seg_reduce(sa, sb, t, j, D, p2);
    if (valid) {
        double f = fit_final(func, sa[mi * D], sb[mi * D]);
        bool acc = f < fcur;
        Xout[e] = (real)(acc ? trial : xi);
        if (j == 0) fitOut[i] = acc ? f : fcur;
    }
}

// choose m (members per block).
//  Criterion 1 (spread the work over ALL SMs) : minimise the load of the busiest SM, in warps :
//        ceil(grid / numSM) * ceil(m*D / 32)
//      (maximising occupancy alone picks m = 1024/D : P=50, D=100 -> only 5 blocks = 5 busy SMs out of 40)
//  Criterion 2 : higher occupancy given by the occupancy API (registers / shared memory / block limits)
//  Criterion 3 : larger m (fewer blocks, as in the article)
//  Override for experiments : environment variable DE_OPT_M=<m>
static int choose_members_per_block(int P, int D, int* occPct)
{
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp pr; CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
    const int mmax = std::min(P, pr.maxThreadsPerBlock / D);
    if (mmax < 1) { fprintf(stderr, "cuda_de_opt: dim %d exceeds max threads per block\n", D); exit(1); }

    auto eval = [&](int m, long long& occ, long long& load) -> bool {
        size_t smem = 3ull * m * D * sizeof(double);
        if (smem > pr.sharedMemPerBlock) return false;
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, de_MCER, m * D, smem));
        long long grid = (P + m - 1) / m, warps = ((long long)m * D + 31) / 32;
        occ  = (long long)nb * m * D;
        load = ((grid + pr.multiProcessorCount - 1) / pr.multiProcessorCount) * warps;
        return true;
    };

    int bestM = 1; long long bestLoad = -1, bestOcc = -1;
    for (int m = 1; m <= mmax; m++) {
        long long occ, load;
        if (!eval(m, occ, load)) break;
        if (bestLoad < 0 || load < bestLoad || (load == bestLoad && occ >= bestOcc)) { bestM = m; bestLoad = load; bestOcc = occ; }
    }
    if (const char* ev = getenv("DE_OPT_M")) {
        int m = atoi(ev);
        if (m >= 1) { bestM = std::min(m, mmax); long long l; if (!eval(bestM, bestOcc, l)) { bestM = 1; eval(1, bestOcc, l); } }
    }
    *occPct = (int)(100.0 * bestOcc / pr.maxThreadsPerMultiProcessor);
    return bestM;
}

OptResult cuda_de_opt(const OptParams& prm, const std::vector<real>& shift)
{
    const int P = prm.pop, D = prm.dim;
    if (P < 4 || D < 1 || (int)shift.size() < D) { fprintf(stderr, "cuda_de_opt: invalid parameters (pop >= 4)\n"); exit(1); }
    const BenchInfo bi = bench_info(prm.func);
    long long G = prm.maxFE / P - 1; if (G < 0) G = 0;
    const real lo = (real)bi.lo, hi = (real)bi.hi;
    const int n = P * D;

    int occ = 0;
    const int m = choose_members_per_block(P, D, &occ);
    const int T = m * D, grid = (P + m - 1) / m;
    const size_t smem = 3ull * T * sizeof(double);

    real *X[2] = { dmalloc<real>(n), dmalloc<real>(n) };
    double *fit[2] = { dmalloc<double>(P), dmalloc<double>(P) };
    int* idx[2] = { dmalloc<int>(4 * P), dmalloc<int>(4 * P) };
    RNG *stE = dmalloc<RNG>(n), *stP = dmalloc<RNG>(2 * P);          // stP[0..P) stream 0, stP[P..2P) stream 1
    real* o = dmalloc<real>(D);
    CUDA_CHECK(cudaMemcpy(o, shift.data(), D * sizeof(real), cudaMemcpyHostToDevice));

    cudaStream_t S[2]; cudaEvent_t done[2];
    for (int k = 0; k < 2; k++) {
        CUDA_CHECK(cudaStreamCreate(&S[k]));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[k], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    de_IE<<<grid, T, smem, S[0]>>>(X[0], fit[0], stE, stP, P, D, m, lo, hi, prm.seed, prm.func, o);
    if (G > 0) de_P<<<nblk(P), TB, 0, S[0]>>>(stP, idx[0], P, D);
    for (long long g = 0; g < G; g++) {
        const int s = (int)(g & 1), c = s, nx = c ^ 1;               // population buffer in = c, out = nx
        if (g > 0) CUDA_CHECK(cudaStreamWaitEvent(S[s], done[s ^ 1], 0));   // MCER(g-1) finished
        de_MCER<<<grid, T, smem, S[s]>>>(X[c], X[nx], fit[c], fit[nx], idx[s], stE, P, D, m,
                                         prm.func, o, (real)prm.F, (real)prm.CR);
        CUDA_CHECK(cudaEventRecord(done[s], S[s]));
        if (g + 1 < G)                                               // kernel(P) of gen g+1, other stream, concurrent
            de_P<<<nblk(P), TB, 0, S[s ^ 1]>>>(stP + (size_t)(s ^ 1) * P, idx[s ^ 1], P, D);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_ms();

    const int fin = (int)(G & 1);
    OptResult r = de_collect(X[fin], fit[fin], P, D, t0, t1, G);
    r.popPerBlock = m;
    (void)occ;
    for (int k = 0; k < 2; k++) { cudaStreamDestroy(S[k]); cudaEventDestroy(done[k]); cudaFree(X[k]); cudaFree(fit[k]); cudaFree(idx[k]); }
    cudaFree(stE); cudaFree(stP); cudaFree(o);
    return r;
}
