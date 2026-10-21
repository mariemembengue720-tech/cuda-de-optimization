#include "kernel.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

static const double SUCCESS_EFV = 1e-8;   // a run succeeds if error < 1e-8 (article, Section 5.3)

static void usage(const char* p)
{
    printf(
        "Usage: %s [options]\n"
        "  --algo  all | list      pso-cpu,pso-gpu,de-cpu,de-gpu,de-opt   (comma separated)  [all]\n"
        "  --func  all | 1..4      1 Rastrigin 2 Rosenbrock 3 Griewank 4 Sphere               [all]\n"
        "  --dim   D               problem dimension (<=100 with cec shift)                   [10]\n"
        "  --pop   P               population size (PSO 2..1024, DE >= 4)                     [50]\n"
        "  --runs  R               independent runs (subject : 10)                           [10]\n"
        "  --seed  S               base seed (run k uses S+k, same for every algorithm)       [1234]\n"
        "  --shift cec|zero|rand|file:PATH   shift vector o (same for all algorithms)         [cec]\n"
        "  --F x --CR x            DE parameters                                              [0.5 / 0.3]\n"
        "  --sweep                 dim {10,50,100} x pop {50,100,500}  (subject protocol)\n"
        "  --out   FILE.csv        every run (for Wilcoxon / Kruskal-Wallis)\n"
        "  --verify                self-test f(o) == bias, then exit\n"
        "MaxFEs = 10^4 * dim.\n", p);
}

static std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}

static int parse_algo(const std::string& s)
{
    if (s == "pso-cpu") return A_PSO_CPU;
    if (s == "pso-gpu") return A_PSO_GPU;
    if (s == "de-cpu")  return A_DE_CPU;
    if (s == "de-gpu")  return A_DE_GPU;
    if (s == "de-opt")  return A_DE_OPT;
    return -1;
}

static OptResult run(int algo, const OptParams& p, const std::vector<real>& o)
{
    switch (algo) {
        case A_PSO_CPU: return seq_pso(p, o);
        case A_PSO_GPU: return cuda_pso(p, o);
        case A_DE_CPU:  return seq_de(p, o);
        case A_DE_GPU:  return cuda_de(p, o);
        default:        return cuda_de_opt(p, o);
    }
}

static int verify()
{
    int fail = 0;
    printf("=== self-test : f(o) == bias (CEC2005 shifts, D = 10/50/100) ===\n");
    for (int f = 1; f <= 4; f++)
        for (int D : { 10, 50, 100 }) {
            std::vector<real> o; make_shift(f, D, "cec", o);
            double v = host_fitness(o, f, o), b = bench_info(f).bias;
            bool ok = std::fabs(v - b) < 1e-9;
            printf("  [%s] %-10s D=%3d  f(o) = %.12f  bias = %.1f\n", ok ? "OK " : "FAIL", bench_info(f).name, D, v, b);
            fail += !ok;
        }
    return fail ? 1 : 0;
}

int main(int argc, char** argv)
{
    std::vector<int> algos = { A_PSO_CPU, A_PSO_GPU, A_DE_CPU, A_DE_GPU, A_DE_OPT };
    std::vector<int> funcs = { F_SPHERE, F_RASTRIGIN, F_ROSENBROCK, F_GRIEWANK };
    std::vector<int> dims = { 10 }, pops = { 50 };
    int runs = 10;
    unsigned long long seed = 1234ULL;
    std::string shiftMode = "cec";
    std::string outPath;
    double F = 0.5, CR = 0.3;
    bool doVerify = false;

    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        auto next = [&]() -> std::string { if (a + 1 >= argc) { usage(argv[0]); exit(1); } return argv[++a]; };
        if (s == "--algo") {
            std::string v = next();
            if (v != "all") {
                algos.clear();
                for (auto& t : split(v)) { int id = parse_algo(t); if (id < 0) { fprintf(stderr, "bad --algo '%s'\n", t.c_str()); return 1; } algos.push_back(id); }
            }
        } else if (s == "--func")  { std::string v = next(); if (v != "all") funcs = { atoi(v.c_str()) }; }
        else if (s == "--dim")     dims = { atoi(next().c_str()) };
        else if (s == "--pop")     pops = { atoi(next().c_str()) };
        else if (s == "--runs")    runs = atoi(next().c_str());
        else if (s == "--seed")    seed = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--shift")   shiftMode = next();
        else if (s == "--F")       F = atof(next().c_str());
        else if (s == "--CR")      CR = atof(next().c_str());
        else if (s == "--sweep")   { dims = { 10, 50, 100 }; pops = { 50, 100, 500 }; }
        else if (s == "--out")     outPath = next();
        else if (s == "--verify")  doVerify = true;
        else { usage(argv[0]); return 1; }
    }
    if (doVerify) return verify();

    bool fileShift = shiftMode.rfind("file:", 0) == 0;
    if (fileShift && funcs.size() != 1) { fprintf(stderr, "--shift file:... requires a single --func\n"); return 1; }
    for (int f : funcs) if (f < 1 || f > 4) { fprintf(stderr, "bad --func (1..4)\n"); return 1; }
    for (int d : dims)  if (d < 1) { fprintf(stderr, "--dim must be >= 1\n"); return 1; }
    for (int p : pops)  if (p < 2 || p > 1024) { fprintf(stderr, "--pop must be in [2,1024]\n"); return 1; }
    if (runs < 1) { fprintf(stderr, "--runs must be >= 1\n"); return 1; }

    FILE* csv = nullptr;
    if (!outPath.empty()) {
        csv = fopen(outPath.c_str(), "w");
        if (!csv) { perror("open --out"); return 1; }
        fprintf(csv, "algo,func,dim,pop,run,seed,fe,best,error,time_ms,success,pop_per_block\n");
    }

#ifndef NO_CUDA
    // warm-up (context creation + lazy module loading), discarded
    {
        OptParams w; w.dim = 2; w.pop = 32; w.maxFE = 3200; std::vector<real> o(2, 0);
        for (int a : algos) if (a == A_PSO_GPU || a == A_DE_GPU || a == A_DE_OPT) { (void)run(a, w, o); }
    }
#endif

    int mismatches = 0;
    for (int algo : algos) {
        printf("\n=== %s ===  (%d runs, shift = %s, MaxFEs = 1e4*D)\n", algo_name(algo), runs, shiftMode.c_str());
        printf("%-10s %4s %5s | %-12s %-12s | %-10s %-10s | %5s\n",
               "function", "dim", "pop", "mean_EFV", "std_EFV", "time_s", "std_s", "SR");
        for (int func : funcs) {
            BenchInfo bi = bench_info(func);
            for (int dim : dims) {
                std::vector<real> o;
                if (!make_shift(func, dim, shiftMode, o)) return 1;
                for (int pop : pops) {
                    if (algo == A_PSO_CPU || algo == A_PSO_GPU) { if (pop < 2) continue; }
                    else if (pop < 4) { fprintf(stderr, "DE needs pop >= 4, skipping pop=%d\n", pop); continue; }
                    OptParams prm;
                    prm.func = func; prm.dim = dim; prm.pop = pop; prm.maxFE = 10000LL * dim;
                    prm.F = F; prm.CR = CR;

                    std::vector<double> efv(runs), tms(runs);
                    int succ = 0, ppb = 0;
                    for (int r = 0; r < runs; r++) {
                        prm.seed = seed + (unsigned long long)r;
                        OptResult res = run(algo, prm, o);
                        efv[r] = res.bestFit - bi.bias;
                        tms[r] = res.timeMs;
                        ppb = res.popPerBlock;
                        bool ok = efv[r] < SUCCESS_EFV; succ += ok;

                        double hf = host_fitness(res.gBest, func, o);          // GPU <-> host consistency
                        if (std::fabs(hf - res.bestFit) > 1e-6 * (1.0 + std::fabs(hf))) mismatches++;

                        if (csv) fprintf(csv, "%s,%s,%d,%d,%d,%llu,%lld,%.12e,%.12e,%.3f,%d,%d\n", algo_name(algo), bi.name,
                                         dim, pop, r, prm.seed, res.fe, res.bestFit, efv[r], res.timeMs, ok ? 1 : 0, res.popPerBlock);
                    }
                    auto ms = [&](const std::vector<double>& v, double& m, double& sd) {
                        m = 0; for (double x : v) m += x; m /= v.size();
                        double q = 0; for (double x : v) q += (x - m) * (x - m);
                        sd = v.size() > 1 ? std::sqrt(q / (v.size() - 1)) : 0.0;
                    };
                    double me, se, mt, st;
                    ms(efv, me, se); ms(tms, mt, st);
                    printf("%-10s %4d %5d | %-12.3e %-12.3e | %-10.3f %-10.3f | %5.2f", bi.name, dim, pop, me, se, mt / 1000.0, st / 1000.0,
                           (double)succ / runs);
                    if (ppb) printf("   (m=%d members/block)", ppb);
                    printf("\n");
                    fflush(stdout);
                }
            }
        }
    }
    if (csv) fclose(csv);
    if (mismatches) printf("WARNING: %d run(s) with GPU/host fitness mismatch\n", mismatches);
    return 0;
}
