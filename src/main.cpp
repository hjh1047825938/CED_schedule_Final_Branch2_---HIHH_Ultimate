#include "Multimethod.h"
#include "CC_HIHH.h"
#include "GA_SLHH.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <random>
#include <vector>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <cmath>
using namespace std;

// Default parameters
#define DEFAULT_ENUM 100
#define DEFAULT_CNUM 100
#define DEFAULT_DNUM 300
#define DEFAULT_TNUM 100
#define DEFAULT_MOPT_NUM 5
#define DEFAULT_MAXGEN 500
#define DEFAULT_POPSIZE 40
#define DEFAULT_PINI 0.4
#define DEFAULT_SEED 42

void print_usage(const char* prog_name) {
    cout << "Usage: " << prog_name << " [OPTIONS]\n";
    cout << "\nOptions:\n";
    cout << "  --data_dir <path>    Data directory (default: ./data)\n";
    cout << "  --data_file <name>   Data file name (default: data_matrix_100.txt)\n";
    cout << "  --generations <n>    Number of generations (default: " << DEFAULT_MAXGEN << ")\n";
    cout << "  --popsize <n>        Population size (default: " << DEFAULT_POPSIZE << ")\n";
    cout << "  --seed <n>           Random seed (default: " << DEFAULT_SEED << ")\n";
    cout << "  --pini <f>           Heuristic init probability 0-1 (default: " << DEFAULT_PINI << ")\n";
    cout << "  --solver <name>      Solver: GA, DE, GDE, CCHIHH, GA-SLHH (default: GA)\n";
    cout << "  --cnum <n>           Number of cloud servers (default: " << DEFAULT_CNUM << ")\n";
    cout << "  --enum <n>           Number of edge servers (default: " << DEFAULT_ENUM << ")\n";
    cout << "  --dnum <n>           Number of devices (default: " << DEFAULT_DNUM << ")\n";
    cout << "  --tnum <n>           Number of tasks (default: " << DEFAULT_TNUM << ")\n";
    cout << "  --mopt <n>           Operations per task (default: " << DEFAULT_MOPT_NUM << ")\n";
    cout << "  --migration          Enable rotated-ring subpopulation migration\n";
    cout << "  --nsubpop <n>        Number of subpopulations for migration (default: 8)\n";
    cout << "  --log_every <n>      Log best_fit every n generations (or evals if --max_evals is set, default: 50)\n";
    cout << "  --max_evals <n>      Stop after N evaluation calls (0 = disabled)\n";
    cout << "  --stable             Enable CCHIHH-Stable mode\n";
    cout << "  --cchihh_no_migration  Disable CCHIHH intra-block migration\n";
    cout << "  --cchihh_random_ops    Disable contextual bandit, random operators\n";
    cout << "  --cchihh_no_blocks     Disable CC blocks, run on full variable space\n";
    cout << "  --cchihh_op_stats <p>  Write operator frequency CSV to path\n";
    cout << "  --cchihh_op_stats_every <n>  Operator stats logging interval (default: log_every)\n";
    cout << "  --resample_gate <n>  Stagnation gate for block resample (default: 15)\n";
    cout << "  --reward_clip <f>    Stable reward clip (default: 0.2)\n";
    cout << "  --eps0 <f>           Stable epsilon start (default: 0.2)\n";
    cout << "  --eps_min <f>        Stable epsilon min (default: 0.02)\n";
    cout << "  --eps_k <f>          Stable epsilon decay k (default: 0.01)\n";
    cout << "  --lr0 <f>            Stable learning rate start (default: 0.05)\n";
    cout << "  --lr_k <f>           Stable learning rate decay k (default: 0.002)\n";
    cout << "  --bench_eval <n>     Run evaluation benchmark with N iterations\n";
    cout << "  --init_only          Only run initialization and print Pini comparisons\n";
    cout << "  --synthetic          Run synthetic phi encoding/decoding self-check\n";
    cout << "  --help               Show this help message\n";
}

#ifdef PROFILE_EVAL
static void PrintEvalProfile(const MultiMet& solver)
{
    const auto& p = solver.workspace.profile;
    if (p.samples == 0) {
        cout << "\n=== Eval Profiling Report ===" << endl;
        cout << "No samples collected." << endl;
        return;
    }
    auto avg = [samples = p.samples](uint64_t total) -> double {
        return (samples > 0) ? (double)total / (double)samples : 0.0;
    };
    cout << "\n=== Eval Profiling Report (avg us per eval) ===" << endl;
    cout << "Samples: " << p.samples << endl;
    cout << "decode   : " << fixed << setprecision(3) << avg(p.decode_us) << " us" << endl;
    cout << "sort     : " << fixed << setprecision(3) << avg(p.sort_us) << " us" << endl;
    cout << "assign   : " << fixed << setprecision(3) << avg(p.assign_us) << " us" << endl;
    cout << "schedule : " << fixed << setprecision(3) << avg(p.schedule_us) << " us" << endl;
    cout << "devices  : " << fixed << setprecision(3) << avg(p.devices_us) << " us" << endl;
    cout << "comm     : " << fixed << setprecision(3) << avg(p.comm_us) << " us" << endl;
    cout << "tasks    : " << fixed << setprecision(3) << avg(p.tasks_us) << " us" << endl;
}
#endif

static uint64_t HashVar(const double* v, int n)
{
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++)
    {
        uint64_t x;
        static_assert(sizeof(double) == sizeof(uint64_t), "double size mismatch");
        memcpy(&x, &v[i], sizeof(uint64_t));
        h ^= x;
        h *= 1099511628211ull;
    }
    return h;
}

static void RunSynthetic(unsigned int seed)
{
    const int N = 10;
    const int pop = 20;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<int> E_sizes(N);
    for (int i = 0; i < N; i++)
        E_sizes[i] = 5 + (i % 3);

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < pop; i++)
    {
        std::vector<double> phi(2 * N);
        for (int j = 0; j < 2 * N; j++)
            phi[j] = dist(rng);

        double fit = 0.0;
        for (int j = 0; j < N; j++)
        {
            int X = (int)floor(phi[j] * 3.0);
            if (X > 2) X = 2;
            int Y = (int)floor(phi[N + j] * E_sizes[j]);
            if (Y >= E_sizes[j]) Y = E_sizes[j] - 1;
            fit += X + Y;
        }
        if (fit < best) best = fit;
    }

    cout << "=== Synthetic 2N phi decode check ===" << endl;
    cout << "N=" << N << ", pop=" << pop << ", seed=" << seed << endl;
    cout << "Best synthetic fitness (1 round): " << best << endl;
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments
    filesystem::path data_dir = "data";  // Default: ./data
    string data_file = "data_matrix_100.txt";
    int max_generations = DEFAULT_MAXGEN;
    int popsize = DEFAULT_POPSIZE;
    unsigned int seed = DEFAULT_SEED;
    double pini = DEFAULT_PINI;
    string solver_name = "GA";
    int bench_eval = 0;
    bool migration_enabled = false;
    int nsubpop = 8;
    int log_every = 50;
    uint64_t max_evals = 0;
    bool init_only = false;
    bool synthetic_mode = false;
    bool stable_mode = false;
    bool cchihh_migration = true;
    bool cchihh_random_ops = false;
    bool cchihh_no_blocks = false;
    string cchihh_op_stats_path;
    int cchihh_op_stats_every = 0;
    int resample_gate = 15;
    double stable_reward_clip = 0.2;
    double eps0 = 0.2;
    double eps_min = 0.02;
    double eps_k = 0.01;
    double lr0 = 0.05;
    double lr_k = 0.002;
    int Cnum = DEFAULT_CNUM;
    int Enum = DEFAULT_ENUM;
    int Dnum = DEFAULT_DNUM;
    int Tnum = DEFAULT_TNUM;
    int Mopt_num = DEFAULT_MOPT_NUM;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data_dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--data_file") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
            max_generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--popsize") == 0 && i + 1 < argc) {
            popsize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pini") == 0 && i + 1 < argc) {
            pini = atof(argv[++i]);
        } else if (strcmp(argv[i], "--solver") == 0 && i + 1 < argc) {
            solver_name = argv[++i];
        } else if (strcmp(argv[i], "--cnum") == 0 && i + 1 < argc) {
            Cnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--enum") == 0 && i + 1 < argc) {
            Enum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dnum") == 0 && i + 1 < argc) {
            Dnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tnum") == 0 && i + 1 < argc) {
            Tnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mopt") == 0 && i + 1 < argc) {
            Mopt_num = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench_eval") == 0 && i + 1 < argc) {
            bench_eval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--migration") == 0) {
            migration_enabled = true;
        } else if (strcmp(argv[i], "--nsubpop") == 0 && i + 1 < argc) {
            nsubpop = atoi(argv[++i]);
            if (nsubpop < 1) {
                cerr << "Error: --nsubpop must be >= 1" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--log_every") == 0 && i + 1 < argc) {
            log_every = atoi(argv[++i]);
            if (log_every < 1) log_every = 1;
        } else if (strcmp(argv[i], "--max_evals") == 0 && i + 1 < argc) {
            max_evals = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--stable") == 0) {
            stable_mode = true;
        } else if (strcmp(argv[i], "--cchihh_no_migration") == 0) {
            cchihh_migration = false;
        } else if (strcmp(argv[i], "--cchihh_random_ops") == 0) {
            cchihh_random_ops = true;
        } else if (strcmp(argv[i], "--cchihh_no_blocks") == 0) {
            cchihh_no_blocks = true;
        } else if (strcmp(argv[i], "--cchihh_op_stats") == 0 && i + 1 < argc) {
            cchihh_op_stats_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_op_stats_every") == 0 && i + 1 < argc) {
            cchihh_op_stats_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--resample_gate") == 0 && i + 1 < argc) {
            resample_gate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reward_clip") == 0 && i + 1 < argc) {
            stable_reward_clip = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eps0") == 0 && i + 1 < argc) {
            eps0 = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eps_min") == 0 && i + 1 < argc) {
            eps_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eps_k") == 0 && i + 1 < argc) {
            eps_k = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr0") == 0 && i + 1 < argc) {
            lr0 = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr_k") == 0 && i + 1 < argc) {
            lr_k = atof(argv[++i]);
        } else if (strcmp(argv[i], "--init_only") == 0) {
            init_only = true;
        } else if (strcmp(argv[i], "--synthetic") == 0) {
            synthetic_mode = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Print configuration
    cout << "=== CED_Schedule Configuration ===" << endl;
    cout << "Data directory: " << data_dir << endl;
    cout << "Data file: " << data_file << endl;
    cout << "Generations: " << max_generations << endl;
    cout << "Population size: " << popsize << endl;
    cout << "Random seed: " << seed << endl;
    cout << "Pini (heuristic prob): " << pini << endl;
    cout << "Solver: " << solver_name << endl;
    cout << "Cnum/Enum/Dnum/Tnum/Mopt: " << Cnum << "/" << Enum << "/" << Dnum
         << "/" << Tnum << "/" << Mopt_num << endl;
    cout << "Migration: " << (migration_enabled ? "enabled" : "disabled") << endl;
    if (migration_enabled) cout << "  Subpopulations: " << nsubpop << endl;
    if (solver_name == "CCHIHH" && stable_mode) {
        cout << "Stable mode: enabled" << endl;
        cout << "  resample_gate=" << resample_gate
             << " reward_clip=" << stable_reward_clip
             << " eps0=" << eps0
             << " eps_min=" << eps_min
             << " eps_k=" << eps_k
             << " lr0=" << lr0
             << " lr_k=" << lr_k << endl;
    }
    if (solver_name == "CCHIHH") {
        cout << "CCHIHH blocks: " << (cchihh_no_blocks ? "disabled" : "enabled") << endl;
        cout << "CCHIHH migration: " << (cchihh_migration ? "enabled" : "disabled") << endl;
        cout << "CCHIHH bandit: " << (cchihh_random_ops ? "random" : "enabled") << endl;
        if (!cchihh_op_stats_path.empty()) {
            cout << "CCHIHH op stats: " << cchihh_op_stats_path << endl;
        }
    }
    cout << "=================================" << endl;

    if (Cnum <= 0 || Enum <= 0 || Dnum <= 0 || Tnum <= 0 || Mopt_num <= 0) {
        cerr << "Error: --cnum/--enum/--dnum/--tnum/--mopt must be positive integers." << endl;
        return 1;
    }
    
    if (synthetic_mode) {
        RunSynthetic(seed);
        return 0;
    }

    if (init_only) {
        cout << "\n=== Init Pini Comparison ===" << endl;
        std::vector<unsigned int> seeds = { seed, seed + 1, seed + 2 };
        for (unsigned int s : seeds) {
            srand(s);
            MultiMet solver_a(popsize, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                              Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
            solver_a.SetSeed(s);
            solver_a.SetPini(1.0);
            solver_a.Initial();
            double best_a = solver_a.gbest_fit;

            srand(s);
            MultiMet solver_b(popsize, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                              Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
            solver_b.SetSeed(s);
            solver_b.SetPini(DEFAULT_PINI);
            solver_b.Initial();
            double best_b = solver_b.gbest_fit;

            cout << "Seed " << s << ": Pini=1.0 best=" << best_a << ", Pini=" << DEFAULT_PINI << " best=" << best_b << endl;
        }
        return 0;
    }

    srand(seed);
    
    // Create solver with data directory
    MultiMet solver(popsize, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                    Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
    solver.SetSeed(seed);
    solver.SetPini(pini);
    solver.Initial();
    solver.ResetEvalCount();
    
    // Initialize migration if enabled (nG=nsubpop, nCircle=5, pElitist=0.8)
    if (migration_enabled) {
        if (nsubpop < 2) {
            cerr << "Error: --nsubpop must be >= 2 when migration is enabled" << endl;
            return 1;
        }
        solver.InitMigration(nsubpop, 5, 0.8);
    }
    
    // Benchmark mode
    if (bench_eval > 0) {
        cout << "\n=== Benchmark Evaluation Mode ===" << endl;
        cout << "Running " << bench_eval << " evaluations..." << endl;

        std::vector<double> fixed_var(solver.Nvar);
        for (int j = 0; j < solver.Nvar; j++)
            fixed_var[j] = solver.pop[0][j];
        uint64_t h = HashVar(fixed_var.data(), solver.Nvar);
        cout << "Fixed var hash: 0x" << std::hex << h << std::dec << endl;
        cout << "Fixed var head: ";
        for (int j = 0; j < std::min(5, solver.Nvar); j++)
            cout << fixed_var[j] << " ";
        cout << endl;

        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_eval; i++) {
            solver.Eval(fixed_var.data());
        }
        auto t2 = std::chrono::steady_clock::now();

        double total_time = std::chrono::duration<double>(t2 - t1).count();
        double per_eval = total_time / bench_eval;
        cout << "Total time: " << std::fixed << std::setprecision(6) << total_time << " s" << endl;
        cout << "Mean time: " << std::fixed << std::setprecision(3) << per_eval * 1000 << " ms" << endl;
        
        return 0;
    }
    
    clock_t t1 = clock();
    
    // CC-HIHH-UCB Solver (Cooperative Coevolution + Heterogeneous Island Hyper-Heuristic + UCB1)
    if (solver_name == "CCHIHH") {
        cout << "\n=== Running CC-HIHH-UCB Solver ===" << endl;
        
        // Create CC-HIHH solver with nsubpop islands per block
        CC_HIHH_Solver cc_solver(&solver, popsize, nsubpop, 5 /*nCircle*/, 0.8 /*pElitist*/);
        cc_solver.SetMaxGenerations(max_generations);
        cc_solver.SetUseBlocks(!cchihh_no_blocks);
        cc_solver.SetMigrationEnabled(cchihh_migration);
        cc_solver.SetUseBandit(!cchihh_random_ops);
        if (!cchihh_op_stats_path.empty()) {
            int stats_every = cchihh_op_stats_every > 0 ? cchihh_op_stats_every : log_every;
            cc_solver.SetOpStats(cchihh_op_stats_path, stats_every);
        }
        if (stable_mode) {
            cc_solver.SetStableMode(true);
            cc_solver.SetResampleGate(resample_gate);
            cc_solver.SetStableRewardClip(stable_reward_clip);
            cc_solver.SetEpsilonParams(eps0, eps_min, eps_k);
            cc_solver.SetLearningRateParams(lr0, lr_k);
        }
        cc_solver.Init();
        solver.ResetEvalCount();
        uint64_t next_log_eval = (uint64_t)log_every;
        
        for (int gen = 0; gen < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals); gen++) {
            cc_solver.RunGeneration(gen);
            cc_solver.LogOpStatsIfNeeded(gen, gen == max_generations - 1);
            
            if (max_evals > 0) {
                while (solver.GetEvalCount() >= next_log_eval) {
                    cout << "Eval " << next_log_eval << ": best_fit = " << cc_solver.GetGlobalBestFit() << endl;
                    next_log_eval += (uint64_t)log_every;
                }
            } else if ((gen + 1) % log_every == 0 || gen == max_generations - 1) {
                cout << "Gen " << (gen + 1) << ": best_fit = " << cc_solver.GetGlobalBestFit() << endl;
            }
        }
        
        clock_t t2 = clock();
        
        cout << "\n=== Final Results (CC-HIHH-UCB) ===" << endl;
        cout << "Solver: " << solver_name << endl;
        cout << "Subpopulations per block: " << nsubpop << endl;
        cout << "Generations = " << max_generations << endl;
        cout << "The best solution = " << cc_solver.GetGlobalBestFit() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;

#ifdef PROFILE_EVAL
        PrintEvalProfile(solver);
#endif
        return 0;
    }

    if (solver_name == "GA-SLHH") {
        cout << "\n=== Running GA-SLHH Solver ===" << endl;
        GA_SLHH_Solver slhh(&solver, popsize);
        slhh.SetMaxGenerations(max_generations);
        slhh.SetCrossoverRate(0.8);
        slhh.SetMutationRate(0.3);
        slhh.SetGeneMutationRate(0.02);
        slhh.SetImmigrantRate(0.1);
        slhh.SetElitismCount(2);
        slhh.SetLocalSearchTrials(50);
        slhh.Init();
        solver.ResetEvalCount();
        uint64_t next_log_eval = (uint64_t)log_every;

        for (int gen = 0; gen < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals); gen++) {
            slhh.RunGeneration(gen);

            if (max_evals > 0) {
                while (solver.GetEvalCount() >= next_log_eval) {
                    cout << "Eval " << next_log_eval << ": best_fit = " << slhh.GetBestFit()
                         << " mean_fit = " << slhh.GetLastMeanFit()
                         << " unique_llh = " << slhh.GetLastUniqueLLH() << endl;
                    next_log_eval += (uint64_t)log_every;
                }
            } else if ((gen + 1) % log_every == 0 || gen == max_generations - 1) {
                cout << "Gen " << (gen + 1) << ": best_fit = " << slhh.GetBestFit()
                     << " mean_fit = " << slhh.GetLastMeanFit()
                     << " unique_llh = " << slhh.GetLastUniqueLLH() << endl;
            }
        }

        clock_t t2 = clock();
        cout << "\n=== Final Results (GA-SLHH) ===" << endl;
        cout << "Solver: " << solver_name << endl;
        cout << "Generation = " << max_generations << endl;
        cout << "The best solution = " << slhh.GetBestFit() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;
#ifdef PROFILE_EVAL
        PrintEvalProfile(solver);
#endif
        return 0;
    }
    
    // Standard GA/DE/GDE solvers
    int Gen_count = 0;
    double best = solver.gbest_fit;
    double* record = new double[max_generations];
    int generation = 0;
    uint64_t next_log_eval = (uint64_t)log_every;
    
    while (generation < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals))
    {
        // Select solver based on command-line argument
        if (solver_name == "GA") {
            solver.GA(0.8, 0.15, popsize / 3, popsize);
        } else if (solver_name == "DE") {
            solver.DE(0.5, 1, 0.5, 0, popsize - 1);
        } else if (solver_name == "GDE") {
            // GDE: Pmu=0.5, n_centric=6
            solver.GDE(0.5, 6, 0, popsize);
        }
        
        solver.Evaluation(1, 0, popsize);
        solver.pop_update(0, popsize);
        solver.worst_and_best();
        solver.Elist();
        
        // Migration: update subpop bests and perform ring migration
        if (migration_enabled) {
            solver.UpdateSubpopBest();
            solver.RingMigration(generation);
        }
        
        if (solver.gbest_fit < best)
            Gen_count = 0;
        else
            Gen_count++;
        
        generation++;
        best = solver.gbest_fit;
        
        if (max_evals > 0) {
            while (solver.GetEvalCount() >= next_log_eval) {
                cout << "Eval " << next_log_eval << ": best_fit = " << solver.gbest_fit << endl;
                next_log_eval += (uint64_t)log_every;
            }
        } else if (generation % log_every == 0 || generation == max_generations) {
            cout << "Gen " << generation << ": best_fit = " << solver.gbest_fit << endl;
        }
        
        record[generation - 1] = solver.gbest_fit;
    }
    clock_t t2 = clock();
    
    cout << "\n=== Final Results ===" << endl;
    cout << "Solver: " << solver_name << endl;
    if (migration_enabled) cout << "Migration: " << nsubpop << " subpops" << endl;
    cout << "Generation = " << generation << endl;
    cout << "The best solution = " << solver.gbest_fit << endl;
    cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;
#ifdef PROFILE_EVAL
    PrintEvalProfile(solver);
#endif
    
    delete[] record;
    return 0;
}
