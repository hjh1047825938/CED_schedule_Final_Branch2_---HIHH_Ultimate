#include "Multimethod.h"
#include "PPO.h"
#include "Problems.h"
#include "Rng.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

#define DEFAULT_ENUM 100
#define DEFAULT_CNUM 100
#define DEFAULT_DNUM 300
#define DEFAULT_TNUM 100
#define DEFAULT_MOPT_NUM 5
#define DEFAULT_EPISODES 10000
#define DEFAULT_SEED 1

static void PrintUsage(const char* prog) {
    cout << "Usage: " << prog << " [OPTIONS]\n";
    cout << "\nOptions:\n";
    cout << "  --data_dir <path>       Data directory (default: ./data)\n";
    cout << "  --data_file <name>      Data file name (default: data_matrix_100.txt)\n";
    cout << "  --episodes <n>          Total episodes (default: 10000)\n";
    cout << "  --seed <n>              Single seed (default: 1)\n";
    cout << "  --seed_start <n>        Multi-seed start (inclusive)\n";
    cout << "  --seed_end <n>          Multi-seed end (inclusive)\n";
    cout << "  --update_every <n>      PPO update frequency (default: 64)\n";
    cout << "  --ppo_epochs <n>        PPO epochs per update (default: 4)\n";
    cout << "  --actor_lr <f>          Actor learning rate (default: 3e-4)\n";
    cout << "  --critic_lr <f>         Critic learning rate (default: 3e-4)\n";
    cout << "  --gamma <f>             Discount factor (default: 0.99)\n";
    cout << "  --lambda <f>            GAE lambda (default: 0.95)\n";
    cout << "  --clip_eps <f>          PPO clip epsilon (default: 0.2)\n";
    cout << "  --entropy_coef <f>      Entropy coefficient (default: 0.01)\n";
    cout << "  --max_grad_norm <f>     Gradient clipping norm (default: 0.5)\n";
    cout << "  --alpha <f>             Objective alpha for makespan/energy (default: 0.5)\n";
    cout << "  --cnum <n>              Number of cloud servers (default: 100)\n";
    cout << "  --enum <n>              Number of edge servers (default: 100)\n";
    cout << "  --dnum <n>              Number of devices (default: 300)\n";
    cout << "  --tnum <n>              Number of tasks (default: 100)\n";
    cout << "  --mopt <n>              Operations per task (default: 5)\n";
    cout << "  --scale_tag <name>      Output prefix tag (default: infer from tnum, e.g. T100)\n";
    cout << "  --results_dir <path>    Output directory (default: results/PPO)\n";
    cout << "  --help                  Show this help\n";
}

static string InferScaleTag(int tnum) {
    return string("T") + to_string(tnum);
}

static void EnsureDir(const filesystem::path& p) {
    std::error_code ec;
    filesystem::create_directories(p, ec);
}

static bool WriteCurve(const filesystem::path& path, const vector<double>& curve) {
    ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs.setf(std::ios::fixed);
    ofs << setprecision(10);
    for (double v : curve) ofs << v << '\n';
    return true;
}

static pair<double, double> MeanStd(const vector<double>& x) {
    if (x.empty()) return {0.0, 0.0};
    const double mean = accumulate(x.begin(), x.end(), 0.0) / (double)x.size();
    double var = 0.0;
    for (double v : x) {
        const double d = v - mean;
        var += d * d;
    }
    var /= (double)x.size();
    return {mean, sqrt(var)};
}

static bool IsFileEmpty(const filesystem::path& p) {
    std::error_code ec;
    if (!filesystem::exists(p, ec)) return true;
    return filesystem::file_size(p, ec) == 0;
}

int main(int argc, char* argv[]) {
    filesystem::path data_dir = "./data";
    string data_file = "data_matrix_100.txt";
    filesystem::path results_dir = "results/PPO";

    int episodes = DEFAULT_EPISODES;
    int seed = DEFAULT_SEED;
    int seed_start = -1;
    int seed_end = -1;

    int Cnum = DEFAULT_CNUM;
    int Enum = DEFAULT_ENUM;
    int Dnum = DEFAULT_DNUM;
    int Tnum = DEFAULT_TNUM;
    int Mopt = DEFAULT_MOPT_NUM;

    double objective_alpha = 0.5;
    string scale_tag;

    PPOConfig cfg;
    cfg.episodes = DEFAULT_EPISODES;
    cfg.update_every = 64;
    cfg.gamma = 0.99;
    cfg.gae_lambda = 0.95;
    cfg.clip_eps = 0.2;
    cfg.actor_lr = 3e-4;
    cfg.critic_lr = 3e-4;
    cfg.entropy_coef = 0.01;
    cfg.max_grad_norm = 0.5;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data_dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--data_file") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "--results_dir") == 0 && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (strcmp(argv[i], "--episodes") == 0 && i + 1 < argc) {
            episodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed_start") == 0 && i + 1 < argc) {
            seed_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed_end") == 0 && i + 1 < argc) {
            seed_end = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--update_every") == 0 && i + 1 < argc) {
            cfg.update_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ppo_epochs") == 0 && i + 1 < argc) {
            cfg.ppo_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--actor_lr") == 0 && i + 1 < argc) {
            cfg.actor_lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--critic_lr") == 0 && i + 1 < argc) {
            cfg.critic_lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            cfg.gamma = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) {
            cfg.gae_lambda = atof(argv[++i]);
        } else if (strcmp(argv[i], "--clip_eps") == 0 && i + 1 < argc) {
            cfg.clip_eps = atof(argv[++i]);
        } else if (strcmp(argv[i], "--entropy_coef") == 0 && i + 1 < argc) {
            cfg.entropy_coef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max_grad_norm") == 0 && i + 1 < argc) {
            cfg.max_grad_norm = atof(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            objective_alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cnum") == 0 && i + 1 < argc) {
            Cnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--enum") == 0 && i + 1 < argc) {
            Enum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dnum") == 0 && i + 1 < argc) {
            Dnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tnum") == 0 && i + 1 < argc) {
            Tnum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mopt") == 0 && i + 1 < argc) {
            Mopt = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scale_tag") == 0 && i + 1 < argc) {
            scale_tag = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    if (episodes <= 0 || Cnum <= 0 || Enum <= 0 || Dnum <= 0 || Tnum <= 0 || Mopt <= 0) {
        cerr << "Error: invalid numeric arguments." << endl;
        return 1;
    }

    cfg.episodes = episodes;
    if (cfg.minibatch_size <= 0) cfg.minibatch_size = cfg.update_every;

    if (scale_tag.empty()) scale_tag = InferScaleTag(Tnum);

    vector<int> seeds;
    if (seed_start > 0 && seed_end >= seed_start) {
        for (int s = seed_start; s <= seed_end; ++s) seeds.push_back(s);
    } else {
        seeds.push_back(seed);
    }

    EnsureDir(results_dir);

    cout << "=== PPO Scheduler Configuration ===" << endl;
    cout << "Data: " << (data_dir / data_file) << endl;
    cout << "Scale tag: " << scale_tag << endl;
    cout << "Episodes: " << cfg.episodes << endl;
    cout << "Dims C/E/D/T/Mopt: " << Cnum << "/" << Enum << "/" << Dnum << "/" << Tnum << "/" << Mopt << endl;
    cout << "Seeds: " << seeds.front() << ".." << seeds.back() << endl;
    cout << "PPO update_every=" << cfg.update_every << " epochs=" << cfg.ppo_epochs
         << " actor_lr=" << cfg.actor_lr << " critic_lr=" << cfg.critic_lr << endl;
    cout << "==================================" << endl;

    vector<double> final_best_all;

    for (int run_seed : seeds) {
        Rng::getInstance().setSeed((unsigned int)run_seed);
        srand(run_seed);

        MultiMet solver(40, Tnum * 2 + Tnum * Mopt * 2, 0, 1,
                        Cnum, Enum, Dnum, Tnum, Tnum, Mopt, CED_Schedule, data_dir, data_file);
        solver.SetSeed((unsigned long)run_seed);
        solver.SetPini(0.4);
        solver.workspace.set_alpha(objective_alpha);
        solver.Initial();

        PPOScheduler ppo(&solver, cfg, (uint32_t)run_seed);
        PPORunResult r = ppo.Train();

        const filesystem::path out_curve = results_dir / (scale_tag + "_seed" + to_string(run_seed) + ".txt");
        if (!WriteCurve(out_curve, r.best_curve)) {
            cerr << "Error: failed to write curve file: " << out_curve << endl;
            return 1;
        }

        final_best_all.push_back(r.final_best);
        cout << "Seed " << run_seed << " done. final_best=" << std::setprecision(10) << r.final_best << endl;
    }

    const auto ms = MeanStd(final_best_all);
    const filesystem::path stat_path = results_dir / "statistics.txt";
    const bool write_header = IsFileEmpty(stat_path);
    ofstream sofs(stat_path, ios::app);
    if (!sofs.is_open()) {
        cerr << "Error: failed to write statistics file: " << stat_path << endl;
        return 1;
    }
    sofs.setf(std::ios::fixed);
    sofs << setprecision(10);
    if (write_header) {
        sofs << "scale\tseeds\tmean\tstd\tmean+-std\n";
    }
    sofs << scale_tag << '\t'
         << seeds.front() << "-" << seeds.back() << '\t'
         << ms.first << '\t'
         << ms.second << '\t'
         << ms.first << " +- " << ms.second << "\n";

    cout << "Statistics written: " << stat_path << endl;
    cout << scale_tag << " final_best mean+-std = " << ms.first << " +- " << ms.second << endl;

    return 0;
}
