#include "Multimethod.h"
#include "CC_HIHH.h"
#include "GA_SLHH.h"
#include "QPHH.h"
#include "IMOMA.h"
#include "CGA.h"
#include "DSAC_DE.h"
#include "Rng.h"
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
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <utility>
using namespace std;

// Default parameters
#define DEFAULT_ENUM 100
#define DEFAULT_CNUM 100
#define DEFAULT_DNUM 300
#define DEFAULT_TNUM 100
#define DEFAULT_MOPT_NUM 5
#define DEFAULT_MAXGEN 10000
#define DEFAULT_POPSIZE 40
#define DEFAULT_PINI 0.4
#define DEFAULT_SEED 42
#define DEFAULT_CGA_VM_RATIO 0.35

void print_usage(const char* prog_name) {
    cout << "Usage: " << prog_name << " [OPTIONS]\n";
    cout << "\nOptions:\n";
    cout << "  --data_dir <path>    Data directory (default: ./data)\n";
    cout << "  --data_file <name>   Data file name (default: data_matrix_100.txt)\n";
    cout << "  --generations <n>    Number of generations (default: " << DEFAULT_MAXGEN << ")\n";
    cout << "  --popsize <n>        Population size (default: " << DEFAULT_POPSIZE << ")\n";
    cout << "  --seed <n>           Random seed (default: " << DEFAULT_SEED << ")\n";
    cout << "  --pini <f>           Heuristic init probability 0-1 (default: " << DEFAULT_PINI << ")\n";
    cout << "  --alpha <f>          Weight for makespan vs energy (default: 0.5, range: [0,1])\n";
    cout << "  --solver <name>      Solver: GA, DE, GDE, DSAC-DE, CCHIHH, QHH, GA-SLHH, IMOMA, CGA (default: GA)\n";
    cout << "  --cnum <n>           Number of cloud servers (default: " << DEFAULT_CNUM << ")\n";
    cout << "  --enum <n>           Number of edge servers (default: " << DEFAULT_ENUM << ")\n";
    cout << "  --dnum <n>           Number of devices (default: " << DEFAULT_DNUM << ")\n";
    cout << "  --tnum <n>           Number of tasks (default: " << DEFAULT_TNUM << ")\n";
    cout << "  --mopt <n>           Operations per task (default: " << DEFAULT_MOPT_NUM << ")\n";
    cout << "  --migration          Enable rotated-ring subpopulation migration\n";
    cout << "  --nsubpop <n>        Number of subpopulations for migration (default: 8)\n";
    cout << "  --log_every <n>      Log best_fit every n generations (or evals if --max_evals is set, default: 50)\n";
    cout << "  --max_evals <n>      Stop after N evaluation calls (0 = disabled)\n";
    cout << "  --qphh_p0_factor <n> QPHH init pool multiplier P0 = P * n (default: 3)\n";
    cout << "  --qphh_tasksn <n>    QPHH greedy-insert tasks per LS (default: 1)\n";
    cout << "  --qphh_gi_cap <n>    QPHH greedy-insert position cap (0=all, default: 30)\n";
    cout << "  --qphh_map_cap <n>   QPHH mapping candidate cap (0=all, default: 40)\n";
    cout << "  --qphh_threads <n>   QPHH OpenMP thread count (default: 1)\n";
    cout << "  --imoma_arc_ratio <f> IMOMA archive ratio (default: 0.5)\n";
    cout << "  --cga_input <path>   CGA input file path (task/vm/rate format)\n";
    cout << "  --cga_rate <f>       CGA fixed transmission rate override (>0)\n";
    cout << "  --cga_vm_count <n>   Raw-mode VM count (default: round(0.35 * Tnum))\n";
    cout << "  --cga_comm_scale <f> Raw-mode communication scale (default: 0.01)\n";
    cout << "  --cga_edge_ratio <f> Raw-mode edge mips ratio to avg VM (default: 0.6)\n";
    cout << "  --cga_deadline_factor <f>  Raw-mode deadline factor (default: 1.5)\n";
    cout << "  --stable             Enable CCHIHH-Stable mode\n";
    cout << "  --cchihh_no_migration  Disable CCHIHH intra-block migration\n";
    cout << "  --cchihh_random_ops    Disable contextual bandit, random operators\n";
    cout << "  --cchihh_fixed_ops     Fixed operators per block: offload=GA, seq=GA, dev=DE\n";
    cout << "  --op_mode <m>          Operator selection: bandit|random|roundrobin (default: bandit)\n";
    cout << "  --cchihh_no_blocks     Disable CC blocks, run on full variable space\n";
    cout << "  --no_blocks            Alias of --cchihh_no_blocks\n";
    cout << "  --use_blocks <bool>    Enable/disable CC blocks (true/false, default: true)\n";
    cout << "  --cchihh_op_stats <p>  Write operator frequency CSV to path\n";
    cout << "  --cchihh_op_stats_every <n>  Operator stats logging interval (default: log_every)\n";
    cout << "  --cchihh_weight_log_offload <p>  Write offload operator weights CSV\n";
    cout << "  --cchihh_weight_log_seq <p>      Write sequence operator weights CSV\n";
    cout << "  --cchihh_weight_log_dev <p>      Write device operator weights CSV\n";
    cout << "  --cchihh_weight_log_every <n>    Weight logging interval (default: log_every)\n";
    cout << "  --cchihh_reward_log <p>          Write operator rewards CSV\n";
    cout << "  --cchihh_global_stats <p>        Write global stats CSV\n";
    cout << "  --cchihh_global_stats_every <n>  Global stats interval (default: log_every)\n";
    cout << "  --cchihh_diversity_log <p>       Write block diversity CSV\n";
    cout << "  --cchihh_diversity_log_every <n> Diversity logging interval (default: 50)\n";
    cout << "  --reward_variance_log <p>        Write per-generation reward variance CSV\n";
    cout << "  --shared_bandit                  Share one bandit across all three CC blocks\n";
    cout << "  --schedule_export <p>            Export decoded best schedule CSV at the end\n";
    cout << "  --resample_gate <n>  Stagnation gate for block resample (default: 15, 0=disable gate)\n";
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

static bool LoadCGAInputFile(const std::string& file_path,
                             std::vector<CGATask>& tasks,
                             std::vector<CGAVM>& vms,
                             double& rate)
{
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open CGA input file: " << file_path << std::endl;
        return false;
    }

    int n_tasks = 0;
    int n_vms = 0;
    if (!(ifs >> n_tasks >> n_vms) || n_tasks <= 0 || n_vms <= 0) {
        std::cerr << "Error: invalid CGA header. Expected: <num_tasks> <num_vms>" << std::endl;
        return false;
    }

    tasks.assign(n_tasks, CGATask{});
    for (int i = 0; i < n_tasks; ++i) {
        if (!(ifs >> tasks[i].data_length >> tasks[i].input_data_size >> tasks[i].deadline)) {
            std::cerr << "Error: invalid task row at index " << i
                      << ". Expected: <data_length> <input_data_size> <deadline>" << std::endl;
            return false;
        }
        if (tasks[i].data_length <= 0.0 || tasks[i].input_data_size < 0.0 || tasks[i].deadline <= 0.0) {
            std::cerr << "Error: task values out of range at index " << i << std::endl;
            return false;
        }
    }

    vms.assign(n_vms, CGAVM{});
    for (int i = 0; i < n_vms; ++i) {
        if (!(ifs >> vms[i].mips)) {
            std::cerr << "Error: invalid VM MIPS value at index " << i << std::endl;
            return false;
        }
        if (vms[i].mips <= 0.0) {
            std::cerr << "Error: VM MIPS must be > 0 at index " << i << std::endl;
            return false;
        }
    }

    // Optional trailing rate in file. If present and valid, it overrides default rate.
    double file_rate = 0.0;
    if (ifs >> file_rate) {
        if (file_rate <= 0.0) {
            std::cerr << "Error: CGA rate in input file must be > 0." << std::endl;
            return false;
        }
        rate = file_rate;
    }

    return true;
}

static bool SkipTokens(std::ifstream& ifs, int n)
{
    double tmp = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!(ifs >> tmp)) return false;
    }
    return true;
}

static bool LoadCGAFromCEDRaw(const std::filesystem::path& data_dir,
                              const std::string& data_file,
                              int enum_num,
                              int dnum,
                              int ce_tnum,
                              int m_jnum,
                              int m_optnum,
                              int vm_count_override,
                              double rate,
                              double comm_scale,
                              double edge_mips_ratio,
                              double deadline_factor,
                              std::vector<CGATask>& tasks,
                              std::vector<CGAVM>& vms)
{
    if (enum_num <= 0 || dnum <= 0 || ce_tnum <= 0 || m_jnum <= 0 || m_optnum <= 0) {
        std::cerr << "Error: invalid dimensions for raw CED parsing." << std::endl;
        return false;
    }

    const std::filesystem::path matrix_path = data_dir / data_file;
    std::ifstream ifs(matrix_path);
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open raw CED matrix file: " << matrix_path << std::endl;
        return false;
    }

    // Section A: EtoD [Enum][Dnum]
    if (!SkipTokens(ifs, enum_num * dnum)) {
        std::cerr << "Error: malformed EtoD section in " << matrix_path << std::endl;
        return false;
    }
    // Section B: DtoD [Dnum][Dnum]
    if (!SkipTokens(ifs, dnum * dnum)) {
        std::cerr << "Error: malformed DtoD section in " << matrix_path << std::endl;
        return false;
    }
    // Section C: MTask_Time [M_Jnum*M_OPTnum]
    if (!SkipTokens(ifs, m_jnum * m_optnum)) {
        std::cerr << "Error: malformed MTask_Time section in " << matrix_path << std::endl;
        return false;
    }

    // Section D: CETask_Property
    tasks.clear();
    tasks.reserve(ce_tnum);
    for (int i = 0; i < ce_tnum; ++i) {
        double comp = 0.0;
        double comm = 0.0;
        if (!(ifs >> comp >> comm)) {
            std::cerr << "Error: malformed CETask_Property at task " << i << std::endl;
            return false;
        }

        for (int g = 0; g < 4; ++g) {
            int k = 0;
            if (!(ifs >> k) || k < 0) {
                std::cerr << "Error: malformed dependency length in task " << i << std::endl;
                return false;
            }
            if (!SkipTokens(ifs, k)) {
                std::cerr << "Error: malformed dependency list in task " << i << std::endl;
                return false;
            }
        }

        int job_constraints = 0;
        if (!(ifs >> job_constraints)) {
            std::cerr << "Error: malformed Job_Constraints in task " << i << std::endl;
            return false;
        }
        (void)job_constraints;

        CGATask t;
        t.data_length = std::max(1e-9, comp);
        // Keep communication in a compatible scale with existing CED timing model.
        t.input_data_size = std::max(0.0, comm * comm_scale);
        t.deadline = 0.0;  // filled after VM loading
        tasks.push_back(t);
    }

    // Section E: AvailDeviceList for all operations
    for (int i = 0; i < m_jnum; ++i) {
        for (int j = 0; j < m_optnum; ++j) {
            int k = 0;
            if (!(ifs >> k) || k < 0) {
                std::cerr << "Error: malformed AvailDeviceList length at op (" << i << "," << j << ")" << std::endl;
                return false;
            }
            if (!SkipTokens(ifs, k)) {
                std::cerr << "Error: malformed AvailDeviceList values at op (" << i << "," << j << ")" << std::endl;
                return false;
            }
        }
    }

    // Section F: AvailEdgeServerList for each task
    for (int i = 0; i < ce_tnum; ++i) {
        int k = 0;
        if (!(ifs >> k) || k < 0) {
            std::cerr << "Error: malformed AvailEdgeServerList length at task " << i << std::endl;
            return false;
        }
        if (!SkipTokens(ifs, k)) {
            std::cerr << "Error: malformed AvailEdgeServerList values at task " << i << std::endl;
            return false;
        }
    }

    // Section G: EnergyList[11]
    if (!SkipTokens(ifs, 11)) {
        std::cerr << "Error: malformed EnergyList section in " << matrix_path << std::endl;
        return false;
    }

    // VM MIPS source: Machines_3000.txt.
    // Accept only plausible MIPS values; otherwise use deterministic fallback.
    int vm_target = vm_count_override > 0 ? vm_count_override : (int)std::lround(DEFAULT_CGA_VM_RATIO * (double)ce_tnum);
    if (vm_target < 1) vm_target = 1;
    if (vm_target > enum_num) vm_target = enum_num;

    std::vector<double> mips_values;
    mips_values.reserve(vm_target);
    const std::filesystem::path machine_path = data_dir / "Machines_3000.txt";
    std::ifstream mfs(machine_path);
    if (mfs.is_open()) {
        std::string line;
        while (std::getline(mfs, line) && (int)mips_values.size() < vm_target) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::vector<double> cols;
            double v = 0.0;
            while (iss >> v) cols.push_back(v);
            if (cols.empty()) continue;

            // Prefer values in a practical VM-MIPS range.
            // Avoid accidentally using IDs or non-performance fields.
            double picked = -1.0;
            for (double c : cols) {
                if (c >= 100.0 && c <= 10000.0) {
                    picked = c;
                    break;
                }
            }
            if (picked > 0.0) {
                mips_values.push_back(picked);
            }
        }
    }
    if (mips_values.empty()) {
        // Fallback deterministic VM capacities (roughly 1000~2000 MIPS scale).
        for (int i = 0; i < vm_target; ++i) {
            mips_values.push_back(1000.0 + 10.0 * i);
        }
    } else if ((int)mips_values.size() < vm_target) {
        const int cur = (int)mips_values.size();
        for (int i = cur; i < vm_target; ++i) {
            mips_values.push_back(mips_values[i % cur]);
        }
    }

    vms.assign(vm_target, CGAVM{});
    double mips_sum = 0.0;
    for (int i = 0; i < vm_target; ++i) {
        vms[i].mips = std::max(1e-9, mips_values[i]);
        mips_sum += vms[i].mips;
    }
    const double avg_mips = mips_sum / (double)vm_target;

    // Derive deadline from raw fields so CGA can run on original format directly.
    for (CGATask& t : tasks) {
        const double base = (t.data_length / std::max(1e-9, avg_mips)) + (t.input_data_size / std::max(1e-9, rate));
        t.deadline = std::max(1e-9, base * deadline_factor);
    }

    // Paper-consistent task classification: only tasks that violate local deadline are offloaded.
    const double edge_mips = std::max(1e-9, avg_mips * edge_mips_ratio);
    std::vector<CGATask> cloud_tasks;
    cloud_tasks.reserve(tasks.size());
    for (const CGATask& t : tasks) {
        const double local_exec = t.data_length / edge_mips;
        if (local_exec > t.deadline) {
            cloud_tasks.push_back(t);
        }
    }
    // Keep solver stable when classification yields empty cloud set.
    if (!cloud_tasks.empty()) {
        tasks.swap(cloud_tasks);
    }

    return true;
}

int main(int argc, char* argv[])
{
    std::ostringstream cmd_oss;
    cmd_oss << "Command:";
    for (int i = 0; i < argc; ++i) {
        cmd_oss << " '" << argv[i] << "'";
    }
    std::cout << cmd_oss.str() << std::endl;

    // Parse command-line arguments
    filesystem::path data_dir = "data";  // Default: ./data
    string data_file = "data_matrix_100.txt";
    int max_generations = DEFAULT_MAXGEN;
    int popsize = DEFAULT_POPSIZE;
    bool generations_set = false;
    bool popsize_set = false;
    unsigned int seed = DEFAULT_SEED;
    double pini = DEFAULT_PINI;
    double objective_alpha = 0.5;
    string solver_name = "GA";
    int qphh_p0_factor = 3;
    int qphh_tasksn = 1;
    int qphh_gi_cap = 30;
    int qphh_map_cap = 40;
    int qphh_threads = 1;
    double imoma_arc_ratio = 0.5;
    string cga_input_file;
    double cga_rate = 10.0;
    bool cga_rate_set = false;
    int cga_vm_count = 0;
    double cga_comm_scale = 0.01;
    double cga_edge_ratio = 0.6;
    double cga_deadline_factor = 1.5;
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
    bool cchihh_fixed_ops = false;
    bool cchihh_no_blocks = false;
    bool shared_bandit = false;
    string cchihh_op_mode = "bandit";
    string cchihh_op_stats_path;
    int cchihh_op_stats_every = 0;
    string cchihh_weight_log_offload_path;
    string cchihh_weight_log_seq_path;
    string cchihh_weight_log_dev_path;
    int cchihh_weight_log_every = 0;
    string cchihh_reward_log_path;
    string cchihh_global_stats_path;
    int cchihh_global_stats_every = 0;
    string cchihh_diversity_log_path;
    int cchihh_diversity_log_every = 50;
    string reward_variance_log_path;
    string schedule_export_path;
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
            generations_set = true;
        } else if (strcmp(argv[i], "--popsize") == 0 && i + 1 < argc) {
            popsize = atoi(argv[++i]);
            popsize_set = true;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pini") == 0 && i + 1 < argc) {
            pini = atof(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            objective_alpha = atof(argv[++i]);
            if (objective_alpha < 0.0 || objective_alpha > 1.0) {
                cerr << "Error: --alpha must be in [0, 1]" << endl;
                return 1;
            }
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
        } else if (strcmp(argv[i], "--qphh_p0_factor") == 0 && i + 1 < argc) {
            qphh_p0_factor = atoi(argv[++i]);
            if (qphh_p0_factor < 1) qphh_p0_factor = 1;
        } else if (strcmp(argv[i], "--qphh_tasksn") == 0 && i + 1 < argc) {
            qphh_tasksn = atoi(argv[++i]);
            if (qphh_tasksn < 1) qphh_tasksn = 1;
        } else if (strcmp(argv[i], "--qphh_gi_cap") == 0 && i + 1 < argc) {
            qphh_gi_cap = atoi(argv[++i]);
            if (qphh_gi_cap < 0) qphh_gi_cap = 0;
        } else if (strcmp(argv[i], "--qphh_map_cap") == 0 && i + 1 < argc) {
            qphh_map_cap = atoi(argv[++i]);
            if (qphh_map_cap < 0) qphh_map_cap = 0;
        } else if (strcmp(argv[i], "--qphh_threads") == 0 && i + 1 < argc) {
            qphh_threads = atoi(argv[++i]);
            if (qphh_threads < 1) qphh_threads = 1;
        } else if (strcmp(argv[i], "--imoma_arc_ratio") == 0 && i + 1 < argc) {
            imoma_arc_ratio = atof(argv[++i]);
            if (imoma_arc_ratio <= 0.0) imoma_arc_ratio = 0.5;
            if (imoma_arc_ratio > 1.0) imoma_arc_ratio = 1.0;
        } else if (strcmp(argv[i], "--cga_input") == 0 && i + 1 < argc) {
            cga_input_file = argv[++i];
        } else if (strcmp(argv[i], "--cga_rate") == 0 && i + 1 < argc) {
            cga_rate = atof(argv[++i]);
            cga_rate_set = true;
            if (cga_rate <= 0.0) {
                cerr << "Error: --cga_rate must be > 0" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cga_vm_count") == 0 && i + 1 < argc) {
            cga_vm_count = atoi(argv[++i]);
            if (cga_vm_count < 0) {
                cerr << "Error: --cga_vm_count must be >= 0" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cga_comm_scale") == 0 && i + 1 < argc) {
            cga_comm_scale = atof(argv[++i]);
            if (cga_comm_scale <= 0.0) {
                cerr << "Error: --cga_comm_scale must be > 0" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cga_edge_ratio") == 0 && i + 1 < argc) {
            cga_edge_ratio = atof(argv[++i]);
            if (cga_edge_ratio <= 0.0) {
                cerr << "Error: --cga_edge_ratio must be > 0" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cga_deadline_factor") == 0 && i + 1 < argc) {
            cga_deadline_factor = atof(argv[++i]);
            if (cga_deadline_factor <= 0.0) {
                cerr << "Error: --cga_deadline_factor must be > 0" << endl;
                return 1;
            }
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
        } else if (strcmp(argv[i], "--cchihh_fixed_ops") == 0) {
            cchihh_fixed_ops = true;
        } else if (strcmp(argv[i], "--shared_bandit") == 0) {
            shared_bandit = true;
        } else if (strcmp(argv[i], "--op_mode") == 0 && i + 1 < argc) {
            cchihh_op_mode = argv[++i];
            std::transform(cchihh_op_mode.begin(), cchihh_op_mode.end(), cchihh_op_mode.begin(),
                           [](unsigned char ch) { return (char)std::tolower(ch); });
            if (cchihh_op_mode != "bandit" && cchihh_op_mode != "random" && cchihh_op_mode != "roundrobin") {
                cerr << "Error: --op_mode must be one of: bandit, random, roundrobin" << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cchihh_no_blocks") == 0 || strcmp(argv[i], "--no_blocks") == 0) {
            cchihh_no_blocks = true;
        } else if (strcmp(argv[i], "--use_blocks") == 0 && i + 1 < argc) {
            string v = argv[++i];
            transform(v.begin(), v.end(), v.begin(),
                      [](unsigned char ch) { return (char)std::tolower(ch); });
            if (v == "true" || v == "1" || v == "on" || v == "yes") {
                cchihh_no_blocks = false;
            } else if (v == "false" || v == "0" || v == "off" || v == "no") {
                cchihh_no_blocks = true;
            } else {
                cerr << "Error: --use_blocks expects true/false (or 1/0, on/off)." << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--cchihh_op_stats") == 0 && i + 1 < argc) {
            cchihh_op_stats_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_op_stats_every") == 0 && i + 1 < argc) {
            cchihh_op_stats_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cchihh_weight_log_offload") == 0 && i + 1 < argc) {
            cchihh_weight_log_offload_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_weight_log_seq") == 0 && i + 1 < argc) {
            cchihh_weight_log_seq_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_weight_log_dev") == 0 && i + 1 < argc) {
            cchihh_weight_log_dev_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_weight_log_every") == 0 && i + 1 < argc) {
            cchihh_weight_log_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cchihh_reward_log") == 0 && i + 1 < argc) {
            cchihh_reward_log_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_global_stats") == 0 && i + 1 < argc) {
            cchihh_global_stats_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_global_stats_every") == 0 && i + 1 < argc) {
            cchihh_global_stats_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cchihh_diversity_log") == 0 && i + 1 < argc) {
            cchihh_diversity_log_path = argv[++i];
        } else if (strcmp(argv[i], "--cchihh_diversity_log_every") == 0 && i + 1 < argc) {
            cchihh_diversity_log_every = atoi(argv[++i]);
            if (cchihh_diversity_log_every < 1) cchihh_diversity_log_every = 1;
        } else if (strcmp(argv[i], "--reward_variance_log") == 0 && i + 1 < argc) {
            reward_variance_log_path = argv[++i];
        } else if (strcmp(argv[i], "--schedule_export") == 0 && i + 1 < argc) {
            schedule_export_path = argv[++i];
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
    cout << "Objective alpha: " << objective_alpha << endl;
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
        if (cchihh_random_ops) cchihh_op_mode = "random";
        cout << "CCHIHH gate: " << (resample_gate > 0 ? "enabled" : "disabled") << endl;
        cout << "CCHIHH blocks: " << (cchihh_no_blocks ? "disabled" : "enabled") << endl;
        cout << "CCHIHH migration: " << (cchihh_migration ? "enabled" : "disabled") << endl;
        cout << "CCHIHH shared bandit: " << (shared_bandit ? "enabled" : "disabled") << endl;
        cout << "CCHIHH op mode: " << cchihh_op_mode << endl;
        cout << "CCHIHH fixed ops: " << (cchihh_fixed_ops ? "enabled" : "disabled") << endl;
        if (!cchihh_op_stats_path.empty()) {
            cout << "CCHIHH op stats: " << cchihh_op_stats_path << endl;
        }
        if (!cchihh_weight_log_offload_path.empty() || !cchihh_weight_log_seq_path.empty() || !cchihh_weight_log_dev_path.empty()) {
            cout << "CCHIHH weight logs: " << cchihh_weight_log_offload_path << " | "
                 << cchihh_weight_log_seq_path << " | " << cchihh_weight_log_dev_path << endl;
        }
        if (!cchihh_reward_log_path.empty()) {
            cout << "CCHIHH reward log: " << cchihh_reward_log_path << endl;
        }
        if (!cchihh_diversity_log_path.empty()) {
            cout << "CCHIHH diversity log: " << cchihh_diversity_log_path << endl;
        }
        if (!reward_variance_log_path.empty()) {
            cout << "CCHIHH reward variance log: " << reward_variance_log_path << endl;
        }
        if (!cchihh_global_stats_path.empty()) {
            cout << "CCHIHH global stats: " << cchihh_global_stats_path << endl;
        }
        if (!schedule_export_path.empty()) {
            cout << "Schedule export: " << schedule_export_path << endl;
        }
    }
    if (solver_name == "QHH" || solver_name == "QPHH") {
        cout << "QPHH params: p0_factor=" << qphh_p0_factor
             << " tasksn=" << qphh_tasksn
             << " gi_cap=" << qphh_gi_cap
             << " map_cap=" << qphh_map_cap
             << " threads=" << qphh_threads << endl;
    }
    if (solver_name == "IMOMA") {
        cout << "IMOMA params: pop=" << popsize
             << " arc_ratio=" << imoma_arc_ratio
             << " generations=" << max_generations << endl;
    }
    if (solver_name == "CGA") {
        int cga_vm_default = (int)std::lround(DEFAULT_CGA_VM_RATIO * (double)Tnum);
        if (cga_vm_default < 1) cga_vm_default = 1;
        if (cga_vm_default > Enum) cga_vm_default = Enum;
        cout << "CGA input: " << (cga_input_file.empty() ? "(raw data mode via --data_dir/--data_file)" : cga_input_file) << endl;
        if (cga_rate_set) {
            cout << "CGA rate override: " << cga_rate << endl;
        }
        cout << "CGA VM count (raw mode): " << (cga_vm_count > 0 ? cga_vm_count : cga_vm_default) << endl;
        cout << "CGA communication scale (raw mode): " << cga_comm_scale << endl;
        cout << "CGA edge ratio (raw mode): " << cga_edge_ratio << endl;
        cout << "CGA deadline factor (raw mode): " << cga_deadline_factor << endl;
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
            solver_a.workspace.set_alpha(objective_alpha);
            solver_a.Initial();
            double best_a = solver_a.gbest_fit;

            srand(s);
            MultiMet solver_b(popsize, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                              Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
            solver_b.SetSeed(s);
            solver_b.SetPini(DEFAULT_PINI);
            solver_b.workspace.set_alpha(objective_alpha);
            solver_b.Initial();
            double best_b = solver_b.gbest_fit;

            cout << "Seed " << s << ": Pini=1.0 best=" << best_a << ", Pini=" << DEFAULT_PINI << " best=" << best_b << endl;
        }
        return 0;
    }

    if (solver_name == "CGA") {
        // Path A: legacy standalone CGA input file mode.
        if (!cga_input_file.empty()) {
            std::vector<CGATask> cga_tasks;
            std::vector<CGAVM> cga_vms;
            double final_rate = cga_rate;
            if (!LoadCGAInputFile(cga_input_file, cga_tasks, cga_vms, final_rate)) {
                return 1;
            }
            if (cga_rate_set) final_rate = cga_rate;

            CGAConfig cfg;
            cfg.population_size = popsize_set ? popsize : 100;
            cfg.max_generations = generations_set ? max_generations : 10000;
            cfg.crossover_prob = 0.75;
            cfg.crossover_similarity_threshold = 0.8;
            cfg.mutation_prob_early = 0.03;
            cfg.mutation_prob_late = 0.01;
            cfg.mutation_switch_generation = 6667;
            cfg.catastrophe_threshold = 150;
            cfg.catastrophe_apply_generations = 5000;

            cout << "\n=== Running CGA Solver ===" << endl;
            cout << "Tasks: " << cga_tasks.size() << ", VMs: " << cga_vms.size() << ", Rate: " << final_rate << endl;
            cout << "CGA params: pop=" << cfg.population_size
                 << ", gen=" << cfg.max_generations
                 << ", pc=" << cfg.crossover_prob
                 << ", pm(early/late)=" << cfg.mutation_prob_early << "/" << cfg.mutation_prob_late
                 << ", cat=" << cfg.catastrophe_threshold
                 << ", cat_apply_gen<=" << cfg.catastrophe_apply_generations << endl;

            clock_t cga_t1 = clock();
            CGA cga(std::move(cga_tasks), std::move(cga_vms), final_rate, cfg, seed);
            CGAResult cga_result = cga.Run(log_every);
            clock_t cga_t2 = clock();

            cout << "\n=== Final Results (CGA) ===" << endl;
            cout << "Solver: CGA" << endl;
            cout << "Generations = " << cfg.max_generations << endl;
            cout << "Best generation = " << cga_result.best_generation << endl;
            cout << "Catastrophe triggers = " << cga_result.catastrophe_count << endl;
            cout << "Best fitness = " << cga_result.best_fitness << endl;
            cout << "The best solution = " << cga_result.best_fitness << endl;
            cout << "Minimum completion time = " << cga_result.min_completion_time << endl;
            cout << "Total punish = " << cga_result.total_punish << endl;
            cout << "Delay satisfaction rate = " << cga_result.delay_satisfaction_rate
                 << " (" << cga_result.satisfied_tasks << "/" << cga_result.total_tasks << ")" << endl;
            cout << "Time = " << (double)(cga_t2 - cga_t1) / CLOCKS_PER_SEC << " s" << endl;
            cout << "\nexit code 0" << endl;
            return 0;
        }

        // Path B: default raw mode uses the same objective as other solvers (MultiMet::Eval).
        srand(seed);
        Rng::getInstance().setSeed(seed);
        const int cga_pop = popsize_set ? popsize : 300;
        const int cga_gen = generations_set ? max_generations : 10000;

        MultiMet cga_solver(cga_pop, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                            Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
        cga_solver.SetSeed(seed);
        cga_solver.SetPini(pini);
        cga_solver.workspace.set_alpha(objective_alpha);
        cga_solver.Initial();
        cga_solver.ResetEvalCount();

        struct CGA_MM_Ind {
            std::vector<double> var;
            double fit = std::numeric_limits<double>::infinity();
            double score = 0.0;
        };

        std::mt19937 rng_local(seed + 7919u);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::uniform_real_distribution<double> gene_rand(0.0, 1.0);

        auto eval_ind = [&](CGA_MM_Ind& ind) {
            ind.fit = cga_solver.Eval(ind.var.data());
            const double safe = ind.fit > 1e-15 ? ind.fit : 1e-15;
            ind.score = 1.0 / safe;
        };

        auto similarity = [&](const CGA_MM_Ind& a, const CGA_MM_Ind& b) {
            int same = 0;
            for (int i = 0; i < (int)a.var.size(); ++i) {
                if (std::fabs(a.var[i] - b.var[i]) <= 1e-12) ++same;
            }
            return (double)same / (double)a.var.size();
        };

        auto mutate_once = [&](CGA_MM_Ind& ind) {
            const int n = (int)ind.var.size();
            if (n <= 1) return;
            std::uniform_int_distribution<int> p_pick(0, n - 1);
            int p1 = p_pick(rng_local);
            int p2 = p_pick(rng_local);
            if (std::fabs(ind.var[p1] - ind.var[p2]) <= 1e-12) {
                int tries = 0;
                while (tries < n && std::fabs(ind.var[p1] - ind.var[p2]) <= 1e-12) {
                    p2 = (p2 + 1) % n;
                    ++tries;
                }
            }
            if (std::fabs(ind.var[p1] - ind.var[p2]) > 1e-12) {
                std::swap(ind.var[p1], ind.var[p2]);
            } else {
                ind.var[p1] = gene_rand(rng_local);
            }
        };

        std::vector<CGA_MM_Ind> pop(cga_pop);
        for (int i = 0; i < cga_pop; ++i) {
            pop[i].var.assign(cga_solver.pop[i], cga_solver.pop[i] + cga_solver.Nvar);
            eval_ind(pop[i]);
        }

        auto best_it = std::min_element(pop.begin(), pop.end(),
                                        [](const CGA_MM_Ind& a, const CGA_MM_Ind& b) { return a.fit < b.fit; });
        CGA_MM_Ind gbest = *best_it;
        int best_gen = 0;
        int stagnation = 0;
        int cat_count = 0;

        cout << "\n=== Running CGA Solver ===" << endl;
        cout << "CGA mode: objective-compatible with CED_Schedule (best_fit smaller is better)" << endl;
        cout << "CGA params: pop=" << cga_pop
             << ", gen=" << cga_gen
             << ", pc=0.75, pm(early/late)=0.03/0.01, cat=150, cat_apply_gen<=5000" << endl;

        clock_t cga_t1 = clock();
        for (int gen = 1; gen <= cga_gen; ++gen) {
            std::vector<CGA_MM_Ind> next;
            next.reserve(cga_pop);
            next.push_back(gbest);  // elitism

            double total_score = 0.0;
            for (const auto& ind : pop) total_score += ind.score;
            auto roulette_pick = [&]() {
                if (total_score <= 0.0) {
                    std::uniform_int_distribution<int> pick(0, cga_pop - 1);
                    return pick(rng_local);
                }
                const double tgt = u01(rng_local) * total_score;
                double acc = 0.0;
                for (int i = 0; i < cga_pop; ++i) {
                    acc += pop[i].score;
                    if (acc >= tgt) return i;
                }
                return cga_pop - 1;
            };

            while ((int)next.size() < cga_pop) {
                next.push_back(pop[roulette_pick()]);
            }

            // Crossover with similarity gate.
            if (cga_solver.Nvar > 1) {
                std::uniform_int_distribution<int> cpick(1, cga_solver.Nvar - 1);
                for (int i = 1; i + 1 < cga_pop; i += 2) {
                    if (u01(rng_local) > 0.75) continue;
                    if (similarity(next[i], next[i + 1]) >= 0.8) continue;
                    int cp = cpick(rng_local);
                    for (int p = cp; p < cga_solver.Nvar; ++p) {
                        std::swap(next[i].var[p], next[i + 1].var[p]);
                    }
                }
            }

            // Mutation (two-stage probability).
            const double pm = (gen < 6667) ? 0.03 : 0.01;
            for (int i = 1; i < cga_pop; ++i) {
                if (u01(rng_local) < pm) mutate_once(next[i]);
            }

            for (auto& ind : next) eval_ind(ind);

            // Catastrophe in early generations.
            auto cur_best_it = std::min_element(next.begin(), next.end(),
                                                [](const CGA_MM_Ind& a, const CGA_MM_Ind& b) { return a.fit < b.fit; });
            if (cur_best_it->fit + 1e-12 < gbest.fit) {
                gbest = *cur_best_it;
                best_gen = gen;
                stagnation = 0;
            } else {
                ++stagnation;
            }

            if (gen <= 5000 && stagnation >= 150) {
                std::vector<int> idx(cga_pop);
                for (int i = 0; i < cga_pop; ++i) idx[i] = i;
                std::sort(idx.begin(), idx.end(), [&](int a, int b) { return next[a].fit < next[b].fit; });
                int ntop = std::max(1, cga_pop / 3);
                for (int k = 0; k < ntop; ++k) {
                    int id = idx[k];
                    if (u01(rng_local) < 0.8) {
                        mutate_once(next[id]);
                        eval_ind(next[id]);
                    }
                }
                ++cat_count;
                stagnation = 0;
                cur_best_it = std::min_element(next.begin(), next.end(),
                                               [](const CGA_MM_Ind& a, const CGA_MM_Ind& b) { return a.fit < b.fit; });
                if (cur_best_it->fit + 1e-12 < gbest.fit) {
                    gbest = *cur_best_it;
                    best_gen = gen;
                }
            }

            pop.swap(next);
            if (log_every > 0 && (gen % log_every == 0 || gen == cga_gen)) {
                cout << "Gen " << gen << ": best_fit = " << gbest.fit << endl;
            }
        }
        clock_t cga_t2 = clock();

        cout << "\n=== Final Results (CGA) ===" << endl;
        cout << "Solver: CGA" << endl;
        cout << "Generations = " << cga_gen << endl;
        cout << "Best generation = " << best_gen << endl;
        cout << "Catastrophe triggers = " << cat_count << endl;
        cout << "Best fitness = " << gbest.fit << endl;
        cout << "The best solution = " << gbest.fit << endl;
        cout << "Time = " << (double)(cga_t2 - cga_t1) / CLOCKS_PER_SEC << " s" << endl;
        cout << "\nexit code 0" << endl;
        return 0;
    }

    srand(seed);
    Rng::getInstance().setSeed(seed);
    
    // Create solver with data directory
    MultiMet solver(popsize, Tnum * 2 + Tnum * Mopt_num * 2, 0, 1,
                    Cnum, Enum, Dnum, Tnum, Tnum, Mopt_num, CED_Schedule, data_dir, data_file);
    solver.SetSeed(seed);
    solver.SetPini(pini);
    solver.workspace.set_alpha(objective_alpha);
    solver.Initial();
    solver.ResetEvalCount();
    
    // Initialize migration if enabled (nG=nsubpop, nCircle=5, pElitist=0.8)
    if (migration_enabled) {
        if (nsubpop < 1) {
            cerr << "Error: --nsubpop must be >= 1 when migration is enabled" << endl;
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
        if (cchihh_random_ops) cchihh_op_mode = "random";
        
        // Create CC-HIHH solver with nsubpop islands per block
        CC_HIHH_Solver cc_solver(&solver, popsize, nsubpop, 5 /*nCircle*/, 0.8 /*pElitist*/);
        cc_solver.SetMaxGenerations(max_generations);
        cc_solver.SetUseBlocks(!cchihh_no_blocks);
        cc_solver.SetMigrationEnabled(cchihh_migration);
        cc_solver.SetUseBandit(cchihh_op_mode == "bandit");
        cc_solver.SetSharedBanditMode(shared_bandit);
        if (cchihh_op_mode == "roundrobin") cc_solver.SetSelectionMode(MODE_ROUND_ROBIN);
        else if (cchihh_op_mode == "random") cc_solver.SetSelectionMode(MODE_RANDOM);
        else cc_solver.SetSelectionMode(MODE_CONTEXTUAL_BANDIT);
        cc_solver.SetFixedOpsPerBlock(cchihh_fixed_ops);
        cc_solver.SetResampleGate(resample_gate);
        if (!cchihh_op_stats_path.empty()) {
            int stats_every = cchihh_op_stats_every > 0 ? cchihh_op_stats_every : log_every;
            cc_solver.SetOpStats(cchihh_op_stats_path, stats_every);
        }
        if (!cchihh_weight_log_offload_path.empty() &&
            !cchihh_weight_log_seq_path.empty() &&
            !cchihh_weight_log_dev_path.empty()) {
            int weight_every = cchihh_weight_log_every > 0 ? cchihh_weight_log_every : log_every;
            cc_solver.SetWeightLogging(cchihh_weight_log_offload_path, cchihh_weight_log_seq_path, cchihh_weight_log_dev_path, weight_every);
        }
        if (!cchihh_reward_log_path.empty()) {
            cc_solver.SetRewardLogging(cchihh_reward_log_path);
        }
        if (!reward_variance_log_path.empty()) {
            cc_solver.SetRewardVarianceLogging(reward_variance_log_path);
        }
        if (!cchihh_global_stats_path.empty()) {
            int global_every = cchihh_global_stats_every > 0 ? cchihh_global_stats_every : log_every;
            cc_solver.SetGlobalStatsLogging(cchihh_global_stats_path, global_every);
        }
        if (!cchihh_diversity_log_path.empty()) {
            cc_solver.SetDiversityLogging(cchihh_diversity_log_path, cchihh_diversity_log_every);
        }
        if (stable_mode) {
            cc_solver.SetStableMode(true);
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
            cc_solver.LogWeightsIfNeeded(gen, gen == max_generations - 1);
            cc_solver.LogGlobalStatsIfNeeded(gen, gen == max_generations - 1);
            cc_solver.LogDiversityIfNeeded(gen, gen == max_generations - 1);
            
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
        cout << "CCHIHH gate_blocked_total = " << cc_solver.GetGateBlockedTotal() << endl;
        cout << "CCHIHH gate_fallback_total = " << cc_solver.GetGateFallbackTotal() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;

        if (!schedule_export_path.empty()) {
            const string instance_tag = "T" + std::to_string(Tnum);
            bool ok = solver.ExportScheduleCSV(cc_solver.GetGlobalBest(), schedule_export_path, seed, cc_solver.GetGlobalBestFit(), instance_tag);
            if (ok) {
                cout << "Schedule export written to " << schedule_export_path << endl;
            } else {
                cout << "Schedule export failed for " << schedule_export_path << endl;
            }
        }

#ifdef PROFILE_EVAL
        PrintEvalProfile(solver);
#endif
        return 0;
    }

    if (solver_name == "GA-SLHH") {
        cout << "\n=== Running GA-SLHH Solver ===" << endl;
        GA_SLHH_Solver slhh(&solver, popsize);
        slhh.SetMaxGenerations(max_generations);
        slhh.SetCrossoverRate(0.9);
        slhh.SetMutationRate(0.2);
        slhh.SetGeneMutationRate(0.03);
        slhh.SetImmigrantRate(0.15);
        slhh.SetElitismCount(std::max(2, popsize / 50));
        slhh.SetLocalSearchTrials(100);
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

    if (solver_name == "QHH" || solver_name == "QPHH") {
        cout << "\n=== Running QPHH Solver ===" << endl;
        QPHH_Solver qphh(&solver, popsize);
        qphh.SetMaxIterations(max_generations);
        qphh.SetInitPoolFactor(qphh_p0_factor);
        qphh.SetTasksN(qphh_tasksn);
        qphh.SetGreedyInsertCap(qphh_gi_cap);
        qphh.SetMappingCap(qphh_map_cap);
        qphh.SetNumThreads(qphh_threads);
        qphh.SetLog(false, false, log_every);
        qphh.Init();
        solver.ResetEvalCount();
        uint64_t next_log_eval = (uint64_t)log_every;

        for (int gen = 0; gen < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals); gen++) {
            qphh.RunIteration(gen);

            if (max_evals > 0) {
                while (solver.GetEvalCount() >= next_log_eval) {
                    cout << "Eval " << next_log_eval << ": best_fit = " << qphh.GetBestFit() << endl;
                    next_log_eval += (uint64_t)log_every;
                }
            } else if ((gen + 1) % log_every == 0 || gen == max_generations - 1) {
                cout << "Gen " << (gen + 1) << ": best_fit = " << qphh.GetBestFit() << endl;
            }
        }

        clock_t t2 = clock();
        cout << "\n=== Final Results (QPHH) ===" << endl;
        cout << "Solver: " << solver_name << endl;
        cout << "Generation = " << max_generations << endl;
        cout << "The best solution = " << qphh.GetBestFit() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;
#ifdef PROFILE_EVAL
        PrintEvalProfile(solver);
#endif
        return 0;
    }

    if (solver_name == "IMOMA") {
        cout << "\n=== Running IMOMA Solver ===" << endl;
        IMOMA_Solver imoma(&solver, popsize, imoma_arc_ratio, max_generations);
        imoma.Init();
        solver.ResetEvalCount();
        uint64_t next_log_eval = (uint64_t)log_every;

        for (int gen = 0; gen < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals); gen++) {
            imoma.RunGeneration(gen);
            if (max_evals > 0) {
                while (solver.GetEvalCount() >= next_log_eval) {
                    cout << "Eval " << next_log_eval
                         << ": best_fit = " << imoma.GetBestScalarFit()
                         << " archive = " << imoma.GetArchiveSize() << endl;
                    next_log_eval += (uint64_t)log_every;
                }
            } else if ((gen + 1) % log_every == 0 || gen == max_generations - 1) {
                cout << "Gen " << (gen + 1)
                     << ": best_fit = " << imoma.GetBestScalarFit()
                     << " archive = " << imoma.GetArchiveSize() << endl;
            }
        }

        clock_t t2 = clock();
        cout << "\n=== Final Results (IMOMA) ===" << endl;
        cout << "Solver: " << solver_name << endl;
        cout << "Generation = " << max_generations << endl;
        cout << "Archive size = " << imoma.GetArchiveSize() << endl;
        cout << "The best scalar solution = " << imoma.GetBestScalarFit() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;
#ifdef PROFILE_EVAL
        PrintEvalProfile(solver);
#endif
        return 0;
    }

    // DSAC-DE Solver (Discretized Soft Actor-Critic configured Differential Evolution)
    if (solver_name == "DSAC-DE") {
        cout << "\n=== Running DSAC-DE Solver ===" << endl;
        DSAC_DE_Solver dsac_de(&solver, popsize);
        dsac_de.SetMaxGenerations(max_generations);
        dsac_de.SetScalingFactor(0.5);
        dsac_de.SetCrossoverRate(0.5);
        dsac_de.SetLearningRate(0.0001);
        dsac_de.SetDiscountFactor(0.99);
        dsac_de.SetTemperature(0.5);
        dsac_de.SetSoftUpdateRate(0.5);
        dsac_de.SetBufferSize(40000);
        dsac_de.SetBatchSize(512);
        dsac_de.SetTrainingEnabled(true);
        dsac_de.Init();
        solver.ResetEvalCount();
        uint64_t next_log_eval = (uint64_t)log_every;

        for (int gen = 0; gen < max_generations && (max_evals == 0 || solver.GetEvalCount() < max_evals); gen++) {
            dsac_de.RunGeneration(gen);
            if (max_evals > 0) {
                while (solver.GetEvalCount() >= next_log_eval) {
                    cout << "Eval " << next_log_eval
                         << ": best_fit = " << dsac_de.GetGlobalBestFit() << endl;
                    next_log_eval += (uint64_t)log_every;
                }
            } else if ((gen + 1) % log_every == 0 || gen == max_generations - 1) {
                cout << "Gen " << (gen + 1)
                     << ": best_fit = " << dsac_de.GetGlobalBestFit() << endl;
            }
        }

        clock_t t2 = clock();
        cout << "\n=== Final Results (DSAC-DE) ===" << endl;
        cout << "Solver: " << solver_name << endl;
        cout << "Generation = " << max_generations << endl;
        cout << "The best solution = " << dsac_de.GetGlobalBestFit() << endl;
        cout << "Time = " << (double)(t2 - t1) / CLOCKS_PER_SEC << " s" << endl;

        // Print operator statistics
        cout << "\nOperator Statistics:" << endl;
        const OpStats* stats = dsac_de.GetOpStats();
        for (int i = 0; i < NUM_DE_OPS; i++) {
            cout << "  Op" << (i+1) << ": uses=" << stats[i].total_uses
                 << " success_parent=" << stats[i].success_parent
                 << " success_gbest=" << stats[i].success_gbest
                 << " success_avg=" << stats[i].success_avg << endl;
        }
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

