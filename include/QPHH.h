#ifndef QPHH_H
#define QPHH_H

#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include "Multimethod.h"

struct QPHHConfig {
    // Parallel
    int num_threads = 1;

    // Evaluation caps
    int gi_cap = 30;
    int map_cap = 40;

    // Q-learning
    double epsilon_init = 0.2;
    double epsilon_decay = 0.95;
    double gamma = 0.9;

    // Local search
    double p_c_init = 1.0;
    double p_c_decay = 0.05;
    double greedy_decay = 0.95;

    // Early stop placeholders
    bool use_early_stop = true;
    int patience = 500;
    double min_improvement = 1e-6;
};

class QPHH_Solver {
public:
    QPHH_Solver(MultiMet* s, int popsize = 50, int n_tasks = 15, double epsilon = 0.3, double gamma = 0.5);
    ~QPHH_Solver() = default;

    void SetMaxIterations(int iters) { max_iterations = iters > 0 ? iters : 1; }
    void SetEpsilon(double e) { epsilon = e; config.epsilon_init = e; }
    void SetGamma(double g) { gamma = g; config.gamma = g; }
    void SetTasksN(int n) { N = n > 0 ? n : 1; }
    void SetPopSize(int p) { P = p > 1 ? p : 2; }
    void SetInitPoolFactor(int factor) { init_pool_factor = factor > 0 ? factor : 1; }
    void SetGreedyInsertCap(int cap) { gi_cap = cap > 0 ? cap : 0; config.gi_cap = gi_cap; }
    void SetMappingCap(int cap) { map_cap = cap > 0 ? cap : 0; config.map_cap = map_cap; }
    void SetNumThreads(int n) { num_threads = n > 0 ? n : 1; config.num_threads = num_threads; }
    void SetConfig(const QPHHConfig& cfg);
    void SetLog(bool llh_log, bool q_log, int every = 50) {
        log_llh = llh_log;
        log_q = q_log;
        log_every = every > 0 ? every : 1;
    }

    void Init();
    void RunIteration(int iter);

    double GetBestFit() const { return gbest_fit; }
    const std::vector<double>& GetBestVar() const { return gbest; }

private:
    struct Individual {
        std::vector<double> var;
        double fit = 1e30;
    };

    // Core pointers and sizes
    MultiMet* solver;
    int P;
    int init_pool_factor;
    int P0;
    int N;
    int max_iterations;
    int CE_Tnum;
    int M_Jnum;
    int M_OPTnum;
    int ops;
    int Nvar;

    // Q-learning
    std::vector<std::vector<double>> Q;
    int prev_state;
    int prev_action;
    double epsilon;
    double gamma;

    // Adaptive local search
    double p_c;

    // Greedy/ETRM
    double p_greedy;
    double greedy_decay;
    int gi_cap;
    int map_cap;
    int num_threads;
    QPHHConfig config;

    // Population
    std::vector<Individual> pop;
    std::vector<double> gbest;
    double gbest_fit;
    double prev_best_fit;
    int current_iter;
    int no_improve_count;
    double last_best;
    std::vector<double> tmp_var;
    std::vector<double> trial_var;
    std::vector<int> reusable_order;
    std::vector<int> reusable_mapping;
    std::vector<int> reusable_rank_idx;
    std::vector<int> reusable_apply_ids;
    std::vector<int> reusable_idx;
    struct ThreadScratch {
        std::vector<int> order;
        std::vector<int> dev_idx_by_op;
        std::vector<double> local_tmp_var;
    };
    std::vector<ThreadScratch> thread_scratch;

    // Logging
    bool log_llh;
    bool log_q;
    int log_every;

private:
    // Mapping helpers
    void DecodeOrder(const std::vector<double>& var, std::vector<int>& order) const;
    void DecodeDevIdxByOp(const std::vector<double>& var, const std::vector<int>& order, std::vector<int>& dev_idx_by_op) const;
    void EncodeOrderAndMapping(std::vector<double>& var, const std::vector<int>& order, const std::vector<int>& dev_idx_by_op) const;
    void RepairOrder(std::vector<int>& order) const;

    // LLHs (operate on TEO)
    void LLH_3Swap(std::vector<int>& order) const;
    void LLH_IntraLevelSwap(std::vector<int>& order) const;
    void LLH_3Insert(std::vector<int>& order);
    void LLH_ParentAwareInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var);
    void LLH_GreedyInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, const std::vector<double>& base_var, std::vector<double>& tmp_var, bool pick_suboptimal);
    void LLH_PairInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var);

    // Local search operators
    void TwoPointCrossover(Individual& ind);
    void NTasksGreedyInsert(Individual& ind, std::vector<double>& tmp_var);

    // Initialization helpers
    void BuildRandomFeasibleOrder(std::vector<int>& order) const;
    void BuildMappingGreedyETRM(std::vector<double>& var, const std::vector<int>& order);

    // Evaluation helpers
    double EvalWithOrder(const std::vector<double>& base_var, const std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var) const;

    void GreedyInsertTask(std::vector<int>& order, int task, const std::vector<int>& dev_idx_by_op, const std::vector<double>& base_var, std::vector<double>& tmp_var, bool pick_suboptimal);

    // Q-learning helpers
    int SelectAction(int state) const;
    int ComputeState(double delta) const;
    double ComputeReward(double delta) const;
    double ComputeAlpha(int iter) const;
    double ComputeEpsilon(int iter) const;
    double EvalVarSafe(const double* var) const;

    // Utility
    double randval(double low, double high) const;
    int randint(int low, int high_exclusive) const;
    void Shuffle(std::vector<int>& data) const;
};

#endif // QPHH_H
