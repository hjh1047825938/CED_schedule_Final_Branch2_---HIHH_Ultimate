#ifndef SOLVER_RDE_H
#define SOLVER_RDE_H

#include "Multimethod.h"
#include <cstdint>
#include <vector>

struct RDEConfig {
    uint64_t eval_budget = 400000;
    int np_max = 0;
    int np_min = 4;
    int memory_size = 5;
    double archive_rate = 1.0;
    double p_max = 0.25;
    double rank_pressure = 3.0;
    double gamma1 = 0.5;
    double gamma2 = 0.5;
    double eta_gamma = 0.2;
    double init_memory_f = 0.3;
    double init_memory_cr = 0.8;
};

struct RDEIndividual {
    std::vector<double> x;
    double fitness = 0.0;
};

struct RDESuccessRecord {
    int strategy_id = 0;
    double F = 0.0;
    double CR = 0.0;
    double delta = 0.0;
    double normalized_delta = 0.0;
};

class SolverRDE {
public:
    SolverRDE(MultiMet* solver, const RDEConfig& config = RDEConfig{});

    void SetConfig(const RDEConfig& config);
    void Init();
    void RunGeneration(int gen);

    double GetBestFit() const { return best_f; }
    const std::vector<double>& GetBestVar() const { return best_x; }
    int GetPopulationSize() const { return (int)population.size(); }
    int GetArchiveSize() const { return (int)archive.size(); }
    bool HasBest() const { return !best_x.empty(); }
    double GetGamma1() const { return gamma1; }
    double GetGamma2() const { return gamma2; }
    uint64_t GetEvalBudget() const { return cfg.eval_budget; }

private:
    struct UnionRef {
        bool from_archive = false;
        int index = -1;
    };

    MultiMet* solver = nullptr;
    RDEConfig cfg{};
    int nvar = 0;
    double lower_bound = 0.0;
    double upper_bound = 1.0;

    std::vector<RDEIndividual> population;
    std::vector<std::vector<double>> archive;
    std::vector<double> memory_f;
    std::vector<double> memory_cr;
    int memory_index = 0;
    double gamma1 = 0.5;
    double gamma2 = 0.5;
    std::vector<double> best_x;
    double best_f = 0.0;

private:
    int default_np_max() const;
    uint64_t eval_count() const;
    double rand01() const;
    int rand_int(int lo, int hi_exclusive) const;
    double clip(double value, double lo, double hi) const;
    double sample_cauchy_positive_truncated(double mean) const;
    double sample_normal_clipped(double mean) const;
    double sample_F() const;
    double sample_CR() const;
    double evaluate(const std::vector<double>& x) const;
    void update_best(const std::vector<double>& x, double fitness);
    void initialize_population();
    std::vector<int> sorted_indices() const;
    void reduce_population();
    int sample_pbest_index(const std::vector<int>& order, int top_count) const;
    int sample_order_pbest_index(const std::vector<int>& order, int top_count) const;
    int sample_population_index_excluding(int exclude_a, int exclude_b = -1) const;
    UnionRef sample_r2_source(int exclude_a, int exclude_b) const;
    const std::vector<double>& source_vector(const UnionRef& ref) const;
    std::vector<double> mutate_strategy1(int target_idx, int pbest_idx, int r1_idx, const UnionRef& r2_ref, double F) const;
    std::vector<double> mutate_strategy2(int target_idx, int order_pbest_idx, int r1_idx, const UnionRef& r2_ref, double F) const;
    std::vector<double> crossover_binomial(const std::vector<double>& target, const std::vector<double>& donor, double CR) const;
    void repair_vector(std::vector<double>& x) const;
    void update_archive(const std::vector<double>& parent);
    void trim_archive();
    void update_memories(const std::vector<RDESuccessRecord>& successes);
    void update_strategy_shares(const std::vector<RDESuccessRecord>& successes);
};

#endif
