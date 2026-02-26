#ifndef CGA_H
#define CGA_H

#include <cstdint>
#include <random>
#include <vector>

struct CGATask {
    double data_length = 0.0;
    double input_data_size = 0.0;
    double deadline = 0.0;
};

struct CGAVM {
    double mips = 0.0;
};

struct CGAConfig {
    int population_size = 100;
    int max_generations = 10000;
    double crossover_prob = 0.75;
    double crossover_similarity_threshold = 0.8;
    double mutation_prob_early = 0.03;
    double mutation_prob_late = 0.01;
    int mutation_switch_generation = 6667;
    int catastrophe_threshold = 150;
    int catastrophe_apply_generations = 5000;
    double catastrophe_mutation_prob = 0.8;
    double stagnation_epsilon = 1e-12;
};

struct CGAResult {
    std::vector<int> best_mapping;  // 1-based VM index for each task
    double best_fitness = 0.0;
    double min_completion_time = 0.0;
    double total_punish = 0.0;
    double delay_satisfaction_rate = 0.0;
    int satisfied_tasks = 0;
    int total_tasks = 0;
    int best_generation = 0;
    int catastrophe_count = 0;
};

class CGA {
public:
    CGA(std::vector<CGATask> tasks, std::vector<CGAVM> vms, double rate, const CGAConfig& config, uint32_t seed);
    CGAResult Run(int log_every = 0);

private:
    struct Individual {
        std::vector<int> genes;  // 0-based VM index
        double fitness = 0.0;         // Objective value: AllNTime + sum(punish), minimized.
        double roulette_score = 0.0;  // Selection score: 1 / fitness, maximized in roulette.
        double makespan = 0.0;
        double total_punish = 0.0;
        int satisfied_tasks = 0;
    };

    std::vector<CGATask> tasks_;
    std::vector<CGAVM> vms_;
    double rate_;
    CGAConfig cfg_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> real01_;

    std::vector<Individual> population_;
    double fitness_ref_;
    mutable std::vector<double> rt_buf_;
    std::vector<int> select_idx_buf_;
    std::vector<int> catastrophe_idx_buf_;

private:
    void InitializePopulation();
    void EvaluateIndividual(Individual& ind) const;
    void EvaluatePopulation(std::vector<Individual>& pop) const;
    int FindBestIndex(const std::vector<Individual>& pop) const;
    double Similarity(const Individual& a, const Individual& b) const;
    int RouletteSelect(const std::vector<Individual>& pop, double total_score);
    std::vector<Individual> SelectPopulation();
    void ApplyCrossover(std::vector<Individual>& pop);
    void MutatePopulation(std::vector<Individual>& pop, int generation);
    void MutateIndividual(Individual& ind);
    void ApplyCatastrophe(std::vector<Individual>& pop);
};

#endif  // CGA_H
