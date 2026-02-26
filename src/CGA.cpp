#include "CGA.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <utility>

CGA::CGA(std::vector<CGATask> tasks, std::vector<CGAVM> vms, double rate, const CGAConfig& config, uint32_t seed)
    : tasks_(std::move(tasks)),
      vms_(std::move(vms)),
      rate_(rate),
      cfg_(config),
      rng_(seed),
      real01_(0.0, 1.0),
      fitness_ref_(1.0) {}

void CGA::InitializePopulation()
{
    const int task_count = static_cast<int>(tasks_.size());
    const int vm_count = static_cast<int>(vms_.size());
    std::uniform_int_distribution<int> vm_pick(0, vm_count - 1);

    population_.assign(cfg_.population_size, Individual{});
    for (int i = 0; i < cfg_.population_size; ++i) {
        population_[i].genes.resize(task_count);
        for (int t = 0; t < task_count; ++t) {
            population_[i].genes[t] = vm_pick(rng_);
        }
    }

    // First pass (with ref=1.0) to collect a stable normalization reference.
    fitness_ref_ = 1.0;
    EvaluatePopulation(population_);

    double max_obj = 0.0;
    for (const Individual& ind : population_) {
        // Here fitness is raw objective because fitness_ref_ == 1.0.
        if (ind.fitness > max_obj) max_obj = ind.fitness;
    }
    fitness_ref_ = (max_obj > 1e-12) ? max_obj : 1.0;

    // Second pass to produce normalized best_fit comparable across scales.
    EvaluatePopulation(population_);
}

void CGA::EvaluateIndividual(Individual& ind) const
{
    const int task_count = static_cast<int>(tasks_.size());
    const int vm_count = static_cast<int>(vms_.size());

    if ((int)rt_buf_.size() != vm_count) rt_buf_.assign(vm_count, 0.0);
    else std::fill(rt_buf_.begin(), rt_buf_.end(), 0.0);
    double punish_sum = 0.0;
    int satisfied = 0;

    for (int t = 0; t < task_count; ++t) {
        int vm = ind.genes[t];
        if (vm < 0) vm = 0;
        if (vm >= vm_count) vm = vm_count - 1;

        const double exec_time = tasks_[t].data_length / vms_[vm].mips;
        const double trans_time = tasks_[t].input_data_size / rate_;
        const double finish_time = exec_time + trans_time;

        rt_buf_[vm] += exec_time;

        if (finish_time > tasks_[t].deadline) {
            punish_sum += std::fabs(finish_time - tasks_[t].deadline);
        } else {
            ++satisfied;
        }
    }

    const double all_ntime = *std::max_element(rt_buf_.begin(), rt_buf_.end());
    const double objective = all_ntime + punish_sum;
    const double safe_obj = (objective > 1e-15) ? objective : 1e-15;
    const double safe_ref = (fitness_ref_ > 1e-15) ? fitness_ref_ : 1.0;

    ind.makespan = all_ntime;
    ind.total_punish = punish_sum;
    ind.satisfied_tasks = satisfied;
    ind.fitness = safe_obj / safe_ref;
    ind.roulette_score = 1.0 / safe_obj;
}

void CGA::EvaluatePopulation(std::vector<Individual>& pop) const
{
    for (Individual& ind : pop) {
        EvaluateIndividual(ind);
    }
}

int CGA::FindBestIndex(const std::vector<Individual>& pop) const
{
    int best = 0;
    for (int i = 1; i < static_cast<int>(pop.size()); ++i) {
        if (pop[i].fitness < pop[best].fitness) {
            best = i;
        }
    }
    return best;
}

double CGA::Similarity(const Individual& a, const Individual& b) const
{
    if (a.genes.empty()) return 1.0;
    int same = 0;
    for (int i = 0; i < static_cast<int>(a.genes.size()); ++i) {
        if (a.genes[i] == b.genes[i]) ++same;
    }
    return static_cast<double>(same) / static_cast<double>(a.genes.size());
}

int CGA::RouletteSelect(const std::vector<Individual>& pop, double total_score)
{
    if (total_score <= 0.0) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
        return pick(rng_);
    }
    const double target = real01_(rng_) * total_score;
    double accum = 0.0;
    for (int i = 0; i < static_cast<int>(pop.size()); ++i) {
        accum += pop[i].roulette_score;
        if (accum >= target) return i;
    }
    return static_cast<int>(pop.size()) - 1;
}

std::vector<CGA::Individual> CGA::SelectPopulation()
{
    std::vector<Individual> selected;
    selected.reserve(population_.size());

    const int elite_idx = FindBestIndex(population_);
    selected.push_back(population_[elite_idx]);  // elitism

    const double total_score = std::accumulate(
        population_.begin(),
        population_.end(),
        0.0,
        [](double s, const Individual& ind) { return s + ind.roulette_score; });

    while (static_cast<int>(selected.size()) < cfg_.population_size) {
        const int idx = RouletteSelect(population_, total_score);
        selected.push_back(population_[idx]);
    }
    return selected;
}

void CGA::ApplyCrossover(std::vector<Individual>& pop)
{
    const int task_count = static_cast<int>(tasks_.size());
    if (task_count <= 1) return;

    std::uniform_int_distribution<int> point_pick(1, task_count - 1);
    for (int i = 1; i + 1 < static_cast<int>(pop.size()); i += 2) {
        if (real01_(rng_) > cfg_.crossover_prob) continue;
        if (Similarity(pop[i], pop[i + 1]) >= cfg_.crossover_similarity_threshold) continue;

        const int cp = point_pick(rng_);
        for (int p = cp; p < task_count; ++p) {
            std::swap(pop[i].genes[p], pop[i + 1].genes[p]);
        }
    }
}

void CGA::MutateIndividual(Individual& ind)
{
    const int task_count = static_cast<int>(ind.genes.size());
    const int vm_count = static_cast<int>(vms_.size());
    if (task_count <= 0 || vm_count <= 1) return;

    std::uniform_int_distribution<int> pos_pick(0, task_count - 1);
    int p1 = pos_pick(rng_);
    int p2 = pos_pick(rng_);

    // Ensure effective mutation: if same-gene values are selected, advance position until different.
    if (ind.genes[p1] == ind.genes[p2]) {
        int tries = 0;
        while (tries < task_count && ind.genes[p1] == ind.genes[p2]) {
            p2 = (p2 + 1) % task_count;
            ++tries;
        }
    }

    if (ind.genes[p1] != ind.genes[p2]) {
        std::swap(ind.genes[p1], ind.genes[p2]);
        return;
    }

    // If all genes are same VM, force a valid VM change.
    std::uniform_int_distribution<int> vm_pick(0, vm_count - 1);
    int new_vm = vm_pick(rng_);
    if (new_vm == ind.genes[p1]) {
        new_vm = (new_vm + 1) % vm_count;
    }
    ind.genes[p1] = new_vm;
}

void CGA::MutatePopulation(std::vector<Individual>& pop, int generation)
{
    const double pm = (generation < cfg_.mutation_switch_generation)
                          ? cfg_.mutation_prob_early
                          : cfg_.mutation_prob_late;
    for (int i = 1; i < static_cast<int>(pop.size()); ++i) {  // keep elite untouched
        if (real01_(rng_) < pm) {
            MutateIndividual(pop[i]);
        }
    }
}

void CGA::ApplyCatastrophe(std::vector<Individual>& pop)
{
    const int n = static_cast<int>(pop.size());
    const int top_n = std::max(1, n / 3);

    if ((int)catastrophe_idx_buf_.size() != n) catastrophe_idx_buf_.resize(n);
    for (int i = 0; i < n; ++i) catastrophe_idx_buf_[i] = i;
    std::sort(catastrophe_idx_buf_.begin(), catastrophe_idx_buf_.end(), [&](int a, int b) { return pop[a].fitness < pop[b].fitness; });

    for (int i = 0; i < top_n; ++i) {
        int id = catastrophe_idx_buf_[i];
        if (real01_(rng_) < cfg_.catastrophe_mutation_prob) {
            MutateIndividual(pop[id]);
            EvaluateIndividual(pop[id]);
        }
    }
}

CGAResult CGA::Run(int log_every)
{
    CGAResult result;
    if (tasks_.empty() || vms_.empty() || rate_ <= 0.0) {
        return result;
    }

    InitializePopulation();
    int best_idx = FindBestIndex(population_);
    Individual global_best = population_[best_idx];
    int best_gen = 0;
    int stagnation_count = 0;
    int catastrophe_count = 0;

    for (int gen = 1; gen <= cfg_.max_generations; ++gen) {
        std::vector<Individual> next = SelectPopulation();
        ApplyCrossover(next);
        MutatePopulation(next, gen);
        EvaluatePopulation(next);
        population_.swap(next);

        const int cur_best_idx = FindBestIndex(population_);
        const Individual& cur_best = population_[cur_best_idx];
        if (cur_best.fitness + cfg_.stagnation_epsilon < global_best.fitness) {
            global_best = cur_best;
            best_gen = gen;
            stagnation_count = 0;
        } else {
            ++stagnation_count;
        }

        if (gen <= cfg_.catastrophe_apply_generations &&
            stagnation_count >= cfg_.catastrophe_threshold) {
            ApplyCatastrophe(population_);
            ++catastrophe_count;
            stagnation_count = 0;

            const int after_cat_idx = FindBestIndex(population_);
            if (population_[after_cat_idx].fitness + cfg_.stagnation_epsilon < global_best.fitness) {
                global_best = population_[after_cat_idx];
                best_gen = gen;
            }
        }

        if (log_every > 0 && (gen % log_every == 0 || gen == cfg_.max_generations)) {
            std::cout << "Gen " << gen << ": best_fit = " << global_best.fitness << std::endl;
        }
    }

    result.total_tasks = static_cast<int>(tasks_.size());
    result.satisfied_tasks = global_best.satisfied_tasks;
    result.best_fitness = global_best.fitness;
    result.min_completion_time = global_best.makespan;
    result.total_punish = global_best.total_punish;
    result.best_generation = best_gen;
    result.catastrophe_count = catastrophe_count;
    result.delay_satisfaction_rate = (result.total_tasks > 0)
                                         ? static_cast<double>(result.satisfied_tasks) /
                                               static_cast<double>(result.total_tasks)
                                         : 0.0;

    result.best_mapping.resize(global_best.genes.size());
    for (int i = 0; i < static_cast<int>(global_best.genes.size()); ++i) {
        result.best_mapping[i] = global_best.genes[i] + 1;  // convert to 1-based VM id
    }
    return result;
}
