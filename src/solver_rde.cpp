#include "solver_rde.h"

#include "Rng.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

SolverRDE::SolverRDE(MultiMet* solver_ptr, const RDEConfig& config)
    : solver(solver_ptr),
      cfg(config),
      nvar(solver_ptr ? solver_ptr->Nvar : 0),
      lower_bound(solver_ptr ? solver_ptr->Lbound : 0.0),
      upper_bound(solver_ptr ? solver_ptr->Ubound : 1.0),
      gamma1(config.gamma1),
      gamma2(config.gamma2),
      best_f(std::numeric_limits<double>::infinity()) {
    SetConfig(config);
}

void SolverRDE::SetConfig(const RDEConfig& config)
{
    cfg = config;
    cfg.eval_budget = std::max<uint64_t>(1, cfg.eval_budget);
    cfg.np_max = (cfg.np_max > 0) ? cfg.np_max : default_np_max();
    cfg.np_max = std::max(4, cfg.np_max);
    cfg.np_min = std::max(4, std::min(cfg.np_min, cfg.np_max));
    cfg.memory_size = std::max(1, cfg.memory_size);
    cfg.archive_rate = std::max(0.0, cfg.archive_rate);
    cfg.p_max = clip(cfg.p_max, 0.05, 1.0);
    cfg.rank_pressure = std::max(0.0, cfg.rank_pressure);
    cfg.init_memory_f = clip(cfg.init_memory_f, 1e-8, 1.0);
    cfg.init_memory_cr = clip(cfg.init_memory_cr, 0.0, 1.0);
    gamma1 = clip(cfg.gamma1, 0.1, 0.9);
    gamma2 = clip(cfg.gamma2, 0.1, 0.9);
    const double gamma_sum = gamma1 + gamma2;
    if (gamma_sum > 0.0) {
        gamma1 /= gamma_sum;
        gamma2 /= gamma_sum;
    } else {
        gamma1 = 0.5;
        gamma2 = 0.5;
    }
}

int SolverRDE::default_np_max() const
{
    return std::min(std::max(18 * std::max(1, nvar), 36), 300);
}

uint64_t SolverRDE::eval_count() const
{
    return solver ? solver->GetEvalCount() : 0;
}

double SolverRDE::rand01() const
{
    return Rng::getInstance().uniform01();
}

int SolverRDE::rand_int(int lo, int hi_exclusive) const
{
    if (hi_exclusive <= lo) return lo;
    return Rng::getInstance().uniformInt(lo, hi_exclusive - 1);
}

double SolverRDE::clip(double value, double lo, double hi) const
{
    return std::max(lo, std::min(hi, value));
}

double SolverRDE::sample_cauchy_positive_truncated(double mean) const
{
    std::cauchy_distribution<double> dist(mean, 0.1);
    double value = -1.0;
    int tries = 0;
    while (value <= 0.0 && tries < 256) {
        value = dist(Rng::getInstance().getEngine());
        ++tries;
    }
    if (value <= 0.0) value = std::max(1e-8, mean);
    return clip(value, 1e-8, 1.0);
}

double SolverRDE::sample_normal_clipped(double mean) const
{
    std::normal_distribution<double> dist(mean, 0.1);
    return clip(dist(Rng::getInstance().getEngine()), 0.0, 1.0);
}

double SolverRDE::sample_F() const
{
    const int slot = rand_int(0, cfg.memory_size);
    return sample_cauchy_positive_truncated(memory_f[slot]);
}

double SolverRDE::sample_CR() const
{
    const int slot = rand_int(0, cfg.memory_size);
    return sample_normal_clipped(memory_cr[slot]);
}

double SolverRDE::evaluate(const std::vector<double>& x) const
{
    return solver->Eval(x.data());
}

void SolverRDE::update_best(const std::vector<double>& x, double fitness)
{
    if (fitness + 1e-12 < best_f) {
        best_f = fitness;
        best_x = x;
    }
}

void SolverRDE::initialize_population()
{
    population.clear();
    archive.clear();
    memory_f.assign(cfg.memory_size, cfg.init_memory_f);
    memory_cr.assign(cfg.memory_size, cfg.init_memory_cr);
    memory_index = 0;
    gamma1 = 0.5;
    gamma2 = 0.5;
    best_f = std::numeric_limits<double>::infinity();
    best_x.clear();

    population.reserve(cfg.np_max);
    for (int i = 0; i < cfg.np_max && eval_count() < cfg.eval_budget; ++i) {
        RDEIndividual ind;
        ind.x.resize(nvar, 0.0);
        for (int j = 0; j < nvar; ++j) {
            ind.x[j] = Rng::getInstance().uniformReal(lower_bound, upper_bound);
        }
        ind.fitness = evaluate(ind.x);
        update_best(ind.x, ind.fitness);
        population.push_back(std::move(ind));
    }
}

void SolverRDE::Init()
{
    initialize_population();
}

std::vector<int> SolverRDE::sorted_indices() const
{
    std::vector<int> order(population.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::fabs(population[a].fitness - population[b].fitness) <= 1e-12) return a < b;
        return population[a].fitness < population[b].fitness;
    });
    return order;
}

void SolverRDE::reduce_population()
{
    if (population.empty()) return;
    const double ratio = (double)eval_count() / (double)cfg.eval_budget;
    int target = (int)std::llround((double)cfg.np_max - (double)(cfg.np_max - cfg.np_min) * ratio);
    target = std::max(cfg.np_min, std::min((int)population.size(), target));
    if (target < (int)population.size()) {
        std::sort(population.begin(), population.end(), [](const RDEIndividual& a, const RDEIndividual& b) {
            if (std::fabs(a.fitness - b.fitness) <= 1e-12) return a.x < b.x;
            return a.fitness < b.fitness;
        });
        population.resize(target);
    }
    trim_archive();
}

int SolverRDE::sample_pbest_index(const std::vector<int>& order, int top_count) const
{
    top_count = std::max(2, std::min((int)order.size(), top_count));
    return order[rand_int(0, top_count)];
}

int SolverRDE::sample_order_pbest_index(const std::vector<int>& order, int top_count) const
{
    top_count = std::max(2, std::min((int)order.size(), top_count));
    std::vector<double> weights(top_count, 0.0);
    double total = 0.0;
    for (int r = 0; r < top_count; ++r) {
        weights[r] = std::pow((double)(top_count - r), cfg.rank_pressure);
        total += weights[r];
    }
    if (total <= 0.0) return order.front();
    const double pick = rand01() * total;
    double acc = 0.0;
    for (int r = 0; r < top_count; ++r) {
        acc += weights[r];
        if (acc >= pick) return order[r];
    }
    return order[top_count - 1];
}

int SolverRDE::sample_population_index_excluding(int exclude_a, int exclude_b) const
{
    for (int tries = 0; tries < 256; ++tries) {
        const int idx = rand_int(0, (int)population.size());
        if (idx != exclude_a && idx != exclude_b) return idx;
    }
    for (int i = 0; i < (int)population.size(); ++i) {
        if (i != exclude_a && i != exclude_b) return i;
    }
    return 0;
}

SolverRDE::UnionRef SolverRDE::sample_r2_source(int exclude_a, int exclude_b) const
{
    const int union_size = (int)population.size() + (int)archive.size();
    for (int tries = 0; tries < 256; ++tries) {
        const int pick = rand_int(0, union_size);
        if (pick < (int)population.size()) {
            if (pick == exclude_a || pick == exclude_b) continue;
            return {false, pick};
        }
        if (!archive.empty()) return {true, pick - (int)population.size()};
    }
    return {false, sample_population_index_excluding(exclude_a, exclude_b)};
}

const std::vector<double>& SolverRDE::source_vector(const UnionRef& ref) const
{
    return ref.from_archive ? archive[ref.index] : population[ref.index].x;
}

std::vector<double> SolverRDE::mutate_strategy1(int target_idx, int pbest_idx, int r1_idx, const UnionRef& r2_ref, double F) const
{
    std::vector<double> donor(nvar, 0.0);
    const std::vector<double>& xi = population[target_idx].x;
    const std::vector<double>& xpbest = population[pbest_idx].x;
    const std::vector<double>& xr1 = population[r1_idx].x;
    const std::vector<double>& xr2 = source_vector(r2_ref);
    for (int j = 0; j < nvar; ++j) {
        donor[j] = xi[j]
                 + F * (xpbest[j] - xi[j])
                 + F * (xr1[j] - xr2[j]);
    }
    return donor;
}

std::vector<double> SolverRDE::mutate_strategy2(int target_idx, int order_pbest_idx, int r1_idx, const UnionRef& r2_ref, double F) const
{
    std::vector<double> donor(nvar, 0.0);
    const std::vector<double>& xi = population[target_idx].x;
    const std::vector<double>& xpbest = population[order_pbest_idx].x;
    const std::vector<double>& xr1 = population[r1_idx].x;
    const std::vector<double>& xr2 = source_vector(r2_ref);
    for (int j = 0; j < nvar; ++j) {
        donor[j] = xi[j]
                 + F * (xpbest[j] - xi[j])
                 + F * (xr1[j] - xr2[j]);
    }
    return donor;
}

std::vector<double> SolverRDE::crossover_binomial(const std::vector<double>& target, const std::vector<double>& donor, double CR) const
{
    std::vector<double> trial(nvar, 0.0);
    const int j_rand = rand_int(0, std::max(1, nvar));
    for (int j = 0; j < nvar; ++j) {
        trial[j] = (rand01() < CR || j == j_rand) ? donor[j] : target[j];
    }
    return trial;
}

void SolverRDE::repair_vector(std::vector<double>& x) const
{
    for (double& value : x) {
        if (value < lower_bound) {
            value = lower_bound + (lower_bound - value);
        }
        if (value > upper_bound) {
            value = upper_bound - (value - upper_bound);
        }
        value = clip(value, lower_bound, upper_bound);
    }
}

void SolverRDE::update_archive(const std::vector<double>& parent)
{
    archive.push_back(parent);
    trim_archive();
}

void SolverRDE::trim_archive()
{
    const int archive_limit = std::max(0, (int)std::llround(cfg.archive_rate * (double)population.size()));
    while ((int)archive.size() > archive_limit) {
        const int idx = rand_int(0, (int)archive.size());
        archive[idx] = archive.back();
        archive.pop_back();
    }
}

void SolverRDE::update_memories(const std::vector<RDESuccessRecord>& successes)
{
    if (successes.empty()) return;

    double delta_sum = 0.0;
    for (const auto& record : successes) delta_sum += record.delta;
    if (delta_sum <= 1e-12) return;

    double mf_num = 0.0;
    double mf_den = 0.0;
    double mcr = 0.0;
    for (const auto& record : successes) {
        const double w = record.delta / delta_sum;
        mf_num += w * record.F * record.F;
        mf_den += w * record.F;
        mcr += w * record.CR;
    }

    if (mf_den > 1e-12) {
        memory_f[memory_index] = clip(mf_num / mf_den, 1e-8, 1.0);
    }
    memory_cr[memory_index] = clip(mcr, 0.0, 1.0);
    memory_index = (memory_index + 1) % cfg.memory_size;
}

void SolverRDE::update_strategy_shares(const std::vector<RDESuccessRecord>& successes)
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    int count1 = 0;
    int count2 = 0;
    for (const auto& record : successes) {
        if (record.strategy_id == 1) {
            sum1 += record.normalized_delta;
            ++count1;
        } else if (record.strategy_id == 2) {
            sum2 += record.normalized_delta;
            ++count2;
        }
    }

    const double s1 = (count1 > 0) ? (sum1 / (double)count1) : 1e-12;
    const double s2 = (count2 > 0) ? (sum2 / (double)count2) : 1e-12;
    const double denom = s1 + s2 + 1e-12;
    const double target1 = s1 / denom;
    const double target2 = s2 / denom;

    gamma1 = (1.0 - cfg.eta_gamma) * gamma1 + cfg.eta_gamma * target1;
    gamma2 = (1.0 - cfg.eta_gamma) * gamma2 + cfg.eta_gamma * target2;
    gamma1 = clip(gamma1, 0.1, 0.9);
    gamma2 = clip(gamma2, 0.1, 0.9);
    const double gamma_sum = gamma1 + gamma2;
    if (gamma_sum > 0.0) {
        gamma1 /= gamma_sum;
        gamma2 /= gamma_sum;
    } else {
        gamma1 = 0.5;
        gamma2 = 0.5;
    }
}

void SolverRDE::RunGeneration(int /*gen*/)
{
    if (population.empty() || eval_count() >= cfg.eval_budget) return;

    reduce_population();
    const std::vector<int> order = sorted_indices();
    const int NP = (int)population.size();
    if (NP < 4) return;

    int n1 = (int)std::llround(gamma1 * (double)NP);
    n1 = std::max(1, std::min(NP - 1, n1));
    std::vector<int> shuffled_indices(NP);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
    Rng::getInstance().shuffle(shuffled_indices);
    std::vector<int> strategy_of(NP, 2);
    for (int k = 0; k < n1; ++k) strategy_of[shuffled_indices[k]] = 1;

    std::vector<RDESuccessRecord> successes;
    successes.reserve(NP);

    for (int i = 0; i < NP && eval_count() < cfg.eval_budget; ++i) {
        const double F = sample_F();
        const double CR = sample_CR();
        const double p_min = std::max(2.0 / (double)NP, 0.05);
        const double p_i = Rng::getInstance().uniformReal(p_min, std::max(p_min, cfg.p_max));
        const int top_count = std::max(2, (int)std::floor(p_i * (double)NP));

        const int r1_idx = sample_population_index_excluding(i);
        const UnionRef r2_ref = sample_r2_source(i, r1_idx);

        std::vector<double> donor;
        int strategy_id = strategy_of[i];
        if (strategy_id == 1) {
            const int pbest_idx = sample_pbest_index(order, top_count);
            donor = mutate_strategy1(i, pbest_idx, r1_idx, r2_ref, F);
        } else {
            const int order_pbest_idx = sample_order_pbest_index(order, top_count);
            donor = mutate_strategy2(i, order_pbest_idx, r1_idx, r2_ref, F);
        }

        std::vector<double> trial = crossover_binomial(population[i].x, donor, CR);
        repair_vector(trial);
        const double trial_fitness = evaluate(trial);

        if (trial_fitness <= population[i].fitness) {
            const double old_fitness = population[i].fitness;
            const double delta = old_fitness - trial_fitness;
            update_archive(population[i].x);
            population[i].x = std::move(trial);
            population[i].fitness = trial_fitness;
            update_best(population[i].x, population[i].fitness);

            RDESuccessRecord record;
            record.strategy_id = strategy_id;
            record.F = F;
            record.CR = CR;
            record.delta = std::max(0.0, delta);
            record.normalized_delta = std::max(0.0, delta) / std::max(std::fabs(old_fitness), 1.0);
            successes.push_back(record);
        }
    }

    if (!successes.empty()) {
        update_memories(successes);
    }
    update_strategy_shares(successes);
    trim_archive();
}
