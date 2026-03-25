#include "NL_SHADE_LBC.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

NL_SHADE_LBC_Solver::NL_SHADE_LBC_Solver(MultiMet* solver, int pop_size, uint64_t max_evals)
    : solver(solver),
      init_pop_size(std::max(4, pop_size)),
      min_pop_size(4),
      nvar(solver ? solver->Nvar : 0),
      memory_size(6),
      max_evals(max_evals),
      bias_start(0.0),
      bias_end(-0.15),
      p_max(0.25),
      memory_f(memory_size, 0.5),
      memory_cr(memory_size, 0.5),
      memory_ptr(0),
      best_fit(std::numeric_limits<double>::infinity()),
      rng(static_cast<uint32_t>(solver ? solver->seed : 1u)) {
    if (!memory_f.empty()) memory_f.back() = 0.9;
    if (!memory_cr.empty()) memory_cr.back() = 0.9;
}

void NL_SHADE_LBC_Solver::SetPopulationBounds(int init_size, int min_size)
{
    init_pop_size = std::max(4, init_size);
    min_pop_size = std::max(4, std::min(init_pop_size, min_size));
}

double NL_SHADE_LBC_Solver::Rand01()
{
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

int NL_SHADE_LBC_Solver::RandInt(int lo, int hi_exclusive)
{
    if (hi_exclusive <= lo) return lo;
    return std::uniform_int_distribution<int>(lo, hi_exclusive - 1)(rng);
}

double NL_SHADE_LBC_Solver::Clip01(double value) const
{
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

double NL_SHADE_LBC_Solver::ProgressRatio() const
{
    if (max_evals == 0) return 0.0;
    return std::clamp((double)solver->GetEvalCount() / (double)max_evals, 0.0, 1.0);
}

double NL_SHADE_LBC_Solver::SampleF()
{
    const int k = RandInt(0, memory_size);
    const double bias = bias_start + (bias_end - bias_start) * ProgressRatio();
    const double loc = (k == memory_size - 1) ? 0.9 : (memory_f[k] + bias);
    std::cauchy_distribution<double> dist(loc, 0.1);
    double f = -1.0;
    int tries = 0;
    while (f <= 0.0 && tries < 64) {
        f = dist(rng);
        tries++;
    }
    if (f <= 0.0) f = std::max(0.05, loc);
    return std::min(1.0, f);
}

double NL_SHADE_LBC_Solver::SampleCR()
{
    const int k = RandInt(0, memory_size);
    std::normal_distribution<double> dist(memory_cr[k], 0.1);
    return Clip01(dist(rng));
}

double NL_SHADE_LBC_Solver::Evaluate(const std::vector<double>& var)
{
    return solver->Eval(var.data());
}

void NL_SHADE_LBC_Solver::UpdateBest(const std::vector<double>& var, double fit)
{
    if (fit + 1e-12 < best_fit) {
        best_fit = fit;
        best_var = var;
    }
}

void NL_SHADE_LBC_Solver::AddToArchive(const NL_SHADE_LBC_Individual& ind)
{
    int max_size = std::max(1, (int)std::llround(2.6 * std::max(1, (int)population.size())));
    if ((int)archive.size() < max_size) {
        archive.push_back(ind);
        return;
    }
    archive[RandInt(0, (int)archive.size())] = ind;
}

void NL_SHADE_LBC_Solver::TrimArchive(int max_size)
{
    max_size = std::max(1, max_size);
    while ((int)archive.size() > max_size) {
        int idx = RandInt(0, (int)archive.size());
        archive[idx] = archive.back();
        archive.pop_back();
    }
}

std::vector<int> NL_SHADE_LBC_Solver::SortedIndices() const
{
    std::vector<int> order(population.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::fabs(population[a].fit - population[b].fit) <= 1e-12) return a < b;
        return population[a].fit < population[b].fit;
    });
    return order;
}

int NL_SHADE_LBC_Solver::SelectPBestIndex(const std::vector<int>& sorted_indices, int pbest_count)
{
    pbest_count = std::max(2, std::min((int)sorted_indices.size(), pbest_count));
    return sorted_indices[RandInt(0, pbest_count)];
}

int NL_SHADE_LBC_Solver::SelectRankBiasedIndex(const std::vector<int>& sorted_indices, int exclude)
{
    if (sorted_indices.empty()) return 0;
    std::vector<double> weights(sorted_indices.size(), 0.0);
    double total = 0.0;
    for (int r = 0; r < (int)sorted_indices.size(); ++r) {
        int idx = sorted_indices[r];
        if (idx == exclude) continue;
        weights[r] = (double)((int)sorted_indices.size() - r);
        total += weights[r];
    }
    if (total <= 0.0) {
        for (int idx : sorted_indices) {
            if (idx != exclude) return idx;
        }
        return sorted_indices.front();
    }
    double pick = Rand01() * total;
    double acc = 0.0;
    for (int r = 0; r < (int)sorted_indices.size(); ++r) {
        if (weights[r] <= 0.0) continue;
        acc += weights[r];
        if (acc >= pick) return sorted_indices[r];
    }
    return sorted_indices.back();
}

NL_SHADE_LBC_Individual NL_SHADE_LBC_Solver::SelectUnionIndividualExcluding(int exclude_a, int exclude_b)
{
    const int union_size = (int)population.size() + (int)archive.size();
    for (int guard = 0; guard < 128; ++guard) {
        int pick = RandInt(0, std::max(1, union_size));
        if (pick < (int)population.size()) {
            if (pick == exclude_a || pick == exclude_b) continue;
            return population[pick];
        }
        if (!archive.empty()) return archive[pick - (int)population.size()];
    }
    for (int i = 0; i < (int)population.size(); ++i) {
        if (i != exclude_a && i != exclude_b) return population[i];
    }
    return population.front();
}

double NL_SHADE_LBC_Solver::WeightedLehmerMean(const std::vector<double>& values, const std::vector<double>& weights) const
{
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < (int)values.size(); ++i) {
        num += weights[i] * values[i] * values[i];
        den += weights[i] * values[i];
    }
    if (den <= 1e-12) return 0.5;
    return num / den;
}

void NL_SHADE_LBC_Solver::ApplyPopulationReduction()
{
    if (max_evals == 0 || population.empty()) return;
    const double ratio = ProgressRatio();
    int target = (int)std::llround((double)init_pop_size + (double)(min_pop_size - init_pop_size) * ratio * ratio);
    target = std::max(min_pop_size, std::min((int)population.size(), target));
    if (target < (int)population.size()) {
        std::sort(population.begin(), population.end(), [](const auto& a, const auto& b) {
            return a.fit < b.fit;
        });
        population.resize(target);
    }
    TrimArchive((int)std::llround(2.6 * std::max(1, (int)population.size())));
}

void NL_SHADE_LBC_Solver::Init()
{
    population.clear();
    archive.clear();
    memory_f.assign(memory_size, 0.5);
    memory_cr.assign(memory_size, 0.5);
    memory_f.back() = 0.9;
    memory_cr.back() = 0.9;
    memory_ptr = 0;
    best_fit = std::numeric_limits<double>::infinity();
    best_var.clear();

    population.reserve(init_pop_size);
    int seeded = 0;
    if (solver && solver->pop && solver->pop_fit && solver->Popsize > 0) {
        seeded = std::min(init_pop_size, solver->Popsize);
        for (int i = 0; i < seeded; ++i) {
            NL_SHADE_LBC_Individual ind;
            ind.var.assign(solver->pop[i], solver->pop[i] + nvar);
            ind.fit = solver->pop_fit[i];
            UpdateBest(ind.var, ind.fit);
            population.push_back(std::move(ind));
        }
    }

    for (int i = seeded; i < init_pop_size; ++i) {
        if (max_evals > 0 && solver->GetEvalCount() >= max_evals) break;
        NL_SHADE_LBC_Individual ind;
        ind.var.resize(nvar, 0.0);
        for (int j = 0; j < nvar; ++j) ind.var[j] = Rand01();
        ind.fit = Evaluate(ind.var);
        UpdateBest(ind.var, ind.fit);
        population.push_back(std::move(ind));
    }
}

void NL_SHADE_LBC_Solver::RunGeneration(int /*gen*/)
{
    if (population.empty()) return;
    if (max_evals > 0 && solver->GetEvalCount() >= max_evals) return;

    const std::vector<int> sorted_indices = SortedIndices();
    const int current_n = (int)population.size();
    const double ratio = ProgressRatio();
    const double pb = std::max(2.0 / std::max(1, current_n), p_max - (p_max - 2.0 / std::max(1, current_n)) * ratio);
    const int pbest_count = std::max(2, (int)std::ceil(pb * current_n));

    std::vector<double> success_f;
    std::vector<double> success_cr;
    std::vector<double> success_delta;

    for (int i = 0; i < current_n; ++i) {
        if (max_evals > 0 && solver->GetEvalCount() >= max_evals) break;

        const double Fi = SampleF();
        const double CRi = SampleCR();
        const int pbest_idx = SelectPBestIndex(sorted_indices, pbest_count);
        int r1_idx = SelectRankBiasedIndex(sorted_indices, i);
        if (r1_idx == pbest_idx && current_n > 2) {
            r1_idx = SelectRankBiasedIndex(sorted_indices, i == pbest_idx ? -1 : i);
            if (r1_idx == i) r1_idx = pbest_idx;
        }
        const NL_SHADE_LBC_Individual xr2 = SelectUnionIndividualExcluding(i, r1_idx);

        std::vector<double> mutant(nvar, 0.0);
        std::vector<double> trial(nvar, 0.0);
        const int j_rand = RandInt(0, std::max(1, nvar));
        for (int j = 0; j < nvar; ++j) {
            double v = population[i].var[j]
                     + Fi * (population[pbest_idx].var[j] - population[i].var[j])
                     + Fi * (population[r1_idx].var[j] - xr2.var[j]);
            if (v < 0.0) v = population[i].var[j] * 0.5;
            else if (v > 1.0) v = (population[i].var[j] + 1.0) * 0.5;
            mutant[j] = Clip01(v);

            if (Rand01() < CRi || j == j_rand) trial[j] = mutant[j];
            else trial[j] = population[i].var[j];
        }

        const double trial_fit = Evaluate(trial);
        if (trial_fit <= population[i].fit) {
            success_f.push_back(Fi);
            success_cr.push_back(CRi);
            success_delta.push_back(std::fabs(population[i].fit - trial_fit));
            AddToArchive(population[i]);
            population[i].var = std::move(trial);
            population[i].fit = trial_fit;
            UpdateBest(population[i].var, population[i].fit);
        }
    }

    if (!success_f.empty()) {
        double delta_sum = std::accumulate(success_delta.begin(), success_delta.end(), 0.0);
        if (delta_sum <= 1e-12) delta_sum = (double)success_delta.size();
        std::vector<double> weights(success_delta.size(), 0.0);
        for (int i = 0; i < (int)success_delta.size(); ++i) weights[i] = success_delta[i] / delta_sum;

        memory_f[memory_ptr] = Clip01(WeightedLehmerMean(success_f, weights));
        double mean_cr = 0.0;
        for (int i = 0; i < (int)success_cr.size(); ++i) mean_cr += weights[i] * success_cr[i];
        memory_cr[memory_ptr] = Clip01(mean_cr);
        memory_ptr = (memory_ptr + 1) % std::max(1, memory_size - 1);
    }

    memory_f.back() = 0.9;
    memory_cr.back() = 0.9;
    ApplyPopulationReduction();
}
