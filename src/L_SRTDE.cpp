#include "L_SRTDE.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

L_SRTDE_Solver::L_SRTDE_Solver(MultiMet* solver, int pop_size, uint64_t max_evals)
    : solver(solver),
      init_pop_size(std::max(4, pop_size)),
      min_pop_size(4),
      nvar(solver ? solver->Nvar : 0),
      memory_size(5),
      max_evals(max_evals),
      memory_cr(memory_size, 0.5),
      memory_ptr(0),
      success_rate(0.5),
      best_fit(std::numeric_limits<double>::infinity()),
      rng(static_cast<uint32_t>(solver ? solver->seed : 1u)) {}

void L_SRTDE_Solver::SetPopulationBounds(int init_size, int min_size)
{
    init_pop_size = std::max(4, init_size);
    min_pop_size = std::max(4, std::min(init_pop_size, min_size));
}

double L_SRTDE_Solver::Rand01()
{
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

int L_SRTDE_Solver::RandInt(int lo, int hi_exclusive)
{
    if (hi_exclusive <= lo) return lo;
    return std::uniform_int_distribution<int>(lo, hi_exclusive - 1)(rng);
}

double L_SRTDE_Solver::Clip01(double value) const
{
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

double L_SRTDE_Solver::SampleF(double mean_f)
{
    const double loc = std::clamp(mean_f, 0.0, 1.0);
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

double L_SRTDE_Solver::SampleCR()
{
    int k = RandInt(0, memory_size);
    std::normal_distribution<double> dist(memory_cr[k], 0.05);
    return Clip01(dist(rng));
}

double L_SRTDE_Solver::Evaluate(const std::vector<double>& var)
{
    return solver->Eval(var.data());
}

void L_SRTDE_Solver::UpdateBest(const std::vector<double>& var, double fit)
{
    if (fit + 1e-12 < best_fit) {
        best_fit = fit;
        best_var = var;
    }
}

void L_SRTDE_Solver::AddToArchive(const L_SRTDE_Individual& ind)
{
    int max_size = std::max(1, (int)population.size());
    if ((int)archive.size() < max_size) {
        archive.push_back(ind);
        return;
    }
    archive[RandInt(0, (int)archive.size())] = ind;
}

std::vector<int> L_SRTDE_Solver::SortedIndices() const
{
    std::vector<int> order(population.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::fabs(population[a].fit - population[b].fit) <= 1e-12) return a < b;
        return population[a].fit < population[b].fit;
    });
    return order;
}

int L_SRTDE_Solver::SelectPBestIndex(int pbest_count)
{
    std::vector<int> order = SortedIndices();
    pbest_count = std::max(2, std::min((int)order.size(), pbest_count));
    return order[RandInt(0, pbest_count)];
}

int L_SRTDE_Solver::SelectPopulationIndexExcluding(int exclude_a, int exclude_b)
{
    if ((int)population.size() <= 1) return std::max(0, exclude_a);
    int idx = exclude_a;
    int guard = 0;
    while ((idx == exclude_a || idx == exclude_b) && guard < 128) {
        idx = RandInt(0, (int)population.size());
        guard++;
    }
    if (idx == exclude_a || idx == exclude_b) {
        for (int i = 0; i < (int)population.size(); ++i) {
            if (i != exclude_a && i != exclude_b) return i;
        }
    }
    return idx;
}

L_SRTDE_Individual L_SRTDE_Solver::SelectUnionIndividualExcluding(int exclude_a, int exclude_b)
{
    const int union_size = (int)population.size() + (int)archive.size();
    if (union_size <= 0) return population[0];
    for (int guard = 0; guard < 128; ++guard) {
        int pick = RandInt(0, union_size);
        if (pick < (int)population.size()) {
            if (pick == exclude_a || pick == exclude_b) continue;
            return population[pick];
        }
        return archive[pick - (int)population.size()];
    }
    int fallback = SelectPopulationIndexExcluding(exclude_a, exclude_b);
    return population[fallback];
}

void L_SRTDE_Solver::TrimArchive(int max_size)
{
    max_size = std::max(1, max_size);
    while ((int)archive.size() > max_size) {
        int idx = RandInt(0, (int)archive.size());
        archive[idx] = archive.back();
        archive.pop_back();
    }
}

void L_SRTDE_Solver::ApplyPopulationReduction()
{
    if (max_evals == 0 || population.empty()) return;
    const double ratio = std::clamp((double)solver->GetEvalCount() / (double)max_evals, 0.0, 1.0);
    int target = (int)std::llround((double)init_pop_size + (double)(min_pop_size - init_pop_size) * ratio);
    target = std::max(min_pop_size, std::min((int)population.size(), target));
    if (target >= (int)population.size()) {
        TrimArchive((int)population.size());
        return;
    }
    std::sort(population.begin(), population.end(), [](const auto& a, const auto& b) {
        return a.fit < b.fit;
    });
    population.resize(target);
    TrimArchive(target);
}

void L_SRTDE_Solver::Init()
{
    population.clear();
    archive.clear();
    memory_cr.assign(memory_size, 0.5);
    memory_ptr = 0;
    success_rate = 0.5;
    best_fit = std::numeric_limits<double>::infinity();
    best_var.clear();

    population.reserve(init_pop_size);
    int seeded = 0;
    if (solver && solver->pop && solver->pop_fit && solver->Popsize > 0) {
        seeded = std::min(init_pop_size, solver->Popsize);
        for (int i = 0; i < seeded; ++i) {
            L_SRTDE_Individual ind;
            ind.var.assign(solver->pop[i], solver->pop[i] + nvar);
            ind.fit = solver->pop_fit[i];
            UpdateBest(ind.var, ind.fit);
            population.push_back(std::move(ind));
        }
    }

    for (int i = seeded; i < init_pop_size; ++i) {
        if (max_evals > 0 && solver->GetEvalCount() >= max_evals) break;
        L_SRTDE_Individual ind;
        ind.var.resize(nvar, 0.0);
        for (int j = 0; j < nvar; ++j) ind.var[j] = Rand01();
        ind.fit = Evaluate(ind.var);
        UpdateBest(ind.var, ind.fit);
        population.push_back(std::move(ind));
    }
}

void L_SRTDE_Solver::RunGeneration(int /*gen*/)
{
    if (population.empty()) return;
    if (max_evals > 0 && solver->GetEvalCount() >= max_evals) return;

    const int current_n = (int)population.size();
    const double mean_f = 0.5 * (1.0 - std::clamp(success_rate, 0.0, 1.0));
    const double pb = std::max(2.0 / std::max(1, current_n), 0.2 * (1.0 - success_rate) + 2.0 / std::max(1, current_n));
    const int pbest_count = std::max(2, (int)std::ceil(pb * current_n));

    std::vector<double> success_cr;
    std::vector<double> success_delta;
    int success_count = 0;

    for (int i = 0; i < current_n; ++i) {
        if (max_evals > 0 && solver->GetEvalCount() >= max_evals) break;

        const double Fi = SampleF(mean_f);
        const double CRi = SampleCR();
        const int pbest_idx = SelectPBestIndex(pbest_count);
        const int r1_idx = SelectPopulationIndexExcluding(i, pbest_idx);
        const L_SRTDE_Individual xr2 = SelectUnionIndividualExcluding(i, r1_idx);

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
            success_count++;
            success_cr.push_back(CRi);
            success_delta.push_back(std::fabs(population[i].fit - trial_fit));
            AddToArchive(population[i]);
            population[i].var = std::move(trial);
            population[i].fit = trial_fit;
            UpdateBest(population[i].var, population[i].fit);
        }
    }

    success_rate = (double)success_count / (double)std::max(1, current_n);
    if (!success_cr.empty()) {
        double delta_sum = std::accumulate(success_delta.begin(), success_delta.end(), 0.0);
        if (delta_sum <= 1e-12) delta_sum = (double)success_delta.size();
        double mean = 0.0;
        for (int i = 0; i < (int)success_cr.size(); ++i) {
            const double w = success_delta[i] / delta_sum;
            mean += w * success_cr[i];
        }
        memory_cr[memory_ptr] = Clip01(mean);
        memory_ptr = (memory_ptr + 1) % memory_size;
    }

    ApplyPopulationReduction();
}
