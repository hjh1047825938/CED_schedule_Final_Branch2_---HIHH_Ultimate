#ifndef NL_SHADE_LBC_H
#define NL_SHADE_LBC_H

#include "Multimethod.h"
#include <cstdint>
#include <random>
#include <vector>

struct NL_SHADE_LBC_Individual {
    std::vector<double> var;
    double fit = 0.0;
};

class NL_SHADE_LBC_Solver {
public:
    NL_SHADE_LBC_Solver(MultiMet* solver, int pop_size = 40, uint64_t max_evals = 0);

    void Init();
    void RunGeneration(int gen);

    void SetMaxEvaluations(uint64_t budget) { max_evals = budget; }
    void SetPopulationBounds(int init_size, int min_size);

    double GetBestFit() const { return best_fit; }
    const std::vector<double>& GetBestVar() const { return best_var; }
    int GetPopulationSize() const { return (int)population.size(); }
    int GetArchiveSize() const { return (int)archive.size(); }
    bool HasBest() const { return !best_var.empty(); }

private:
    MultiMet* solver;
    int init_pop_size;
    int min_pop_size;
    int nvar;
    int memory_size;
    uint64_t max_evals;
    double bias_start;
    double bias_end;
    double p_max;

    std::vector<NL_SHADE_LBC_Individual> population;
    std::vector<NL_SHADE_LBC_Individual> archive;
    std::vector<double> memory_f;
    std::vector<double> memory_cr;
    int memory_ptr;
    std::vector<double> best_var;
    double best_fit;
    std::mt19937 rng;

private:
    double Rand01();
    int RandInt(int lo, int hi_exclusive);
    double Clip01(double value) const;
    double ProgressRatio() const;
    double SampleF();
    double SampleCR();
    double Evaluate(const std::vector<double>& var);
    void UpdateBest(const std::vector<double>& var, double fit);
    void AddToArchive(const NL_SHADE_LBC_Individual& ind);
    void TrimArchive(int max_size);
    std::vector<int> SortedIndices() const;
    int SelectPBestIndex(const std::vector<int>& sorted_indices, int pbest_count);
    int SelectRankBiasedIndex(const std::vector<int>& sorted_indices, int exclude);
    NL_SHADE_LBC_Individual SelectUnionIndividualExcluding(int exclude_a, int exclude_b);
    void ApplyPopulationReduction();
    double WeightedLehmerMean(const std::vector<double>& values, const std::vector<double>& weights) const;
};

#endif
