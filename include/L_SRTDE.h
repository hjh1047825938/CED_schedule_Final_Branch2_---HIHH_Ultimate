#ifndef L_SRTDE_H
#define L_SRTDE_H

#include "Multimethod.h"
#include <cstdint>
#include <random>
#include <vector>

struct L_SRTDE_Individual {
    std::vector<double> var;
    double fit = 0.0;
};

class L_SRTDE_Solver {
public:
    L_SRTDE_Solver(MultiMet* solver, int pop_size = 40, uint64_t max_evals = 0);

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

    std::vector<L_SRTDE_Individual> population;
    std::vector<L_SRTDE_Individual> archive;
    std::vector<double> memory_cr;
    int memory_ptr;
    double success_rate;
    std::vector<double> best_var;
    double best_fit;
    std::mt19937 rng;

private:
    double Rand01();
    int RandInt(int lo, int hi_exclusive);
    double Clip01(double value) const;
    double SampleF(double mean_f);
    double SampleCR();
    double Evaluate(const std::vector<double>& var);
    void UpdateBest(const std::vector<double>& var, double fit);
    void AddToArchive(const L_SRTDE_Individual& ind);
    int SelectPBestIndex(int pbest_count);
    int SelectPopulationIndexExcluding(int exclude_a, int exclude_b = -1);
    L_SRTDE_Individual SelectUnionIndividualExcluding(int exclude_a, int exclude_b);
    std::vector<int> SortedIndices() const;
    void ApplyPopulationReduction();
    void TrimArchive(int max_size);
};

#endif
