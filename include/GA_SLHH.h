#ifndef GA_SLHH_H
#define GA_SLHH_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "Multimethod.h"

class GA_SLHH_Solver {
public:
    GA_SLHH_Solver(MultiMet* s, int psize);

    void SetMaxGenerations(int max_gen) { max_generations = (max_gen > 0) ? max_gen : 1; }
    void SetCrossoverRate(double v) { pc = v; }
    void SetMutationRate(double v) { pm = v; }
    void SetGeneMutationRate(double v) { gene_pm = v; }
    void SetImmigrantRate(double v) { immigrant_rate = v; }
    void SetElitismCount(int v) { elitism = v < 0 ? 0 : v; }
    void SetLocalSearchTrials(int v) { ls_trials = v < 0 ? 0 : v; }

    void Init();
    void RunGeneration(int gen);

    double GetBestFit() const { return gbest_fit; }
    const double* GetBest() const { return gbest.data(); }
    double GetLastMeanFit() const { return last_mean_fit; }
    int GetLastUniqueLLH() const { return last_unique_llh; }

private:
    struct ArchiveEntry {
        std::vector<int> llh;
        double fit;
    };

    MultiMet* solver;
    int popsize;
    int CE_Tnum;
    int M_Jnum;
    int M_OPTnum;
    int ops;
    int Nvar;
    int num_rules;
    int max_generations;
    int archive_size;

    double pc;
    double pm;
    double gene_pm;
    double immigrant_rate;
    int elitism;
    int ls_trials;
    int stagnation;
    int stagnation_trigger;

    std::vector<std::vector<double>> pop;
    std::vector<std::vector<double>> newpop;
    std::vector<double> pop_fit;
    std::vector<double> newpop_fit;

    std::vector<std::vector<int>> llh_pop;
    std::vector<std::vector<int>> llh_newpop;

    std::vector<ArchiveEntry> archive;

    std::vector<double> gbest;
    double gbest_fit;
    double last_mean_fit;
    int last_unique_llh;

    double pm_base;
    double gene_pm_base;
    double immigrant_base;

    std::vector<double> uniform_probs;

    // Learning buffers
    std::vector<double> probs_cur;
    std::vector<double> probs_his;
    std::vector<double> probs_mix;

private:
    double randval(double low, double high) const {
        return low + (double)rand() / RAND_MAX * (high - low);
    }

    double clip01(double v) const {
        return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }

    void InitializePopulation();
    void EvaluatePopulation();
    void UpdateArchive();
    void BuildLearningProbabilities(int iter);

    int RouletteSelect(const std::vector<double>& fitness) const;
    void ApplyCrossover();
    void ApplyMutation();

    void DecodeLLH(const std::vector<double>& chrom,
                   const std::vector<double>& probs,
                   std::vector<int>& llh_out) const;
    void DecodeToSolution(const std::vector<int>& llh, std::vector<double>& var_out) const;

    std::vector<double> ComputeProbabilities(const std::vector<std::vector<int>>& seqs) const;

    double EvalLLH(const std::vector<int>& llh, std::vector<double>& var_buf) const;
    void LocalSearch();

    // Helpers for decoding
    int PickMinLoad(const std::vector<int>& loads) const;
    int PickMaxLoad(const std::vector<int>& loads) const;
    int PickRandomIndex(int n) const { return n > 0 ? (rand() % n) : 0; }
    int PickNearestDevice(const std::vector<int>& devices, int ref_dev, double** DtoD) const;
    int PickNearestEdge(const std::vector<int>& edges, const std::vector<int>& task_devices, double** EtoD) const;
    int PickMinLoadEdge(const std::vector<int>& edges, const std::vector<int>& edge_load) const;
    int PickEnergyAwareEdge(const std::vector<int>& edges, const std::vector<int>& edge_load,
                            const std::vector<int>& cloud_load, const double* EnergyList) const;
    double VarForIndex(int count, int idx) const;
};

#endif
