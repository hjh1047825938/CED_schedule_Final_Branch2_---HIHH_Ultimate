#ifndef IMOMA_H
#define IMOMA_H

#include <vector>
#include <string>
#include <random>
#include "Multimethod.h"

struct IMOMAIndividual {
    std::vector<double> var;
    double energy = 0.0;
    double makespan = 0.0;
    int rank = 0;
    double crowding_dist = 0.0;
};

struct IMOMAOperatorStats {
    std::string name;
    double weight = 1.0;
    double score = 0.0;
    int usage_count = 0;

    explicit IMOMAOperatorStats(const std::string& n) : name(n) {}
    void Update(bool improved_pf, int archive_size_change, double eta = 0.5);
};

class IMOMA_Solver {
public:
    IMOMA_Solver(MultiMet* solver, int pop_size = 60, double arc_ratio = 0.5, int max_generations = 10000);

    void Init();
    void RunGeneration(int gen);

    double GetBestScalarFit() const { return best_scalar_fit; }
    int GetArchiveSize() const { return (int)archive.size(); }
    const std::vector<IMOMAIndividual>& GetArchive() const { return archive; }
    const std::vector<double>& GetBestVar() const { return scalar_best_ind.var; }
    double GetBestMakespan() const { return scalar_best_ind.makespan; }
    double GetBestEnergy() const { return scalar_best_ind.energy; }
    bool HasBest() const { return has_scalar_best; }

private:
    MultiMet* solver;
    int pop_size;
    double arc_ratio;
    int max_generations;
    int archive_capacity;
    int Nvar;
    int CE_Tnum;
    int M_Jnum;
    int M_OPTnum;
    int ops;

    std::vector<IMOMAIndividual> population;
    std::vector<IMOMAIndividual> archive;
    std::vector<IMOMAOperatorStats> operators;

    std::vector<int> tmp_front_indices;
    std::vector<std::vector<int>> fronts;
    mutable std::vector<int> nd_dom_count_buf;
    mutable std::vector<std::vector<int>> nd_dom_set_buf;
    mutable std::vector<int> nd_cur_front_buf;
    mutable std::vector<int> nd_next_front_buf;
    mutable std::vector<int> crowd_order_buf;
    mutable std::vector<int> select_order_buf;

    std::vector<int> task_order_buf;

    std::mt19937 rng;
    double best_scalar_fit;
    double last_best_scalar_fit;
    int stagnation_count;
    IMOMAIndividual scalar_best_ind;
    bool has_scalar_best;
    int restart_count;
    double suppression_factor;

private:
    double Rand01();
    int RandInt(int lo, int hi_exclusive);

    void Evaluate(IMOMAIndividual& ind);
    void EvaluateObjectives(const std::vector<double>& var, double& makespan, double& energy) const;
    IMOMAIndividual RandomIndividual();
    std::vector<IMOMAIndividual> RandomInitialize(int size);

    double CalculateOmega(int g) const;
    IMOMAIndividual GenerateOpposite(const IMOMAIndividual& x, double omega);
    IMOMAIndividual BiasedCrossover(const IMOMAIndividual& a, const IMOMAIndividual& b, double bias);

    IMOMAIndividual Mutation(const IMOMAIndividual& x);
    IMOMAIndividual DifferentialMutation(const IMOMAIndividual& x);
    IMOMAIndividual BestGuidedMutation(const IMOMAIndividual& x, const IMOMAIndividual& best_ref);
    IMOMAIndividual DiversifyFromBest(double intensity);
    IMOMAIndividual EnergyOptimization(const IMOMAIndividual& x);
    IMOMAIndividual MakespanOptimization(const IMOMAIndividual& x);

    void FastNonDominatedSort(std::vector<IMOMAIndividual>& pop) const;
    bool Dominates(const IMOMAIndividual& a, const IMOMAIndividual& b) const;
    void CalculateCrowdingDistance(std::vector<IMOMAIndividual>& pop, const std::vector<int>& front) const;
    std::vector<IMOMAIndividual> SelectNextGeneration(std::vector<IMOMAIndividual>& combined, int next_size);
    void UpdateArchive(const std::vector<IMOMAIndividual>& candidates);
    std::vector<IMOMAIndividual> SelectByRankFromArchive(int count);
    int SelectOperatorIndex();
    double TriggerProbability(int g, double S) const;
    double ScalarFit(const IMOMAIndividual& ind) const;

    int DecodeCloudIndex(const std::vector<double>& var, int task) const;
    int DecodeEdgeIndex(const std::vector<double>& var, int task) const;
    bool DecodeIsEdge(const std::vector<double>& var, int task) const;
    void SetTaskCloud(std::vector<double>& var, int task, int cloud_idx) const;
    void SetTaskEdge(std::vector<double>& var, int task, int edge_idx) const;
    int PickNearestAllowedEdgeForTask(int task) const;
    int PickTaskLayerCode(const std::vector<double>& var, int task) const;
    void InjectExplorers(std::vector<IMOMAIndividual>& pop, int count, double intensity);
};

#endif
