#ifndef _DSAC_DE_H
#define _DSAC_DE_H

#include "Multimethod.h"
#include <vector>
#include <random>
#include <cmath>
#include <deque>
#include <array>

/**
 * DSAC-DE: Discretized Soft Actor-Critic configured Differential Evolution
 *
 * Based on: "DSAC-Configured Differential Evolution for Cloud-Edge-Device
 * Collaborative Task Scheduling" (IEEE TII, 2024)
 *
 * Key features:
 * - 5 DE mutation operators selected by DSAC policy
 * - Dec-POMDP modeling of evolutionary process
 * - Maximum entropy learning for stable operator selection
 */

// Number of DE mutation operators
constexpr int NUM_DE_OPS = 5;
// State dimension: mean(1) + std(1) + 5 ops * 3 success metrics = 17
constexpr int STATE_DIM = 17;
// Observation dimension: 6 (diff to gbest + 5 random individuals)
constexpr int OBS_DIM = 6;
// Hidden layer size
constexpr int HIDDEN_DIM = 128;

// Experience replay buffer entry
struct DSACTransition {
    std::vector<double> state;      // g(t) || omega(t)
    int action;                      // selected operator index
    double reward;                   // r(t)
    std::vector<double> next_state; // g(t+1) || omega(t+1)
    bool done;                       // episode end flag
};

// Operator success statistics
struct OpStats {
    int success_parent;   // NS_op,1: offspring better than parent
    int success_gbest;    // NS_op,2: offspring beats global best
    int success_avg;      // NS_op,3: offspring better than average
    int total_uses;       // total times this operator was used
    double sum_improvement; // sum of fitness improvements

    OpStats() : success_parent(0), success_gbest(0), success_avg(0),
                total_uses(0), sum_improvement(0.0) {}
    void reset() {
        success_parent = success_gbest = success_avg = total_uses = 0;
        sum_improvement = 0.0;
    }

    // UCB1 score for operator selection
    double ucb_score(int total_selections, double c = 1.414) const {
        if (total_uses == 0) return 1e9; // Unexplored operator
        double avg_reward = sum_improvement / total_uses;
        double exploration = c * std::sqrt(std::log(total_selections + 1) / total_uses);
        return avg_reward + exploration;
    }
};

// Simple MLP for policy network
class PolicyNetwork {
public:
    // Weights and biases
    std::vector<std::vector<double>> w1; // [HIDDEN_DIM x (STATE_DIM + OBS_DIM)]
    std::vector<double> b1;              // [HIDDEN_DIM]
    std::vector<std::vector<double>> w2; // [NUM_DE_OPS x HIDDEN_DIM]
    std::vector<double> b2;              // [NUM_DE_OPS]

    PolicyNetwork();
    void init_weights(std::mt19937& rng);
    std::vector<double> forward(const std::vector<double>& state_obs);
    std::vector<double> get_action_probs(const std::vector<double>& state_obs);
    int sample_action(const std::vector<double>& probs, std::mt19937& rng);
};

// Simple MLP for Q-network (critic)
class QNetwork {
public:
    std::vector<std::vector<double>> w1; // [HIDDEN_DIM x (STATE_DIM + OBS_DIM)]
    std::vector<double> b1;
    std::vector<std::vector<double>> w2; // [NUM_DE_OPS x HIDDEN_DIM]
    std::vector<double> b2;

    QNetwork();
    void init_weights(std::mt19937& rng);
    std::vector<double> forward(const std::vector<double>& state_obs);
    double get_q_value(const std::vector<double>& state_obs, int action);
    void copy_from(const QNetwork& other);
    void soft_update(const QNetwork& other, double tau);
};

class DSAC_DE_Solver {
public:
    DSAC_DE_Solver(MultiMet* solver, int popsize);
    ~DSAC_DE_Solver();

    // Main interface
    void Init();
    void RunGeneration(int gen);
    double GetGlobalBestFit() const { return global_best_fit; }
    double* GetGlobalBest() const { return global_best; }

    // Configuration
    void SetMaxGenerations(int max_gen) { max_generations = max_gen; }
    void SetScalingFactor(double f) { F_base = f; }
    void SetCrossoverRate(double cr) { CR = cr; }
    void SetLearningRate(double lr) { learning_rate = lr; }
    void SetDiscountFactor(double gamma) { discount_factor = gamma; }
    void SetTemperature(double alpha) { temperature = alpha; }
    void SetBufferSize(int size) { buffer_size = size; }
    void SetBatchSize(int size) { batch_size = size; }
    void SetTrainingEnabled(bool enabled) { training_enabled = enabled; }
    void SetK1(double k1) { K1 = k1; }
    void SetK2(double k2) { K2 = k2; }

    // Statistics
    const OpStats* GetOpStats() const { return op_stats; }

private:
    MultiMet* solver;
    int popsize;
    int nvar;
    int max_generations;

    // DE parameters
    double F_base;      // Base scaling factor [0, 2]
    double CR;          // Crossover rate

    // DSAC parameters
    double learning_rate;
    double discount_factor;  // gamma
    double temperature;      // alpha for entropy
    double tau;              // soft update rate
    int buffer_size;
    int batch_size;
    bool training_enabled;

    // Reward parameters (Eq. 17)
    double K1;
    double K2;
    double initial_gbest_fit;

    // Neural networks
    PolicyNetwork policy;
    QNetwork q1, q1_target;
    QNetwork q2, q2_target;

    // Experience replay buffer
    std::deque<DSACTransition> replay_buffer;

    // Operator statistics
    OpStats op_stats[NUM_DE_OPS];

    // Population state
    double* global_best;
    double global_best_fit;
    double pop_mean;
    double pop_std;
    double pop_worst;
    int stagnation_count;      // Generations without improvement
    int total_op_selections;   // Total operator selections for UCB

    // Random number generator
    std::mt19937 rng;

    // Temporary buffers
    std::vector<double> trial;
    std::vector<int> r_indices;

    // Core methods
    void compute_population_stats();
    std::vector<double> get_global_state();
    std::vector<double> get_observation(int ind_idx);
    void apply_mutation_operator(int ind_idx, int op_idx, double F);
    void apply_crossover(int ind_idx);
    double compute_reward(int gen, double prev_gbest);
    void store_transition(const DSACTransition& trans);
    void train_step();
    void update_op_stats(int op_idx, double parent_fit, double offspring_fit);
    int select_operator_ucb();  // UCB-based operator selection
    double adaptive_F(int gen); // Adaptive scaling factor
    double adaptive_CR(int gen); // Adaptive crossover rate

    // DE mutation operators (Eq. 16)
    void mutation_op1(int ind_idx, double F); // gbest + F*(r1-r2)
    void mutation_op2(int ind_idx, double F); // r1 + F*(r2-r3) + F*(r4-r5)
    void mutation_op3(int ind_idx, double F); // gbest + F*(r1-r2) + F*(r3-r4)
    void mutation_op4(int ind_idx, double F); // xi + F*(gbest-xi) + F*(r1-r2)
    void mutation_op5(int ind_idx, double F); // r1 + F*(r2-r3)

    // Helper methods
    void select_random_indices(int exclude, int count);
    double clamp(double val, double lb, double ub);
};

#endif // _DSAC_DE_H
