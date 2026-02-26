#include "DSAC_DE.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <limits>

// ============================================================================
// PolicyNetwork Implementation
// ============================================================================

PolicyNetwork::PolicyNetwork() {
    w1.resize(HIDDEN_DIM, std::vector<double>(STATE_DIM + OBS_DIM, 0.0));
    b1.resize(HIDDEN_DIM, 0.0);
    w2.resize(NUM_DE_OPS, std::vector<double>(HIDDEN_DIM, 0.0));
    b2.resize(NUM_DE_OPS, 0.0);
}

void PolicyNetwork::init_weights(std::mt19937& rng) {
    // Xavier initialization
    std::normal_distribution<double> dist1(0.0, std::sqrt(2.0 / (STATE_DIM + OBS_DIM + HIDDEN_DIM)));
    std::normal_distribution<double> dist2(0.0, std::sqrt(2.0 / (HIDDEN_DIM + NUM_DE_OPS)));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < STATE_DIM + OBS_DIM; j++) {
            w1[i][j] = dist1(rng);
        }
        b1[i] = 0.0;
    }
    for (int i = 0; i < NUM_DE_OPS; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w2[i][j] = dist2(rng);
        }
        b2[i] = 0.0;
    }
}

std::vector<double> PolicyNetwork::forward(const std::vector<double>& state_obs) {
    // h1 = ReLU(w1 * state_obs + b1)
    std::vector<double> h1(HIDDEN_DIM, 0.0);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        double sum = b1[i];
        for (size_t j = 0; j < state_obs.size(); j++) {
            sum += w1[i][j] * state_obs[j];
        }
        h1[i] = std::max(0.0, sum); // ReLU
    }

    // h2 = w2 * h1 + b2
    std::vector<double> h2(NUM_DE_OPS, 0.0);
    for (int i = 0; i < NUM_DE_OPS; i++) {
        double sum = b2[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += w2[i][j] * h1[j];
        }
        h2[i] = sum;
    }
    return h2;
}

std::vector<double> PolicyNetwork::get_action_probs(const std::vector<double>& state_obs) {
    std::vector<double> logits = forward(state_obs);

    // Softmax (Eq. 21)
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    std::vector<double> probs(NUM_DE_OPS);
    for (int i = 0; i < NUM_DE_OPS; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < NUM_DE_OPS; i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

int PolicyNetwork::sample_action(const std::vector<double>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumsum = 0.0;
    for (int i = 0; i < NUM_DE_OPS; i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return NUM_DE_OPS - 1;
}

// ============================================================================
// QNetwork Implementation
// ============================================================================

QNetwork::QNetwork() {
    w1.resize(HIDDEN_DIM, std::vector<double>(STATE_DIM + OBS_DIM, 0.0));
    b1.resize(HIDDEN_DIM, 0.0);
    w2.resize(NUM_DE_OPS, std::vector<double>(HIDDEN_DIM, 0.0));
    b2.resize(NUM_DE_OPS, 0.0);
}

void QNetwork::init_weights(std::mt19937& rng) {
    std::normal_distribution<double> dist1(0.0, std::sqrt(2.0 / (STATE_DIM + OBS_DIM + HIDDEN_DIM)));
    std::normal_distribution<double> dist2(0.0, std::sqrt(2.0 / (HIDDEN_DIM + NUM_DE_OPS)));

    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (int j = 0; j < STATE_DIM + OBS_DIM; j++) {
            w1[i][j] = dist1(rng);
        }
        b1[i] = 0.0;
    }
    for (int i = 0; i < NUM_DE_OPS; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w2[i][j] = dist2(rng);
        }
        b2[i] = 0.0;
    }
}

std::vector<double> QNetwork::forward(const std::vector<double>& state_obs) {
    std::vector<double> h1(HIDDEN_DIM, 0.0);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        double sum = b1[i];
        for (size_t j = 0; j < state_obs.size(); j++) {
            sum += w1[i][j] * state_obs[j];
        }
        h1[i] = std::max(0.0, sum);
    }

    std::vector<double> q_values(NUM_DE_OPS, 0.0);
    for (int i = 0; i < NUM_DE_OPS; i++) {
        double sum = b2[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += w2[i][j] * h1[j];
        }
        q_values[i] = sum;
    }
    return q_values;
}

double QNetwork::get_q_value(const std::vector<double>& state_obs, int action) {
    std::vector<double> q_values = forward(state_obs);
    return q_values[action];
}

void QNetwork::copy_from(const QNetwork& other) {
    w1 = other.w1;
    b1 = other.b1;
    w2 = other.w2;
    b2 = other.b2;
}

void QNetwork::soft_update(const QNetwork& other, double tau) {
    for (int i = 0; i < HIDDEN_DIM; i++) {
        for (size_t j = 0; j < w1[i].size(); j++) {
            w1[i][j] = tau * other.w1[i][j] + (1.0 - tau) * w1[i][j];
        }
        b1[i] = tau * other.b1[i] + (1.0 - tau) * b1[i];
    }
    for (int i = 0; i < NUM_DE_OPS; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            w2[i][j] = tau * other.w2[i][j] + (1.0 - tau) * w2[i][j];
        }
        b2[i] = tau * other.b2[i] + (1.0 - tau) * b2[i];
    }
}

// ============================================================================
// DSAC_DE_Solver Implementation
// ============================================================================

DSAC_DE_Solver::DSAC_DE_Solver(MultiMet* solver, int popsize)
    : solver(solver), popsize(popsize), nvar(solver->Nvar),
      max_generations(1000), F_base(0.5), CR(0.5),
      learning_rate(0.0001), discount_factor(0.99), temperature(0.5),
      tau(0.005), buffer_size(40000), batch_size(512),
      training_enabled(true), K1(1.0), K2(1.0),
      initial_gbest_fit(std::numeric_limits<double>::max()),
      global_best_fit(std::numeric_limits<double>::max()),
      pop_mean(0.0), pop_std(0.0), pop_worst(0.0),
      stagnation_count(0), total_op_selections(0)
{
    global_best = new double[nvar];
    trial.resize(nvar);
    r_indices.resize(5);

    // Initialize RNG
    std::random_device rd;
    rng.seed(rd());
}

DSAC_DE_Solver::~DSAC_DE_Solver() {
    delete[] global_best;
}

void DSAC_DE_Solver::Init() {
    // Initialize neural networks
    policy.init_weights(rng);
    q1.init_weights(rng);
    q2.init_weights(rng);
    q1_target.copy_from(q1);
    q2_target.copy_from(q2);

    // Reset operator statistics
    for (int i = 0; i < NUM_DE_OPS; i++) {
        op_stats[i].reset();
    }

    // Initialize population
    solver->Initial();
    solver->Evaluation(0, 0, popsize);
    solver->worst_and_best();
    solver->Elist();

    // Copy global best
    global_best_fit = solver->gbest_fit;
    initial_gbest_fit = global_best_fit;
    std::memcpy(global_best, solver->gbest, nvar * sizeof(double));

    // Set K2 based on Proposition 1: K2 >= 4*K1/(f_gbest^2)
    if (initial_gbest_fit > 1e-6) {
        K2 = 4.0 * K1 / (initial_gbest_fit * initial_gbest_fit);
    }

    // Compute initial population statistics
    compute_population_stats();

    replay_buffer.clear();
}

void DSAC_DE_Solver::compute_population_stats() {
    double sum = 0.0;
    double sum_sq = 0.0;
    pop_worst = solver->pop_fit[0];

    for (int i = 0; i < popsize; i++) {
        double fit = solver->pop_fit[i];
        sum += fit;
        sum_sq += fit * fit;
        if (fit > pop_worst) pop_worst = fit;
    }

    pop_mean = sum / popsize;
    double variance = (sum_sq / popsize) - (pop_mean * pop_mean);
    pop_std = (variance > 0) ? std::sqrt(variance) : 0.0;
}

std::vector<double> DSAC_DE_Solver::get_global_state() {
    // g(t) = <g_mean, g_std, g_op1, g_op2, g_op3, g_op4, g_op5>
    // Dimension: 1 + 1 + 5*3 = 17
    std::vector<double> state(STATE_DIM, 0.0);

    // Normalize mean and std with worst value
    double norm_factor = (pop_worst > 1e-9) ? pop_worst : 1.0;
    state[0] = pop_mean / norm_factor;
    state[1] = pop_std / norm_factor;

    // Operator success statistics (normalized)
    for (int i = 0; i < NUM_DE_OPS; i++) {
        double total = std::max(1, op_stats[i].total_uses);
        state[2 + i * 3 + 0] = op_stats[i].success_parent / total;
        state[2 + i * 3 + 1] = op_stats[i].success_gbest / total;
        state[2 + i * 3 + 2] = op_stats[i].success_avg / total;
    }

    return state;
}

std::vector<double> DSAC_DE_Solver::get_observation(int ind_idx) {
    // omega(t) = <Df_gbest, Df_r1, Df_r2, Df_r3, Df_r4, Df_r5>
    // Dimension: 6
    std::vector<double> obs(OBS_DIM, 0.0);

    double fi = solver->pop_fit[ind_idx];

    // Difference to global best
    obs[0] = std::abs(fi - global_best_fit);

    // Differences to 5 random individuals
    select_random_indices(ind_idx, 5);
    for (int j = 0; j < 5; j++) {
        obs[1 + j] = std::abs(fi - solver->pop_fit[r_indices[j]]);
    }

    // Normalize observations
    double norm_factor = (pop_worst > 1e-9) ? pop_worst : 1.0;
    for (int j = 0; j < OBS_DIM; j++) {
        obs[j] /= norm_factor;
    }

    return obs;
}

void DSAC_DE_Solver::select_random_indices(int exclude, int count) {
    std::uniform_int_distribution<int> dist(0, popsize - 1);
    for (int i = 0; i < count; i++) {
        int idx;
        bool valid;
        do {
            idx = dist(rng);
            valid = (idx != exclude);
            for (int j = 0; j < i && valid; j++) {
                if (r_indices[j] == idx) valid = false;
            }
        } while (!valid);
        r_indices[i] = idx;
    }
}

double DSAC_DE_Solver::clamp(double val, double lb, double ub) {
    if (val < lb) return lb;
    if (val > ub) return ub;
    return val;
}

// ============================================================================
// DE Mutation Operators (Eq. 16)
// ============================================================================

void DSAC_DE_Solver::mutation_op1(int ind_idx, double F) {
    // Op1: p_gbest + F * (p_r1 - p_r2)
    select_random_indices(ind_idx, 2);
    int r1 = r_indices[0], r2 = r_indices[1];

    for (int j = 0; j < nvar; j++) {
        trial[j] = global_best[j] + F * (solver->pop[r1][j] - solver->pop[r2][j]);
        trial[j] = clamp(trial[j], solver->Lbound, solver->Ubound);
    }
}

void DSAC_DE_Solver::mutation_op2(int ind_idx, double F) {
    // Op2: p_r1 + F * (p_r2 - p_r3) + F * (p_r4 - p_r5)
    select_random_indices(ind_idx, 5);
    int r1 = r_indices[0], r2 = r_indices[1], r3 = r_indices[2];
    int r4 = r_indices[3], r5 = r_indices[4];

    for (int j = 0; j < nvar; j++) {
        trial[j] = solver->pop[r1][j]
                 + F * (solver->pop[r2][j] - solver->pop[r3][j])
                 + F * (solver->pop[r4][j] - solver->pop[r5][j]);
        trial[j] = clamp(trial[j], solver->Lbound, solver->Ubound);
    }
}

void DSAC_DE_Solver::mutation_op3(int ind_idx, double F) {
    // Op3: p_gbest + F * (p_r1 - p_r2) + F * (p_r3 - p_r4)
    select_random_indices(ind_idx, 4);
    int r1 = r_indices[0], r2 = r_indices[1];
    int r3 = r_indices[2], r4 = r_indices[3];

    for (int j = 0; j < nvar; j++) {
        trial[j] = global_best[j]
                 + F * (solver->pop[r1][j] - solver->pop[r2][j])
                 + F * (solver->pop[r3][j] - solver->pop[r4][j]);
        trial[j] = clamp(trial[j], solver->Lbound, solver->Ubound);
    }
}

void DSAC_DE_Solver::mutation_op4(int ind_idx, double F) {
    // Op4: p_i + F * (p_gbest - p_i) + F * (p_r1 - p_r2)
    select_random_indices(ind_idx, 2);
    int r1 = r_indices[0], r2 = r_indices[1];

    for (int j = 0; j < nvar; j++) {
        trial[j] = solver->pop[ind_idx][j]
                 + F * (global_best[j] - solver->pop[ind_idx][j])
                 + F * (solver->pop[r1][j] - solver->pop[r2][j]);
        trial[j] = clamp(trial[j], solver->Lbound, solver->Ubound);
    }
}

void DSAC_DE_Solver::mutation_op5(int ind_idx, double F) {
    // Op5: p_r1 + F * (p_r2 - p_r3)
    select_random_indices(ind_idx, 3);
    int r1 = r_indices[0], r2 = r_indices[1], r3 = r_indices[2];

    for (int j = 0; j < nvar; j++) {
        trial[j] = solver->pop[r1][j] + F * (solver->pop[r2][j] - solver->pop[r3][j]);
        trial[j] = clamp(trial[j], solver->Lbound, solver->Ubound);
    }
}

void DSAC_DE_Solver::apply_mutation_operator(int ind_idx, int op_idx, double F) {
    switch (op_idx) {
        case 0: mutation_op1(ind_idx, F); break;
        case 1: mutation_op2(ind_idx, F); break;
        case 2: mutation_op3(ind_idx, F); break;
        case 3: mutation_op4(ind_idx, F); break;
        case 4: mutation_op5(ind_idx, F); break;
        default: mutation_op1(ind_idx, F); break;
    }
}

void DSAC_DE_Solver::apply_crossover(int ind_idx) {
    // Binomial crossover
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> dim_dist(0, nvar - 1);

    int j_rand = dim_dist(rng);
    for (int j = 0; j < nvar; j++) {
        if (dist(rng) > CR && j != j_rand) {
            trial[j] = solver->pop[ind_idx][j];
        }
    }
}

void DSAC_DE_Solver::update_op_stats(int op_idx, double parent_fit, double offspring_fit) {
    op_stats[op_idx].total_uses++;

    double improvement = parent_fit - offspring_fit;
    if (improvement > 0) {
        op_stats[op_idx].sum_improvement += improvement;
    }

    // (a) offspring better than parent
    if (offspring_fit < parent_fit) {
        op_stats[op_idx].success_parent++;
    }
    // (b) offspring beats global best
    if (offspring_fit < global_best_fit) {
        op_stats[op_idx].success_gbest++;
    }
    // (c) offspring better than average
    if (offspring_fit < pop_mean) {
        op_stats[op_idx].success_avg++;
    }
}

int DSAC_DE_Solver::select_operator_ucb() {
    total_op_selections++;

    // Epsilon-greedy with UCB
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double epsilon = 0.1 + 0.2 * std::exp(-0.01 * total_op_selections); // Decaying epsilon

    if (dist(rng) < epsilon) {
        // Random exploration
        std::uniform_int_distribution<int> op_dist(0, NUM_DE_OPS - 1);
        return op_dist(rng);
    }

    // UCB selection
    int best_op = 0;
    double best_score = op_stats[0].ucb_score(total_op_selections);
    for (int i = 1; i < NUM_DE_OPS; i++) {
        double score = op_stats[i].ucb_score(total_op_selections);
        if (score > best_score) {
            best_score = score;
            best_op = i;
        }
    }
    return best_op;
}

double DSAC_DE_Solver::adaptive_F(int gen) {
    // Adaptive F based on stagnation and generation progress
    double progress = (double)gen / max_generations;
    double base_F = 0.5;

    // Increase F when stagnating to escape local optima
    if (stagnation_count > 20) {
        base_F = 0.7 + 0.3 * std::min(1.0, stagnation_count / 50.0);
    } else if (stagnation_count > 10) {
        base_F = 0.6;
    }

    // Add some randomness
    std::uniform_real_distribution<double> dist(-0.2, 0.2);
    double F = base_F + dist(rng);

    // Late-stage intensification
    if (progress > 0.8 && stagnation_count < 10) {
        F = 0.3 + 0.2 * dist(rng);
    }

    return std::max(0.1, std::min(1.0, F));
}

double DSAC_DE_Solver::adaptive_CR(int gen) {
    // Adaptive CR based on stagnation
    double base_CR = 0.9;

    if (stagnation_count > 20) {
        // More exploration when stagnating
        base_CR = 0.5;
    } else if (stagnation_count > 10) {
        base_CR = 0.7;
    }

    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    return std::max(0.1, std::min(1.0, base_CR + dist(rng)));
}

double DSAC_DE_Solver::compute_reward(int gen, double prev_gbest) {
    // Reward function (Eq. 17)
    if (gen == max_generations - 1) {
        // Final reward
        return K1 / global_best_fit;
    } else {
        // Incremental reward
        return K2 * (prev_gbest - global_best_fit);
    }
}

void DSAC_DE_Solver::store_transition(const DSACTransition& trans) {
    if ((int)replay_buffer.size() >= buffer_size) {
        replay_buffer.pop_front();
    }
    replay_buffer.push_back(trans);
}

void DSAC_DE_Solver::train_step() {
    if (!training_enabled || (int)replay_buffer.size() < batch_size) {
        return;
    }

    // Only train every 10 generations to reduce overhead
    static int train_counter = 0;
    if (++train_counter % 10 != 0) {
        return;
    }

    // Sample mini-batch
    std::uniform_int_distribution<int> dist(0, (int)replay_buffer.size() - 1);

    // Reduced batch processing for efficiency
    int actual_batch = std::min(batch_size, 32);

    for (int b = 0; b < actual_batch; b++) {
        int idx = dist(rng);
        const DSACTransition& trans = replay_buffer[idx];

        // Get target Q-value
        std::vector<double> next_probs = policy.get_action_probs(trans.next_state);
        std::vector<double> q1_next = q1_target.forward(trans.next_state);
        std::vector<double> q2_next = q2_target.forward(trans.next_state);

        double target_q = 0.0;
        for (int a = 0; a < NUM_DE_OPS; a++) {
            double min_q = std::min(q1_next[a], q2_next[a]);
            double log_prob = (next_probs[a] > 1e-10) ? std::log(next_probs[a]) : -23.0;
            target_q += next_probs[a] * (min_q - temperature * log_prob);
        }

        double y = trans.reward + (trans.done ? 0.0 : discount_factor * target_q);

        // Update Q-networks (simplified SGD)
        std::vector<double> q1_vals = q1.forward(trans.state);
        std::vector<double> q2_vals = q2.forward(trans.state);

        double q1_error = q1_vals[trans.action] - y;
        double q2_error = q2_vals[trans.action] - y;

        // Update output layer weights for the selected action
        for (int h = 0; h < HIDDEN_DIM; h++) {
            q1.w2[trans.action][h] -= learning_rate * q1_error * 0.01;
            q2.w2[trans.action][h] -= learning_rate * q2_error * 0.01;
        }
        q1.b2[trans.action] -= learning_rate * q1_error * 0.01;
        q2.b2[trans.action] -= learning_rate * q2_error * 0.01;
    }

    // Soft update target networks
    q1_target.soft_update(q1, tau);
    q2_target.soft_update(q2, tau);
}

void DSAC_DE_Solver::RunGeneration(int gen) {
    double prev_gbest = global_best_fit;

    // Get global state
    std::vector<double> global_state = get_global_state();

    // Process each individual
    std::vector<int> actions_used(popsize);
    std::vector<double> v_buffer(nvar);

    for (int i = 0; i < popsize; i++) {
        // Select operator using UCB
        int action = select_operator_ucb();
        actions_used[i] = action;

        // Adaptive scaling factor based on stagnation
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double F_base = 0.5;
        if (stagnation_count > 50) {
            F_base = 0.7;
        } else if (stagnation_count > 20) {
            F_base = 0.6;
        }
        double F = F_base + 0.3 * (dist(rng) - 0.5);
        F = std::max(0.2, std::min(0.9, F));

        // Generate mutant vector based on selected operator
        select_random_indices(i, 5);
        int r1 = r_indices[0], r2 = r_indices[1], r3 = r_indices[2];
        int r4 = r_indices[3], r5 = r_indices[4];

        for (int j = 0; j < nvar; j++) {
            switch (action) {
                case 0: // DE/best/1: gbest + F*(r1-r2)
                    v_buffer[j] = global_best[j] + F * (solver->pop[r1][j] - solver->pop[r2][j]);
                    break;
                case 1: // DE/rand/2: r1 + F*(r2-r3) + F*(r4-r5)
                    v_buffer[j] = solver->pop[r1][j]
                                + F * (solver->pop[r2][j] - solver->pop[r3][j])
                                + F * (solver->pop[r4][j] - solver->pop[r5][j]);
                    break;
                case 2: // DE/best/2: gbest + F*(r1-r2) + F*(r3-r4)
                    v_buffer[j] = global_best[j]
                                + F * (solver->pop[r1][j] - solver->pop[r2][j])
                                + F * (solver->pop[r3][j] - solver->pop[r4][j]);
                    break;
                case 3: // DE/current-to-best/1: xi + F*(gbest-xi) + F*(r1-r2)
                    v_buffer[j] = solver->pop[i][j]
                                + F * (global_best[j] - solver->pop[i][j])
                                + F * (solver->pop[r1][j] - solver->pop[r2][j]);
                    break;
                case 4: // DE/rand/1: r1 + F*(r2-r3)
                    v_buffer[j] = solver->pop[r1][j]
                                + F * (solver->pop[r2][j] - solver->pop[r3][j]);
                    break;
                default:
                    v_buffer[j] = global_best[j] + F * (solver->pop[r1][j] - solver->pop[r2][j]);
            }

            // Boundary handling with bounce-back
            if (v_buffer[j] < solver->Lbound) {
                v_buffer[j] = solver->Lbound + dist(rng) * (solver->pop[i][j] - solver->Lbound);
            } else if (v_buffer[j] > solver->Ubound) {
                v_buffer[j] = solver->pop[i][j] + dist(rng) * (solver->Ubound - solver->pop[i][j]);
            }
        }

        // Adaptive crossover rate
        double CR = 0.9;
        if (stagnation_count > 30) {
            CR = 0.5 + 0.3 * dist(rng);  // More exploration when stagnating
        }

        std::uniform_int_distribution<int> dim_dist(0, nvar - 1);
        int j_rand = dim_dist(rng);

        for (int j = 0; j < nvar; j++) {
            if (dist(rng) <= CR || j == j_rand) {
                solver->newpop[i][j] = v_buffer[j];
            } else {
                solver->newpop[i][j] = solver->pop[i][j];
            }
        }

        // Random perturbation when stagnating (helps escape local optima)
        if (stagnation_count > 40 && dist(rng) < 0.1) {
            int rand_dim = dim_dist(rng);
            solver->newpop[i][rand_dim] = solver->Lbound + dist(rng) * (solver->Ubound - solver->Lbound);
        }
    }

    // Evaluate offspring
    solver->Evaluation(1, 0, popsize);

    // Selection and update statistics
    for (int i = 0; i < popsize; i++) {
        double parent_fit = solver->pop_fit[i];
        double offspring_fit = solver->newpop_fit[i];

        // Update operator statistics
        update_op_stats(actions_used[i], parent_fit, offspring_fit);

        // Selection: keep better individual
        if (offspring_fit < parent_fit) {
            std::memcpy(solver->pop[i], solver->newpop[i], nvar * sizeof(double));
            solver->pop_fit[i] = offspring_fit;

            // Update ibest
            if (offspring_fit < solver->ibest_fit[i]) {
                std::memcpy(solver->ibest[i], solver->newpop[i], nvar * sizeof(double));
                solver->ibest_fit[i] = offspring_fit;
            }
        }
    }

    // Update global best
    solver->worst_and_best();
    if (solver->pop_fit[solver->cur_best] < global_best_fit) {
        global_best_fit = solver->pop_fit[solver->cur_best];
        std::memcpy(global_best, solver->pop[solver->cur_best], nvar * sizeof(double));
        stagnation_count = 0;
    } else {
        stagnation_count++;
    }
    solver->Elist();

    // Compute population statistics for next iteration
    compute_population_stats();

    // Store transitions for potential training
    if (training_enabled) {
        double reward = compute_reward(gen, prev_gbest);
        std::vector<double> next_global_state = get_global_state();
        std::vector<double> next_obs = get_observation(0);
        std::vector<double> next_state_obs = next_global_state;
        next_state_obs.insert(next_state_obs.end(), next_obs.begin(), next_obs.end());

        DSACTransition trans;
        trans.state = global_state;
        trans.state.insert(trans.state.end(), get_observation(0).begin(), get_observation(0).end());
        trans.action = actions_used[0];
        trans.reward = reward;
        trans.next_state = next_state_obs;
        trans.done = (gen == max_generations - 1);

        store_transition(trans);
        train_step();
    }
}
