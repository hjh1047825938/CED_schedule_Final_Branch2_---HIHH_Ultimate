#include "QPHH.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
    inline int clamp_int(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    inline double bucket_value(int idx, int count) {
        if (count <= 1) return 0.0;
        if (idx >= count - 1) return 1.0;
        return (idx + 0.5) / (count - 1);
    }

    inline int decode_idx(double v, int count) {
        if (count <= 1) return 0;
        int idx = (int)(v * (count - 1));
        return clamp_int(idx, 0, count - 1);
    }

    inline std::mt19937& tls_rng() {
        static thread_local std::mt19937 eng(std::random_device{}());
        return eng;
    }

}

QPHH_Solver::QPHH_Solver(MultiMet* s, int popsize, int n_tasks, double eps, double g)
    : solver(s),
      P(popsize),
      init_pool_factor(10),
      P0(popsize * 10),
      N(n_tasks),
      max_iterations(500),
      CE_Tnum(s->CE_Tnum),
      M_Jnum(s->M_Jnum),
      M_OPTnum(s->M_OPTnum),
      ops(s->M_Jnum * s->M_OPTnum),
      Nvar(s->Nvar),
      Q(10, std::vector<double>(6, 1.0)),
      prev_state(1),
      prev_action(0),
      epsilon(eps),
      gamma(g),
      p_c(1.0),
      p_greedy(1.0),
      greedy_decay(0.95),
      gi_cap(30),
      map_cap(40),
      num_threads(1),
      gbest_fit(1e30),
      prev_best_fit(1e30),
      current_iter(0),
      no_improve_count(0),
      last_best(1e30),
      log_llh(false),
      log_q(false),
      log_every(50)
{
    config.gi_cap = gi_cap;
    config.map_cap = map_cap;
    config.epsilon_init = eps;
    config.gamma = g;
    config.greedy_decay = greedy_decay;
    config.num_threads = num_threads;
}

void QPHH_Solver::SetConfig(const QPHHConfig& cfg)
{
    config = cfg;
    gi_cap = cfg.gi_cap > 0 ? cfg.gi_cap : 0;
    map_cap = cfg.map_cap > 0 ? cfg.map_cap : 0;
    epsilon = cfg.epsilon_init;
    gamma = cfg.gamma;
    p_c = cfg.p_c_init;
    greedy_decay = cfg.greedy_decay;
    num_threads = cfg.num_threads > 0 ? cfg.num_threads : 1;
}

double QPHH_Solver::randval(double low, double high) const
{
    if (num_threads <= 1) {
        return low + (double)rand() / RAND_MAX * (high - low);
    }
    std::uniform_real_distribution<double> dist(low, high);
    return dist(tls_rng());
}

int QPHH_Solver::randint(int low, int high_exclusive) const
{
    if (high_exclusive <= low) return low;
    if (num_threads <= 1) {
        return low + (rand() % (high_exclusive - low));
    }
    std::uniform_int_distribution<int> dist(low, high_exclusive - 1);
    return dist(tls_rng());
}

void QPHH_Solver::Shuffle(std::vector<int>& data) const
{
    if (num_threads <= 1) {
        for (int i = (int)data.size() - 1; i > 0; i--) {
            int r = randint(0, i + 1);
            std::swap(data[i], data[r]);
        }
        return;
    }
    std::shuffle(data.begin(), data.end(), tls_rng());
}

double QPHH_Solver::EvalVarSafe(const double* var) const
{
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    Workspace& ws = solver->ws_pool.get(tid);
    solver->IncrementEvalCount();
    return solver->EvalWithWorkspace(var, ws);
}

double QPHH_Solver::ComputeEpsilon(int iter) const
{
    const double ratio = (max_iterations > 0) ? (iter / (max_iterations * 0.3)) : 0.0;
    return config.epsilon_init * std::exp(-ratio);
}

void QPHH_Solver::Init()
{
    P0 = P * init_pool_factor;
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#endif
    std::cout << "[QPHH] Init start: P=" << P << " P0=" << P0
              << " Nvar=" << Nvar << " ops=" << ops << std::endl;
    std::cout.flush();
    pop.clear();
    pop.reserve(P);
    gbest.assign(Nvar, 0.0);
    gbest_fit = 1e30;
    prev_best_fit = 1e30;
    p_c = config.p_c_init;
    p_greedy = 1.0;
    prev_state = 1;
    prev_action = 0;
    Q.assign(10, std::vector<double>(6, 1.0));
    tmp_var.assign(Nvar, 0.0);
    trial_var.assign(Nvar, 0.0);
    reusable_order.clear();
    reusable_mapping.clear();
    reusable_order.reserve(ops);
    reusable_mapping.reserve(ops);

    std::vector<Individual> pool(P0);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < P0; i++) {
        Individual ind_local;
        ind_local.var.assign(Nvar, 0.0);

        // Start from random base to keep segments in [0,1]
        for (int j = 0; j < Nvar; j++) {
            ind_local.var[j] = randval(0.0, 1.0);
        }

        std::vector<int> order;
        BuildRandomFeasibleOrder(order);
        // Set TEO segment before mapping evaluation
        EncodeOrderAndMapping(ind_local.var, order, std::vector<int>());
        BuildMappingGreedyETRM(ind_local.var, order);

        ind_local.fit = EvalVarSafe(ind_local.var.data());
        pool[i] = std::move(ind_local);

#ifdef _OPENMP
        #pragma omp critical
#endif
        if ((i + 1) % 50 == 0) {
            std::cout << "[QPHH] Init pool " << (i + 1) << "/" << P0 << std::endl;
            std::cout.flush();
        }
    }

    std::sort(pool.begin(), pool.end(), [](const Individual& a, const Individual& b) {
        return a.fit < b.fit;
    });

    for (int i = 0; i < P; i++) {
        pop.push_back(pool[i]);
    }

    gbest = pop[0].var;
    gbest_fit = pop[0].fit;
    for (int i = 1; i < P; i++) {
        if (pop[i].fit < gbest_fit) {
            gbest_fit = pop[i].fit;
            gbest = pop[i].var;
        }
    }
    prev_best_fit = gbest_fit;
    current_iter = 0;
    no_improve_count = 0;
    last_best = 1e30;
    std::cout << "[QPHH] Init done. gbest=" << gbest_fit << std::endl;
    std::cout.flush();
}

void QPHH_Solver::RunIteration(int iter)
{
    if ((int)thread_scratch.size() < std::max(1, num_threads)) {
        thread_scratch.resize(std::max(1, num_threads));
    }
    for (auto& sc : thread_scratch) {
        if ((int)sc.dev_idx_by_op.size() != ops) sc.dev_idx_by_op.assign(ops, 0);
        if ((int)sc.local_tmp_var.size() != Nvar) sc.local_tmp_var.assign(Nvar, 0.0);
    }

    current_iter = iter;
    double alpha = ComputeAlpha(iter);
    epsilon = ComputeEpsilon(iter);
    int action = SelectAction(prev_state);
    double iter_ratio = (max_iterations > 0) ? (iter / (double)max_iterations) : 0.0;
    if (iter_ratio < 0.0) iter_ratio = 0.0;
    if (iter_ratio > 1.0) iter_ratio = 1.0;

    if (log_llh && ((iter + 1) % log_every == 0)) {
        static const char* names[6] = {
            "3-swap", "intra-level swap", "3-insert",
            "parent-aware insert", "greedy insert", "pair insert"
        };
        std::cout << "[QPHH] iter=" << (iter + 1)
                  << " select LLH=" << action
                  << " (" << names[action] << ")" << std::endl;
    }

    // Speed/quality balance: update mostly inferior individuals, keep elites stable.
    reusable_rank_idx.resize(P);
    std::iota(reusable_rank_idx.begin(), reusable_rank_idx.end(), 0);
    std::sort(reusable_rank_idx.begin(), reusable_rank_idx.end(), [&](int a, int b) {
        return pop[a].fit > pop[b].fit;
    });
    int apply_count = std::max(2, (int)(P * (0.45 + 0.2 * (1.0 - iter_ratio))));
    if (apply_count > P) apply_count = P;
    reusable_apply_ids.clear();
    reusable_apply_ids.reserve(apply_count);
    for (int k = 0; k < apply_count; k++) {
        reusable_apply_ids.push_back(reusable_rank_idx[k]);
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < (int)reusable_apply_ids.size(); n++) {
        int i = reusable_apply_ids[n];
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        ThreadScratch& sc = thread_scratch[tid];
        std::vector<int>& order = sc.order;
        std::vector<int>& dev_idx_by_op = sc.dev_idx_by_op;
        std::vector<double>& local_tmp_var = sc.local_tmp_var;

        DecodeOrder(pop[i].var, order);
        DecodeDevIdxByOp(pop[i].var, order, dev_idx_by_op);

        if (action == 0) {
            LLH_3Swap(order);
        } else if (action == 1) {
            LLH_IntraLevelSwap(order);
        } else if (action == 2) {
            LLH_3Insert(order);
        } else if (action == 3) {
            LLH_ParentAwareInsert(order, dev_idx_by_op, local_tmp_var);
        } else if (action == 4) {
            LLH_GreedyInsert(order, dev_idx_by_op, pop[i].var, local_tmp_var, true);
        } else {
            LLH_PairInsert(order, dev_idx_by_op, local_tmp_var);
        }

        EncodeOrderAndMapping(pop[i].var, order, dev_idx_by_op);
        pop[i].fit = EvalVarSafe(pop[i].var.data());
    }

    // Update global best after LLH
    for (int i = 0; i < P; i++) {
        if (pop[i].fit < gbest_fit) {
            gbest_fit = pop[i].fit;
            gbest = pop[i].var;
        }
    }

    // Adaptive local search on P/2 inferior individuals
    reusable_idx.resize(P);
    std::iota(reusable_idx.begin(), reusable_idx.end(), 0);
    std::sort(reusable_idx.begin(), reusable_idx.end(), [&](int a, int b) {
        return pop[a].fit > pop[b].fit;
    });

    if ((iter % 3) == 0) {
        int num_improve = std::max(1, (int)(P * (0.11 + 0.09 * (1.0 - iter_ratio))));
        if (num_improve > P) num_improve = P;
        for (int k = 0; k < num_improve; k++) {
            int i = reusable_idx[k];
            double r = randval(0.0, 1.0);
            if (r < p_c) {
                TwoPointCrossover(pop[i]);
            } else {
                NTasksGreedyInsert(pop[i], tmp_var);
            }
        }
    }

    // Update best after local search
    for (int i = 0; i < P; i++) {
        if (pop[i].fit < gbest_fit) {
            gbest_fit = pop[i].fit;
            gbest = pop[i].var;
        }
    }

    // Light elite intensification to improve convergence quality.
    if ((iter % 200) == 0) {
        int elite_count = std::max(1, P / 10);
        if (elite_count > 2) elite_count = 2;
        for (int e = 0; e < elite_count; e++) {
            int i = reusable_idx[P - 1 - e];
            NTasksGreedyInsert(pop[i], tmp_var);
            if (pop[i].fit < gbest_fit) {
                gbest_fit = pop[i].fit;
                gbest = pop[i].var;
            }
        }

        // Very low-cost refinement around current global best.
        for (int r = 0; r < 3; r++) {
            trial_var = gbest;
            int changes = std::max(4, Nvar / 80);
            for (int c = 0; c < changes; c++) {
                int pos = randint(0, Nvar);
                double v = trial_var[pos] + randval(-0.08, 0.08);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                trial_var[pos] = v;
            }
            double fit = EvalVarSafe(trial_var.data());
            if (fit < gbest_fit) {
                gbest_fit = fit;
                gbest = trial_var;
            }
        }
    }

    // Q-learning update
    double delta = 0.0;
    if (prev_best_fit > 0.0 && gbest_fit > 0.0) {
        delta = std::log(gbest_fit) - std::log(prev_best_fit);
    }
    int cur_state = ComputeState(delta);
    double reward = ComputeReward(delta);

    double max_q = Q[cur_state][0];
    for (int a = 1; a < 6; a++) {
        if (Q[cur_state][a] > max_q) max_q = Q[cur_state][a];
    }
    Q[prev_state][action] = Q[prev_state][action] +
        alpha * (reward + gamma * max_q - Q[prev_state][action]);

    if (log_q && ((iter + 1) % log_every == 0)) {
        std::cout << "[QPHH] iter=" << (iter + 1)
                  << " state=" << prev_state
                  << " action=" << action
                  << " reward=" << reward
                  << " next_state=" << cur_state
                  << " best=" << gbest_fit << std::endl;
    }

    prev_state = cur_state;
    prev_action = action;
    prev_best_fit = gbest_fit;

    // Improved p_c decay strategy
    p_c = std::max(0.1, p_c * 0.998);
    if (iter % 500 == 0) p_c = 0.8;

    // Stagnation detection
    if (std::fabs(gbest_fit - last_best) < config.min_improvement) {
        no_improve_count++;
    } else {
        no_improve_count = 0;
        last_best = gbest_fit;
    }

    // Population restart mechanism
    if (no_improve_count >= config.patience) {
        std::cout << "[QPHH] Stagnation detected at iter " << iter
                  << ", restarting 25% population..." << std::endl;
        std::cout.flush();

        int restart_count = P / 4;
        if (restart_count < 1) restart_count = 1;

        for (int k = 0; k < restart_count; k++) {
            int i = reusable_idx[k];
            // Half guided restart around gbest, half full random.
            const bool guided = (k < restart_count / 2);
            for (int j = 0; j < Nvar; j++) {
                if (guided) {
                    double v = gbest[j] + randval(-0.12, 0.12);
                    if (v < 0.0) v = 0.0;
                    if (v > 1.0) v = 1.0;
                    pop[i].var[j] = v;
                } else {
                    pop[i].var[j] = randval(0.0, 1.0);
                }
            }

            std::vector<int> order;
            BuildRandomFeasibleOrder(order);
            EncodeOrderAndMapping(pop[i].var, order, std::vector<int>());
            BuildMappingGreedyETRM(pop[i].var, order);
            pop[i].fit = EvalVarSafe(pop[i].var.data());
        }

        // Reset exploration parameters
        p_c = 0.8;
        p_greedy = 0.8;
        no_improve_count = 0;
        last_best = gbest_fit;

        // Update gbest
        for (int i = 0; i < P; i++) {
            if (pop[i].fit < gbest_fit) {
                gbest_fit = pop[i].fit;
                gbest = pop[i].var;
            }
        }

        std::cout << "[QPHH] Restart complete. New best: " << gbest_fit << std::endl;
        std::cout.flush();
    }
}

// ---------------------- decoding / encoding ----------------------
void QPHH_Solver::DecodeOrder(const std::vector<double>& var, std::vector<int>& order) const
{
    order.resize(ops);
    std::vector<int> idx(ops);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        double va = var[2 * CE_Tnum + a];
        double vb = var[2 * CE_Tnum + b];
        if (va == vb) return a < b;
        return va < vb;
    });
    order.assign(idx.begin(), idx.end());
}

void QPHH_Solver::DecodeDevIdxByOp(const std::vector<double>& var, const std::vector<int>& order, std::vector<int>& dev_idx_by_op) const
{
    dev_idx_by_op.assign(ops, 0);
    for (int rank = 0; rank < ops; rank++) {
        int op = order[rank];
        int avail = (int)solver->AvailDeviceList[op].size();
        int idx = decode_idx(var[2 * CE_Tnum + ops + rank], avail);
        dev_idx_by_op[op] = idx;
    }
}

void QPHH_Solver::EncodeOrderAndMapping(std::vector<double>& var, const std::vector<int>& order, const std::vector<int>& dev_idx_by_op) const
{
    for (int rank = 0; rank < ops; rank++) {
        int op = order[rank];
        var[2 * CE_Tnum + op] = (rank + 0.5) / ops;
    }

    if (!dev_idx_by_op.empty()) {
        for (int rank = 0; rank < ops; rank++) {
            int op = order[rank];
            int avail = (int)solver->AvailDeviceList[op].size();
            int idx = dev_idx_by_op[op];
            var[2 * CE_Tnum + ops + rank] = bucket_value(idx, avail);
        }
    }
}

void QPHH_Solver::RepairOrder(std::vector<int>& order) const
{
    std::vector<int> pos(ops, -1);
    for (int i = 0; i < ops; i++) pos[order[i]] = i;

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < ops; i++) {
            int op = order[i];
            int job = op / M_OPTnum;
            int step = op % M_OPTnum;
            if (step == 0) continue;
            int parent = job * M_OPTnum + (step - 1);
            if (pos[op] < pos[parent]) {
                // move op after parent
                order.erase(order.begin() + pos[op]);
                int insert_pos = pos[parent] + 1;
                if (insert_pos > (int)order.size()) insert_pos = (int)order.size();
                order.insert(order.begin() + insert_pos, op);
                for (int k = 0; k < ops; k++) pos[order[k]] = k;
                changed = true;
                break;
            }
        }
    }
}

// ---------------------- LLHs ----------------------
void QPHH_Solver::LLH_3Swap(std::vector<int>& order) const
{
    for (int k = 0; k < 3; k++) {
        int a = randint(0, ops);
        int b = randint(0, ops);
        if (a == b) continue;
        std::swap(order[a], order[b]);
    }
    RepairOrder(order);
}

void QPHH_Solver::LLH_IntraLevelSwap(std::vector<int>& order) const
{
    int level = randint(0, M_OPTnum);
    std::vector<int> positions;
    positions.reserve(M_Jnum);
    for (int i = 0; i < ops; i++) {
        if ((order[i] % M_OPTnum) == level) positions.push_back(i);
    }
    if (positions.size() < 2) return;
    int ia = positions[randint(0, (int)positions.size())];
    int ib = positions[randint(0, (int)positions.size())];
    if (ia == ib) return;
    std::swap(order[ia], order[ib]);
    RepairOrder(order);
}

void QPHH_Solver::LLH_3Insert(std::vector<int>& order)
{
    int task = order[randint(0, ops)];
    int job = task / M_OPTnum;
    int step = task % M_OPTnum;
    int parent = (step > 0) ? (job * M_OPTnum + (step - 1)) : -1;
    int child = (step + 1 < M_OPTnum) ? (job * M_OPTnum + (step + 1)) : -1;

    std::vector<int> pos(ops, -1);
    for (int i = 0; i < ops; i++) pos[order[i]] = i;

    int q1 = (parent >= 0) ? pos[parent] : -1;
    int q2 = (child >= 0) ? pos[child] : (ops - 1);

    // Remove task and adjust bounds
    int remove_pos = pos[task];
    order.erase(order.begin() + remove_pos);
    if (q1 > remove_pos) q1--;
    if (q2 > remove_pos) q2--;

    int mid = (int)order.size() / 2;
    int modes[3] = {0, 1, 2}; // any, first half, second half

    for (int m = 0; m < 3; m++) {
        std::vector<int> candidates;
        for (int p = q1 + 1; p <= q2; p++) {
            int pos_after = p;
            if (pos_after > (int)order.size()) pos_after = (int)order.size();
            if (modes[m] == 1 && pos_after > mid) continue;
            if (modes[m] == 2 && pos_after <= mid) continue;
            candidates.push_back(pos_after);
        }
        if (!candidates.empty()) {
            int insert_pos = candidates[randint(0, (int)candidates.size())];
            order.insert(order.begin() + insert_pos, task);
            RepairOrder(order);
            return;
        }
    }

    // No feasible insertion found, restore
    order.insert(order.begin() + clamp_int(q1 + 1, 0, (int)order.size()), task);
}

void QPHH_Solver::LLH_ParentAwareInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var)
{
    (void)dev_idx_by_op;
    (void)tmp_var;
    int task = order[randint(0, ops)];
    int job = task / M_OPTnum;
    int step = task % M_OPTnum;
    if (step == 0) return;
    int parent = job * M_OPTnum + (step - 1);

    std::vector<int> pos(ops, -1);
    for (int i = 0; i < ops; i++) pos[order[i]] = i;

    int parent_pos = pos[parent];
    order.erase(order.begin() + pos[task]);
    int insert_pos = parent_pos + 1;
    if (insert_pos > (int)order.size()) insert_pos = (int)order.size();
    order.insert(order.begin() + insert_pos, task);
    RepairOrder(order);
}

void QPHH_Solver::LLH_GreedyInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, const std::vector<double>& base_var, std::vector<double>& tmp_var, bool pick_suboptimal)
{
    int task = order[randint(0, ops)];
    GreedyInsertTask(order, task, dev_idx_by_op, base_var, tmp_var, pick_suboptimal);
}

void QPHH_Solver::LLH_PairInsert(std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var)
{
    (void)dev_idx_by_op;
    (void)tmp_var;
    if (M_OPTnum < 2) return;
    int job = randint(0, M_Jnum);
    int step = 1 + randint(0, M_OPTnum - 1);
    int parent = job * M_OPTnum + (step - 1);
    int child = job * M_OPTnum + step;

    std::vector<int> pos(ops, -1);
    for (int i = 0; i < ops; i++) pos[order[i]] = i;

    int parent_parent = (step - 2 >= 0) ? (job * M_OPTnum + (step - 2)) : -1;
    int child_child = (step + 1 < M_OPTnum) ? (job * M_OPTnum + (step + 1)) : -1;

    int q1 = (parent_parent >= 0) ? pos[parent_parent] : -1;
    int q2 = (child_child >= 0) ? pos[child_child] : (ops - 1);

    int pos_parent = pos[parent];
    int pos_child = pos[child];
    if (pos_parent > pos_child) std::swap(pos_parent, pos_child);

    std::vector<int> trial = order;
    // remove child then parent (higher index first)
    trial.erase(trial.begin() + pos_child);
    trial.erase(trial.begin() + pos_parent);

    std::vector<int> candidates;
    for (int p = q1 + 1; p <= q2; p++) {
        int insert_pos = p;
        if (insert_pos > (int)trial.size()) insert_pos = (int)trial.size();
        candidates.push_back(insert_pos);
    }
    if (candidates.empty()) return;
    int insert_pos = candidates[randint(0, (int)candidates.size())];
    trial.insert(trial.begin() + insert_pos, parent);
    trial.insert(trial.begin() + insert_pos + 1, child);
    RepairOrder(trial);
    order.swap(trial);
}

// ---------------------- local search ----------------------
void QPHH_Solver::TwoPointCrossover(Individual& ind)
{
    int a = randint(0, Nvar);
    int b = randint(0, Nvar);
    if (a > b) std::swap(a, b);
    if (a == b) return;
    std::vector<double> trial = ind.var;
    for (int i = a; i <= b; i++) {
        trial[i] = gbest[i];
    }
    double fit = EvalVarSafe(trial.data());
    if (fit < ind.fit) {
        ind.var.swap(trial);
        ind.fit = fit;
    }
}

void QPHH_Solver::NTasksGreedyInsert(Individual& ind, std::vector<double>& tmp_var)
{
    std::vector<double> base_var = ind.var;
    double base_fit = ind.fit;

    std::vector<int> order;
    std::vector<int> dev_idx_by_op(ops, 0);
    DecodeOrder(ind.var, order);
    DecodeDevIdxByOp(ind.var, order, dev_idx_by_op);

    std::vector<int> tasks;
    tasks.reserve(ops);
    for (int i = 0; i < ops; i++) tasks.push_back(order[i]);
    for (int i = ops - 1; i > 0; i--) {
        int r = randint(0, i + 1);
        std::swap(tasks[i], tasks[r]);
    }
    if ((int)tasks.size() > N) tasks.resize(N);

    std::sort(tasks.begin(), tasks.end(), [&](int a, int b) {
        int sa = a % M_OPTnum;
        int sb = b % M_OPTnum;
        int imp_a = (M_OPTnum - 1 - sa);
        int imp_b = (M_OPTnum - 1 - sb);
        return imp_a > imp_b;
    });

    for (int t : tasks) {
        GreedyInsertTask(order, t, dev_idx_by_op, ind.var, tmp_var, false);
    }

    EncodeOrderAndMapping(ind.var, order, dev_idx_by_op);
    double fit = EvalVarSafe(ind.var.data());
    if (fit < ind.fit) {
        ind.fit = fit;
    } else {
        ind.var.swap(base_var);
        ind.fit = base_fit;
    }
}

// ---------------------- initialization ----------------------
void QPHH_Solver::BuildRandomFeasibleOrder(std::vector<int>& order) const
{
    // Ready-queue random topological ordering for job-chain precedence
    std::vector<int> next_step(M_Jnum, 0);
    std::vector<int> ready;
    ready.reserve(M_Jnum);
    for (int j = 0; j < M_Jnum; j++) {
        ready.push_back(j * M_OPTnum);
    }

    order.clear();
    order.reserve(ops);

    while (!ready.empty()) {
        int idx = randint(0, (int)ready.size());
        int op = ready[idx];
        ready.erase(ready.begin() + idx);
        order.push_back(op);
        int job = op / M_OPTnum;
        int step = op % M_OPTnum;
        if (step + 1 < M_OPTnum) {
            ready.push_back(job * M_OPTnum + (step + 1));
        }
    }
}

void QPHH_Solver::BuildMappingGreedyETRM(std::vector<double>& var, const std::vector<int>& order)
{
    // Initialize CE task mapping (cloud/edge) + device mapping for ops
    // Greedy with probability p_greedy, ETRM otherwise
    std::vector<int> freq_cloud(solver->Cnum, 0);
    std::vector<int> freq_edge(solver->Enum, 0);
    std::vector<int> cloud_load(solver->Cnum, 0);
    std::vector<int> edge_load(solver->Enum, 0);

    // CE task mapping
    for (int t = 0; t < CE_Tnum; t++) {
        double iter_ratio = (current_iter > 0 && max_iterations > 0)
            ? ((double)current_iter / (double)max_iterations) : 0.0;
        if (iter_ratio < 0.0) iter_ratio = 0.0;
        if (iter_ratio > 1.0) iter_ratio = 1.0;
        double p_greedy_local = p_greedy * (0.3 + 0.7 * (1.0 - iter_ratio));
        bool use_greedy = (randval(0.0, 1.0) < p_greedy_local);
        const std::vector<int>& edge_list = solver->CETask_Property[t].AvailEdgeServerList;

        double best_fit = std::numeric_limits<double>::infinity();
        bool best_edge = false;
        int best_idx = 0;

        std::vector<double> trial = var;
        auto try_candidate = [&](bool edge_mode, int idx) {
            trial[t] = edge_mode ? 0.75 : 0.25;
            if (edge_mode) {
                trial[CE_Tnum + t] = bucket_value(idx, (int)edge_list.size());
            } else {
                trial[CE_Tnum + t] = bucket_value(idx, solver->Cnum);
            }
            double fit = EvalVarSafe(trial.data());
            if (fit < best_fit) {
                best_fit = fit;
                best_edge = edge_mode;
                best_idx = idx;
            }
        };

        if (use_greedy) {
            std::vector<int> cloud_candidates(solver->Cnum);
            std::iota(cloud_candidates.begin(), cloud_candidates.end(), 0);
            if (map_cap > 0 && (int)cloud_candidates.size() > map_cap) {
                Shuffle(cloud_candidates);
                cloud_candidates.resize(map_cap);
            }
            for (int c : cloud_candidates) try_candidate(false, c);

            std::vector<int> edge_candidates_all((int)edge_list.size());
            std::iota(edge_candidates_all.begin(), edge_candidates_all.end(), 0);
            if (map_cap > 0 && (int)edge_candidates_all.size() > map_cap) {
                Shuffle(edge_candidates_all);
                edge_candidates_all.resize(map_cap);
            }
            for (int k : edge_candidates_all) try_candidate(true, k);
        } else {
            std::vector<int> edge_candidates;
            for (int k = 0; k < (int)edge_list.size(); k++) {
                int edge = edge_list[k];
                if (freq_edge[edge] > 0 && edge_load[edge] == 0) edge_candidates.push_back(k);
            }
            if (map_cap > 0 && (int)edge_candidates.size() > map_cap) {
                Shuffle(edge_candidates);
                edge_candidates.resize(map_cap);
            }
            if (!edge_candidates.empty()) {
                for (int k : edge_candidates) try_candidate(true, k);
            } else {
                std::vector<int> cloud_candidates(solver->Cnum);
                std::iota(cloud_candidates.begin(), cloud_candidates.end(), 0);
                if (map_cap > 0 && (int)cloud_candidates.size() > map_cap) {
                    Shuffle(cloud_candidates);
                    cloud_candidates.resize(map_cap);
                }
                for (int c : cloud_candidates) try_candidate(false, c);

                std::vector<int> edge_candidates_all((int)edge_list.size());
                std::iota(edge_candidates_all.begin(), edge_candidates_all.end(), 0);
                if (map_cap > 0 && (int)edge_candidates_all.size() > map_cap) {
                    Shuffle(edge_candidates_all);
                    edge_candidates_all.resize(map_cap);
                }
                for (int k : edge_candidates_all) try_candidate(true, k);
            }
        }

        var[t] = best_edge ? 0.75 : 0.25;
        if (best_edge) {
            var[CE_Tnum + t] = bucket_value(best_idx, (int)edge_list.size());
            int edge = edge_list[best_idx];
            freq_edge[edge]++;
            edge_load[edge]++;
        } else {
            var[CE_Tnum + t] = bucket_value(best_idx, solver->Cnum);
            freq_cloud[best_idx]++;
            cloud_load[best_idx]++;
        }
    }

    // Device mapping for ops
    std::vector<int> freq_dev(solver->Dnum, 0);
    std::vector<int> dev_load(solver->Dnum, 0);

    for (int rank = 0; rank < ops; rank++) {
        int op = order[rank];
        const std::vector<int>& dev_list = solver->AvailDeviceList[op];
        if (dev_list.empty()) continue;

        double iter_ratio = (current_iter > 0 && max_iterations > 0)
            ? ((double)current_iter / (double)max_iterations) : 0.0;
        if (iter_ratio < 0.0) iter_ratio = 0.0;
        if (iter_ratio > 1.0) iter_ratio = 1.0;
        double p_greedy_local = p_greedy * (0.3 + 0.7 * (1.0 - iter_ratio));
        bool use_greedy = (randval(0.0, 1.0) < p_greedy_local);
        double best_fit = std::numeric_limits<double>::infinity();
        int best_idx = 0;

        std::vector<double> trial = var;
        auto try_candidate = [&](int idx) {
            trial[2 * CE_Tnum + ops + rank] = bucket_value(idx, (int)dev_list.size());
            double fit = EvalVarSafe(trial.data());
            if (fit < best_fit) {
                best_fit = fit;
                best_idx = idx;
            }
        };

        if (use_greedy) {
            std::vector<int> candidates((int)dev_list.size());
            std::iota(candidates.begin(), candidates.end(), 0);
            if (map_cap > 0 && (int)candidates.size() > map_cap) {
                Shuffle(candidates);
                candidates.resize(map_cap);
            }
            for (int k : candidates) try_candidate(k);
        } else {
            std::vector<int> candidates;
            for (int k = 0; k < (int)dev_list.size(); k++) {
                int dev = dev_list[k];
                if (freq_dev[dev] > 0 && dev_load[dev] == 0) candidates.push_back(k);
            }
            if (map_cap > 0 && (int)candidates.size() > map_cap) {
                Shuffle(candidates);
                candidates.resize(map_cap);
            }
            if (!candidates.empty()) {
                for (int k : candidates) try_candidate(k);
            } else {
                std::vector<int> all_candidates((int)dev_list.size());
                std::iota(all_candidates.begin(), all_candidates.end(), 0);
                if (map_cap > 0 && (int)all_candidates.size() > map_cap) {
                    Shuffle(all_candidates);
                    all_candidates.resize(map_cap);
                }
                for (int k : all_candidates) try_candidate(k);
            }
        }

        var[2 * CE_Tnum + ops + rank] = bucket_value(best_idx, (int)dev_list.size());
        int dev = dev_list[best_idx];
        freq_dev[dev]++;
        dev_load[dev]++;
    }
}

// ---------------------- evaluation helpers ----------------------
double QPHH_Solver::EvalWithOrder(const std::vector<double>& base_var, const std::vector<int>& order, const std::vector<int>& dev_idx_by_op, std::vector<double>& tmp_var) const
{
    tmp_var = base_var;
    EncodeOrderAndMapping(tmp_var, order, dev_idx_by_op);
    return EvalVarSafe(tmp_var.data());
}

void QPHH_Solver::GreedyInsertTask(std::vector<int>& order, int task, const std::vector<int>& dev_idx_by_op, const std::vector<double>& base_var, std::vector<double>& tmp_var, bool pick_suboptimal)
{
    int job = task / M_OPTnum;
    int step = task % M_OPTnum;
    int parent = (step > 0) ? (job * M_OPTnum + (step - 1)) : -1;
    int child = (step + 1 < M_OPTnum) ? (job * M_OPTnum + (step + 1)) : -1;

    std::vector<int> pos(ops, -1);
    for (int i = 0; i < ops; i++) pos[order[i]] = i;

    int q1 = (parent >= 0) ? pos[parent] : -1;
    int q2 = (child >= 0) ? pos[child] : (ops - 1);

    int remove_pos = pos[task];
    order.erase(order.begin() + remove_pos);
    if (q1 > remove_pos) q1--;
    if (q2 > remove_pos) q2--;

    struct Candidate {
        int pos;
        double fit;
    };
    std::vector<int> positions;
    for (int p = q1 + 1; p <= q2; p++) {
        int insert_pos = p;
        if (insert_pos > (int)order.size()) insert_pos = (int)order.size();
        positions.push_back(insert_pos);
    }
    int adaptive_cap = 0;
    if (gi_cap > 0) {
        adaptive_cap = std::max(5, std::min(gi_cap, std::max(1, (q2 - q1) / 3)));
    }
    if (adaptive_cap > 0 && (int)positions.size() > adaptive_cap) {
        Shuffle(positions);
        positions.resize(adaptive_cap);
    }

    std::vector<Candidate> cands;
    for (int insert_pos : positions) {
        std::vector<int> trial = order;
        trial.insert(trial.begin() + insert_pos, task);
        RepairOrder(trial);
        double fit = EvalWithOrder(base_var, trial, dev_idx_by_op, tmp_var);
        cands.push_back({insert_pos, fit});
    }

    if (cands.empty()) {
        order.insert(order.begin() + clamp_int(q1 + 1, 0, (int)order.size()), task);
        return;
    }

    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        return a.fit < b.fit;
    });

    int pick = 0;
    if (pick_suboptimal && cands.size() >= 2) pick = 1;
    int insert_pos = cands[pick].pos;
    order.insert(order.begin() + insert_pos, task);
    RepairOrder(order);
}

// ---------------------- Q-learning helpers ----------------------
int QPHH_Solver::SelectAction(int state) const
{
    double iter_ratio = (current_iter > 0 && max_iterations > 0)
        ? ((double)current_iter / (double)max_iterations) : 0.0;
    if (iter_ratio < 0.0) iter_ratio = 0.0;
    if (iter_ratio > 1.0) iter_ratio = 1.0;
    double adaptive_epsilon = std::max(0.05, epsilon * std::exp(-3.0 * iter_ratio));

    double r = randval(0.0, 1.0);
    if (r < adaptive_epsilon) {
        return randint(0, 6);
    }
    int best = 0;
    double best_q = Q[state][0];
    for (int a = 1; a < 6; a++) {
        if (Q[state][a] > best_q) {
            best_q = Q[state][a];
            best = a;
        }
    }
    return best;
}

int QPHH_Solver::ComputeState(double delta) const
{
    if (delta < 0.0) return 0;
    if (std::fabs(delta) < 1e-12) return 1;
    for (int z = 1; z <= 6; z++) {
        double lo = std::pow(10.0, -z);
        double hi = std::pow(10.0, -(z - 1));
        if (delta >= lo && delta < hi) return z + 1;
    }
    if (delta >= 1.0) return 8;
    return 9;
}

double QPHH_Solver::ComputeReward(double delta) const
{
    if (delta < 0.0) return -10.0;
    if (delta >= std::pow(10.0, -1)) return 10.0;
    for (int z = 1; z <= 5; z++) {
        double lo = std::pow(10.0, -(z + 1));
        double hi = std::pow(10.0, -z);
        if (delta >= lo && delta <= hi) return 10.0 - z;
    }
    return 0.0;
}

double QPHH_Solver::ComputeAlpha(int iter) const
{
    double T = (double)(iter + 1);
    double Ttotal = (double)max_iterations;
    double alpha = 1.0 - 0.9 * (T / Ttotal);
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    return alpha;
}
