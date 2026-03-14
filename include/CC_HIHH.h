#ifndef CC_HIHH_H
#define CC_HIHH_H

#include <vector>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include "Multimethod.h"

// Debug output for contextual bandit
#define HIHH_CB_DEBUG 0
#define HIHH_CB_DEBUG_INTERVAL 50

//=============================================================================
// ContextualBanditSelector - Linear contextual bandit for operator selection
//=============================================================================
struct ContextualBanditSelector {
    int num_ops;
    int feat_dim;
    std::vector<double> weights;        // [num_ops * feat_dim]
    std::vector<double> total_reward;   // Cumulative reward per operator (raw)
    std::vector<int> selection_count;   // Times each operator was selected
    int total_updates;

    ContextualBanditSelector() : num_ops(0), feat_dim(0), total_updates(0) {}

    void init(int n_ops, int fdim) {
        num_ops = n_ops;
        feat_dim = fdim;
        weights.assign(num_ops * feat_dim, 0.0);
        total_reward.assign(num_ops, 0.0);
        selection_count.assign(num_ops, 0);
        total_updates = 0;
    }

    double score_op(int op_idx, const std::array<double, 7>& s) const {
        double score = 0.0;
        const int base = op_idx * feat_dim;
        const int nfeat = std::min(feat_dim, 7);
        for (int k = 0; k < nfeat; k++) {
            score += weights[base + k] * s[k];
        }
        return score;
    }

    int select(const std::array<double, 7>& s, double epsilon) {
        if (num_ops == 0) return 0;

        // Initial exploration: try each operator once
        for (int i = 0; i < num_ops; i++) {
            if (selection_count[i] == 0) return i;
        }

        double r = (double)rand() / RAND_MAX;
        if (r < epsilon) {
            return rand() % num_ops;
        }

        double best_score = -1e30;
        int best_op = 0;
        for (int i = 0; i < num_ops; i++) {
            double sc = score_op(i, s);
            if (sc > best_score) {
                best_score = sc;
                best_op = i;
            }
        }
        return best_op;
    }

    int select_masked(const std::array<double, 7>& s, double epsilon, int excluded_op) const {
        if (num_ops <= 0) return 0;
        int allowed = 0;
        for (int i = 0; i < num_ops; ++i) {
            if (i != excluded_op) ++allowed;
        }
        if (allowed <= 0) return 0;

        double r = (double)rand() / RAND_MAX;
        if (r < epsilon) {
            int pick = rand() % allowed;
            for (int i = 0; i < num_ops; ++i) {
                if (i == excluded_op) continue;
                if (pick == 0) return i;
                --pick;
            }
        }
        return select_best_masked(s, excluded_op);
    }

    int select_best_masked(const std::array<double, 7>& s, int excluded_op) const {
        if (num_ops <= 0) return 0;
        double best_score = -1e30;
        int best_op = 0;
        bool found = false;
        for (int i = 0; i < num_ops; i++) {
            if (i == excluded_op) continue;
            double sc = score_op(i, s);
            if (!found || sc > best_score) {
                best_score = sc;
                best_op = i;
                found = true;
            }
        }
        return found ? best_op : 0;
    }

    void update(const std::array<double, 7>& s, int op_idx, double reward, double lr) {
        if (op_idx < 0 || op_idx >= num_ops) return;
        double pred = score_op(op_idx, s);
        double err = reward - pred;
        int base = op_idx * feat_dim;
        const int nfeat = std::min(feat_dim, 7);
        for (int k = 0; k < nfeat; k++) {
            weights[base + k] += lr * err * s[k];
        }
        total_reward[op_idx] += reward;
        selection_count[op_idx]++;
        total_updates++;
    }

    void reset() {
        std::fill(weights.begin(), weights.end(), 0.0);
        std::fill(total_reward.begin(), total_reward.end(), 0.0);
        std::fill(selection_count.begin(), selection_count.end(), 0);
        total_updates = 0;
    }
};

struct BanditSnapshot {
    int num_ops;
    int feat_dim;
    std::vector<double> weights;  // [num_ops * feat_dim]

    BanditSnapshot() : num_ops(0), feat_dim(0) {}

    void clear() {
        num_ops = 0;
        feat_dim = 0;
        weights.clear();
    }

    void take_snapshot(const ContextualBanditSelector& sel) {
        num_ops = sel.num_ops;
        feat_dim = sel.feat_dim;
        weights = sel.weights;
    }

    double score_op(int op_idx, const std::array<double, 7>& s) const {
        if (op_idx < 0 || op_idx >= num_ops) return -1e30;
        double score = 0.0;
        const int base = op_idx * feat_dim;
        const int nfeat = std::min(feat_dim, 7);
        for (int k = 0; k < nfeat; ++k) {
            score += weights[base + k] * s[k];
        }
        return score;
    }

    int select_from_snapshot(const std::array<double, 7>& s, double epsilon,
                             int n_ops_block, int block_offset = 0) const {
        if (n_ops_block <= 0) return 0;
        double r = (double)rand() / RAND_MAX;
        if (r < epsilon) return rand() % n_ops_block;

        int best_op = 0;
        double best_score = score_op(block_offset + 0, s);
        for (int op = 1; op < n_ops_block; ++op) {
            double sc = score_op(block_offset + op, s);
            if (sc > best_score) {
                best_score = sc;
                best_op = op;
            }
        }
        return best_op;
    }

    int select_from_snapshot_masked(const std::array<double, 7>& s, double epsilon,
                                    int n_ops_block, int excluded_local_op,
                                    int block_offset = 0) const {
        if (n_ops_block <= 0) return 0;
        int allowed = 0;
        for (int op = 0; op < n_ops_block; ++op) {
            if (op != excluded_local_op) ++allowed;
        }
        if (allowed <= 0) return 0;

        double r = (double)rand() / RAND_MAX;
        if (r < epsilon) {
            int pick = rand() % allowed;
            for (int op = 0; op < n_ops_block; ++op) {
                if (op == excluded_local_op) continue;
                if (pick == 0) return op;
                --pick;
            }
        }
        return select_best_from_snapshot_masked(s, n_ops_block, excluded_local_op, block_offset);
    }

    int select_best_from_snapshot_masked(const std::array<double, 7>& s,
                                         int n_ops_block,
                                         int excluded_local_op,
                                         int block_offset = 0) const {
        if (n_ops_block <= 0) return 0;
        double best_score = -1e30;
        int best_op = 0;
        bool found = false;
        for (int op = 0; op < n_ops_block; ++op) {
            if (op == excluded_local_op) continue;
            double sc = score_op(block_offset + op, s);
            if (!found || sc > best_score) {
                best_score = sc;
                best_op = op;
                found = true;
            }
        }
        return found ? best_op : 0;
    }
};

struct DeferredBanditUpdate {
    std::array<double, 7> state;
    int op;
    double reward;
};

//=============================================================================
// BlockContext - Manages context vectors for cooperative coevolution
//=============================================================================
struct BlockContext {
    std::vector<double> offload_ctx;  // Block 0: [0, 2*CE_Tnum)
    std::vector<double> seq_ctx;      // Block 1: [2*CE_Tnum, 2*CE_Tnum + ops)
    std::vector<double> dev_ctx;      // Block 2: [2*CE_Tnum + ops, end)
    
    int CE_Tnum, ops, Nvar;
    
    BlockContext() : CE_Tnum(0), ops(0), Nvar(0) {}
    
    void init(int ce_tnum, int m_jnum, int m_optnum) {
        CE_Tnum = ce_tnum;
        ops = m_jnum * m_optnum;
        Nvar = 2 * CE_Tnum + 2 * ops;
        
        offload_ctx.resize(2 * CE_Tnum);
        seq_ctx.resize(ops);
        dev_ctx.resize(ops);
    }
    
    void set_random(double lb = 0.0, double ub = 1.0) {
        for (auto& v : offload_ctx) v = lb + (double)rand() / RAND_MAX * (ub - lb);
        for (auto& v : seq_ctx) v = lb + (double)rand() / RAND_MAX * (ub - lb);
        for (auto& v : dev_ctx) v = lb + (double)rand() / RAND_MAX * (ub - lb);
    }
    
    void update_block(int block_id, const double* new_ctx) {
        if (block_id == 0) {
            std::memcpy(offload_ctx.data(), new_ctx, 2 * CE_Tnum * sizeof(double));
        } else if (block_id == 1) {
            std::memcpy(seq_ctx.data(), new_ctx, ops * sizeof(double));
        } else {
            std::memcpy(dev_ctx.data(), new_ctx, ops * sizeof(double));
        }
    }
    
    void assemble_full(int block_id, const double* block_data, double* var_full) const {
        const int offload_len = 2 * CE_Tnum;
        if (block_id == 0) {
            std::memcpy(var_full, block_data, offload_len * sizeof(double));
        } else {
            std::memcpy(var_full, offload_ctx.data(), offload_len * sizeof(double));
        }
        
        if (block_id == 1) {
            std::memcpy(var_full + offload_len, block_data, ops * sizeof(double));
        } else {
            std::memcpy(var_full + offload_len, seq_ctx.data(), ops * sizeof(double));
        }
        
        if (block_id == 2) {
            std::memcpy(var_full + offload_len + ops, block_data, ops * sizeof(double));
        } else {
            std::memcpy(var_full + offload_len + ops, dev_ctx.data(), ops * sizeof(double));
        }
    }
    
    void get_full(double* var_full) const {
        const int offload_len = 2 * CE_Tnum;
        std::memcpy(var_full, offload_ctx.data(), offload_len * sizeof(double));
        std::memcpy(var_full + offload_len, seq_ctx.data(), ops * sizeof(double));
        std::memcpy(var_full + offload_len + ops, dev_ctx.data(), ops * sizeof(double));
    }
    
    int get_block_len(int block_id) const {
        if (block_id == 0) return 2 * CE_Tnum;
        return ops;  // Both seq and dev have length = ops
    }
};

//=============================================================================
// BlockPopulation - Manages population for one block with islands and contextual bandit
//=============================================================================
struct BlockPopulation {
    int block_id;           // 0=offload, 1=seq, 2=dev
    int block_start;        // Start index in full var
    int block_len;          // Length of this block
    int popsize;
    int nSubpop;
    
    double** pop;           // Current population [popsize][block_len]
    double** newpop;        // Offspring population
    double* pop_fit;        // Fitness values
    double* newpop_fit;
    double* block_gbest;    // Best individual for this block
    double block_gbest_fit;
    
    std::vector<ContextualBanditSelector> cb_selectors;  // One per island
    
    // Island-level best tracking for migration
    std::vector<double*> island_gbest;
    std::vector<double> island_gbest_fit;

    // Island stats for contextual state
    int recent_k;
    std::vector<int> island_stagnation;
    std::vector<double> island_last_best_fit;
    std::vector<double> island_last_reward;
    std::vector<int> island_last_op;
    std::vector<int> recent_pos;
    std::vector<int> recent_count;
    std::vector<int> recent_success_count;
    std::vector<double> recent_improve_sum;
    std::vector<std::vector<int>> recent_success_hist;
    std::vector<std::vector<double>> recent_improve_hist;
    
    BlockPopulation() : block_id(0), block_start(0), block_len(0), 
                        popsize(0), nSubpop(0), pop(nullptr), newpop(nullptr),
                        pop_fit(nullptr), newpop_fit(nullptr), 
                        block_gbest(nullptr), block_gbest_fit(1e30) {}
    
    void init(int bid, int bstart, int blen, int psize, int nsub, int num_ops, int feat_dim, int recent_window) {
        block_id = bid;
        block_start = bstart;
        block_len = blen;
        popsize = psize;
        nSubpop = nsub;
        recent_k = recent_window;
        
        // Allocate arrays
        pop = new double*[popsize];
        newpop = new double*[popsize];
        for (int i = 0; i < popsize; i++) {
            pop[i] = new double[block_len];
            newpop[i] = new double[block_len];
        }
        pop_fit = new double[popsize];
        newpop_fit = new double[popsize];
        block_gbest = new double[block_len];
        block_gbest_fit = 1e30;
        
        // Initialize contextual bandit selectors for each island
        cb_selectors.resize(nsub);
        for (int i = 0; i < nsub; i++) {
            cb_selectors[i].init(num_ops, feat_dim);
        }
        
        // Initialize island gbest tracking
        island_gbest.resize(nsub);
        island_gbest_fit.resize(nsub, 1e30);
        for (int i = 0; i < nsub; i++) {
            island_gbest[i] = new double[block_len];
        }

        island_stagnation.assign(nsub, 0);
        island_last_best_fit.assign(nsub, 1e30);
        island_last_reward.assign(nsub, 0.0);
        island_last_op.assign(nsub, -1);
        recent_pos.assign(nsub, 0);
        recent_count.assign(nsub, 0);
        recent_success_count.assign(nsub, 0);
        recent_improve_sum.assign(nsub, 0.0);
        recent_success_hist.assign(nsub, std::vector<int>(recent_k, 0));
        recent_improve_hist.assign(nsub, std::vector<double>(recent_k, 0.0));
        
        // Random initialization
        for (int i = 0; i < popsize; i++) {
            for (int j = 0; j < block_len; j++) {
                pop[i][j] = (double)rand() / RAND_MAX;
                newpop[i][j] = pop[i][j];
            }
            pop_fit[i] = 1e30;
            newpop_fit[i] = 1e30;
        }
    }
    
    void destroy() {
        if (pop) {
            for (int i = 0; i < popsize; i++) {
                delete[] pop[i];
                delete[] newpop[i];
            }
            delete[] pop;
            delete[] newpop;
            delete[] pop_fit;
            delete[] newpop_fit;
            delete[] block_gbest;
            
            for (int i = 0; i < nSubpop; i++) {
                delete[] island_gbest[i];
            }
            
            pop = nullptr;
            newpop = nullptr;
        }
    }
    
    void get_island_range(int isl, int& p_start, int& p_end) const {
        int subpop_size = popsize / nSubpop;
        p_start = isl * subpop_size;
        p_end = (isl == nSubpop - 1) ? popsize : (isl + 1) * subpop_size;
    }
    
    double get_island_best_fit(int p_start, int p_end) const {
        double best = pop_fit[p_start];
        for (int i = p_start + 1; i < p_end; i++) {
            if (pop_fit[i] < best) best = pop_fit[i];
        }
        return best;
    }
    
    int get_island_best_idx(int p_start, int p_end) const {
        int best_idx = p_start;
        for (int i = p_start + 1; i < p_end; i++) {
            if (pop_fit[i] < pop_fit[best_idx]) best_idx = i;
        }
        return best_idx;
    }
    
    double compute_diversity(int p_start, int p_end) const {
        if (p_end <= p_start + 1) return 0.0;
        
        // Average variance across dimensions
        double total_var = 0.0;
        for (int j = 0; j < block_len; j++) {
            double mean = 0.0;
            for (int i = p_start; i < p_end; i++) {
                mean += pop[i][j];
            }
            mean /= (p_end - p_start);
            
            double var = 0.0;
            for (int i = p_start; i < p_end; i++) {
                var += (pop[i][j] - mean) * (pop[i][j] - mean);
            }
            total_var += var / (p_end - p_start);
        }
        return total_var / block_len;
    }

    double compute_global_diversity() const {
        return compute_diversity(0, popsize);
    }

    double compute_intra_island_diversity() const {
        if (nSubpop <= 1) return compute_global_diversity();
        double total = 0.0;
        int valid = 0;
        for (int isl = 0; isl < nSubpop; ++isl) {
            int p_start, p_end;
            get_island_range(isl, p_start, p_end);
            if (p_end <= p_start) continue;
            total += compute_diversity(p_start, p_end);
            ++valid;
        }
        return valid > 0 ? total / (double)valid : 0.0;
    }

    double compute_inter_island_diversity() const {
        if (nSubpop <= 1 || block_len <= 0) return 0.0;

        std::vector<std::vector<double>> centroids;
        centroids.reserve(nSubpop);
        for (int isl = 0; isl < nSubpop; ++isl) {
            int p_start, p_end;
            get_island_range(isl, p_start, p_end);
            const int island_size = p_end - p_start;
            if (island_size <= 0) continue;

            centroids.emplace_back(block_len, 0.0);
            std::vector<double>& centroid = centroids.back();
            for (int i = p_start; i < p_end; ++i) {
                for (int j = 0; j < block_len; ++j) {
                    centroid[j] += pop[i][j];
                }
            }
            for (int j = 0; j < block_len; ++j) {
                centroid[j] /= (double)island_size;
            }
        }

        if (centroids.size() <= 1) return 0.0;

        double total_var = 0.0;
        for (int j = 0; j < block_len; ++j) {
            double mean = 0.0;
            for (const auto& centroid : centroids) mean += centroid[j];
            mean /= (double)centroids.size();

            double var = 0.0;
            for (const auto& centroid : centroids) {
                const double diff = centroid[j] - mean;
                var += diff * diff;
            }
            total_var += var / (double)centroids.size();
        }
        return total_var / (double)block_len;
    }
    
    void update_block_gbest() {
        const size_t bytes = block_len * sizeof(double);
        for (int i = 0; i < popsize; i++) {
            if (pop_fit[i] < block_gbest_fit) {
                std::memcpy(block_gbest, pop[i], bytes);
                block_gbest_fit = pop_fit[i];
            }
        }
    }
    
    void update_island_gbest() {
        const size_t bytes = block_len * sizeof(double);
        for (int isl = 0; isl < nSubpop; isl++) {
            int p_start, p_end;
            get_island_range(isl, p_start, p_end);

            for (int i = p_start; i < p_end; i++) {
                if (pop_fit[i] < island_gbest_fit[isl]) {
                    std::memcpy(island_gbest[isl], pop[i], bytes);
                    island_gbest_fit[isl] = pop_fit[i];
                }
            }
        }
    }
    
    int get_island_worst_idx(int p_start, int p_end) const {
        int worst = p_start;
        for (int i = p_start + 1; i < p_end; i++) {
            if (pop_fit[i] > pop_fit[worst]) worst = i;
        }
        return worst;
    }
    
    void copy_pop_to_newpop(int p_start, int p_end) {
        const size_t bytes = block_len * sizeof(double);
        for (int i = p_start; i < p_end; i++) {
            std::memcpy(newpop[i], pop[i], bytes);
            newpop_fit[i] = pop_fit[i];
        }
    }

    void update_recent_stats(int isl, int success, double improve_norm) {
        if (recent_k <= 0) return;
        int pos = recent_pos[isl];
        if (recent_count[isl] < recent_k) {
            recent_count[isl]++;
        } else {
            recent_success_count[isl] -= recent_success_hist[isl][pos];
            recent_improve_sum[isl] -= recent_improve_hist[isl][pos];
        }
        recent_success_hist[isl][pos] = success;
        recent_improve_hist[isl][pos] = improve_norm;
        recent_success_count[isl] += success;
        recent_improve_sum[isl] += improve_norm;
        recent_pos[isl] = (pos + 1) % recent_k;
    }
};

struct DEArchive {
    std::vector<std::vector<double>> buf;
    int max_size;
    int dim;
    int count;
    int next_slot;

    DEArchive() : max_size(0), dim(0), count(0), next_slot(0) {}

    void init(int max_sz, int d) {
        max_size = max_sz > 0 ? max_sz : 0;
        dim = d > 0 ? d : 0;
        count = 0;
        next_slot = 0;
        buf.assign(max_size, std::vector<double>(dim, 0.0));
    }

    void add(const double* sol) {
        if (max_size <= 0 || dim <= 0 || sol == nullptr) {
            return;
        }
        std::copy(sol, sol + dim, buf[next_slot].begin());
        next_slot = (next_slot + 1) % max_size;
        if (count < max_size) {
            count++;
        }
    }

    const double* random_get() const {
        if (count <= 0) {
            return nullptr;
        }
        int idx = rand() % count;
        return buf[idx].data();
    }

    bool empty() const {
        return count <= 0;
    }
};

//=============================================================================
// Operator IDs (per block)
//=============================================================================
enum OffloadOps {
    OFF_OP_GA = 0,
    OFF_OP_DE = 1,
    OFF_OP_BITFLIP = 2,
    OFF_OP_BLOCK_RESAMPLE = 3
};

enum SeqOps {
    SEQ_OP_GA = 0,
    SEQ_OP_SWAP = 1,
    SEQ_OP_VNS = 2,
    SEQ_OP_BLOCK_RESAMPLE = 3
};

enum DevOps {
    DEV_OP_DE = 0,
    DEV_OP_GDE = 1,
    DEV_OP_LEVY = 2,
    DEV_OP_BLOCK_RESAMPLE = 3
};

// Full-space ops for no-block ablation
enum FullOps {
    FULL_OP_GA = 0,
    FULL_OP_DE = 1,
    FULL_OP_GDE = 2,
    FULL_OP_LEVY = 3,
    FULL_OP_BLOCK_RESAMPLE = 4
};

enum OperatorSelectionMode {
    MODE_CONTEXTUAL_BANDIT = 0,
    MODE_RANDOM = 1,
    MODE_ROUND_ROBIN = 2
};

//=============================================================================
// CC_HIHH_Solver - Main solver class for CC-HIHH-UCB algorithm
//=============================================================================
class CC_HIHH_Solver {
public:
    // Problem parameters from MultiMet
    MultiMet* solver;
    int CE_Tnum, M_Jnum, M_OPTnum, Nvar;
    int ops;
    
    // Populations per block
    BlockPopulation P_offload;
    BlockPopulation P_seq;
    BlockPopulation P_dev;
    BlockPopulation P_full;
    
    // Shared context
    BlockContext context;
    
    // Full var buffer for evaluation
    std::vector<double> var_full;
    
    // Global best
    std::vector<double> gbest;
    double gbest_fit;
    
    // Parameters
    int popsize;
    int nSubpop;
    int nCircle;
    double pElitist;
    double lambda;  // Diversity bonus weight

    // Contextual bandit parameters
    int max_generations;
    int state_dim;
    int recent_k;
    double lr0;
    double epsilon0;
    double epsilon_decay;
    double epsilon_min;
    double epsilon_k;
    double lr_k;
    double reward_clip;
    double stable_reward_clip;
    int seq_swap_count;
    double resample_rate;
    bool stable_mode;
    bool gate_enabled;
    int resample_gate;
    int gate_blocked_total;
    int gate_fallback_total;
    
    // Operator counts per block
    int num_ops_offload;
    int num_ops_seq;
    int num_ops_dev;
    int num_ops_full;
    
    // Stagnation tracking
    int stagnation_count;
    double prev_gbest_fit;

    // Ablation toggles
    bool use_blocks;
    bool enable_intra_migration;
    bool use_bandit;
    bool shared_bandit_mode;
    bool fixed_ops_per_block;
    OperatorSelectionMode op_selection_mode;
    std::vector<int> round_robin_counters;
    // Per-island shared bandits: each island's bandit is shared across all
    // 3 blocks so that cross-block interference is present while keeping
    // per-island data density identical to the full (independent) baseline.
    std::vector<ContextualBanditSelector> shared_cb_selectors;          // [nSubpop]
    std::vector<BanditSnapshot> shared_snapshots;                       // [nSubpop]
    std::vector<std::vector<DeferredBanditUpdate>> shared_island_deferred; // [nSubpop]

    // Operator stats logging
    bool op_stats_enabled;
    int op_stats_every;
    int op_stats_interval_gens;
    std::string op_stats_path;
    std::ofstream op_stats_out;
    std::vector<long long> op_counts_offload;
    std::vector<long long> op_counts_seq;
    std::vector<long long> op_counts_dev;
    std::vector<long long> op_counts_overall;
    std::vector<long long> op_counts_full;
    DEArchive archive_offload;
    DEArchive archive_dev;
    DEArchive archive_full;
    std::vector<double> migration_buffer;      // [nSubpop * block_len]
    std::vector<double> migration_fit_buffer;  // [nSubpop]

    // Weight logging
    bool weight_log_enabled;
    int weight_log_every;
    std::string weight_log_path_offload;
    std::string weight_log_path_seq;
    std::string weight_log_path_dev;
    std::ofstream weight_log_out_offload;
    std::ofstream weight_log_out_seq;
    std::ofstream weight_log_out_dev;

    // Reward logging
    bool reward_log_enabled;
    std::string reward_log_path;
    std::ofstream reward_log_out;

    // Reward variance logging
    bool reward_variance_log_enabled;
    std::string reward_variance_log_path;
    std::ofstream reward_variance_log_out;

    // Global stats logging
    bool global_stats_enabled;
    int global_stats_every;
    std::string global_stats_path;
    std::ofstream global_stats_out;

    // Diversity logging
    bool diversity_log_enabled;
    int diversity_log_every;
    std::string diversity_log_path;
    std::string diversity_log_config_name;
    std::ofstream diversity_log_out;
    
    CC_HIHH_Solver(MultiMet* s, int psize, int nsub, int ncircle, double pelite = 0.8);
    ~CC_HIHH_Solver();

    void SetMaxGenerations(int max_gen) { max_generations = max_gen > 0 ? max_gen : 1; }
    void SetStableMode(bool v) { stable_mode = v; }
    void SetResampleGate(int t) { resample_gate = t; gate_enabled = (t > 0); }
    void SetStableRewardClip(double v) { stable_reward_clip = v > 0.0 ? v : stable_reward_clip; }
    void SetEpsilonParams(double e0, double emin, double k) {
        epsilon0 = e0;
        epsilon_min = emin;
        epsilon_k = k;
    }
    void SetLearningRateParams(double l0, double lk) {
        lr0 = l0;
        lr_k = lk;
    }
    void SetUseBlocks(bool v) { use_blocks = v; }
    void SetMigrationEnabled(bool v) { enable_intra_migration = v; }
    void SetUseBandit(bool v) { use_bandit = v; }
    void SetSharedBanditMode(bool v) { shared_bandit_mode = v; }
    void SetFixedOpsPerBlock(bool v) { fixed_ops_per_block = v; }
    void SetSelectionMode(OperatorSelectionMode mode) { op_selection_mode = mode; }
    void SetOpStats(const std::string& path, int every);
    void SetWeightLogging(const std::string& offload_path, const std::string& seq_path, const std::string& dev_path, int every);
    void SetRewardLogging(const std::string& path);
    void SetRewardVarianceLogging(const std::string& path);
    void SetGlobalStatsLogging(const std::string& path, int every);
    void SetDiversityLogging(const std::string& path, int every);
    
    void Init();
    void RunGeneration(int gen);
    void LogOpStatsIfNeeded(int gen, bool is_last);
    void LogWeightsIfNeeded(int gen, bool is_last);
    void LogGlobalStatsIfNeeded(int gen, bool is_last);
    void LogDiversityIfNeeded(int gen, bool is_last);
    void CloseOpStats();
    void CloseWeightLogging();
    void CloseRewardLogging();
    void CloseRewardVarianceLogging();
    void CloseGlobalStatsLogging();
    void CloseDiversityLogging();
    void MigrationWithinBlock(BlockPopulation& bp, int dispara);
    
    // Operator application
    void ApplyOperator(int op, BlockPopulation& bp, int p_start, int p_end);
    void ApplyOperatorFull(int op, BlockPopulation& bp, int p_start, int p_end);
    
    // Block-specific operators
    void ApplyGA(BlockPopulation& bp, int p_start, int p_end);
    void ApplyDE(BlockPopulation& bp, int p_start, int p_end);
    void ApplyGDE(BlockPopulation& bp, int p_start, int p_end);
    void ApplyBitFlip(BlockPopulation& bp, int p_start, int p_end);
    void ApplySeqSwap(BlockPopulation& bp, int p_start, int p_end, int n_swaps);
    void ApplyVNS(BlockPopulation& bp, int p_start, int p_end);
    void ApplyLevy(BlockPopulation& bp, int p_start, int p_end);
    void ApplyBlockResample(BlockPopulation& bp, int p_start, int p_end, double rate);
    
    // Cooperative evaluation
    void EvaluateBlock(BlockPopulation& bp);
    void EvaluateBlockIsland(BlockPopulation& bp, int p_start, int p_end);
    
    // Selection: update pop from newpop where improved
    void SelectionUpdate(BlockPopulation& bp, int p_start, int p_end);
    
    double GetGlobalBestFit() const { return gbest_fit; }
    const double* GetGlobalBest() const { return gbest.data(); }
    int GetGateBlockedTotal() const { return gate_blocked_total; }
    int GetGateFallbackTotal() const { return gate_fallback_total; }
    
private:
    double levy_beta;
    double levy_step_coeff;
    double levy_dim_ratio;
    int vns_max_k;
    int vns_samples_per_k;
    double vns_elite_ratio;

    inline double randval(double low, double high) {
        return low + (double)rand() / RAND_MAX * (high - low);
    }
    
    inline double clip01(double v) {
        return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }

    std::array<double, 7> ComputeState(const BlockPopulation& bp, int isl, int gen, double diversity) const;
    double ComputeEpsilon(int gen) const;
    double ComputeLearningRate(const ContextualBanditSelector& sel) const;
    bool IsResampleOp(int op, int block_id) const;
    bool ShouldBlockResample(const BlockPopulation& bp, int isl, int gen, double diversity) const;
    double LevyFlight(double beta);
    int SelectBestNonResampleOp(int block_id,
                                const std::array<double, 7>& state,
                                ContextualBanditSelector* selector,
                                const BanditSnapshot* snapshot) const;
    void ApplyGateBlockedPenalty(ContextualBanditSelector* selector,
                                 const std::array<double, 7>& state,
                                 int blocked_op,
                                 double clip_val);

    void InitOpStats();
    void ResetOpStatsInterval();
    void RecordOpSelection(int block_id, int op_id);
    void InitWeightLogging();
    void InitRewardLogging();
    void InitRewardVarianceLogging();
    void InitGlobalStatsLogging();
    void InitDiversityLogging();
    void LogReward(int gen, int block_id, int island_id, int op_id, double reward, double improvement, double div_change);
    void LogRewardVariance(int generation, int block_id, const std::vector<double>& rewards);
    int SelectOperatorRandom(int block_id) const;
    int SelectOperatorRoundRobin(int block_id);
    double ComputeGlobalAvgFitness() const;
    double ComputeGlobalDiversity() const;
    void WriteDiversityRow(int generation, const char* block_name, double intra_raw, double global_raw, double inter_raw);
};

#endif // CC_HIHH_H
