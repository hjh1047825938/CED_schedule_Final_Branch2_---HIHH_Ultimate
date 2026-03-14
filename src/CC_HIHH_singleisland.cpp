#include "CC_HIHH.h"
#include "Rng.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <utility>

namespace {

constexpr double kPi = 3.14159265358979323846;

double unit_open(double u) {
    if (u <= 0.0) return 1e-10;
    if (u >= 1.0) return 1.0 - 1e-10;
    return u;
}

void segment_shift_values(double* values, int block_len, int seg_len) {
    if (values == nullptr || block_len < 2) {
        return;
    }
    seg_len = std::max(1, std::min(seg_len, block_len));
    if (seg_len >= block_len) {
        return;
    }

    int seg_start = rand() % (block_len - seg_len + 1);
    int remaining = block_len - seg_len;
    int insert_pos = rand() % (remaining + 1);

    std::vector<double> segment(seg_len);
    std::vector<double> temp;
    temp.reserve(remaining);

    for (int s = 0; s < seg_len; ++s) {
        segment[s] = values[seg_start + s];
    }
    for (int j = 0; j < block_len; ++j) {
        if (j < seg_start || j >= seg_start + seg_len) {
            temp.push_back(values[j]);
        }
    }

    int wi = 0;
    for (int j = 0; j < block_len; ++j) {
        if (j >= insert_pos && j < insert_pos + seg_len) {
            values[j] = segment[j - insert_pos];
        } else {
            values[j] = temp[wi++];
        }
    }
}

void apply_vns_neighborhood(double* values, int block_len, int k) {
    if (values == nullptr || block_len < 2) {
        return;
    }

    if (k == 1) {
        int j1 = rand() % block_len;
        int j2 = rand() % block_len;
        if (j1 != j2) {
            std::swap(values[j1], values[j2]);
        }
        return;
    }

    if (k == 2) {
        if (block_len < 2) {
            return;
        }
        int j = rand() % (block_len - 1);
        std::swap(values[j], values[j + 1]);
        return;
    }

    if (k == 3) {
        int seg_len = std::min(block_len, 2 + rand() % 3);
        int start = rand() % (block_len - seg_len + 1);
        std::reverse(values + start, values + start + seg_len);
        return;
    }

    if (k == 4) {
        int seg_len = std::min(block_len, 2 + rand() % 2);
        segment_shift_values(values, block_len, seg_len);
        return;
    }

    int seg_len = std::min(block_len, 3 + rand() % 3);
    int start = rand() % (block_len - seg_len + 1);
    for (int t = seg_len - 1; t > 0; --t) {
        int r = rand() % (t + 1);
        std::swap(values[start + t], values[start + r]);
    }
}

}  // namespace

//=============================================================================
// CC_HIHH_Solver Implementation
//=============================================================================

CC_HIHH_Solver::CC_HIHH_Solver(MultiMet* s, int psize, int nsub, int ncircle, double pelite)
    : solver(s), popsize(psize), nSubpop(nsub), nCircle(ncircle), pElitist(pelite)
{
    CE_Tnum = solver->CE_Tnum;
    M_Jnum = solver->M_Jnum;
    M_OPTnum = solver->M_OPTnum;
    ops = M_Jnum * M_OPTnum;
    Nvar = 2 * CE_Tnum + 2 * ops;
    
    lambda = 0.1;  // Diversity bonus weight
    stagnation_count = 0;
    prev_gbest_fit = 1e30;
    gbest_fit = 1e30;

    // Contextual bandit defaults
    max_generations = 1;
    state_dim = 7;        // gen_ratio, stagnation, improve_recent, diversity, success_rate, last_reward, bias
    recent_k = 10;
    lr0 = 0.05;
    epsilon0 = 0.2;
    epsilon_decay = 0.995;
    epsilon_min = 0.02;
    epsilon_k = 0.01;
    lr_k = 0.002;
    reward_clip = 2.0;
    stable_reward_clip = 0.2;
    seq_swap_count = 3;
    resample_rate = 0.2;
    stable_mode = false;
    gate_enabled = true;
    resample_gate = 15;
    gate_blocked_total = 0;
    gate_fallback_total = 0;

    // Lévy Flight parameters
    levy_beta = 1.5;
    levy_step_coeff = 0.01;
    levy_dim_ratio = 0.1;

    // VNS parameters
    vns_max_k = 5;
    vns_samples_per_k = 5;
    vns_elite_ratio = 0.2;
    
    // Operator counts: GA, DE for offload; GA, SWAP for seq; DE, GDE for dev
    num_ops_offload = 4;  // GA, DE, BITFLIP, BLOCK_RESAMPLE
    num_ops_seq = 4;      // GA, SEQ_SWAP, VNS, BLOCK_RESAMPLE
    num_ops_dev = 4;      // DE, GDE, LEVY, BLOCK_RESAMPLE
    num_ops_full = 5;     // GA, DE, GDE, LEVY, BLOCK_RESAMPLE

    use_blocks = true;
    enable_intra_migration = true;
    use_bandit = true;
    shared_bandit_mode = false;
    fixed_ops_per_block = false;
    op_selection_mode = MODE_CONTEXTUAL_BANDIT;
    round_robin_counters.assign(4, 0);

    op_stats_enabled = false;
    op_stats_every = 50;
    op_stats_interval_gens = 0;

    weight_log_enabled = false;
    weight_log_every = 50;
    reward_log_enabled = false;
    reward_variance_log_enabled = false;
    global_stats_enabled = false;
    global_stats_every = 50;
    diversity_log_enabled = false;
    diversity_log_every = 50;
    diversity_log_config_name.clear();
    
    var_full.resize(Nvar);
    gbest.resize(Nvar);
}

CC_HIHH_Solver::~CC_HIHH_Solver()
{
    CloseOpStats();
    CloseWeightLogging();
    CloseRewardLogging();
    CloseRewardVarianceLogging();
    CloseGlobalStatsLogging();
    CloseDiversityLogging();
    P_offload.destroy();
    P_seq.destroy();
    P_dev.destroy();
    P_full.destroy();
}

void CC_HIHH_Solver::Init()
{
    std::fill(round_robin_counters.begin(), round_robin_counters.end(), 0);
    if (nSubpop < 1) {
        std::cerr << "[CC-HIHH] Error: nSubpop must be >= 1" << std::endl;
        std::exit(1);
    }
    if (enable_intra_migration && nSubpop < 1) {
        std::cerr << "[CC-HIHH] Error: nSubpop must be >= 1 when migration is enabled" << std::endl;
        std::exit(1);
    }
    
    std::cout << "[CC-HIHH] Initializing solver..." << std::endl;
    std::cout << "  CE_Tnum=" << CE_Tnum << ", M_Jnum=" << M_Jnum 
              << ", M_OPTnum=" << M_OPTnum << ", ops=" << ops << std::endl;
    std::cout << "  Nvar=" << Nvar << ", popsize=" << popsize 
              << ", nSubpop=" << nSubpop << std::endl;

    if (!use_blocks) {
        shared_cb_selectors.clear();
        int full_start = 0;
        int full_len = Nvar;
        P_full.init(0, full_start, full_len, popsize, nSubpop, num_ops_full, state_dim, recent_k);
        archive_full.init(std::max(1, popsize / nSubpop), full_len);
        archive_offload.init(0, 0);
        archive_dev.init(0, 0);

        for (int i = 0; i < popsize && i < solver->Popsize; i++) {
            std::copy(solver->pop[i], solver->pop[i] + full_len, P_full.pop[i]);
        }

        std::cout << "  Full-space block: len=" << full_len << std::endl;

        std::cout << "[CC-HIHH] Initial evaluation..." << std::endl;
        EvaluateBlock(P_full);

        P_full.update_block_gbest();
        P_full.update_island_gbest();

        gbest_fit = P_full.block_gbest_fit;
        std::copy(P_full.block_gbest, P_full.block_gbest + full_len, gbest.begin());

        std::cout << "[CC-HIHH] Initial gbest_fit = " << gbest_fit << std::endl;

        if (op_stats_enabled) InitOpStats();
        if (weight_log_enabled) InitWeightLogging();
        if (reward_log_enabled) InitRewardLogging();
        if (reward_variance_log_enabled) InitRewardVarianceLogging();
        if (global_stats_enabled) InitGlobalStatsLogging();
        if (diversity_log_enabled) InitDiversityLogging();
        return;
    }

    if (shared_bandit_mode) {
        const int shared_ops = std::max(num_ops_offload, std::max(num_ops_seq, num_ops_dev));
        shared_cb_selectors.resize(nSubpop);
        shared_snapshots.resize(nSubpop);
        shared_island_deferred.resize(nSubpop);
        for (int i = 0; i < nSubpop; i++) {
            shared_cb_selectors[i].init(shared_ops, state_dim);
            shared_snapshots[i].clear();
            shared_island_deferred[i].clear();
        }
    } else {
        shared_cb_selectors.clear();
        shared_snapshots.clear();
        shared_island_deferred.clear();
    }
    
    // Initialize context
    context.init(CE_Tnum, M_Jnum, M_OPTnum);
    
    // Block 0 (Offload): [0, 2*CE_Tnum)
    int offload_start = 0;
    int offload_len = 2 * CE_Tnum;
    P_offload.init(0, offload_start, offload_len, popsize, nSubpop, num_ops_offload, state_dim, recent_k);
    
    // Block 1 (Sequence): [2*CE_Tnum, 2*CE_Tnum + ops)
    int seq_start = 2 * CE_Tnum;
    int seq_len = ops;
    P_seq.init(1, seq_start, seq_len, popsize, nSubpop, num_ops_seq, state_dim, recent_k);
    
    // Block 2 (Device): [2*CE_Tnum + ops, 2*CE_Tnum + 2*ops)
    int dev_start = 2 * CE_Tnum + ops;
    int dev_len = ops;
    P_dev.init(2, dev_start, dev_len, popsize, nSubpop, num_ops_dev, state_dim, recent_k);
    int archive_size = std::max(1, popsize / nSubpop);
    archive_offload.init(archive_size, offload_len);
    archive_dev.init(archive_size, dev_len);
    archive_full.init(0, 0);
    
    std::cout << "  Block 0 (offload): start=" << offload_start << ", len=" << offload_len << std::endl;
    std::cout << "  Block 1 (seq): start=" << seq_start << ", len=" << seq_len << std::endl;
    std::cout << "  Block 2 (dev): start=" << dev_start << ", len=" << dev_len << std::endl;
    
    // Copy initial populations from solver's initialized population
    // This leverages the heuristic initialization from MultiMet::Initial()
    for (int i = 0; i < popsize && i < solver->Popsize; i++) {
        // Extract block portions from solver's pop
        std::copy(solver->pop[i], solver->pop[i] + offload_len, P_offload.pop[i]);
        std::copy(solver->pop[i] + seq_start, solver->pop[i] + seq_start + seq_len, P_seq.pop[i]);
        std::copy(solver->pop[i] + dev_start, solver->pop[i] + dev_start + dev_len, P_dev.pop[i]);
    }
    
    // Initialize context with first individual
    context.update_block(0, P_offload.pop[0]);
    context.update_block(1, P_seq.pop[0]);
    context.update_block(2, P_dev.pop[0]);
    
    // Initial evaluation of all blocks
    std::cout << "[CC-HIHH] Initial evaluation..." << std::endl;
    EvaluateBlock(P_offload);
    EvaluateBlock(P_seq);
    EvaluateBlock(P_dev);
    
    // Update block gbest and context
    P_offload.update_block_gbest();
    P_seq.update_block_gbest();
    P_dev.update_block_gbest();
    P_offload.update_island_gbest();
    P_seq.update_island_gbest();
    P_dev.update_island_gbest();
    
    context.update_block(0, P_offload.block_gbest);
    context.update_block(1, P_seq.block_gbest);
    context.update_block(2, P_dev.block_gbest);
    
    // Compute global best
    context.get_full(var_full.data());
    gbest_fit = solver->Eval(var_full.data());
    std::copy(var_full.begin(), var_full.end(), gbest.begin());
    
    std::cout << "[CC-HIHH] Initial gbest_fit = " << gbest_fit << std::endl;

    if (op_stats_enabled) InitOpStats();
    if (weight_log_enabled) InitWeightLogging();
    if (reward_log_enabled) InitRewardLogging();
    if (reward_variance_log_enabled) InitRewardVarianceLogging();
    if (global_stats_enabled) InitGlobalStatsLogging();
    if (diversity_log_enabled) InitDiversityLogging();
}

void CC_HIHH_Solver::RunGeneration(int gen)
{
    if (!use_blocks) {
        BlockPopulation& bp = P_full;
        std::vector<double> generation_rewards;
        if (reward_variance_log_enabled) generation_rewards.reserve(nSubpop);
        for (int isl = 0; isl < nSubpop; isl++) {
            int p_start, p_end;
            bp.get_island_range(isl, p_start, p_end);

            double old_best_fit = bp.get_island_best_fit(p_start, p_end);
            double old_diversity = bp.compute_diversity(p_start, p_end);
            bool gate_blocks_resample = ShouldBlockResample(bp, isl, gen, old_diversity);

            int op_sel = 0;
            std::array<double, 7> state{};
            if (op_selection_mode == MODE_CONTEXTUAL_BANDIT && use_bandit) {
                state = ComputeState(bp, isl, gen, old_diversity);
                double eps = ComputeEpsilon(gen);
                op_sel = bp.cb_selectors[isl].select(state, eps);
            } else if (op_selection_mode == MODE_ROUND_ROBIN) {
                op_sel = SelectOperatorRoundRobin(-1);
            } else {
                op_sel = SelectOperatorRandom(-1);
            }
            int op_exec = op_sel;
            if (gate_blocks_resample && op_sel == FULL_OP_BLOCK_RESAMPLE) {
                gate_blocked_total++;
                op_exec = SelectBestNonResampleOp(-1, state, &bp.cb_selectors[isl], nullptr);
                ApplyGateBlockedPenalty(&bp.cb_selectors[isl], state, op_sel,
                                        (stable_mode ? stable_reward_clip : reward_clip) * 0.25);
                gate_fallback_total++;
            }

            bp.copy_pop_to_newpop(p_start, p_end);
            ApplyOperatorFull(op_exec, bp, p_start, p_end);
            EvaluateBlockIsland(bp, p_start, p_end);
            SelectionUpdate(bp, p_start, p_end);

            double new_best_fit = bp.get_island_best_fit(p_start, p_end);
            double new_diversity = bp.compute_diversity(p_start, p_end);
            double improvement_ratio = (old_best_fit - new_best_fit) / (std::abs(old_best_fit) + 1e-9);
            double diversity_bonus = lambda * (new_diversity - old_diversity);
            double reward = improvement_ratio + diversity_bonus;

            double clip_val = stable_mode ? stable_reward_clip : reward_clip;
            if (reward > clip_val) reward = clip_val;
            if (reward < -clip_val) reward = -clip_val;

            if (op_selection_mode == MODE_CONTEXTUAL_BANDIT && use_bandit) {
                double lr = ComputeLearningRate(bp.cb_selectors[isl]);
                bp.cb_selectors[isl].update(state, op_exec, reward, lr);
            }
            LogReward(gen + 1, -1, isl, op_exec, reward, improvement_ratio, (new_diversity - old_diversity));
            if (reward_variance_log_enabled) generation_rewards.push_back(reward);

            int success = (new_best_fit + 1e-12 < old_best_fit) ? 1 : 0;
            double improve_norm = improvement_ratio;
            if (improve_norm < 0.0) improve_norm = 0.0;
            if (improve_norm > 1.0) improve_norm = 1.0;
            bp.update_recent_stats(isl, success, improve_norm);
            bp.island_last_reward[isl] = reward;
            bp.island_last_op[isl] = op_exec;
            if (success) {
                bp.island_stagnation[isl] = 0;
            } else {
                bp.island_stagnation[isl]++;
            }
            bp.island_last_best_fit[isl] = new_best_fit;

            RecordOpSelection(-1, op_exec);
        }

        bp.update_island_gbest();
        bp.update_block_gbest();

        if (enable_intra_migration && nSubpop > 1 && gen > 0 && gen % nCircle == 0) {
            int dispara = (gen / nCircle - 1) % (nSubpop - 1) + 1;
            MigrationWithinBlock(bp, dispara);
        }

        if (bp.block_gbest_fit < gbest_fit) {
            gbest_fit = bp.block_gbest_fit;
            std::copy(bp.block_gbest, bp.block_gbest + bp.block_len, gbest.begin());
            stagnation_count = 0;
        } else {
            stagnation_count++;
        }

        prev_gbest_fit = gbest_fit;
        LogRewardVariance(gen + 1, -1, generation_rewards);
        return;
    }

    // Process each block in round-robin fashion
    BlockPopulation* blocks[3] = {&P_offload, &P_seq, &P_dev};
    const char* block_names[3] = {"offload", "seq", "dev"};

    // For shared bandit: record assembled fitness BEFORE block processing
    // so we can compute a global reward (credit-assignment-free) at end of gen.
    double old_assembled_fit = 0.0;
    if (shared_bandit_mode) {
        context.get_full(var_full.data());
        old_assembled_fit = solver->Eval(var_full.data());
        for (int i = 0; i < nSubpop; i++) {
            shared_snapshots[i].take_snapshot(shared_cb_selectors[i]);
            shared_island_deferred[i].clear();
        }
    }
    
    for (int b = 0; b < 3; b++) {
        BlockPopulation& bp = *blocks[b];
        std::vector<double> block_rewards;
        if (reward_variance_log_enabled) block_rewards.reserve(nSubpop);
        
        // Process each island
        for (int isl = 0; isl < nSubpop; isl++) {
            int p_start, p_end;
            bp.get_island_range(isl, p_start, p_end);
            
            // Record pre-application statistics
            double old_best_fit = bp.get_island_best_fit(p_start, p_end);
            double old_diversity = bp.compute_diversity(p_start, p_end);
            bool gate_blocks_resample = ShouldBlockResample(bp, isl, gen, old_diversity);
            
            // Build contextual state and select operator
            std::array<double, 7> state{};
            int op_sel = 0;
            bool allow_bandit = (op_selection_mode == MODE_CONTEXTUAL_BANDIT) && use_bandit && !fixed_ops_per_block;
            ContextualBanditSelector* selector = nullptr;
            if (fixed_ops_per_block) {
                if (bp.block_id == 0) op_sel = OFF_OP_GA;
                else if (bp.block_id == 1) op_sel = SEQ_OP_GA;
                else op_sel = DEV_OP_DE;
            } else if (allow_bandit) {
                state = ComputeState(bp, isl, gen, old_diversity);
                double eps = ComputeEpsilon(gen);
                if (shared_bandit_mode) {
                    int n_ops_block = (bp.block_id == 0) ? num_ops_offload : ((bp.block_id == 1) ? num_ops_seq : num_ops_dev);
                    selector = &shared_cb_selectors[isl];
                    op_sel = shared_snapshots[isl].select_from_snapshot(state, eps, n_ops_block);
                } else {
                    selector = &bp.cb_selectors[isl];
                    op_sel = selector->select(state, eps);
                }
            } else if (op_selection_mode == MODE_ROUND_ROBIN) {
                op_sel = SelectOperatorRoundRobin(bp.block_id);
            } else {
                op_sel = SelectOperatorRandom(bp.block_id);
            }
            int op_exec = op_sel;
            if (gate_blocks_resample && IsResampleOp(op_sel, bp.block_id)) {
                gate_blocked_total++;
                op_exec = SelectBestNonResampleOp(bp.block_id, state, selector,
                                                  shared_bandit_mode ? &shared_snapshots[isl] : nullptr);
                if (!shared_bandit_mode) {
                    double gate_penalty_scale = 1.0;
                    if (bp.block_len >= 2000) {
                        gate_penalty_scale = 0.20;
                    } else if (bp.block_len >= 800) {
                        gate_penalty_scale = 0.60;
                    }
                    ApplyGateBlockedPenalty(selector, state, op_sel,
                                            (stable_mode ? stable_reward_clip : reward_clip) * gate_penalty_scale);
                }
                gate_fallback_total++;
            }
            
            // Copy pop to newpop for operator to work on
            bp.copy_pop_to_newpop(p_start, p_end);
            
            // Apply selected operator
            ApplyOperator(op_exec, bp, p_start, p_end);
            
            // Evaluate newpop with context assembly
            EvaluateBlockIsland(bp, p_start, p_end);
            
            // Selection: update pop where newpop is better
            SelectionUpdate(bp, p_start, p_end);
            
            // Compute post-application statistics
            double new_best_fit = bp.get_island_best_fit(p_start, p_end);
            double new_diversity = bp.compute_diversity(p_start, p_end);
            
            // Compute reward for contextual bandit
            double improvement_ratio = (old_best_fit - new_best_fit) / (std::abs(old_best_fit) + 1e-9);
            double diversity_bonus = lambda * (new_diversity - old_diversity);
            double reward = improvement_ratio + diversity_bonus;

            // Clip reward for stability
            double clip_val = stable_mode ? stable_reward_clip : reward_clip;
            if (reward > clip_val) reward = clip_val;
            if (reward < -clip_val) reward = -clip_val;

            if (allow_bandit && selector != nullptr) {
                if (shared_bandit_mode) {
                    shared_island_deferred[isl].push_back({state, op_exec, reward});
                } else {
                    double lr = ComputeLearningRate(*selector);
                    selector->update(state, op_exec, reward, lr);
                }
            }
            LogReward(gen + 1, bp.block_id, isl, op_exec, reward, improvement_ratio, (new_diversity - old_diversity));
            if (reward_variance_log_enabled) block_rewards.push_back(reward);

            int success = (new_best_fit + 1e-12 < old_best_fit) ? 1 : 0;
            double improve_norm = improvement_ratio;
            if (improve_norm < 0.0) improve_norm = 0.0;
            if (improve_norm > 1.0) improve_norm = 1.0;
            bp.update_recent_stats(isl, success, improve_norm);
            bp.island_last_reward[isl] = reward;
            bp.island_last_op[isl] = op_exec;
            if (success) {
                bp.island_stagnation[isl] = 0;
            } else {
                bp.island_stagnation[isl]++;
            }
            bp.island_last_best_fit[isl] = new_best_fit;

            RecordOpSelection(bp.block_id, op_exec);
        }

#if HIHH_CB_DEBUG
        if ((gen + 1) % HIHH_CB_DEBUG_INTERVAL == 0) {
            double eps_dbg = ComputeEpsilon(gen);
            for (int isl = 0; isl < nSubpop; isl++) {
                int p_start, p_end;
                bp.get_island_range(isl, p_start, p_end);
                double div_now = bp.compute_diversity(p_start, p_end);
                std::array<double, 7> state_dbg = ComputeState(bp, isl, gen, div_now);
                const ContextualBanditSelector& dbg_sel = shared_bandit_mode ? shared_cb_selectors[isl] : bp.cb_selectors[isl];
                double lr_dbg = ComputeLearningRate(dbg_sel);
                std::cout << "[CB][Gen " << (gen + 1) << "][Block " << b << "][Island " << isl
                          << "] stag=" << bp.island_stagnation[isl]
                          << " eps=" << eps_dbg
                          << " lr=" << lr_dbg
                          << " gate_blocked=" << gate_blocked_total
                          << " gate_fallback=" << gate_fallback_total
                          << " ";
                for (int op = 0; op < (int)dbg_sel.selection_count.size(); op++) {
                    int cnt = dbg_sel.selection_count[op];
                    double avg = (cnt > 0) ? dbg_sel.total_reward[op] / cnt : 0.0;
                    double sc = dbg_sel.score_op(op, state_dbg);
                    std::cout << "op" << op << ":cnt=" << cnt << ",avg=" << avg << ",score=" << sc << " ";
                }
                std::cout << std::endl;
            }
        }
#endif
        
        // Update island and block gbest
        bp.update_island_gbest();
        bp.update_block_gbest();
        
        // Intra-block migration every nCircle generations
        if (enable_intra_migration && nSubpop > 1 && gen > 0 && gen % nCircle == 0) {
            int dispara = (gen / nCircle - 1) % (nSubpop - 1) + 1;
            MigrationWithinBlock(bp, dispara);
        }
        
        // Update context for this block
        context.update_block(bp.block_id, bp.block_gbest);
        LogRewardVariance(gen + 1, bp.block_id, block_rewards);
    }

    if (shared_bandit_mode) {
        // Compute global reward: assembled fitness change over the generation.
        // This deliberately removes per-block credit assignment — the shared
        // bandit cannot tell which block's operator caused the improvement.
        context.get_full(var_full.data());
        double new_assembled_fit = solver->Eval(var_full.data());
        double global_reward = (old_assembled_fit - new_assembled_fit)
                             / (std::abs(old_assembled_fit) + 1e-9);
        double clip_val = stable_mode ? stable_reward_clip : reward_clip;
        if (global_reward > clip_val) global_reward = clip_val;
        if (global_reward < -clip_val) global_reward = -clip_val;

        for (int i = 0; i < nSubpop; i++) {
            for (auto& upd : shared_island_deferred[i]) {
                upd.reward = global_reward;   // override per-block reward
            }
            for (const auto& upd : shared_island_deferred[i]) {
                double lr = ComputeLearningRate(shared_cb_selectors[i]);
                shared_cb_selectors[i].update(upd.state, upd.op, upd.reward, lr);
            }
            shared_island_deferred[i].clear();
        }
    }

    // Update global best
    context.get_full(var_full.data());
    double current_fit = solver->Eval(var_full.data());
    
    if (current_fit < gbest_fit) {
        gbest_fit = current_fit;
        std::copy(var_full.begin(), var_full.end(), gbest.begin());
        stagnation_count = 0;
    } else {
        stagnation_count++;
    }

    prev_gbest_fit = gbest_fit;
}

double CC_HIHH_Solver::ComputeEpsilon(int gen) const
{
    if (!stable_mode) {
        double eps = epsilon0 * std::pow(epsilon_decay, gen);
        if (eps < epsilon_min) eps = epsilon_min;
        return eps;
    }
    double eps = epsilon0 * std::exp(-epsilon_k * gen);
    if (eps < epsilon_min) eps = epsilon_min;
    return eps;
}

double CC_HIHH_Solver::ComputeLearningRate(const ContextualBanditSelector& sel) const
{
    double t = (double)sel.total_updates + 1.0;
    if (!stable_mode) {
        return lr0 / std::sqrt(t);
    }
    return lr0 * std::exp(-lr_k * t);
}

bool CC_HIHH_Solver::IsResampleOp(int op, int block_id) const
{
    if (block_id == 0) return op == OFF_OP_BLOCK_RESAMPLE;
    if (block_id == 1) return op == SEQ_OP_BLOCK_RESAMPLE;
    return op == DEV_OP_BLOCK_RESAMPLE;
}

bool CC_HIHH_Solver::ShouldBlockResample(const BlockPopulation& bp, int isl, int gen, double diversity) const
{
    if (!gate_enabled) return false;
    if (isl < 0 || isl >= (int)bp.island_stagnation.size()) return false;

    double success_rate = 0.0;
    if (bp.recent_count[isl] > 0) {
        success_rate = (double)bp.recent_success_count[isl] / (double)bp.recent_count[isl];
    }
    const double diversity_norm = diversity / (diversity + 1.0);
    const double gen_ratio = (max_generations > 0)
        ? (double)(gen + 1) / (double)max_generations
        : 0.0;

    if (bp.block_len >= 2000) {
        const int warmup_generations = std::max(5, max_generations / 50);
        if (gen < warmup_generations && diversity_norm >= 0.05) {
            return true;
        }
        if (gen_ratio < 0.06 &&
            bp.island_stagnation[isl] < std::max(2, resample_gate / 4) &&
            success_rate >= 0.30 &&
            diversity_norm >= 0.15) {
            return true;
        }
        return false;
    }

    if (bp.block_len >= 800) {
        const bool offload_medium = (bp.block_id == 0);
        const int warmup_generations = offload_medium
            ? std::max(resample_gate * 4, max_generations / 12)
            : std::max(resample_gate * 10, max_generations / 8);
        if (gen < warmup_generations) {
            return true;
        }
        if (gen_ratio < (offload_medium ? 0.22 : 0.35) &&
            bp.island_stagnation[isl] < (offload_medium ? std::max(5, (2 * resample_gate) / 3) : resample_gate) &&
            success_rate >= (offload_medium ? 0.18 : 0.15) &&
            diversity_norm >= (offload_medium ? 0.10 : 0.08)) {
            return true;
        }
        return false;
    }

    const int warmup_generations = std::max(resample_gate * 20, max_generations / 6);
    if (gen < warmup_generations) {
        return true;
    }
    if (bp.island_stagnation[isl] < resample_gate &&
        success_rate >= 0.10 &&
        diversity_norm >= 0.06) {
        return true;
    }
    return false;
}

int CC_HIHH_Solver::SelectBestNonResampleOp(int block_id,
                                            const std::array<double, 7>& state,
                                            ContextualBanditSelector* selector,
                                            const BanditSnapshot* snapshot) const
{
    if (block_id < 0) {
        if (op_selection_mode == MODE_CONTEXTUAL_BANDIT && use_bandit && selector != nullptr) {
            return selector->select_best_masked(state, FULL_OP_BLOCK_RESAMPLE);
        }
        if (op_selection_mode == MODE_ROUND_ROBIN) {
            return FULL_OP_DE;
        }
        return rand() % FULL_OP_BLOCK_RESAMPLE;
    }

    const int excluded = 3;
    if (op_selection_mode == MODE_CONTEXTUAL_BANDIT && use_bandit && !fixed_ops_per_block) {
        if (shared_bandit_mode && snapshot != nullptr) {
            const int n_ops_block = (block_id == 0) ? num_ops_offload : ((block_id == 1) ? num_ops_seq : num_ops_dev);
            return snapshot->select_best_from_snapshot_masked(state, n_ops_block, excluded);
        }
        if (selector != nullptr) {
            return selector->select_best_masked(state, excluded);
        }
    }

    if (fixed_ops_per_block) {
        if (block_id == 0) return OFF_OP_GA;
        if (block_id == 1) return SEQ_OP_GA;
        return DEV_OP_DE;
    }

    if (op_selection_mode == MODE_ROUND_ROBIN) {
        if (block_id == 0) return OFF_OP_DE;
        if (block_id == 1) return SEQ_OP_SWAP;
        return DEV_OP_GDE;
    }

    return rand() % 3;
}

void CC_HIHH_Solver::ApplyGateBlockedPenalty(ContextualBanditSelector* selector,
                                             const std::array<double, 7>& state,
                                             int blocked_op,
                                             double clip_val)
{
    if (selector == nullptr) return;
    if (op_selection_mode != MODE_CONTEXTUAL_BANDIT || !use_bandit) return;
    if (blocked_op < 0 || blocked_op >= selector->num_ops) return;

    // Teach the bandit that resample is currently unavailable without letting
    // the fixed fallback dominate the learning signal.
    const double penalty = -0.5 * std::max(clip_val, 1e-6);
    const double lr = 0.5 * ComputeLearningRate(*selector);
    selector->update(state, blocked_op, penalty, lr);
}

std::array<double, 7> CC_HIHH_Solver::ComputeState(const BlockPopulation& bp, int isl, int gen, double diversity) const
{
    std::array<double, 7> s{};
    double gen_ratio = (max_generations > 0) ? (double)gen / (double)max_generations : 0.0;
    if (gen_ratio < 0.0) gen_ratio = 0.0;
    if (gen_ratio > 1.0) gen_ratio = 1.0;

    double stagnation_cap = 0.25 * (double)max_generations + 1.0;
    double stagnation_norm = (stagnation_cap > 0.0) ? (double)bp.island_stagnation[isl] / stagnation_cap : 0.0;
    if (stagnation_norm > 1.0) stagnation_norm = 1.0;
    if (stagnation_norm < 0.0) stagnation_norm = 0.0;

    double improve_recent = 0.0;
    if (bp.recent_count[isl] > 0) {
        improve_recent = bp.recent_improve_sum[isl] / (double)bp.recent_count[isl];
    }
    if (improve_recent < 0.0) improve_recent = 0.0;
    if (improve_recent > 1.0) improve_recent = 1.0;

    double diversity_norm = diversity / (diversity + 1.0);
    if (diversity_norm < 0.0) diversity_norm = 0.0;
    if (diversity_norm > 1.0) diversity_norm = 1.0;

    double success_rate = 0.0;
    if (bp.recent_count[isl] > 0) {
        success_rate = (double)bp.recent_success_count[isl] / (double)bp.recent_count[isl];
    }
    if (success_rate < 0.0) success_rate = 0.0;
    if (success_rate > 1.0) success_rate = 1.0;

    double last_reward_norm = 0.5 + 0.5 * std::tanh(bp.island_last_reward[isl]);
    if (last_reward_norm < 0.0) last_reward_norm = 0.0;
    if (last_reward_norm > 1.0) last_reward_norm = 1.0;

    s[0] = gen_ratio;
    s[1] = stagnation_norm;
    s[2] = improve_recent;
    s[3] = diversity_norm;
    s[4] = success_rate;
    s[5] = last_reward_norm;
    s[6] = 1.0;  // bias
    return s;
}

void CC_HIHH_Solver::MigrationWithinBlock(BlockPopulation& bp, int dispara)
{
    if (bp.nSubpop <= 1) {
        return;
    }

    // Ring migration: island k receives from (k - dispara + nSubpop) mod nSubpop
    const size_t needed = (size_t)nSubpop * (size_t)bp.block_len;
    if (migration_buffer.size() < needed) migration_buffer.resize(needed);
    if ((int)migration_fit_buffer.size() < nSubpop) migration_fit_buffer.resize(nSubpop);
    
    // Prepare migrants
    for (int k = 0; k < nSubpop; k++) {
        double* migrant = migration_buffer.data() + (size_t)k * (size_t)bp.block_len;
        
        if (randval(0, 1) < pElitist) {
            // Send island gbest
            std::copy(bp.island_gbest[k], bp.island_gbest[k] + bp.block_len, migrant);
            migration_fit_buffer[k] = bp.island_gbest_fit[k];
        } else {
            // Send random individual from island
            int p_start, p_end;
            bp.get_island_range(k, p_start, p_end);
            int rand_idx = p_start + rand() % (p_end - p_start);
            std::copy(bp.pop[rand_idx], bp.pop[rand_idx] + bp.block_len, migrant);
            migration_fit_buffer[k] = bp.pop_fit[rand_idx];
        }
    }
    
    // Perform migration
    for (int k = 0; k < nSubpop; k++) {
        int source = ((k - dispara) % nSubpop + nSubpop) % nSubpop;
        int p_start, p_end;
        bp.get_island_range(k, p_start, p_end);
        int worst_idx = bp.get_island_worst_idx(p_start, p_end);
        
        // Replace worst with migrant
        const double* migrant = migration_buffer.data() + (size_t)source * (size_t)bp.block_len;
        std::copy(migrant, migrant + bp.block_len, bp.pop[worst_idx]);
        bp.pop_fit[worst_idx] = migration_fit_buffer[source];
    }
}

void CC_HIHH_Solver::ApplyOperator(int op, BlockPopulation& bp, int p_start, int p_end)
{
    // Map operator ID based on block type
    if (bp.block_id == 0) {  // Offload block
        switch (op) {
            case OFF_OP_GA: ApplyGA(bp, p_start, p_end); break;
            case OFF_OP_DE: ApplyDE(bp, p_start, p_end); break;
            case OFF_OP_BITFLIP: ApplyBitFlip(bp, p_start, p_end); break;
            case OFF_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
            default: ApplyGA(bp, p_start, p_end); break;
        }
    } else if (bp.block_id == 1) {  // Sequence block
        switch (op) {
            case SEQ_OP_GA: ApplyGA(bp, p_start, p_end); break;
            case SEQ_OP_SWAP: ApplySeqSwap(bp, p_start, p_end, seq_swap_count); break;
            case SEQ_OP_VNS: ApplyVNS(bp, p_start, p_end); break;
            case SEQ_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
            default: ApplyGA(bp, p_start, p_end); break;
        }
    } else {  // Device block
        switch (op) {
            case DEV_OP_DE: ApplyDE(bp, p_start, p_end); break;
            case DEV_OP_GDE: ApplyGDE(bp, p_start, p_end); break;
            case DEV_OP_LEVY: ApplyLevy(bp, p_start, p_end); break;
            case DEV_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
            default: ApplyDE(bp, p_start, p_end); break;
        }
    }
}

void CC_HIHH_Solver::ApplyGA(BlockPopulation& bp, int p_start, int p_end)
{
    const int island_size = p_end - p_start;
    const int block_len = bp.block_len;
    if (block_len <= 0 || island_size <= 0) {
        return;
    }

    const double eta_c = 20.0;
    const double eta_m = 20.0;
    const double pc = 0.9;
    const double pm = 1.0 / static_cast<double>(block_len);

    for (int i = p_start; i < p_end; i++) {
        int t1 = p_start + rand() % island_size;
        int t2 = p_start + rand() % island_size;
        int winner = (bp.pop_fit[t1] < bp.pop_fit[t2]) ? t1 : t2;
        std::copy(bp.pop[winner], bp.pop[winner] + block_len, bp.newpop[i]);
    }

    for (int i = p_start; i < p_end; i++) {
        if (randval(0, 1) < pc) {
            int partner = p_start + rand() % island_size;
            if (partner == i) {
                continue;
            }

            for (int j = 0; j < block_len; j++) {
                if (randval(0, 1) < 0.5) {
                    double u = unit_open(randval(0, 1));
                    double beta_q = (u <= 0.5)
                        ? std::pow(2.0 * u, 1.0 / (eta_c + 1.0))
                        : std::pow(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta_c + 1.0));
                    double c1 = 0.5 * ((1.0 + beta_q) * bp.newpop[i][j] + (1.0 - beta_q) * bp.newpop[partner][j]);
                    bp.newpop[i][j] = clip01(c1);
                }
            }
        }
    }

    for (int i = p_start; i < p_end; i++) {
        for (int j = 0; j < block_len; j++) {
            if (randval(0, 1) < pm) {
                double u = randval(0, 1);
                double delta_q = (u < 0.5)
                    ? std::pow(2.0 * u, 1.0 / (eta_m + 1.0)) - 1.0
                    : 1.0 - std::pow(2.0 * (1.0 - u), 1.0 / (eta_m + 1.0));
                bp.newpop[i][j] = clip01(bp.newpop[i][j] + delta_q);
            }
        }
    }
}

void CC_HIHH_Solver::ApplyDE(BlockPopulation& bp, int p_start, int p_end)
{
    const int island_size = p_end - p_start;
    if (island_size < 4) {
        ApplyGA(bp, p_start, p_end);
        return;
    }

    const double F = 0.5;
    const double CR = 0.5;
    const double p = 0.1;
    const int block_len = bp.block_len;
    DEArchive* archive = nullptr;
    if (!use_blocks) {
        archive = &archive_full;
    } else if (bp.block_id == 0) {
        archive = &archive_offload;
    } else if (bp.block_id == 2) {
        archive = &archive_dev;
    }

    std::vector<std::pair<double, int>> fit_idx;
    fit_idx.reserve(island_size);
    for (int i = p_start; i < p_end; ++i) {
        fit_idx.push_back({bp.pop_fit[i], i});
    }
    std::sort(fit_idx.begin(), fit_idx.end());
    int p_count = std::max(1, static_cast<int>(std::floor(p * island_size)));

    for (int i = p_start; i < p_end; i++) {
        int pbest_idx = fit_idx[rand() % p_count].second;
        int r1;
        do {
            r1 = p_start + rand() % island_size;
        } while (r1 == i);

        const double* r2_vals = nullptr;
        int archive_count = (archive != nullptr) ? archive->count : 0;
        double archive_prob = (archive_count > 0)
            ? static_cast<double>(archive_count) / static_cast<double>(island_size + archive_count)
            : 0.0;
        if (archive_count > 0 && randval(0, 1) < archive_prob) {
            r2_vals = archive->random_get();
        } else {
            int r2_idx;
            do {
                r2_idx = p_start + rand() % island_size;
            } while (r2_idx == i || r2_idx == r1);
            r2_vals = bp.pop[r2_idx];
        }

        int jrand = rand() % block_len;
        for (int j = 0; j < block_len; j++) {
            if (randval(0, 1) < CR || j == jrand) {
                double v = bp.pop[i][j]
                    + F * (bp.pop[pbest_idx][j] - bp.pop[i][j])
                    + F * (bp.pop[r1][j] - r2_vals[j]);
                bp.newpop[i][j] = clip01(v);
            } else {
                bp.newpop[i][j] = bp.pop[i][j];
            }
        }
    }
}

void CC_HIHH_Solver::ApplyGDE(BlockPopulation& bp, int p_start, int p_end)
{
    const int island_size = p_end - p_start;
    if (island_size < 3) {
        ApplyGA(bp, p_start, p_end);
        return;
    }

    const double F = 0.5;
    const double CR = 0.5;
    const double p = 0.15;
    const int block_len = bp.block_len;

    std::vector<std::pair<double, int>> fit_idx;
    fit_idx.reserve(island_size);
    for (int i = p_start; i < p_end; ++i) {
        fit_idx.push_back({bp.pop_fit[i], i});
    }
    std::sort(fit_idx.begin(), fit_idx.end());
    int p_count = std::max(1, static_cast<int>(std::floor(p * island_size)));

    for (int i = p_start; i < p_end; i++) {
        int pbest_idx = fit_idx[rand() % p_count].second;
        int cand1;
        int cand2;
        do {
            cand1 = p_start + rand() % island_size;
        } while (cand1 == i);
        do {
            cand2 = p_start + rand() % island_size;
        } while (cand2 == i || cand2 == cand1);
        int r1 = (bp.pop_fit[cand1] < bp.pop_fit[cand2]) ? cand1 : cand2;

        int r2;
        do {
            r2 = p_start + rand() % island_size;
        } while (r2 == i || r2 == r1);

        int jrand = rand() % block_len;
        for (int j = 0; j < block_len; j++) {
            if (randval(0, 1) < CR || j == jrand) {
                double v = bp.pop[i][j]
                    + F * (bp.pop[pbest_idx][j] - bp.pop[i][j])
                    + F * (bp.pop[r1][j] - bp.pop[r2][j]);
                bp.newpop[i][j] = clip01(v);
            } else {
                bp.newpop[i][j] = bp.pop[i][j];
            }
        }
    }
}

void CC_HIHH_Solver::ApplyBitFlip(BlockPopulation& bp, int p_start, int p_end)
{
    const double p_flip = 0.1;
    const double cauchy_scale = 0.1;
    const int half_len = bp.block_len / 2;

    for (int i = p_start; i < p_end; i++) {
        std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
        for (int j = 0; j < half_len; j++) {
            if (randval(0, 1) < p_flip) {
                bp.newpop[i][j] = (bp.newpop[i][j] < 0.5) ? 0.75 : 0.25;
            }
        }
        for (int j = half_len; j < bp.block_len; j++) {
            if (randval(0, 1) < p_flip) {
                double u = unit_open(randval(0, 1));
                double cauchy_sample = cauchy_scale * std::tan(kPi * (u - 0.5));
                bp.newpop[i][j] = clip01(bp.pop[i][j] + cauchy_sample);
            }
        }
    }
}

void CC_HIHH_Solver::ApplySeqSwap(BlockPopulation& bp, int p_start, int p_end, int n_swaps)
{
    if (bp.block_len < 2) {
        for (int i = p_start; i < p_end; ++i) {
            if (bp.block_len > 0) {
                std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
            }
        }
        return;
    }
    if (n_swaps < 1) {
        n_swaps = 1;
    }

    for (int i = p_start; i < p_end; i++) {
        std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
        for (int k = 0; k < n_swaps; k++) {
            int seg_len = 1 + rand() % 3;
            segment_shift_values(bp.newpop[i], bp.block_len, seg_len);
        }
    }
}

void CC_HIHH_Solver::ApplyOperatorFull(int op, BlockPopulation& bp, int p_start, int p_end)
{
    switch (op) {
        case FULL_OP_GA: ApplyGA(bp, p_start, p_end); break;
        case FULL_OP_DE: ApplyDE(bp, p_start, p_end); break;
        case FULL_OP_GDE: ApplyGDE(bp, p_start, p_end); break;
        case FULL_OP_LEVY: ApplyLevy(bp, p_start, p_end); break;
        case FULL_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
        default: ApplyGA(bp, p_start, p_end); break;
    }
}

void CC_HIHH_Solver::ApplyVNS(BlockPopulation& bp, int p_start, int p_end)
{
    if (bp.block_len < 2) {
        ApplySeqSwap(bp, p_start, p_end, seq_swap_count);
        return;
    }

    int island_size = p_end - p_start;
    int elite_count = std::max(1, (int)(island_size * vns_elite_ratio));

    std::vector<std::pair<double, int>> fit_idx;
    fit_idx.reserve(island_size);
    for (int i = p_start; i < p_end; i++) {
        fit_idx.push_back({bp.pop_fit[i], i});
    }
    std::sort(fit_idx.begin(), fit_idx.end());

    for (int e = 0; e < elite_count; e++) {
        int i = fit_idx[e].second;
        for (int k = 1; k <= vns_max_k; k++) {
            for (int sample = 0; sample < vns_samples_per_k; sample++) {
                std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
                apply_vns_neighborhood(bp.newpop[i], bp.block_len, k);
            }
        }
    }

    for (int e = elite_count; e < island_size; e++) {
        int i = fit_idx[e].second;
        std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
        for (int k = 0; k < seq_swap_count; k++) {
            int a = rand() % bp.block_len;
            int b = rand() % bp.block_len;
            if (a != b) {
                std::swap(bp.newpop[i][a], bp.newpop[i][b]);
            }
        }
    }
}

double CC_HIHH_Solver::LevyFlight(double beta)
{
    const double pi = 3.14159265358979323846;
    double sigma_u = std::pow(
        (std::tgamma(1.0 + beta) * std::sin(pi * beta / 2.0)) /
            (std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0)),
        1.0 / beta);

    double u = Rng::getInstance().normal(0.0, sigma_u);
    double v = Rng::getInstance().normal(0.0, 1.0);

    return u / std::pow(std::fabs(v), 1.0 / beta);
}

void CC_HIHH_Solver::ApplyLevy(BlockPopulation& bp, int p_start, int p_end)
{
    if (bp.block_len <= 0) return;

    int n_perturb = std::max(1, (int)(bp.block_len * levy_dim_ratio));

    for (int i = p_start; i < p_end; i++) {
        std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
        for (int k = 0; k < n_perturb; k++) {
            int j = rand() % bp.block_len;
            double u = unit_open(randval(0, 1));
            double cauchy_sample = std::tan(kPi * (u - 0.5));
            double new_val = bp.pop[i][j] + levy_step_coeff * cauchy_sample;
            bp.newpop[i][j] = clip01(new_val);
        }
    }
}

void CC_HIHH_Solver::ApplyBlockResample(BlockPopulation& bp, int p_start, int p_end, double rate)
{
    if (bp.block_len <= 0) return;
    if (rate < 0.0) rate = 0.0;
    if (rate > 1.0) rate = 1.0;
    int count = (int)std::round(rate * bp.block_len);
    if (count < 1) count = 1;

    for (int i = p_start; i < p_end; i++) {
        std::copy(bp.pop[i], bp.pop[i] + bp.block_len, bp.newpop[i]);
        for (int k = 0; k < count; k++) {
            int idx = rand() % bp.block_len;
            if (randval(0, 1) < 0.5) {
                bp.newpop[i][idx] = clip01(1.0 - bp.pop[i][idx]);
            } else {
                bp.newpop[i][idx] = clip01(randval(0, 1));
            }
        }
    }
}

void CC_HIHH_Solver::EvaluateBlock(BlockPopulation& bp)
{
    for (int i = 0; i < bp.popsize; i++) {
        if (use_blocks) {
            context.assemble_full(bp.block_id, bp.pop[i], var_full.data());
            bp.pop_fit[i] = solver->Eval(var_full.data());
        } else {
            bp.pop_fit[i] = solver->Eval(bp.pop[i]);
        }
    }
}

void CC_HIHH_Solver::EvaluateBlockIsland(BlockPopulation& bp, int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i++) {
        if (use_blocks) {
            context.assemble_full(bp.block_id, bp.newpop[i], var_full.data());
            bp.newpop_fit[i] = solver->Eval(var_full.data());
        } else {
            bp.newpop_fit[i] = solver->Eval(bp.newpop[i]);
        }
    }
}

void CC_HIHH_Solver::SelectionUpdate(BlockPopulation& bp, int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i++) {
        if (bp.newpop_fit[i] < bp.pop_fit[i]) {
            if (!use_blocks) {
                archive_full.add(bp.pop[i]);
            } else if (bp.block_id == 0) {
                archive_offload.add(bp.pop[i]);
            } else if (bp.block_id == 2) {
                archive_dev.add(bp.pop[i]);
            }
            std::copy(bp.newpop[i], bp.newpop[i] + bp.block_len, bp.pop[i]);
            bp.pop_fit[i] = bp.newpop_fit[i];
        }
    }
}

void CC_HIHH_Solver::SetOpStats(const std::string& path, int every)
{
    op_stats_path = path;
    op_stats_every = every > 0 ? every : 1;
    op_stats_enabled = !op_stats_path.empty();
}

void CC_HIHH_Solver::SetWeightLogging(const std::string& offload_path, const std::string& seq_path, const std::string& dev_path, int every)
{
    weight_log_path_offload = offload_path;
    weight_log_path_seq = seq_path;
    weight_log_path_dev = dev_path;
    weight_log_every = every > 0 ? every : 1;
    weight_log_enabled = !(weight_log_path_offload.empty() || weight_log_path_seq.empty() || weight_log_path_dev.empty());
}

void CC_HIHH_Solver::SetRewardLogging(const std::string& path)
{
    reward_log_path = path;
    reward_log_enabled = !reward_log_path.empty();
}

void CC_HIHH_Solver::SetRewardVarianceLogging(const std::string& path)
{
    reward_variance_log_path = path;
    reward_variance_log_enabled = !reward_variance_log_path.empty();
}

void CC_HIHH_Solver::SetGlobalStatsLogging(const std::string& path, int every)
{
    global_stats_path = path;
    global_stats_every = every > 0 ? every : 1;
    global_stats_enabled = !global_stats_path.empty();
}

void CC_HIHH_Solver::SetDiversityLogging(const std::string& path, int every)
{
    namespace fs = std::filesystem;
    diversity_log_path = path;
    diversity_log_every = every > 0 ? every : 1;
    diversity_log_enabled = !diversity_log_path.empty();
    diversity_log_config_name = "unknown";
    if (!diversity_log_path.empty()) {
        std::string stem = fs::path(diversity_log_path).stem().string();
        const std::string prefix = "diversity_log_";
        size_t start = stem.rfind(prefix, 0) == 0 ? prefix.size() : 0;
        size_t end = stem.find("_seed", start);
        if (end == std::string::npos) end = stem.size();
        if (end > start) diversity_log_config_name = stem.substr(start, end - start);
    }
}

void CC_HIHH_Solver::InitOpStats()
{
    op_stats_out.open(op_stats_path, std::ios::out | std::ios::trunc);
    if (!op_stats_out.is_open()) {
        std::cerr << "[CC-HIHH] Warning: failed to open op stats file: " << op_stats_path << std::endl;
        op_stats_enabled = false;
        return;
    }

    if (use_blocks) {
        op_counts_offload.assign(num_ops_offload, 0);
        op_counts_seq.assign(num_ops_seq, 0);
        op_counts_dev.assign(num_ops_dev, 0);
        op_counts_overall.assign(8, 0);
        op_stats_out
            << "gen"
            << ",offload_GA,offload_DE,offload_BITFLIP,offload_RESAMPLE"
            << ",seq_GA,seq_SWAP,seq_VNS,seq_RESAMPLE"
            << ",dev_DE,dev_GDE,dev_LEVY,dev_RESAMPLE"
            << ",overall_GA,overall_DE,overall_GDE,overall_BITFLIP,overall_SWAP,overall_VNS,overall_LEVY,overall_RESAMPLE"
            << "\n";
    } else {
        op_counts_full.assign(num_ops_full, 0);
        op_stats_out
            << "gen"
            << ",full_GA,full_DE,full_GDE,full_LEVY,full_RESAMPLE"
            << ",overall_GA,overall_DE,overall_GDE,overall_LEVY,overall_RESAMPLE"
            << "\n";
    }

    op_stats_interval_gens = 0;
}

void CC_HIHH_Solver::CloseOpStats()
{
    if (op_stats_out.is_open()) {
        op_stats_out.close();
    }
}

void CC_HIHH_Solver::InitWeightLogging()
{
    namespace fs = std::filesystem;
    auto open_one = [](std::ofstream& out, const std::string& path) -> bool {
        if (path.empty()) return false;
        out.open(path, std::ios::out | std::ios::trunc);
        return out.is_open();
    };
    if (!weight_log_path_offload.empty()) {
        fs::path p = fs::path(weight_log_path_offload).parent_path();
        if (!p.empty()) fs::create_directories(p);
    }
    if (!weight_log_path_seq.empty()) {
        fs::path p = fs::path(weight_log_path_seq).parent_path();
        if (!p.empty()) fs::create_directories(p);
    }
    if (!weight_log_path_dev.empty()) {
        fs::path p = fs::path(weight_log_path_dev).parent_path();
        if (!p.empty()) fs::create_directories(p);
    }
    bool ok = open_one(weight_log_out_offload, weight_log_path_offload) &&
              open_one(weight_log_out_seq, weight_log_path_seq) &&
              open_one(weight_log_out_dev, weight_log_path_dev);
    if (!ok) {
        std::cerr << "[CC-HIHH] Warning: failed to open one or more weight log files." << std::endl;
        weight_log_enabled = false;
        CloseWeightLogging();
        return;
    }
    const char* header = "gen,op_id,w0,w1,w2,w3,w4,w5,w6,norm\n";
    weight_log_out_offload << header;
    weight_log_out_seq << header;
    weight_log_out_dev << header;
}

void CC_HIHH_Solver::CloseWeightLogging()
{
    if (weight_log_out_offload.is_open()) weight_log_out_offload.close();
    if (weight_log_out_seq.is_open()) weight_log_out_seq.close();
    if (weight_log_out_dev.is_open()) weight_log_out_dev.close();
}

void CC_HIHH_Solver::InitRewardLogging()
{
    if (reward_log_path.empty()) return;
    namespace fs = std::filesystem;
    fs::path p = fs::path(reward_log_path).parent_path();
    if (!p.empty()) fs::create_directories(p);
    reward_log_out.open(reward_log_path, std::ios::out | std::ios::trunc);
    if (!reward_log_out.is_open()) {
        std::cerr << "[CC-HIHH] Warning: failed to open reward log file: " << reward_log_path << std::endl;
        reward_log_enabled = false;
        return;
    }
    reward_log_out << "gen,block_id,island_id,op_id,reward,improvement,diversity_change\n";
}

void CC_HIHH_Solver::CloseRewardLogging()
{
    if (reward_log_out.is_open()) reward_log_out.close();
}

void CC_HIHH_Solver::InitRewardVarianceLogging()
{
    if (reward_variance_log_path.empty()) return;
    namespace fs = std::filesystem;
    fs::path p = fs::path(reward_variance_log_path).parent_path();
    if (!p.empty()) fs::create_directories(p);
    reward_variance_log_out.open(reward_variance_log_path, std::ios::out | std::ios::trunc);
    if (!reward_variance_log_out.is_open()) {
        std::cerr << "[CC-HIHH] Warning: failed to open reward variance log file: " << reward_variance_log_path << std::endl;
        reward_variance_log_enabled = false;
        return;
    }
    reward_variance_log_out << "generation,block_id,reward_mean,reward_variance,reward_min,reward_max\n";
}

void CC_HIHH_Solver::CloseRewardVarianceLogging()
{
    if (reward_variance_log_out.is_open()) reward_variance_log_out.close();
}

void CC_HIHH_Solver::InitGlobalStatsLogging()
{
    if (global_stats_path.empty()) return;
    namespace fs = std::filesystem;
    fs::path p = fs::path(global_stats_path).parent_path();
    if (!p.empty()) fs::create_directories(p);
    global_stats_out.open(global_stats_path, std::ios::out | std::ios::trunc);
    if (!global_stats_out.is_open()) {
        std::cerr << "[CC-HIHH] Warning: failed to open global stats file: " << global_stats_path << std::endl;
        global_stats_enabled = false;
        return;
    }
    global_stats_out << "gen,best_fitness,avg_fitness,diversity,epsilon,stagnation,gate_blocked_count,gate_fallback_count\n";
}

void CC_HIHH_Solver::CloseGlobalStatsLogging()
{
    if (global_stats_out.is_open()) global_stats_out.close();
}

void CC_HIHH_Solver::InitDiversityLogging()
{
    if (diversity_log_path.empty()) return;
    namespace fs = std::filesystem;
    fs::path p = fs::path(diversity_log_path).parent_path();
    if (!p.empty()) fs::create_directories(p);
    diversity_log_out.open(diversity_log_path, std::ios::out | std::ios::trunc);
    if (!diversity_log_out.is_open()) {
        std::cerr << "[CC-HIHH] Warning: failed to open diversity log file: " << diversity_log_path << std::endl;
        diversity_log_enabled = false;
        return;
    }
    diversity_log_out << "generation,block,config,intra_diversity,global_diversity,inter_diversity\n";
}

void CC_HIHH_Solver::CloseDiversityLogging()
{
    if (diversity_log_out.is_open()) diversity_log_out.close();
}

void CC_HIHH_Solver::LogReward(int gen, int block_id, int island_id, int op_id, double reward, double improvement, double div_change)
{
    if (!reward_log_enabled || !reward_log_out.is_open()) return;
    reward_log_out << gen << ","
                   << block_id << ","
                   << island_id << ","
                   << op_id << ","
                   << std::setprecision(16) << reward << ","
                   << std::setprecision(16) << improvement << ","
                   << std::setprecision(16) << div_change << "\n";
}

void CC_HIHH_Solver::LogRewardVariance(int generation, int block_id, const std::vector<double>& rewards)
{
    if (!reward_variance_log_enabled || !reward_variance_log_out.is_open() || rewards.empty()) return;
    double sum = 0.0;
    double min_reward = rewards[0];
    double max_reward = rewards[0];
    for (double reward : rewards) {
        sum += reward;
        min_reward = std::min(min_reward, reward);
        max_reward = std::max(max_reward, reward);
    }
    const double mean = sum / (double)rewards.size();
    double variance = 0.0;
    for (double reward : rewards) {
        const double diff = reward - mean;
        variance += diff * diff;
    }
    variance /= (double)rewards.size();
    reward_variance_log_out << generation << ","
                            << block_id << ","
                            << std::setprecision(16) << mean << ","
                            << std::setprecision(16) << variance << ","
                            << std::setprecision(16) << min_reward << ","
                            << std::setprecision(16) << max_reward << "\n";
}

void CC_HIHH_Solver::RecordOpSelection(int block_id, int op_id)
{
    if (!op_stats_enabled) return;

    if (!use_blocks) {
        if (op_id >= 0 && op_id < num_ops_full && op_id < (int)op_counts_full.size()) {
            op_counts_full[op_id]++;
        }
        return;
    }

    if (block_id == 0) {
        if (op_id >= 0 && op_id < num_ops_offload && op_id < (int)op_counts_offload.size()) {
            op_counts_offload[op_id]++;
        }
    } else if (block_id == 1) {
        if (op_id >= 0 && op_id < num_ops_seq && op_id < (int)op_counts_seq.size()) {
            op_counts_seq[op_id]++;
        }
    } else if (block_id == 2) {
        if (op_id >= 0 && op_id < num_ops_dev && op_id < (int)op_counts_dev.size()) {
            op_counts_dev[op_id]++;
        }
    }

    int overall_idx = -1;
    if (block_id == 0) {
        if (op_id == OFF_OP_GA) overall_idx = 0;
        else if (op_id == OFF_OP_DE) overall_idx = 1;
        else if (op_id == OFF_OP_BITFLIP) overall_idx = 3;
        else if (op_id == OFF_OP_BLOCK_RESAMPLE) overall_idx = 7;
    } else if (block_id == 1) {
        if (op_id == SEQ_OP_GA) overall_idx = 0;
        else if (op_id == SEQ_OP_SWAP) overall_idx = 4;
        else if (op_id == SEQ_OP_VNS) overall_idx = 5;
        else if (op_id == SEQ_OP_BLOCK_RESAMPLE) overall_idx = 7;
    } else if (block_id == 2) {
        if (op_id == DEV_OP_DE) overall_idx = 1;
        else if (op_id == DEV_OP_GDE) overall_idx = 2;
        else if (op_id == DEV_OP_LEVY) overall_idx = 6;
        else if (op_id == DEV_OP_BLOCK_RESAMPLE) overall_idx = 7;
    }

    if (overall_idx >= 0 && overall_idx < (int)op_counts_overall.size()) {
        op_counts_overall[overall_idx]++;
    }
}

void CC_HIHH_Solver::LogOpStatsIfNeeded(int gen, bool is_last)
{
    if (!op_stats_enabled || !op_stats_out.is_open()) return;
    op_stats_interval_gens++;
    bool flush = ((gen + 1) % op_stats_every == 0) || is_last;
    if (!flush) return;

    if (use_blocks) {
        double denom_block = (double)op_stats_interval_gens * (double)nSubpop;
        double denom_overall = denom_block * 3.0;
        if (denom_block <= 0.0) denom_block = 1.0;
        if (denom_overall <= 0.0) denom_overall = 1.0;

        op_stats_out << (gen + 1);
        for (int i = 0; i < num_ops_offload; i++) {
            op_stats_out << "," << (double)op_counts_offload[i] / denom_block;
        }
        for (int i = 0; i < num_ops_seq; i++) {
            op_stats_out << "," << (double)op_counts_seq[i] / denom_block;
        }
        for (int i = 0; i < num_ops_dev; i++) {
            op_stats_out << "," << (double)op_counts_dev[i] / denom_block;
        }
        for (int i = 0; i < (int)op_counts_overall.size(); i++) {
            op_stats_out << "," << (double)op_counts_overall[i] / denom_overall;
        }
        op_stats_out << "\n";
    } else {
        double denom = (double)op_stats_interval_gens * (double)nSubpop;
        if (denom <= 0.0) denom = 1.0;
        op_stats_out << (gen + 1);
        for (int i = 0; i < num_ops_full; i++) {
            double v = (i < (int)op_counts_full.size()) ? (double)op_counts_full[i] / denom : 0.0;
            op_stats_out << "," << v;
        }
        for (int i = 0; i < num_ops_full; i++) {
            double v = (i < (int)op_counts_full.size()) ? (double)op_counts_full[i] / denom : 0.0;
            op_stats_out << "," << v;
        }
        op_stats_out << "\n";
    }

    op_stats_interval_gens = 0;
    if (use_blocks) {
        std::fill(op_counts_offload.begin(), op_counts_offload.end(), 0);
        std::fill(op_counts_seq.begin(), op_counts_seq.end(), 0);
        std::fill(op_counts_dev.begin(), op_counts_dev.end(), 0);
        std::fill(op_counts_overall.begin(), op_counts_overall.end(), 0);
    } else {
        std::fill(op_counts_full.begin(), op_counts_full.end(), 0);
    }
}

int CC_HIHH_Solver::SelectOperatorRandom(int block_id) const
{
    if (!use_blocks || block_id < 0) {
        return rand() % std::max(1, num_ops_full);
    }
    if (block_id == 0) return rand() % std::max(1, num_ops_offload);
    if (block_id == 1) return rand() % std::max(1, num_ops_seq);
    return rand() % std::max(1, num_ops_dev);
}

int CC_HIHH_Solver::SelectOperatorRoundRobin(int block_id)
{
    int idx = (block_id < 0) ? 3 : block_id;
    if (idx < 0 || idx >= (int)round_robin_counters.size()) {
        idx = 0;
    }
    int op_count = num_ops_full;
    if (block_id == 0) op_count = num_ops_offload;
    else if (block_id == 1) op_count = num_ops_seq;
    else if (block_id == 2) op_count = num_ops_dev;
    if (op_count <= 0) op_count = 1;
    int op = round_robin_counters[idx] % op_count;
    round_robin_counters[idx] = (round_robin_counters[idx] + 1) % op_count;
    return op;
}

double CC_HIHH_Solver::ComputeGlobalAvgFitness() const
{
    auto avg_fit = [](const BlockPopulation& bp) {
        if (!bp.pop_fit || bp.popsize <= 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < bp.popsize; ++i) sum += bp.pop_fit[i];
        return sum / (double)bp.popsize;
    };
    if (!use_blocks) return avg_fit(P_full);
    return (avg_fit(P_offload) + avg_fit(P_seq) + avg_fit(P_dev)) / 3.0;
}

double CC_HIHH_Solver::ComputeGlobalDiversity() const
{
    if (!use_blocks) {
        return P_full.compute_global_diversity();
    }
    double d0 = P_offload.compute_global_diversity();
    double d1 = P_seq.compute_global_diversity();
    double d2 = P_dev.compute_global_diversity();
    return (d0 + d1 + d2) / 3.0;
}

void CC_HIHH_Solver::WriteDiversityRow(int generation, const char* block_name, double intra_raw, double global_raw, double inter_raw)
{
    if (!diversity_log_out.is_open()) return;
    const double intra_diversity = intra_raw / (intra_raw + 1.0);
    const double global_diversity = global_raw / (global_raw + 1.0);
    const double inter_diversity = inter_raw / (inter_raw + 1.0);
    diversity_log_out << generation << ","
                      << block_name << ","
                      << diversity_log_config_name << ","
                      << std::setprecision(16) << intra_diversity << ","
                      << std::setprecision(16) << global_diversity << ","
                      << std::setprecision(16) << inter_diversity << "\n";
}

void CC_HIHH_Solver::LogWeightsIfNeeded(int gen, bool is_last)
{
    if (!weight_log_enabled) return;
    bool flush = ((gen + 1) % weight_log_every == 0) || is_last;
    if (!flush) return;
    if (!(weight_log_out_offload.is_open() && weight_log_out_seq.is_open() && weight_log_out_dev.is_open())) return;

    auto log_block = [&](std::ofstream& out, const BlockPopulation& bp, int num_ops) {
        if (!shared_bandit_mode && bp.cb_selectors.empty()) return;
        for (int op = 0; op < num_ops; ++op) {
            std::vector<double> avg_w(state_dim, 0.0);
            int denom = 0;
            if (shared_bandit_mode) {
                for (int sisl = 0; sisl < (int)shared_cb_selectors.size(); ++sisl) {
                    const auto& sel = shared_cb_selectors[sisl];
                    int base = op * sel.feat_dim;
                    for (int k = 0; k < state_dim && k < sel.feat_dim; ++k) {
                        avg_w[k] += sel.weights[base + k];
                    }
                }
                denom = (int)shared_cb_selectors.size();
            } else {
                for (int isl = 0; isl < (int)bp.cb_selectors.size(); ++isl) {
                    const auto& sel = bp.cb_selectors[isl];
                    int base = op * sel.feat_dim;
                    for (int k = 0; k < state_dim && k < sel.feat_dim; ++k) {
                        avg_w[k] += sel.weights[base + k];
                    }
                }
                denom = (int)bp.cb_selectors.size();
            }
            if (denom < 1) denom = 1;
            for (int k = 0; k < state_dim; ++k) {
                avg_w[k] /= (double)denom;
            }
            double norm = 0.0;
            for (int k = 0; k < state_dim; ++k) norm += avg_w[k] * avg_w[k];
            norm = std::sqrt(norm);

            out << (gen + 1) << "," << op;
            for (int k = 0; k < state_dim; ++k) {
                out << "," << std::setprecision(16) << avg_w[k];
            }
            out << "," << std::setprecision(16) << norm << "\n";
        }
    };

    if (use_blocks) {
        log_block(weight_log_out_offload, P_offload, num_ops_offload);
        log_block(weight_log_out_seq, P_seq, num_ops_seq);
        log_block(weight_log_out_dev, P_dev, num_ops_dev);
    } else {
        log_block(weight_log_out_offload, P_full, num_ops_full);
    }
}

void CC_HIHH_Solver::LogGlobalStatsIfNeeded(int gen, bool is_last)
{
    if (!global_stats_enabled || !global_stats_out.is_open()) return;
    bool flush = ((gen + 1) % global_stats_every == 0) || is_last;
    if (!flush) return;
    const double best = GetGlobalBestFit();
    const double avg = ComputeGlobalAvgFitness();
    const double div = ComputeGlobalDiversity();
    const double eps = (op_selection_mode == MODE_CONTEXTUAL_BANDIT && use_bandit) ? ComputeEpsilon(gen) : 0.0;
    global_stats_out << (gen + 1) << ","
                     << std::setprecision(16) << best << ","
                     << std::setprecision(16) << avg << ","
                     << std::setprecision(16) << div << ","
                     << std::setprecision(16) << eps << ","
                     << stagnation_count << ","
                     << gate_blocked_total << ","
                     << gate_fallback_total << "\n";
}

void CC_HIHH_Solver::LogDiversityIfNeeded(int gen, bool is_last)
{
    if (!diversity_log_enabled || !diversity_log_out.is_open()) return;
    const bool flush = ((gen + 1) % diversity_log_every == 0) || is_last;
    if (!flush) return;

    if (!use_blocks) {
        const double global = P_full.compute_global_diversity();
        WriteDiversityRow(gen + 1, "full", global, global, 0.0);
        return;
    }

    WriteDiversityRow(
        gen + 1,
        "offload",
        P_offload.compute_intra_island_diversity(),
        P_offload.compute_global_diversity(),
        P_offload.compute_inter_island_diversity()
    );
    WriteDiversityRow(
        gen + 1,
        "seq",
        P_seq.compute_intra_island_diversity(),
        P_seq.compute_global_diversity(),
        P_seq.compute_inter_island_diversity()
    );
    WriteDiversityRow(
        gen + 1,
        "dev",
        P_dev.compute_intra_island_diversity(),
        P_dev.compute_global_diversity(),
        P_dev.compute_inter_island_diversity()
    );
}
