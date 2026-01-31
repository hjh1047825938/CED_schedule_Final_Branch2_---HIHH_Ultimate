#include "CC_HIHH.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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
    resample_gate = 15;
    gate_blocked_total = 0;
    gate_fallback_total = 0;
    
    // Operator counts: GA, DE for offload; GA, SWAP for seq; DE, GDE for dev
    num_ops_offload = 4;  // GA, DE, BITFLIP, BLOCK_RESAMPLE
    num_ops_seq = 3;      // GA, SEQ_SWAP, BLOCK_RESAMPLE
    num_ops_dev = 3;      // DE, GDE, BLOCK_RESAMPLE
    
    var_full.resize(Nvar);
    gbest.resize(Nvar);
}

CC_HIHH_Solver::~CC_HIHH_Solver()
{
    P_offload.destroy();
    P_seq.destroy();
    P_dev.destroy();
}

void CC_HIHH_Solver::Init()
{
    // Validate nSubpop
    if (nSubpop < 2) {
        std::cerr << "[CC-HIHH] Error: nSubpop must be >= 2 for migration to work" << std::endl;
        std::exit(1);
    }
    
    std::cout << "[CC-HIHH] Initializing solver..." << std::endl;
    std::cout << "  CE_Tnum=" << CE_Tnum << ", M_Jnum=" << M_Jnum 
              << ", M_OPTnum=" << M_OPTnum << ", ops=" << ops << std::endl;
    std::cout << "  Nvar=" << Nvar << ", popsize=" << popsize 
              << ", nSubpop=" << nSubpop << std::endl;
    
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
}

void CC_HIHH_Solver::RunGeneration(int gen)
{
    // Process each block in round-robin fashion
    BlockPopulation* blocks[3] = {&P_offload, &P_seq, &P_dev};
    const char* block_names[3] = {"offload", "seq", "dev"};
    
    for (int b = 0; b < 3; b++) {
        BlockPopulation& bp = *blocks[b];
        
        // Process each island
        for (int isl = 0; isl < nSubpop; isl++) {
            int p_start, p_end;
            bp.get_island_range(isl, p_start, p_end);
            
            // Record pre-application statistics
            double old_best_fit = bp.get_island_best_fit(p_start, p_end);
            double old_diversity = bp.compute_diversity(p_start, p_end);
            
            // Build contextual state and select operator
            std::vector<double> state = ComputeState(bp, isl, gen, old_diversity);
            double eps = ComputeEpsilon(gen);
            int op_sel = bp.cb_selectors[isl].select(state, eps);
            int op_exec = op_sel;
            if (stable_mode && IsResampleOp(op_sel, bp.block_id) &&
                bp.island_stagnation[isl] < resample_gate) {
                gate_blocked_total++;
                if (bp.block_id == 1) {
                    op_exec = SEQ_OP_SWAP;
                } else if (bp.block_id == 2) {
                    op_exec = DEV_OP_GDE;
                } else {
                    op_exec = OFF_OP_DE;
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

            double lr = ComputeLearningRate(bp.cb_selectors[isl]);
            bp.cb_selectors[isl].update(state, op_exec, reward, lr);

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
        }

#if HIHH_CB_DEBUG
        if ((gen + 1) % HIHH_CB_DEBUG_INTERVAL == 0) {
            double eps_dbg = ComputeEpsilon(gen);
            for (int isl = 0; isl < nSubpop; isl++) {
                int p_start, p_end;
                bp.get_island_range(isl, p_start, p_end);
                double div_now = bp.compute_diversity(p_start, p_end);
                std::vector<double> state_dbg = ComputeState(bp, isl, gen, div_now);
                double lr_dbg = ComputeLearningRate(bp.cb_selectors[isl]);
                std::cout << "[CB][Gen " << (gen + 1) << "][Block " << b << "][Island " << isl
                          << "] stag=" << bp.island_stagnation[isl]
                          << " eps=" << eps_dbg
                          << " lr=" << lr_dbg
                          << " gate_blocked=" << gate_blocked_total
                          << " gate_fallback=" << gate_fallback_total
                          << " ";
                for (int op = 0; op < (int)bp.cb_selectors[isl].selection_count.size(); op++) {
                    int cnt = bp.cb_selectors[isl].selection_count[op];
                    double avg = (cnt > 0) ? bp.cb_selectors[isl].total_reward[op] / cnt : 0.0;
                    double sc = bp.cb_selectors[isl].score_op(op, state_dbg);
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
        if (gen > 0 && gen % nCircle == 0) {
            int dispara = (gen / nCircle - 1) % (nSubpop - 1) + 1;
            MigrationWithinBlock(bp, dispara);
        }
        
        // Update context for this block
        context.update_block(bp.block_id, bp.block_gbest);
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

std::vector<double> CC_HIHH_Solver::ComputeState(const BlockPopulation& bp, int isl, int gen, double diversity) const
{
    std::vector<double> s(state_dim, 0.0);
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
    // Ring migration: island k receives from (k - dispara + nSubpop) mod nSubpop
    std::vector<double*> migrants(nSubpop);
    std::vector<double> migrant_fit(nSubpop);
    
    // Prepare migrants
    for (int k = 0; k < nSubpop; k++) {
        migrants[k] = new double[bp.block_len];
        
        if (randval(0, 1) < pElitist) {
            // Send island gbest
            std::copy(bp.island_gbest[k], bp.island_gbest[k] + bp.block_len, migrants[k]);
            migrant_fit[k] = bp.island_gbest_fit[k];
        } else {
            // Send random individual from island
            int p_start, p_end;
            bp.get_island_range(k, p_start, p_end);
            int rand_idx = p_start + rand() % (p_end - p_start);
            std::copy(bp.pop[rand_idx], bp.pop[rand_idx] + bp.block_len, migrants[k]);
            migrant_fit[k] = bp.pop_fit[rand_idx];
        }
    }
    
    // Perform migration
    for (int k = 0; k < nSubpop; k++) {
        int source = ((k - dispara) % nSubpop + nSubpop) % nSubpop;
        int p_start, p_end;
        bp.get_island_range(k, p_start, p_end);
        int worst_idx = bp.get_island_worst_idx(p_start, p_end);
        
        // Replace worst with migrant
        std::copy(migrants[source], migrants[source] + bp.block_len, bp.pop[worst_idx]);
        bp.pop_fit[worst_idx] = migrant_fit[source];
    }
    
    // Cleanup
    for (int k = 0; k < nSubpop; k++) {
        delete[] migrants[k];
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
            case SEQ_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
            default: ApplyGA(bp, p_start, p_end); break;
        }
    } else {  // Device block
        switch (op) {
            case DEV_OP_DE: ApplyDE(bp, p_start, p_end); break;
            case DEV_OP_GDE: ApplyGDE(bp, p_start, p_end); break;
            case DEV_OP_BLOCK_RESAMPLE: ApplyBlockResample(bp, p_start, p_end, resample_rate); break;
            default: ApplyDE(bp, p_start, p_end); break;
        }
    }
}

void CC_HIHH_Solver::ApplyGA(BlockPopulation& bp, int p_start, int p_end)
{
    double pc = 0.8;  // Crossover probability
    double pm = 0.15; // Mutation probability
    
    // Selection (tournament)
    for (int i = p_start; i < p_end; i++) {
        int t1 = p_start + rand() % (p_end - p_start);
        int t2 = p_start + rand() % (p_end - p_start);
        int winner = (bp.pop_fit[t1] < bp.pop_fit[t2]) ? t1 : t2;
        std::copy(bp.pop[winner], bp.pop[winner] + bp.block_len, bp.newpop[i]);
    }
    
    // Crossover (blend crossover)
    for (int i = p_start; i < p_end; i++) {
        if (randval(0, 1) < pc) {
            int partner = p_start + rand() % (p_end - p_start);
            if (partner == i) continue;
            
            int point = rand() % bp.block_len;
            for (int j = 0; j < point; j++) {
                double r = randval(0, 1);
                double temp = bp.newpop[i][j] * r + (1 - r) * bp.newpop[partner][j];
                bp.newpop[i][j] = clip01(temp);
            }
        }
    }
    
    // Mutation
    for (int i = p_start; i < p_end; i++) {
        if (randval(0, 1) < pm) {
            int r = rand() % bp.block_len;
            bp.newpop[i][r] = randval(0, 1);
        }
    }
}

void CC_HIHH_Solver::ApplyDE(BlockPopulation& bp, int p_start, int p_end)
{
    int island_size = p_end - p_start;
    
    // Safety check: DE requires at least 4 distinct individuals (target + 3 donors)
    if (island_size < 4) {
        ApplyGA(bp, p_start, p_end);  // Fallback to GA
        return;
    }
    
    double F = 0.5;   // Scale factor
    double CR = 0.5;  // Crossover rate
    
    for (int i = p_start; i < p_end; i++) {
        // Select three distinct individuals from island
        int r1, r2, r3;
        do { r1 = p_start + rand() % island_size; } while (r1 == i);
        do { r2 = p_start + rand() % island_size; } while (r2 == i || r2 == r1);
        do { r3 = p_start + rand() % island_size; } while (r3 == i || r3 == r1 || r3 == r2);
        
        // Mutation and crossover
        int jrand = rand() % bp.block_len;
        for (int j = 0; j < bp.block_len; j++) {
            if (randval(0, 1) < CR || j == jrand) {
                double v = bp.pop[r1][j] + F * (bp.pop[r2][j] - bp.pop[r3][j]);
                bp.newpop[i][j] = clip01(v);
            } else {
                bp.newpop[i][j] = bp.pop[i][j];
            }
        }
    }
}

void CC_HIHH_Solver::ApplyGDE(BlockPopulation& bp, int p_start, int p_end)
{
    int island_size = p_end - p_start;
    
    // Safety check: GDE requires at least 3 distinct individuals (best + 2 donors)
    if (island_size < 3) {
        ApplyGA(bp, p_start, p_end);  // Fallback to GA
        return;
    }
    
    double F = randval(0.2, 0.8);
    double CR = randval(0.1, 0.6);
    
    // Find island best for gbest-centric mutation
    int best_idx = bp.get_island_best_idx(p_start, p_end);
    
    for (int i = p_start; i < p_end; i++) {
        int r1, r2;
        do { r1 = p_start + rand() % island_size; } while (r1 == i);
        do { r2 = p_start + rand() % island_size; } while (r2 == i || r2 == r1);
        
        // Gbest-centric mutation
        int jrand = rand() % bp.block_len;
        for (int j = 0; j < bp.block_len; j++) {
            if (randval(0, 1) < CR || j == jrand) {
                double v = bp.pop[best_idx][j] + F * (bp.pop[r1][j] - bp.pop[r2][j]);
                bp.newpop[i][j] = clip01(v);
            } else {
                bp.newpop[i][j] = bp.pop[i][j];
            }
        }
    }
}

void CC_HIHH_Solver::ApplyBitFlip(BlockPopulation& bp, int p_start, int p_end)
{
    // For offload block: flip cloud/edge decision
    // Offload block structure: [cloud/edge selection (CE_Tnum)] + [server assignment (CE_Tnum)]
    // First half of offload block is cloud/edge selection
    double pm = 0.1;  // Flip probability
    int half_len = bp.block_len / 2;  // == CE_Tnum
    
    // Safety assertion: offload block should have even length
    if (half_len * 2 != bp.block_len) {
        std::cerr << "[CC-HIHH] Warning: offload block_len is odd, half_len may be inaccurate" << std::endl;
    }
    
    for (int i = p_start; i < p_end; i++) {
        for (int j = 0; j < half_len; j++) {
            if (randval(0, 1) < pm) {
                // Flip: if < 0.5 (cloud), set to 0.75 (edge); else set to 0.25 (cloud)
                bp.newpop[i][j] = (bp.newpop[i][j] < 0.5) ? 0.75 : 0.25;
            }
        }
        // Small mutation on server selection (second half)
        for (int j = half_len; j < bp.block_len; j++) {
            if (randval(0, 1) < pm) {
                bp.newpop[i][j] = randval(0, 1);
            }
        }
    }
}

void CC_HIHH_Solver::ApplySeqSwap(BlockPopulation& bp, int p_start, int p_end, int n_swaps)
{
    // For sequence block: swap a few pairs (local exploitation)
    if (bp.block_len < 2) return;
    if (n_swaps < 1) n_swaps = 1;

    for (int i = p_start; i < p_end; i++) {
        int swaps = n_swaps;
        for (int k = 0; k < swaps; k++) {
            int j1 = rand() % bp.block_len;
            int j2 = rand() % bp.block_len;
            if (j1 != j2) {
                std::swap(bp.newpop[i][j1], bp.newpop[i][j2]);
            }
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
        for (int k = 0; k < count; k++) {
            int idx = rand() % bp.block_len;
            bp.newpop[i][idx] = clip01(randval(0, 1));
        }
    }
}

void CC_HIHH_Solver::EvaluateBlock(BlockPopulation& bp)
{
    for (int i = 0; i < bp.popsize; i++) {
        context.assemble_full(bp.block_id, bp.pop[i], var_full.data());
        bp.pop_fit[i] = solver->Eval(var_full.data());
    }
}

void CC_HIHH_Solver::EvaluateBlockIsland(BlockPopulation& bp, int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i++) {
        context.assemble_full(bp.block_id, bp.newpop[i], var_full.data());
        bp.newpop_fit[i] = solver->Eval(var_full.data());
    }
}

void CC_HIHH_Solver::SelectionUpdate(BlockPopulation& bp, int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i++) {
        if (bp.newpop_fit[i] < bp.pop_fit[i]) {
            std::copy(bp.newpop[i], bp.newpop[i] + bp.block_len, bp.pop[i]);
            bp.pop_fit[i] = bp.newpop_fit[i];
        }
    }
}
