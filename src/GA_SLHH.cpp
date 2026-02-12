#include "GA_SLHH.h"
#include <limits>

GA_SLHH_Solver::GA_SLHH_Solver(MultiMet* s, int psize)
    : solver(s),
      popsize(psize),
      CE_Tnum(s->CE_Tnum),
      M_Jnum(s->M_Jnum),
      M_OPTnum(s->M_OPTnum),
      ops(s->M_Jnum * s->M_OPTnum),
      Nvar(2 * s->CE_Tnum + 2 * s->M_Jnum * s->M_OPTnum),
      num_rules(14),
      max_generations(1),
      archive_size(psize),
      pc(0.9),
      pm(0.2),
      gene_pm(0.03),
      immigrant_rate(0.15),
      elitism(std::max(2, psize / 50)),
      ls_trials(100),
      ls_interval(10),
      diversity_interval(20),
      learning_interval(3),
      stagnation(0),
      stagnation_trigger(20),
      diversity_threshold(0.1),
      gbest_fit(std::numeric_limits<double>::infinity()),
      last_mean_fit(std::numeric_limits<double>::infinity()),
      last_unique_llh(0),
      pm_base(0.2),
      gene_pm_base(0.03),
      immigrant_base(0.15)
{
    pop.resize(popsize, std::vector<double>(Nvar, 0.0));
    newpop.resize(popsize, std::vector<double>(Nvar, 0.0));
    pop_fit.assign(popsize, std::numeric_limits<double>::infinity());
    newpop_fit.assign(popsize, std::numeric_limits<double>::infinity());
    llh_pop.resize(popsize, std::vector<int>(Nvar, 0));
    llh_newpop.resize(popsize, std::vector<int>(Nvar, 0));

    gbest.resize(Nvar, 0.0);

    uniform_probs.assign(num_rules, 1.0 / (double)num_rules);
    probs_cur.assign(Nvar * num_rules, 1.0 / (double)num_rules);
    probs_his.assign(Nvar * num_rules, 1.0 / (double)num_rules);
    probs_mix.assign(Nvar * num_rules, 1.0 / (double)num_rules);
    eval_buffer.assign(Nvar, 0.0);
    temp_llh.assign(Nvar, 0);

    nearest_edge_for_device.assign(solver ? solver->Dnum : 0, 0);
    job_total_cache.assign(M_Jnum, 0.0);
    cloud_load_buf.assign(CE_Tnum > 0 ? (solver ? solver->Cnum : 0) : 0, 0);
    edge_load_buf.assign(CE_Tnum > 0 ? (solver ? solver->Enum : 0) : 0, 0);
    device_load_buf.assign(solver ? solver->Dnum : 0, 0);
    last_device_per_job_buf.assign(M_Jnum, -1);
    op_score_buf.assign(ops, 0.0);
    op_order_buf.assign(ops, 0);
    rank_buf.assign(ops, 0);
    seen_stamp_buf.assign(solver ? solver->Dnum : 0, 0);
    task_devices_buf.reserve(solver ? std::min(64, solver->Dnum) : 64);
    task_devices_per_task_cache.resize(CE_Tnum);
    task_central_device_cache.assign(CE_Tnum, -1);
    nearest_edge_for_task_cache.assign(CE_Tnum, -1);
    task_edge_allowed_cache.assign((solver ? solver->Enum : 0) * CE_Tnum, (unsigned char)0);
    task_edge_pos_cache.assign((solver ? solver->Enum : 0) * CE_Tnum, -1);
    rem_time_cache.assign(ops, 0.0);
    rem_ops_cache.assign(ops, 1);
    op_device_pos_cache.assign(ops * (solver ? solver->Dnum : 0), -1);
}

void GA_SLHH_Solver::Init()
{
    eval_buffer.assign(Nvar, 0.0);
    temp_llh.assign(Nvar, 0);
    InitializePopulation();
    BuildStaticCaches();
    EvaluatePopulation();
    UpdateArchive();

    // Reset adaptive params
    pm_base = pm;
    gene_pm_base = gene_pm;
    immigrant_base = immigrant_rate;
}

void GA_SLHH_Solver::RunGeneration(int gen)
{
    double prev_best = gbest_fit;
    if (gen == 0 || learning_interval <= 1 || (gen % learning_interval) == 0) {
        BuildLearningProbabilities(gen + 1);
    }

    // Selection with elitism
    std::vector<int> idx(popsize);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

    int elite = std::min(elitism, popsize);
    for (int i = 0; i < elite; i++) {
        newpop[i] = pop[idx[i]];
        newpop_fit[i] = pop_fit[idx[i]];
        llh_newpop[i] = llh_pop[idx[i]];
    }
    for (int i = elite; i < popsize; i++) {
        int sel = RouletteSelect(pop_fit);
        newpop[i] = pop[sel];
    }

    ApplyCrossover();
    ApplyMutation();

    // Decode and evaluate (skip elites that are copied unchanged)
    for (int i = elite; i < popsize; i++) {
        const std::vector<double>& probs = (i < popsize / 2) ? uniform_probs : probs_mix;
        DecodeLLH(newpop[i], probs, llh_newpop[i]);
        newpop_fit[i] = EvalLLH(llh_newpop[i], eval_buffer);
    }

    pop.swap(newpop);
    pop_fit.swap(newpop_fit);
    llh_pop.swap(llh_newpop);

    // Update global best
    for (int i = 0; i < popsize; i++) {
        if (pop_fit[i] < gbest_fit) {
            gbest_fit = pop_fit[i];
            DecodeToSolution(llh_pop[i], gbest);
        }
    }

    // Stats for stagnation detection
    double current_mean_fit = 0.0;
    for (int i = 0; i < popsize; i++) current_mean_fit += pop_fit[i];
    current_mean_fit /= (double)popsize;

    double mean_improvement = std::numeric_limits<double>::infinity();
    if (std::isfinite(last_mean_fit)) {
        mean_improvement = std::abs(last_mean_fit - current_mean_fit);
    }
    double best_improvement = std::abs(prev_best - gbest_fit);

    if (best_improvement > 1e-6 || mean_improvement > 1e-4) {
        stagnation = 0;
    } else {
        stagnation++;
    }

    // Graded response on stagnation
    pm = pm_base;
    gene_pm = gene_pm_base;
    immigrant_rate = immigrant_base;
    if (stagnation >= stagnation_trigger) {
        if (stagnation < stagnation_trigger * 2) {
            pm = std::min(0.9, pm_base * 1.5);
            gene_pm = std::min(0.1, gene_pm_base * 2.0);
        } else if (stagnation < stagnation_trigger * 3) {
            pm = std::min(0.9, pm_base * 2.5);
            gene_pm = std::min(0.15, gene_pm_base * 4.0);
            immigrant_rate = std::min(0.5, immigrant_base * 3.0);
        } else {
            int restart_count = std::max(1, popsize / 2);
            for (int i = popsize - restart_count; i < popsize; i++) {
                for (int j = 0; j < Nvar; j++) {
                    pop[i][j] = randval(0.0, 1.0);
                }
                const std::vector<double>& probs = (i < popsize / 2) ? uniform_probs : probs_mix;
                DecodeLLH(pop[i], probs, llh_pop[i]);
                pop_fit[i] = EvalLLH(llh_pop[i], eval_buffer);
            }
            stagnation = 0;
        }
    }

    // Budgeted local search: run periodically or when stagnation is visible.
    if (ls_trials > 0 && (gen == 0 || (ls_interval > 0 && (gen % ls_interval) == 0) || stagnation >= stagnation_trigger)) {
        LocalSearch();
    }

    // Periodic diversity maintenance
    if (diversity_interval > 0 && (gen % diversity_interval) == 0) {
        MaintainDiversity();
    }

    // Refresh best and stats after local/diversity adjustments
    current_mean_fit = 0.0;
    for (int i = 0; i < popsize; i++) {
        current_mean_fit += pop_fit[i];
        if (pop_fit[i] < gbest_fit) {
            gbest_fit = pop_fit[i];
            DecodeToSolution(llh_pop[i], gbest);
        }
    }
    last_mean_fit = current_mean_fit / (double)popsize;
    {
        std::vector<uint64_t> hashes;
        hashes.reserve(popsize);
        for (int i = 0; i < popsize; i++) {
            uint64_t h = 1469598103934665603ull;
            for (int q = 0; q < Nvar; q++) {
                uint64_t x = (uint64_t)(llh_pop[i][q] + 1);
                h ^= x;
                h *= 1099511628211ull;
            }
            hashes.push_back(h);
        }
        std::sort(hashes.begin(), hashes.end());
        last_unique_llh = (int)std::distance(hashes.begin(), std::unique(hashes.begin(), hashes.end()));
    }

    UpdateArchive();
}

void GA_SLHH_Solver::LocalSearch()
{
    if (ls_trials <= 0) return;
    if (popsize <= 0) return;

    std::vector<int> idx(popsize);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

    int search_count = std::max(1, popsize / 10);
    const int scales[] = {1, 2, 4};
    int scale_count = (int)(sizeof(scales) / sizeof(scales[0]));
    int trials_per_scale = std::max(1, ls_trials / (scale_count * 2));

    for (int s = 0; s < search_count; s++) {
        int target_idx = idx[s];
        std::vector<double> best_chrom = pop[target_idx];
        std::vector<double> cand(best_chrom);
        double best_fit = pop_fit[target_idx];

        for (int si = 0; si < scale_count; si++) {
            int scale = scales[si];
            int no_improve = 0;
            int max_no_improve = std::max(8, trials_per_scale / 2);
            for (int t = 0; t < trials_per_scale; t++) {
                cand = best_chrom;
                for (int k = 0; k < scale; k++) {
                    int pos = rand() % Nvar;
                    cand[pos] = randval(0.0, 1.0);
                }

                DecodeLLH(cand, probs_mix, temp_llh);
                double fit = EvalLLH(temp_llh, eval_buffer);
                if (fit < best_fit) {
                    best_fit = fit;
                    best_chrom.swap(cand);
                    no_improve = 0;
                } else {
                    no_improve++;
                    if (no_improve >= max_no_improve) break;
                }
            }
        }

        if (best_fit + 1e-12 < pop_fit[target_idx]) {
            pop[target_idx] = best_chrom;
            pop_fit[target_idx] = best_fit;
            DecodeLLH(best_chrom, probs_mix, llh_pop[target_idx]);
            if (best_fit < gbest_fit) {
                gbest_fit = best_fit;
                DecodeToSolution(llh_pop[target_idx], gbest);
            }
        }
    }
}

void GA_SLHH_Solver::InitializePopulation()
{
    int seed_count = 0;
    if (solver && solver->pop && solver->Popsize > 0) {
        seed_count = std::min(popsize, solver->Popsize);
        for (int i = 0; i < seed_count; i++) {
            for (int j = 0; j < Nvar; j++) {
                pop[i][j] = solver->pop[i][j];
            }
        }
    }
    for (int i = seed_count; i < popsize; i++) {
        for (int j = 0; j < Nvar; j++) {
            pop[i][j] = randval(0.0, 1.0);
        }
    }
}

void GA_SLHH_Solver::BuildStaticCaches()
{
    if (!solver) return;

    // Precompute nearest edge for each device (stable per instance).
    nearest_edge_for_device.assign(solver->Dnum, 0);
    for (int d = 0; d < solver->Dnum; d++) {
        double min_dis = std::numeric_limits<double>::infinity();
        int min_idx = 0;
        for (int e = 0; e < solver->Enum; e++) {
            double dis = solver->EtoD_Distance[e][d];
            if (dis < min_dis) {
                min_dis = dis;
                min_idx = e;
            }
        }
        nearest_edge_for_device[d] = min_idx;
    }

    // Precompute job-level totals and op-level remaining workload.
    job_total_cache.assign(M_Jnum, 0.0);
    rem_time_cache.assign(ops, 0.0);
    rem_ops_cache.assign(ops, 1);
    for (int j = 0; j < M_Jnum; j++) {
        double total = 0.0;
        double running = 0.0;
        for (int k = M_OPTnum - 1; k >= 0; k--) {
            int op = j * M_OPTnum + k;
            double proc = solver->MTask_Time[op];
            total += proc;
            running += proc;
            rem_time_cache[op] = running;
            rem_ops_cache[op] = M_OPTnum - k;
        }
        job_total_cache[j] = total;
    }

    // Precompute per-task device sets and edge lookup tables.
    task_devices_per_task_cache.clear();
    task_devices_per_task_cache.resize(CE_Tnum);
    task_central_device_cache.assign(CE_Tnum, -1);
    nearest_edge_for_task_cache.assign(CE_Tnum, -1);
    task_edge_allowed_cache.assign(CE_Tnum * solver->Enum, (unsigned char)0);
    task_edge_pos_cache.assign(CE_Tnum * solver->Enum, -1);

    std::vector<int> seen_local(solver->Dnum, 0);
    int seen_local_epoch = 1;
    for (int t = 0; t < CE_Tnum; t++) {
        std::vector<int>& task_devices = task_devices_per_task_cache[t];
        task_devices.clear();

        int stamp = seen_local_epoch++;
        if (seen_local_epoch == std::numeric_limits<int>::max()) {
            std::fill(seen_local.begin(), seen_local.end(), 0);
            seen_local_epoch = 1;
            stamp = seen_local_epoch++;
        }
        for (int j = 0; j < M_OPTnum; j++) {
            int op_idx = t * M_OPTnum + j;
            const std::vector<int>& avail = solver->AvailDeviceList[op_idx];
            for (int dev : avail) {
                if (dev >= 0 && dev < solver->Dnum && seen_local[dev] != stamp) {
                    seen_local[dev] = stamp;
                    task_devices.push_back(dev);
                }
            }
        }

        if (!task_devices.empty()) {
            int center = task_devices[0];
            double best_sum = std::numeric_limits<double>::infinity();
            for (int cand : task_devices) {
                double sum = 0.0;
                for (int other : task_devices) {
                    sum += solver->DtoD_Distance[cand][other];
                }
                if (sum < best_sum) {
                    best_sum = sum;
                    center = cand;
                }
            }
            task_central_device_cache[t] = center;
        }

        const std::vector<int>& edge_list = solver->CETask_Property[t].AvailEdgeServerList;
        int best_edge = -1;
        double best_avg = std::numeric_limits<double>::infinity();
        for (size_t pos = 0; pos < edge_list.size(); pos++) {
            int edge = edge_list[pos];
            if (edge < 0 || edge >= solver->Enum) continue;
            task_edge_allowed_cache[t * solver->Enum + edge] = (unsigned char)1;
            task_edge_pos_cache[t * solver->Enum + edge] = (int)pos;

            if (task_devices.empty()) {
                if (best_edge < 0) best_edge = edge;
                continue;
            }

            double sum = 0.0;
            for (int dev : task_devices) sum += solver->EtoD_Distance[edge][dev];
            double avg = sum / (double)task_devices.size();
            if (avg < best_avg) {
                best_avg = avg;
                best_edge = edge;
            }
        }
        nearest_edge_for_task_cache[t] = best_edge;
    }

    // Precompute device-position lookup per operation.
    op_device_pos_cache.assign(ops * solver->Dnum, -1);
    for (int op = 0; op < ops; op++) {
        const std::vector<int>& avail = solver->AvailDeviceList[op];
        for (size_t idx = 0; idx < avail.size(); idx++) {
            int dev = avail[idx];
            if (dev >= 0 && dev < solver->Dnum) {
                op_device_pos_cache[op * solver->Dnum + dev] = (int)idx;
            }
        }
    }

    // Resize transient decode buffers.
    cloud_load_buf.assign(CE_Tnum > 0 ? solver->Cnum : 0, 0);
    edge_load_buf.assign(CE_Tnum > 0 ? solver->Enum : 0, 0);
    device_load_buf.assign(solver->Dnum, 0);
    last_device_per_job_buf.assign(M_Jnum, -1);
    op_score_buf.assign(ops, 0.0);
    op_order_buf.resize(ops);
    rank_buf.resize(ops);
    seen_stamp_buf.assign(solver->Dnum, 0);
    seen_epoch = 1;
    task_devices_buf.clear();
}

void GA_SLHH_Solver::EvaluatePopulation()
{
    for (int i = 0; i < popsize; i++) {
        DecodeLLH(pop[i], uniform_probs, llh_pop[i]);
        pop_fit[i] = EvalLLH(llh_pop[i], eval_buffer);
        if (pop_fit[i] < gbest_fit) {
            gbest_fit = pop_fit[i];
            DecodeToSolution(llh_pop[i], gbest);
        }
    }
}

void GA_SLHH_Solver::UpdateArchive()
{
    // Take top half from current population
    std::vector<int> idx(popsize);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

    std::vector<ArchiveEntry> candidates = archive;
    int add_count = popsize / 2;
    for (int i = 0; i < add_count; i++) {
        ArchiveEntry e;
        e.llh = llh_pop[idx[i]];
        e.fit = pop_fit[idx[i]];
        candidates.push_back(std::move(e));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const ArchiveEntry& a, const ArchiveEntry& b) { return a.fit < b.fit; });

    archive.clear();
    int keep = std::min(archive_size, (int)candidates.size());
    archive.reserve(keep);
    for (int i = 0; i < keep; i++) {
        archive.push_back(candidates[i]);
    }
}

void GA_SLHH_Solver::BuildLearningProbabilities(int iter)
{
    int take = std::max(1, popsize / 2);
    std::vector<int> idx(popsize);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

    std::vector<std::vector<int>> cur_seqs;
    cur_seqs.reserve(take);
    for (int i = 0; i < take; i++) {
        cur_seqs.push_back(llh_pop[idx[i]]);
    }

    std::vector<std::vector<int>> his_seqs;
    his_seqs.reserve(archive.size());
    for (const auto& e : archive) {
        his_seqs.push_back(e.llh);
    }

    probs_cur = ComputeProbabilities(cur_seqs);
    probs_his = ComputeProbabilities(his_seqs);

    double learning_rate = 0.5;
    if (iter > max_generations / 2) {
        learning_rate = 0.3;
    }

    for (int q = 0; q < Nvar; q++) {
        double row_sum = 0.0;
        for (int h = 0; h < num_rules; h++) {
            double v = learning_rate * probs_cur[q * num_rules + h] +
                       (1.0 - learning_rate) * probs_his[q * num_rules + h];
            probs_mix[q * num_rules + h] = v;
            row_sum += v;
        }
        if (row_sum <= 0.0) {
            for (int h = 0; h < num_rules; h++) {
                probs_mix[q * num_rules + h] = 1.0 / (double)num_rules;
            }
        } else {
            for (int h = 0; h < num_rules; h++) {
                probs_mix[q * num_rules + h] /= row_sum;
            }
        }
    }
}

int GA_SLHH_Solver::RouletteSelect(const std::vector<double>& fitness) const
{
    const double eps = 1e-9;
    double total = 0.0;
    std::vector<double> w(fitness.size(), 0.0);
    for (size_t i = 0; i < fitness.size(); i++) {
        w[i] = 1.0 / (fitness[i] + eps);
        total += w[i];
    }
    double r = randval(0.0, total);
    double acc = 0.0;
    for (size_t i = 0; i < w.size(); i++) {
        acc += w[i];
        if (r <= acc) return (int)i;
    }
    return (int)w.size() - 1;
}

void GA_SLHH_Solver::ApplyCrossover()
{
    int start = elitism;
    if (start < 0) start = 0;
    if (start > popsize) start = popsize;
    for (int i = start; i + 1 < popsize; i += 2) {
        if (randval(0.0, 1.0) < pc) {
            int crossover_type = rand() % 3;
            if (crossover_type == 0) {
                int pt = 1 + (rand() % (Nvar - 1));
                for (int j = pt; j < Nvar; j++) {
                    std::swap(newpop[i][j], newpop[i + 1][j]);
                }
            } else if (crossover_type == 1) {
                int pt1 = rand() % Nvar;
                int pt2 = rand() % Nvar;
                if (pt1 > pt2) std::swap(pt1, pt2);
                for (int j = pt1; j <= pt2; j++) {
                    std::swap(newpop[i][j], newpop[i + 1][j]);
                }
            } else {
                for (int j = 0; j < Nvar; j++) {
                    if (randval(0.0, 1.0) < 0.5) {
                        std::swap(newpop[i][j], newpop[i + 1][j]);
                    }
                }
            }
        }
    }
}

void GA_SLHH_Solver::ApplyMutation()
{
    if (pm <= 0.0) return;
    int offload_len = CE_Tnum;
    int server_len = CE_Tnum;
    int seq_len = ops;
    int dev_len = ops;
    int offload_start = 0;
    int server_start = CE_Tnum;
    int seq_start = 2 * CE_Tnum;
    int dev_start = 2 * CE_Tnum + ops;

    int start = elitism;
    if (start < 0) start = 0;
    if (start > popsize) start = popsize;

    for (int i = start; i < popsize; i++) {
        // Stage-aware mutation
        if (randval(0.0, 1.0) < pm) {
            if (offload_len > 0) {
                int idx = offload_start + rand() % offload_len;
                newpop[i][idx] = randval(0.0, 1.0);
            }
            if (server_len > 0) {
                int idx = server_start + rand() % server_len;
                newpop[i][idx] = randval(0.0, 1.0);
            }
            if (seq_len > 0) {
                int idx = seq_start + rand() % seq_len;
                newpop[i][idx] = randval(0.0, 1.0);
            }
            if (dev_len > 0) {
                int idx = dev_start + rand() % dev_len;
                newpop[i][idx] = randval(0.0, 1.0);
            }
        }

        // Gene-level mutation for exploration
        if (gene_pm > 0.0) {
            for (int j = 0; j < Nvar; j++) {
                if (randval(0.0, 1.0) < gene_pm) {
                    newpop[i][j] = randval(0.0, 1.0);
                }
            }
        }
    }

    // Random immigrants
    if (immigrant_rate > 0.0) {
        if (start >= popsize) return;
        int immigrants = (int)std::round(immigrant_rate * popsize);
        if (immigrants < 1) immigrants = 1;
        for (int k = 0; k < immigrants; k++) {
            int idx = start + (rand() % (popsize - start));
            for (int j = 0; j < Nvar; j++) {
                newpop[idx][j] = randval(0.0, 1.0);
            }
        }
    }
}

double GA_SLHH_Solver::ChromosomeDistance(const std::vector<double>& a, const std::vector<double>& b) const
{
    double dist = 0.0;
    for (int i = 0; i < Nvar; i++) {
        dist += std::abs(a[i] - b[i]);
    }
    return dist / (double)Nvar;
}

void GA_SLHH_Solver::MaintainDiversity()
{
    std::vector<int> idx(popsize);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

    for (int i = popsize / 2; i < popsize; i++) {
        int cur = idx[i];
        double min_dist = std::numeric_limits<double>::max();
        int compare_count = std::min(5, i);
        for (int j = 0; j < compare_count; j++) {
            double d = ChromosomeDistance(pop[cur], pop[idx[j]]);
            if (d < min_dist) min_dist = d;
        }

        if (min_dist < diversity_threshold) {
            for (int j = 0; j < Nvar; j++) {
                pop[cur][j] = randval(0.0, 1.0);
            }
            const std::vector<double>& probs = (cur < popsize / 2) ? uniform_probs : probs_mix;
            DecodeLLH(pop[cur], probs, llh_pop[cur]);
            pop_fit[cur] = EvalLLH(llh_pop[cur], eval_buffer);
        }
    }
}

void GA_SLHH_Solver::DecodeLLH(const std::vector<double>& chrom,
                              const std::vector<double>& probs,
                              std::vector<int>& llh_out) const
{
    if ((int)llh_out.size() != Nvar) llh_out.resize(Nvar);
    for (int q = 0; q < Nvar; q++) {
        double r = chrom[q];
        double cum = 0.0;
        int pick = num_rules - 1;
        for (int h = 0; h < num_rules; h++) {
            double p = probs[(probs.size() == (size_t)num_rules)
                             ? h
                             : (q * num_rules + h)];
            cum += p;
            if (r <= cum) {
                pick = h;
                break;
            }
        }
        llh_out[q] = pick;
    }
}

double GA_SLHH_Solver::EvalLLH(const std::vector<int>& llh, std::vector<double>& var_buf) const
{
    DecodeToSolution(llh, var_buf);
    return solver->Eval(var_buf.data());
}

std::vector<double> GA_SLHH_Solver::ComputeProbabilities(const std::vector<std::vector<int>>& seqs) const
{
    std::vector<double> probs(Nvar * num_rules, 0.0);
    if (seqs.empty()) {
        for (int q = 0; q < Nvar; q++) {
            for (int h = 0; h < num_rules; h++) {
                probs[q * num_rules + h] = 1.0 / (double)num_rules;
            }
        }
        return probs;
    }

    std::vector<int> counts(num_rules, 0);
    std::vector<int> trans_next(num_rules * num_rules, 0);
    std::vector<int> trans_prev(num_rules * num_rules, 0);

    for (int q = 0; q < Nvar; q++) {
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(trans_next.begin(), trans_next.end(), 0);
        std::fill(trans_prev.begin(), trans_prev.end(), 0);

        for (const auto& seq : seqs) {
            int h = seq[q];
            counts[h]++;
            if (q + 1 < Nvar) {
                int r = seq[q + 1];
                trans_next[h * num_rules + r]++;
            }
            if (q > 0) {
                int r = seq[q - 1];
                trans_prev[h * num_rules + r]++;
            }
        }

        int sum_counts = 0;
        for (int h = 0; h < num_rules; h++) sum_counts += counts[h];

        for (int h = 0; h < num_rules; h++) {
            double p1 = (sum_counts > 0) ? (double)counts[h] / (double)sum_counts
                                         : 1.0 / (double)num_rules;

            double p2 = p1;
            double p3 = p1;
            if (q + 1 < Nvar) {
                int trans_sum = 0;
                for (int r = 0; r < num_rules; r++) trans_sum += trans_next[h * num_rules + r];
                if (trans_sum > 0) {
                    p2 = (double)trans_next[h * num_rules + h] / (double)trans_sum;
                }
            }
            if (q > 0) {
                int trans_sum = 0;
                for (int r = 0; r < num_rules; r++) trans_sum += trans_prev[h * num_rules + r];
                if (trans_sum > 0) {
                    p3 = (double)trans_prev[h * num_rules + h] / (double)trans_sum;
                }
            }

            probs[q * num_rules + h] = (p1 + p2 + p3) / 3.0;
        }

        double row_sum = 0.0;
        for (int h = 0; h < num_rules; h++) row_sum += probs[q * num_rules + h];
        if (row_sum <= 0.0) {
            for (int h = 0; h < num_rules; h++) {
                probs[q * num_rules + h] = 1.0 / (double)num_rules;
            }
        } else {
            for (int h = 0; h < num_rules; h++) {
                probs[q * num_rules + h] /= row_sum;
            }
        }
    }

    return probs;
}

int GA_SLHH_Solver::PickMinLoad(const std::vector<int>& loads) const
{
    int best_idx = 0;
    int best_val = loads.empty() ? 0 : loads[0];
    for (int i = 1; i < (int)loads.size(); i++) {
        if (loads[i] < best_val) {
            best_val = loads[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int GA_SLHH_Solver::PickMaxLoad(const std::vector<int>& loads) const
{
    int best_idx = 0;
    int best_val = loads.empty() ? 0 : loads[0];
    for (int i = 1; i < (int)loads.size(); i++) {
        if (loads[i] > best_val) {
            best_val = loads[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int GA_SLHH_Solver::PickNearestDevice(const std::vector<int>& devices, int ref_dev, double** DtoD) const
{
    if (devices.empty()) return 0;
    int best_dev = devices[0];
    double best_dis = std::numeric_limits<double>::infinity();
    for (int dev : devices) {
        double d = DtoD[ref_dev][dev];
        if (d < best_dis) {
            best_dis = d;
            best_dev = dev;
        }
    }
    return best_dev;
}

int GA_SLHH_Solver::PickNearestEdge(const std::vector<int>& edges, const std::vector<int>& task_devices, double** EtoD) const
{
    if (edges.empty()) return 0;
    int best_edge = edges[0];
    double best_avg = std::numeric_limits<double>::infinity();
    for (int edge : edges) {
        double sum = 0.0;
        for (int dev : task_devices) sum += EtoD[edge][dev];
        double avg = task_devices.empty() ? 0.0 : (sum / (double)task_devices.size());
        if (avg < best_avg) {
            best_avg = avg;
            best_edge = edge;
        }
    }
    return best_edge;
}

int GA_SLHH_Solver::PickMinLoadEdge(const std::vector<int>& edges, const std::vector<int>& edge_load) const
{
    if (edges.empty()) return 0;
    int best_edge = edges[0];
    int best_load = edge_load[best_edge];
    for (int edge : edges) {
        if (edge_load[edge] < best_load) {
            best_load = edge_load[edge];
            best_edge = edge;
        }
    }
    return best_edge;
}

int GA_SLHH_Solver::PickEnergyAwareEdge(const std::vector<int>& edges, const std::vector<int>& edge_load,
                                        const std::vector<int>& cloud_load, const double* EnergyList) const
{
    if (edges.empty()) return 0;
    int cloud_pick = PickMinLoad(cloud_load);
    int edge_pick = PickMinLoadEdge(edges, edge_load);
    int cloud_ratio = (int)((cloud_load[cloud_pick] + 1) / 20.0 * 10.0);
    if (cloud_ratio > 10) cloud_ratio = 10;
    int edge_ratio = (int)((edge_load[edge_pick] + 1) / 6.0 * 10.0);
    if (edge_ratio > 10) edge_ratio = 10;
    if (EnergyList[edge_ratio] < EnergyList[cloud_ratio]) return edge_pick;
    return -1;
}

double GA_SLHH_Solver::VarForIndex(int count, int idx) const
{
    if (count <= 1) return 0.0;
    if (idx <= 0) return 0.0;
    if (idx >= count - 1) return 1.0;
    return (double)idx / (double)(count - 1);
}

void GA_SLHH_Solver::DecodeToSolution(const std::vector<int>& llh, std::vector<double>& var_out) const
{
    if ((int)var_out.size() != Nvar) var_out.resize(Nvar);

    // Reset decode buffers.
    if (!cloud_load_buf.empty()) std::fill(cloud_load_buf.begin(), cloud_load_buf.end(), 0);
    if (!edge_load_buf.empty()) std::fill(edge_load_buf.begin(), edge_load_buf.end(), 0);
    if (!device_load_buf.empty()) std::fill(device_load_buf.begin(), device_load_buf.end(), 0);
    if (!last_device_per_job_buf.empty()) std::fill(last_device_per_job_buf.begin(), last_device_per_job_buf.end(), -1);
    int last_global_device = -1;

    // Offload and server decisions
    for (int t = 0; t < CE_Tnum; t++) {
        int rule_off = llh[t] % 7;
        int rule_srv = llh[CE_Tnum + t] % 7;

        const std::vector<int>& edge_list = solver->CETask_Property[t].AvailEdgeServerList;
        const std::vector<int>& task_devices = task_devices_per_task_cache[t];

        bool edge_mode = false;
        int cloud_idx = 0;
        int edge_idx = 0;

        if (rule_off == 0) {
            if (!edge_list.empty()) {
                int nearest = nearest_edge_for_task_cache[t];
                if (nearest >= 0) {
                    edge_mode = true;
                    edge_idx = nearest;
                } else {
                    cloud_idx = PickMinLoad(cloud_load_buf);
                }
            } else {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
        } else if (rule_off == 1) {
            edge_mode = false;
            cloud_idx = PickMinLoad(cloud_load_buf);
        } else if (rule_off == 2) {
            if (!edge_list.empty()) {
                edge_mode = true;
                edge_idx = PickMinLoadEdge(edge_list, edge_load_buf);
            } else {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
        } else if (rule_off == 3) {
            double ratio = solver->CETask_Property[t].Communication /
                           (solver->CETask_Property[t].Computation + 1e-6);
            if (!edge_list.empty() && ratio > 0.5) {
                int nearest = nearest_edge_for_task_cache[t];
                if (nearest >= 0) {
                    edge_mode = true;
                    edge_idx = nearest;
                } else {
                    cloud_idx = PickMinLoad(cloud_load_buf);
                }
            } else {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
        } else if (rule_off == 4) {
            if (!edge_list.empty()) {
                int energy_edge = PickEnergyAwareEdge(edge_list, edge_load_buf, cloud_load_buf, solver->EnergyList);
                if (energy_edge >= 0) {
                    edge_mode = true;
                    edge_idx = energy_edge;
                } else {
                    cloud_idx = PickMinLoad(cloud_load_buf);
                }
            } else {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
        } else if (rule_off == 5) {
            edge_mode = (!edge_list.empty() && randval(0.0, 1.0) > 0.5);
            if (edge_mode) {
                edge_idx = edge_list[PickRandomIndex((int)edge_list.size())];
            } else {
                cloud_idx = PickRandomIndex(solver->Cnum);
            }
        } else {
            if (!edge_list.empty() && !task_devices.empty()) {
                int dev = task_central_device_cache[t];
                if (dev < 0) dev = task_devices[0];
                int near_edge = nearest_edge_for_device[dev];
                if (near_edge >= 0 &&
                    near_edge < solver->Enum &&
                    task_edge_allowed_cache[t * solver->Enum + near_edge]) {
                    edge_mode = true;
                    edge_idx = near_edge;
                } else {
                    cloud_idx = PickMinLoad(cloud_load_buf);
                }
            } else {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
        }

        if (edge_mode) {
            var_out[t] = 0.75;
            int edge_pos = 0;
            if (edge_idx >= 0 && edge_idx < solver->Enum) {
                int cached_pos = task_edge_pos_cache[t * solver->Enum + edge_idx];
                if (cached_pos >= 0) edge_pos = cached_pos;
            }
            if (rule_srv == 1) {
                int edge_min = PickMinLoadEdge(edge_list, edge_load_buf);
                if (edge_min >= 0 && edge_min < solver->Enum) {
                    int cached_pos = task_edge_pos_cache[t * solver->Enum + edge_min];
                    if (cached_pos >= 0) edge_pos = cached_pos;
                }
            } else if (rule_srv == 5 || rule_srv == 6) {
                edge_pos = PickRandomIndex((int)edge_list.size());
            }
            if (!edge_list.empty()) {
                edge_idx = edge_list[edge_pos];
            }
            var_out[CE_Tnum + t] = VarForIndex((int)edge_list.size(), edge_pos);
            edge_load_buf[edge_idx]++;
        } else {
            var_out[t] = 0.25;
            if (rule_srv == 5 || rule_srv == 6) {
                cloud_idx = PickRandomIndex(solver->Cnum);
            } else if (rule_srv == 2 || rule_srv == 3 || rule_srv == 4) {
                cloud_idx = PickMinLoad(cloud_load_buf);
            }
            var_out[CE_Tnum + t] = VarForIndex(solver->Cnum, cloud_idx);
            cloud_load_buf[cloud_idx]++;
        }
    }

    // Sequence block: compute priority scores and order
    for (int op = 0; op < ops; op++) {
        int rule = llh[2 * CE_Tnum + op];
        int job = op / M_OPTnum;
        double proc = solver->MTask_Time[op];
        double rem_time = rem_time_cache[op];
        int rem_ops = rem_ops_cache[op];
        double avg_proc = (M_OPTnum > 0) ? (job_total_cache[job] / (double)M_OPTnum) : proc;

        double score = 0.0;
        switch (rule) {
            case 0: score = (double)op; break; // FIFO
            case 1: score = proc; break;       // SPT
            case 2: score = -proc; break;      // LPT
            case 3: score = rem_time; break;   // LWKR
            case 4: score = -rem_time; break;  // MWKR
            case 5: score = avg_proc; break;   // SPTswm
            case 6: score = -avg_proc; break;  // LPTswm
            case 7: score = (job_total_cache[job] > 0.0) ? (proc / job_total_cache[job]) : proc; break;   // SDT
            case 8: score = (job_total_cache[job] > 0.0) ? -(proc / job_total_cache[job]) : -proc; break; // LDT
            case 9: score = (rem_time > 0.0) ? (proc / rem_time) : proc; break;               // SDR
            case 10: score = (rem_time > 0.0) ? -(proc / rem_time) : -proc; break;            // LDR
            case 11: score = (double)rem_ops; break;  // FRO
            case 12: score = -(double)rem_ops; break; // LRO
            default: score = randval(0.0, 1.0); break; // Random
        }
        op_score_buf[op] = score;
    }

    if ((int)op_order_buf.size() != ops) op_order_buf.resize(ops);
    std::iota(op_order_buf.begin(), op_order_buf.end(), 0);
    std::sort(op_order_buf.begin(), op_order_buf.end(),
              [&](int a, int b) {
                  if (op_score_buf[a] == op_score_buf[b]) return a < b;
                  return op_score_buf[a] < op_score_buf[b];
              });
    if ((int)rank_buf.size() != ops) rank_buf.resize(ops);
    for (int i = 0; i < ops; i++) rank_buf[op_order_buf[i]] = i;
    for (int op = 0; op < ops; op++) {
        var_out[2 * CE_Tnum + op] = (ops <= 1) ? 0.0 : (double)rank_buf[op] / (double)(ops - 1);
    }

    // Device block: follow sequence order
    int dev_start = 2 * CE_Tnum + ops;
    auto pick_min_load_in_avail = [&](const std::vector<int>& avail) -> int {
        int best_dev = avail.empty() ? 0 : avail[0];
        int best_load = avail.empty() ? 0 : device_load_buf[best_dev];
        for (int dev : avail) {
            if (device_load_buf[dev] < best_load) {
                best_load = device_load_buf[dev];
                best_dev = dev;
            }
        }
        return best_dev;
    };
    auto pick_max_load_in_avail = [&](const std::vector<int>& avail) -> int {
        int best_dev = avail.empty() ? 0 : avail[0];
        int best_load = avail.empty() ? 0 : device_load_buf[best_dev];
        for (int dev : avail) {
            if (device_load_buf[dev] > best_load) {
                best_load = device_load_buf[dev];
                best_dev = dev;
            }
        }
        return best_dev;
    };
    for (int pos = 0; pos < ops; pos++) {
        int op = op_order_buf[pos];
        int rule = llh[dev_start + pos] % 7;
        int job = op / M_OPTnum;
        const std::vector<int>& avail = solver->AvailDeviceList[op];

        int chosen_dev = 0;
        if (!avail.empty()) {
            if (rule == 0) {
                if (last_device_per_job_buf[job] >= 0) {
                    chosen_dev = PickNearestDevice(avail, last_device_per_job_buf[job], solver->DtoD_Distance);
                } else {
                    chosen_dev = pick_min_load_in_avail(avail);
                }
            } else if (rule == 1) {
                chosen_dev = pick_min_load_in_avail(avail);
            } else if (rule == 2) {
                chosen_dev = avail[PickRandomIndex((int)avail.size())];
            } else if (rule == 3) {
                if (last_global_device >= 0) {
                    chosen_dev = PickNearestDevice(avail, last_global_device, solver->DtoD_Distance);
                } else {
                    chosen_dev = avail[PickRandomIndex((int)avail.size())];
                }
            } else if (rule == 4) {
                chosen_dev = pick_max_load_in_avail(avail);
            } else if (rule == 5) {
                chosen_dev = avail[0];
            } else {
                chosen_dev = avail[PickRandomIndex((int)avail.size())];
            }
        }

        int idx = 0;
        if (chosen_dev >= 0 && chosen_dev < solver->Dnum) {
            int p = op_device_pos_cache[op * solver->Dnum + chosen_dev];
            if (p >= 0) idx = p;
        }
        var_out[dev_start + pos] = VarForIndex((int)avail.size(), idx);

        if (chosen_dev >= 0 && chosen_dev < solver->Dnum) {
            device_load_buf[chosen_dev]++;
            last_device_per_job_buf[job] = chosen_dev;
            last_global_device = chosen_dev;
        }
    }
}
