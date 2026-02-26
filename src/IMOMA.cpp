#include "IMOMA.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

using std::vector;

void IMOMAOperatorStats::Update(bool improved_pf, int archive_size_change, double eta)
{
    double sigma = 9.0;
    if (improved_pf) sigma = 33.0;
    else if (archive_size_change == 0) sigma = 13.0;

    score += sigma;
    usage_count++;
    // Exponential response to recent operator contribution.
    weight = weight * (1.0 - eta) + eta * sigma;
    if (weight < 1e-6) weight = 1e-6;
}

IMOMA_Solver::IMOMA_Solver(MultiMet* s, int psize, double arc, int gens)
    : solver(s),
      pop_size(psize > 1 ? psize : 2),
      arc_ratio(std::clamp(arc, 0.05, 1.0)),
      max_generations(gens > 0 ? gens : 1),
      archive_capacity(1),
      Nvar(s ? s->Nvar : 0),
      CE_Tnum(s ? s->CE_Tnum : 0),
      M_Jnum(s ? s->M_Jnum : 0),
      M_OPTnum(s ? s->M_OPTnum : 0),
      ops((s ? s->M_Jnum : 0) * (s ? s->M_OPTnum : 0)),
      rng((unsigned int)(s ? s->seed : 1u)),
      best_scalar_fit(std::numeric_limits<double>::infinity()),
      last_best_scalar_fit(std::numeric_limits<double>::infinity()),
      stagnation_count(0),
      has_scalar_best(false),
      restart_count(0),
      suppression_factor(0.3)
{
    archive_capacity = std::max(1, (int)std::round(pop_size * arc_ratio));
    operators.emplace_back("Mutation");
    operators.emplace_back("EO");
    operators.emplace_back("MTO");
    task_order_buf.resize(CE_Tnum);
    std::iota(task_order_buf.begin(), task_order_buf.end(), 0);
}

double IMOMA_Solver::Rand01()
{
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

int IMOMA_Solver::RandInt(int lo, int hi_exclusive)
{
    if (hi_exclusive <= lo) return lo;
    return std::uniform_int_distribution<int>(lo, hi_exclusive - 1)(rng);
}

void IMOMA_Solver::EvaluateObjectives(const vector<double>& var, double& makespan, double& energy) const
{
    const double old_alpha = solver->workspace.alpha;
    solver->workspace.set_alpha(1.0);
    makespan = solver->Eval(var.data());
    solver->workspace.set_alpha(0.0);
    energy = solver->Eval(var.data());
    solver->workspace.set_alpha(old_alpha);
}

void IMOMA_Solver::Evaluate(IMOMAIndividual& ind)
{
    EvaluateObjectives(ind.var, ind.makespan, ind.energy);
    const double alpha = std::clamp(solver->workspace.alpha, 0.0, 1.0);
    const double scalar = alpha * ind.makespan + (1.0 - alpha) * ind.energy;
    if (scalar + 1e-12 < best_scalar_fit) {
        best_scalar_fit = scalar;
        scalar_best_ind = ind;
        has_scalar_best = true;
    }
}

IMOMAIndividual IMOMA_Solver::RandomIndividual()
{
    IMOMAIndividual ind;
    ind.var.resize(Nvar, 0.0);
    for (int j = 0; j < Nvar; j++) ind.var[j] = Rand01();
    Evaluate(ind);
    return ind;
}

vector<IMOMAIndividual> IMOMA_Solver::RandomInitialize(int size)
{
    vector<IMOMAIndividual> pop;
    pop.reserve(size);

    int seeded = 0;
    if (solver && solver->pop && solver->Popsize > 0) {
        seeded = std::min(size, solver->Popsize);
        for (int i = 0; i < seeded; i++) {
            IMOMAIndividual ind;
            ind.var.resize(Nvar);
            for (int j = 0; j < Nvar; j++) ind.var[j] = solver->pop[i][j];
            Evaluate(ind);
            pop.push_back(std::move(ind));
        }
    }
    for (int i = seeded; i < size; i++) pop.push_back(RandomIndividual());
    return pop;
}

void IMOMA_Solver::Init()
{
    best_scalar_fit = std::numeric_limits<double>::infinity();
    last_best_scalar_fit = std::numeric_limits<double>::infinity();
    stagnation_count = 0;
    restart_count = 0;
    has_scalar_best = false;
    population = RandomInitialize(pop_size);
    archive.clear();
    UpdateArchive(population);
    last_best_scalar_fit = best_scalar_fit;
}

double IMOMA_Solver::CalculateOmega(int g) const
{
    const double alpha = 0.5;
    const double beta = 0.5;
    const double lambda = 2.0;
    const double gg = (double)g / (double)max_generations;
    return alpha * std::sin(M_PI * gg) + beta * std::sin(2.0 * M_PI * gg) + lambda;
}

IMOMAIndividual IMOMA_Solver::GenerateOpposite(const IMOMAIndividual& x, double omega)
{
    IMOMAIndividual y = x;
    const double k = std::clamp(omega / 3.0, 0.05, 1.0);
    const double jitter = 0.02 + 0.08 * k;
    for (int i = 0; i < Nvar; i++) {
        const double opposite = 1.0 - x.var[i];
        const double dyn = x.var[i] + k * (opposite - x.var[i]);
        y.var[i] = std::clamp(dyn + jitter * (Rand01() - 0.5), 0.0, 1.0);
    }
    Evaluate(y);
    return y;
}

IMOMAIndividual IMOMA_Solver::BiasedCrossover(const IMOMAIndividual& a, const IMOMAIndividual& b, double bias)
{
    IMOMAIndividual c;
    c.var.resize(Nvar);
    for (int i = 0; i < Nvar; i++) {
        c.var[i] = (Rand01() < bias) ? a.var[i] : b.var[i];
    }
    Evaluate(c);
    return c;
}

int IMOMA_Solver::DecodeCloudIndex(const vector<double>& var, int task) const
{
    int Cnum = solver->Cnum;
    int idx = (Cnum > 1) ? (int)(var[CE_Tnum + task] * (Cnum - 1)) : 0;
    if (idx < 0) idx = 0;
    if (idx >= Cnum) idx = Cnum - 1;
    return idx;
}

int IMOMA_Solver::DecodeEdgeIndex(const vector<double>& var, int task) const
{
    const vector<int>& edges = solver->CETask_Property[task].AvailEdgeServerList;
    if (edges.empty()) return -1;
    int idx = (edges.size() > 1) ? (int)(var[CE_Tnum + task] * (double)(edges.size() - 1)) : 0;
    if (idx < 0) idx = 0;
    if (idx >= (int)edges.size()) idx = (int)edges.size() - 1;
    return edges[idx];
}

bool IMOMA_Solver::DecodeIsEdge(const vector<double>& var, int task) const
{
    return var[task] > 0.5;
}

void IMOMA_Solver::SetTaskCloud(vector<double>& var, int task, int cloud_idx) const
{
    cloud_idx = std::clamp(cloud_idx, 0, solver->Cnum - 1);
    var[task] = 0.25;
    if (solver->Cnum <= 1) {
        var[CE_Tnum + task] = 0.0;
    } else {
        var[CE_Tnum + task] = (double)cloud_idx / (double)(solver->Cnum - 1);
    }
}

void IMOMA_Solver::SetTaskEdge(vector<double>& var, int task, int edge_idx) const
{
    const vector<int>& edges = solver->CETask_Property[task].AvailEdgeServerList;
    if (edges.empty()) {
        SetTaskCloud(var, task, 0);
        return;
    }
    int pos = 0;
    for (int i = 0; i < (int)edges.size(); i++) {
        if (edges[i] == edge_idx) {
            pos = i;
            break;
        }
    }
    var[task] = 0.75;
    if (edges.size() <= 1) var[CE_Tnum + task] = 0.0;
    else var[CE_Tnum + task] = (double)pos / (double)(edges.size() - 1);
}

int IMOMA_Solver::PickNearestAllowedEdgeForTask(int task) const
{
    const vector<int>& edges = solver->CETask_Property[task].AvailEdgeServerList;
    if (edges.empty()) return -1;
    double best_avg = std::numeric_limits<double>::infinity();
    int best_edge = edges[0];
    for (int edge : edges) {
        double sum = 0.0;
        for (int j = 0; j < M_OPTnum; j++) {
            int op = task * M_OPTnum + j;
            const vector<int>& avail = solver->AvailDeviceList[op];
            if (!avail.empty()) sum += solver->EtoD_Distance[edge][avail[0]];
        }
        double avg = sum / std::max(1, M_OPTnum);
        if (avg < best_avg) {
            best_avg = avg;
            best_edge = edge;
        }
    }
    return best_edge;
}

int IMOMA_Solver::PickTaskLayerCode(const vector<double>& var, int task) const
{
    if (DecodeIsEdge(var, task)) return 1;
    return 2;
}

IMOMAIndividual IMOMA_Solver::Mutation(const IMOMAIndividual& x)
{
    IMOMAIndividual y = x;
    if (Nvar <= 0) return y;

    // Paper-style partial segment redistribution on task assignment genes.
    const int seg_max = std::max(1, CE_Tnum / 4);
    const int seg_len = RandInt(1, seg_max + 1);
    const int start_task = RandInt(0, std::max(1, CE_Tnum - seg_len + 1));
    for (int t = start_task; t < start_task + seg_len; t++) {
        if (Rand01() < 0.5) {
            const vector<int>& edges = solver->CETask_Property[t].AvailEdgeServerList;
            if (!edges.empty()) {
                int e = edges[RandInt(0, (int)edges.size())];
                SetTaskEdge(y.var, t, e);
            } else {
                SetTaskCloud(y.var, t, RandInt(0, solver->Cnum));
            }
        } else {
            SetTaskCloud(y.var, t, RandInt(0, solver->Cnum));
        }
    }

    // Keep a light perturbation on remaining dimensions to retain diversity.
    const int extra = std::max(1, Nvar / 100);
    for (int k = 0; k < extra; k++) {
        int idx = RandInt(0, Nvar);
        y.var[idx] = std::clamp(y.var[idx] + 0.20 * (Rand01() - 0.5), 0.0, 1.0);
    }

    Evaluate(y);
    return y;
}

IMOMAIndividual IMOMA_Solver::DifferentialMutation(const IMOMAIndividual& x)
{
    IMOMAIndividual y = x;
    if (population.empty()) {
        Evaluate(y);
        return y;
    }

    const vector<IMOMAIndividual>& pool = archive.empty() ? population : archive;
    if ((int)pool.size() < 3) return Mutation(x);

    int r1 = RandInt(0, (int)pool.size());
    int r2 = RandInt(0, (int)pool.size());
    int r3 = RandInt(0, (int)pool.size());
    if ((int)pool.size() > 3) {
        while (r2 == r1) r2 = RandInt(0, (int)pool.size());
        while (r3 == r1 || r3 == r2) r3 = RandInt(0, (int)pool.size());
    }
    const double F = 0.5 + 0.4 * Rand01();
    const double CR = 0.75;
    const int jrand = RandInt(0, std::max(1, Nvar));
    y.var.resize(Nvar);
    for (int j = 0; j < Nvar; j++) {
        if (Rand01() < CR || j == jrand) {
            const double v = pool[r1].var[j] + F * (pool[r2].var[j] - pool[r3].var[j]);
            y.var[j] = std::clamp(v, 0.0, 1.0);
        } else {
            y.var[j] = x.var[j];
        }
    }
    Evaluate(y);
    return y;
}

IMOMAIndividual IMOMA_Solver::BestGuidedMutation(const IMOMAIndividual& x, const IMOMAIndividual& best_ref)
{
    IMOMAIndividual y = x;
    if (Nvar <= 0) return y;

    const vector<IMOMAIndividual>& pool = archive.empty() ? population : archive;
    if (pool.empty()) {
        Evaluate(y);
        return y;
    }

    int r1 = RandInt(0, (int)pool.size());
    int r2 = RandInt(0, (int)pool.size());
    if ((int)pool.size() > 1) {
        while (r2 == r1) r2 = RandInt(0, (int)pool.size());
    }

    const double F1 = 0.35 + 0.25 * Rand01();
    const double F2 = 0.20 + 0.20 * Rand01();
    const double CR = 0.90;
    const int jrand = RandInt(0, std::max(1, Nvar));

    y.var.resize(Nvar);
    for (int j = 0; j < Nvar; j++) {
        if (Rand01() < CR || j == jrand) {
            const double v = x.var[j]
                           + F1 * (best_ref.var[j] - x.var[j])
                           + F2 * (pool[r1].var[j] - pool[r2].var[j]);
            y.var[j] = std::clamp(v, 0.0, 1.0);
        } else {
            y.var[j] = x.var[j];
        }
    }

    const int copy_tasks = std::max(1, CE_Tnum / 25);
    for (int c = 0; c < copy_tasks; c++) {
        int t = RandInt(0, CE_Tnum);
        y.var[t] = best_ref.var[t];
        y.var[CE_Tnum + t] = best_ref.var[CE_Tnum + t];
    }

    Evaluate(y);
    return y;
}

IMOMAIndividual IMOMA_Solver::DiversifyFromBest(double intensity)
{
    const double scale = std::clamp(intensity, 0.05, 1.0);
    IMOMAIndividual y;
    if (has_scalar_best) {
        const double p_from_best = std::clamp(0.75 - 0.55 * scale, 0.15, 0.85);
        if (Rand01() < p_from_best) y = scalar_best_ind;
        else if (!archive.empty()) y = archive[RandInt(0, (int)archive.size())];
        else if (!population.empty()) y = population[RandInt(0, (int)population.size())];
        else y = RandomIndividual();
    } else if (!archive.empty()) {
        y = archive[RandInt(0, (int)archive.size())];
    } else if (!population.empty()) {
        y = population[RandInt(0, (int)population.size())];
    } else {
        y = RandomIndividual();
    }

    const int task_changes = std::max(1, (int)std::round((0.07 + 0.43 * scale) * CE_Tnum));
    for (int k = 0; k < task_changes; k++) {
        int t = RandInt(0, CE_Tnum);
        if (Rand01() < (0.55 + 0.25 * scale)) {
            if (Rand01() < 0.65) {
                int e = PickNearestAllowedEdgeForTask(t);
                if (e >= 0) SetTaskEdge(y.var, t, e);
                else SetTaskCloud(y.var, t, RandInt(0, solver->Cnum));
            } else {
                int e = DecodeEdgeIndex(y.var, t);
                if (e < 0) e = PickNearestAllowedEdgeForTask(t);
                if (e >= 0) SetTaskEdge(y.var, t, e);
                else SetTaskCloud(y.var, t, RandInt(0, solver->Cnum));
            }
        } else {
            SetTaskCloud(y.var, t, RandInt(0, solver->Cnum));
        }
    }

    const int seg_len = std::max(1, (int)std::round((0.04 + 0.28 * scale) * Nvar));
    const int start = RandInt(0, std::max(1, Nvar - seg_len + 1));
    for (int i = start; i < start + seg_len; i++) {
        y.var[i] = Rand01();
    }

    const int jitter = std::max(1, (int)std::round((0.02 + 0.11 * scale) * Nvar));
    const double amp = 0.10 + 0.50 * scale;
    for (int j = 0; j < jitter; j++) {
        int idx = RandInt(0, Nvar);
        y.var[idx] = std::clamp(y.var[idx] + amp * (Rand01() - 0.5), 0.0, 1.0);
    }

    Evaluate(y);
    return y;
}

void IMOMA_Solver::InjectExplorers(vector<IMOMAIndividual>& pop, int count, double intensity)
{
    if (pop.empty() || count <= 0) return;
    int inject_n = std::min(count, (int)pop.size());
    vector<IMOMAIndividual> injected;
    injected.reserve(inject_n);

    for (int i = 0; i < inject_n; i++) {
        IMOMAIndividual cand;
        const double guided_prob = std::clamp(0.75 - 0.55 * intensity, 0.15, 0.75);
        if (has_scalar_best && Rand01() < guided_prob) {
            const IMOMAIndividual& base = pop[RandInt(0, (int)pop.size())];
            cand = BestGuidedMutation(base, scalar_best_ind);
            if (Rand01() < std::clamp(0.25 + intensity * 0.55, 0.0, 0.95)) {
                cand = Mutation(cand);
            }
        } else if (Rand01() < std::clamp(0.08 + 0.22 * intensity, 0.0, 0.4)) {
            cand = RandomIndividual();
        } else {
            cand = DiversifyFromBest(intensity);
        }

        int worst_idx = 0;
        double worst_sf = ScalarFit(pop[0]);
        for (int w = 1; w < (int)pop.size(); w++) {
            double sf = ScalarFit(pop[w]);
            if (sf > worst_sf) {
                worst_sf = sf;
                worst_idx = w;
            }
        }
        pop[worst_idx] = cand;
        injected.push_back(std::move(cand));
    }

    if (!injected.empty()) UpdateArchive(injected);
}

IMOMAIndividual IMOMA_Solver::EnergyOptimization(const IMOMAIndividual& x)
{
    IMOMAIndividual best = x;
    auto decode_sid = [&](const vector<double>& var, int task) -> int {
        if (DecodeIsEdge(var, task)) {
            int e = DecodeEdgeIndex(var, task);
            if (e >= 0) return solver->Cnum + e;
        }
        return DecodeCloudIndex(var, task);
    };
    auto assign_sid = [&](vector<double>& var, int task, int sid) {
        if (sid < solver->Cnum) {
            SetTaskCloud(var, task, sid);
        } else {
            int edge = sid - solver->Cnum;
            const vector<int>& edges = solver->CETask_Property[task].AvailEdgeServerList;
            bool ok = false;
            for (int e : edges) {
                if (e == edge) {
                    ok = true;
                    break;
                }
            }
            if (ok) SetTaskEdge(var, task, edge);
            else if (!edges.empty()) SetTaskEdge(var, task, edges[RandInt(0, (int)edges.size())]);
            else SetTaskCloud(var, task, RandInt(0, solver->Cnum));
        }
    };

    const int attempts = std::max(3, CE_Tnum / 20);
    for (int a = 0; a < attempts; a++) {
        IMOMAIndividual cand = best;
        vector<int> active;
        active.reserve(CE_Tnum);
        for (int t = 0; t < CE_Tnum; t++) {
            int sid = decode_sid(cand.var, t);
            bool exists = false;
            for (int v : active) {
                if (v == sid) {
                    exists = true;
                    break;
                }
            }
            if (!exists) active.push_back(sid);
        }
        if (active.size() <= 1) break;

        std::shuffle(active.begin(), active.end(), rng);
        int target_n = std::max(1, (int)std::round((0.45 + 0.25 * Rand01()) * (double)active.size()));
        target_n = std::min(target_n, (int)active.size());
        vector<int> targets(active.begin(), active.begin() + target_n);

        for (int t = 0; t < CE_Tnum; t++) {
            int sid = decode_sid(cand.var, t);
            bool is_target = false;
            for (int v : targets) {
                if (v == sid) {
                    is_target = true;
                    break;
                }
            }
            if (is_target) continue;

            vector<int> feasible;
            feasible.reserve(targets.size());
            const vector<int>& edges = solver->CETask_Property[t].AvailEdgeServerList;
            for (int v : targets) {
                if (v < solver->Cnum) {
                    feasible.push_back(v);
                } else {
                    int edge = v - solver->Cnum;
                    for (int e : edges) {
                        if (e == edge) {
                            feasible.push_back(v);
                            break;
                        }
                    }
                }
            }

            if (!feasible.empty()) {
                assign_sid(cand.var, t, feasible[RandInt(0, (int)feasible.size())]);
            } else {
                if (!edges.empty() && Rand01() < 0.7) {
                    SetTaskEdge(cand.var, t, edges[RandInt(0, (int)edges.size())]);
                } else {
                    SetTaskCloud(cand.var, t, RandInt(0, solver->Cnum));
                }
            }
        }

        Evaluate(cand);
        if (cand.energy + 1e-12 < best.energy || Dominates(cand, best) || ScalarFit(cand) + 1e-12 < ScalarFit(best)) {
            best = std::move(cand);
        }
    }
    return best;
}

IMOMAIndividual IMOMA_Solver::MakespanOptimization(const IMOMAIndividual& x)
{
    IMOMAIndividual best = x;
    vector<int> dep_tasks;
    dep_tasks.reserve(CE_Tnum);
    for (int i = 0; i < CE_Tnum; i++) {
        if (!solver->CETask_Property[i].Precedence.empty()) dep_tasks.push_back(i);
    }
    if (dep_tasks.empty()) return best;

    std::shuffle(dep_tasks.begin(), dep_tasks.end(), rng);
    int picks = std::max(1, (int)std::round((0.20 + 0.30 * Rand01()) * (double)dep_tasks.size()));
    picks = std::min(picks, (int)dep_tasks.size());

    IMOMAIndividual cand = best;
    for (int idx = 0; idx < picks; idx++) {
        int i = dep_tasks[idx];
        const vector<int>& preds = solver->CETask_Property[i].Precedence;
        if (preds.empty()) continue;
        int p = preds[RandInt(0, (int)preds.size())];
        if (p < 0 || p >= CE_Tnum) continue;

        int p_layer = PickTaskLayerCode(cand.var, p);
        if (p_layer == 1) {
            int edge = DecodeEdgeIndex(cand.var, p);
            if (edge < 0) edge = PickNearestAllowedEdgeForTask(i);
            if (edge >= 0) SetTaskEdge(cand.var, i, edge);
            else SetTaskCloud(cand.var, i, RandInt(0, solver->Cnum));
        } else {
            int cloud = DecodeCloudIndex(cand.var, p);
            SetTaskCloud(cand.var, i, cloud);
        }
    }

    Evaluate(cand);
    if (cand.makespan + 1e-12 < best.makespan || Dominates(cand, best) || ScalarFit(cand) + 1e-12 < ScalarFit(best)) {
        best = std::move(cand);
    }
    return best;
}

bool IMOMA_Solver::Dominates(const IMOMAIndividual& a, const IMOMAIndividual& b) const
{
    bool one_better = false;
    if (a.energy > b.energy) return false;
    if (a.makespan > b.makespan) return false;
    if (a.energy < b.energy) one_better = true;
    if (a.makespan < b.makespan) one_better = true;
    return one_better;
}

void IMOMA_Solver::FastNonDominatedSort(vector<IMOMAIndividual>& pop) const
{
    int n = (int)pop.size();
    if ((int)nd_dom_count_buf.size() != n) nd_dom_count_buf.assign(n, 0);
    else std::fill(nd_dom_count_buf.begin(), nd_dom_count_buf.end(), 0);
    if ((int)nd_dom_set_buf.size() != n) nd_dom_set_buf.resize(n);
    for (int i = 0; i < n; i++) nd_dom_set_buf[i].clear();
    vector<int>& dom_count = nd_dom_count_buf;
    vector<vector<int>>& dom_set = nd_dom_set_buf;
    for (int i = 0; i < n; i++) pop[i].rank = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (Dominates(pop[i], pop[j])) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            } else if (Dominates(pop[j], pop[i])) {
                dom_set[j].push_back(i);
                dom_count[i]++;
            }
        }
    }

    nd_cur_front_buf.clear();
    nd_cur_front_buf.reserve(n);
    for (int i = 0; i < n; i++) {
        if (dom_count[i] == 0) {
            pop[i].rank = 0;
            nd_cur_front_buf.push_back(i);
        }
    }

    int rank = 0;
    while (!nd_cur_front_buf.empty()) {
        nd_next_front_buf.clear();
        for (int idx : nd_cur_front_buf) {
            for (int j : dom_set[idx]) {
                dom_count[j]--;
                if (dom_count[j] == 0) {
                    pop[j].rank = rank + 1;
                    nd_next_front_buf.push_back(j);
                }
            }
        }
        rank++;
        nd_cur_front_buf.swap(nd_next_front_buf);
    }
}

void IMOMA_Solver::CalculateCrowdingDistance(vector<IMOMAIndividual>& pop, const vector<int>& front) const
{
    if (front.empty()) return;
    if (front.size() == 1) {
        pop[front[0]].crowding_dist = std::numeric_limits<double>::infinity();
        return;
    }
    for (int idx : front) pop[idx].crowding_dist = 0.0;

    auto assign_obj = [&](bool use_energy) {
        crowd_order_buf.assign(front.begin(), front.end());
        std::sort(crowd_order_buf.begin(), crowd_order_buf.end(), [&](int a, int b) {
            return use_energy ? (pop[a].energy < pop[b].energy)
                              : (pop[a].makespan < pop[b].makespan);
        });
        pop[crowd_order_buf.front()].crowding_dist = std::numeric_limits<double>::infinity();
        pop[crowd_order_buf.back()].crowding_dist = std::numeric_limits<double>::infinity();
        double min_v = use_energy ? pop[crowd_order_buf.front()].energy : pop[crowd_order_buf.front()].makespan;
        double max_v = use_energy ? pop[crowd_order_buf.back()].energy : pop[crowd_order_buf.back()].makespan;
        double range = max_v - min_v;
        if (range <= 1e-12) return;
        for (int i = 1; i + 1 < (int)crowd_order_buf.size(); i++) {
            if (!std::isfinite(pop[crowd_order_buf[i]].crowding_dist)) continue;
            double prev = use_energy ? pop[crowd_order_buf[i - 1]].energy : pop[crowd_order_buf[i - 1]].makespan;
            double next = use_energy ? pop[crowd_order_buf[i + 1]].energy : pop[crowd_order_buf[i + 1]].makespan;
            pop[crowd_order_buf[i]].crowding_dist += (next - prev) / range;
        }
    };

    assign_obj(true);
    assign_obj(false);
}

vector<IMOMAIndividual> IMOMA_Solver::SelectNextGeneration(vector<IMOMAIndividual>& combined, int next_size)
{
    FastNonDominatedSort(combined);

    vector<vector<int>> by_rank;
    int max_rank = 0;
    for (const auto& x : combined) max_rank = std::max(max_rank, x.rank);
    by_rank.resize(max_rank + 1);
    for (int i = 0; i < (int)combined.size(); i++) by_rank[combined[i].rank].push_back(i);

    vector<IMOMAIndividual> next;
    next.reserve(next_size);
    for (int r = 0; r <= max_rank; r++) {
        if (by_rank[r].empty()) continue;
        CalculateCrowdingDistance(combined, by_rank[r]);
        if ((int)next.size() + (int)by_rank[r].size() <= next_size) {
            for (int idx : by_rank[r]) next.push_back(combined[idx]);
        } else {
            select_order_buf.assign(by_rank[r].begin(), by_rank[r].end());
            std::sort(select_order_buf.begin(), select_order_buf.end(), [&](int a, int b) {
                return combined[a].crowding_dist > combined[b].crowding_dist;
            });
            int need = next_size - (int)next.size();
            for (int i = 0; i < need; i++) next.push_back(combined[select_order_buf[i]]);
            break;
        }
    }
    return next;
}

void IMOMA_Solver::UpdateArchive(const vector<IMOMAIndividual>& candidates)
{
    vector<IMOMAIndividual> merged = archive;
    merged.insert(merged.end(), candidates.begin(), candidates.end());
    if (merged.empty()) return;

    FastNonDominatedSort(merged);
    vector<IMOMAIndividual> nd;
    nd.reserve(merged.size());
    for (const auto& ind : merged) {
        if (ind.rank == 0) nd.push_back(ind);
    }
    int dyn_capacity = archive_capacity;
    if (pop_size > archive_capacity) {
        const int gate = std::max(20, max_generations / 80);
        const double ratio = std::clamp((double)stagnation_count / (double)gate, 0.0, 1.0);
        const int bonus = (int)std::round((double)(pop_size - archive_capacity) * ratio);
        dyn_capacity = std::clamp(archive_capacity + bonus, archive_capacity, pop_size);
    }

    if ((int)nd.size() > dyn_capacity) {
        vector<int> front(nd.size());
        std::iota(front.begin(), front.end(), 0);
        CalculateCrowdingDistance(nd, front);
        std::sort(nd.begin(), nd.end(), [](const IMOMAIndividual& a, const IMOMAIndividual& b) {
            return a.crowding_dist > b.crowding_dist;
        });
        nd.resize(dyn_capacity);
    }
    archive.swap(nd);
}

vector<IMOMAIndividual> IMOMA_Solver::SelectByRankFromArchive(int count)
{
    vector<IMOMAIndividual> out;
    if (archive.empty() || count <= 0) return out;

    double sum = 0.0;
    for (const auto& ind : archive) sum += 1.0 / (double)(ind.rank + 1);
    if (sum <= 1e-12) sum = 1.0;

    vector<double> probs(archive.size(), 0.0);
    for (int i = 0; i < (int)archive.size(); i++) probs[i] = (1.0 / (archive[i].rank + 1)) / sum;

    for (int c = 0; c < count; c++) {
        double r = Rand01();
        double acc = 0.0;
        int pick = (int)archive.size() - 1;
        for (int i = 0; i < (int)archive.size(); i++) {
            acc += probs[i];
            if (r <= acc) {
                pick = i;
                break;
            }
        }
        out.push_back(archive[pick]);
    }
    return out;
}

int IMOMA_Solver::SelectOperatorIndex()
{
    double sum = 0.0;
    for (const auto& op : operators) sum += op.weight;
    if (sum <= 1e-12) return 0;
    double r = Rand01() * sum;
    double acc = 0.0;
    for (int i = 0; i < (int)operators.size(); i++) {
        acc += operators[i].weight;
        if (r <= acc) return i;
    }
    return 0;
}

double IMOMA_Solver::TriggerProbability(int g, double S) const
{
    const double gg = std::clamp((double)g / (double)max_generations, 0.0, 0.999999);
    const double s = std::clamp(S, 1e-6, 0.5);
    const double t = std::tan(0.5 * M_PI * gg);
    const double rho = std::pow(std::abs(t), s);
    if (!std::isfinite(rho)) return 1.0;
    return std::clamp(rho, 0.0, 1.0);
}

double IMOMA_Solver::ScalarFit(const IMOMAIndividual& ind) const
{
    const double alpha = std::clamp(solver->workspace.alpha, 0.0, 1.0);
    return alpha * ind.makespan + (1.0 - alpha) * ind.energy;
}

void IMOMA_Solver::RunGeneration(int gen)
{
    const int g = gen + 1;
    const double omega = CalculateOmega(g);

    vector<int> non_pf_indices;
    if ((int)population.size() > 1) {
        vector<IMOMAIndividual> ranked = population;
        FastNonDominatedSort(ranked);
        for (int i = 0; i < (int)ranked.size(); i++) {
            if (ranked[i].rank > 0) non_pf_indices.push_back(i);
        }
    }

    vector<IMOMAIndividual> dol_pop;
    dol_pop.reserve(pop_size);
    for (const auto& ind : population) dol_pop.push_back(GenerateOpposite(ind, omega));

    vector<IMOMAIndividual> offspring;
    offspring.reserve(pop_size);
    for (int i = 0; i < pop_size; i++) {
        const IMOMAIndividual& pA = archive.empty() ? population[RandInt(0, (int)population.size())]
                                                    : archive[RandInt(0, (int)archive.size())];
        const IMOMAIndividual& pB = !non_pf_indices.empty()
                                  ? population[non_pf_indices[RandInt(0, (int)non_pf_indices.size())]]
                                  : population[RandInt(0, (int)population.size())];
        offspring.push_back(BiasedCrossover(pA, pB, 0.6));
    }

    vector<IMOMAIndividual> combined = population;
    combined.insert(combined.end(), dol_pop.begin(), dol_pop.end());
    combined.insert(combined.end(), offspring.begin(), offspring.end());
    combined.insert(combined.end(), archive.begin(), archive.end());
    FastNonDominatedSort(combined);

    const int old_archive = (int)archive.size();
    UpdateArchive(combined);
    const int archive_change = (int)archive.size() - old_archive;

    vector<IMOMAIndividual> guided_candidates;
    // Scale stagnation handling for long runs (e.g. 1e4 generations).
    const int stagnation_gate = std::max(30, max_generations / 50);
    const double stag_ratio = std::min(1.0, (double)stagnation_count / (double)stagnation_gate);
    double rho = TriggerProbability(g, suppression_factor);
    if (stagnation_count >= stagnation_gate / 3) {
        rho = std::max(rho, 0.25 + 0.55 * stag_ratio);
    }
    if (!archive.empty() && (Rand01() < rho || stagnation_count >= stagnation_gate / 2)) {
        vector<IMOMAIndividual> selected = SelectByRankFromArchive(std::max(5, pop_size / 8));
        guided_candidates.reserve(selected.size());
        for (const auto& ind : selected) {
            int op_idx = SelectOperatorIndex();
            const double best_before_op = best_scalar_fit;
            IMOMAIndividual cand;
            if (operators[op_idx].name == "Mutation") cand = Mutation(ind);
            else if (operators[op_idx].name == "EO") cand = EnergyOptimization(ind);
            else cand = MakespanOptimization(ind);
            guided_candidates.push_back(cand);

            bool improved_pf = false;
            for (const auto& a : archive) {
                if (Dominates(cand, a)) {
                    improved_pf = true;
                    break;
                }
            }
            int old_sz = (int)archive.size();
            UpdateArchive({cand});
            int local_archive_change = (int)archive.size() - old_sz;
            const bool improved_scalar = (best_scalar_fit + 1e-12 < best_before_op);
            operators[op_idx].Update(improved_pf || improved_scalar, local_archive_change, 0.5);
        }
    }

    if (!guided_candidates.empty()) {
        combined.insert(combined.end(), guided_candidates.begin(), guided_candidates.end());
    }

    if (has_scalar_best && !population.empty()) {
        vector<IMOMAIndividual> scalar_guided;
        const int scalar_trials = std::max(6, pop_size / 5);
        scalar_guided.reserve(scalar_trials);
        for (int k = 0; k < scalar_trials; k++) {
            const IMOMAIndividual& base = population[RandInt(0, (int)population.size())];
            scalar_guided.push_back(BestGuidedMutation(base, scalar_best_ind));
        }
        combined.insert(combined.end(), scalar_guided.begin(), scalar_guided.end());
        UpdateArchive(scalar_guided);
    }

    const int pulse_interval = std::max(120, max_generations / 40);
    const bool pulse = (g % pulse_interval) == 0;
    const bool archive_stuck = archive_change <= 0;
    if ((pulse || archive_stuck || stagnation_count >= stagnation_gate / 4) && (!population.empty() || has_scalar_best)) {
        const double pulse_boost = pulse ? 0.25 : 0.0;
        const double intensity = std::clamp(0.20 + 0.65 * stag_ratio + pulse_boost, 0.15, 1.0);
        int explore_num = std::max(4, pop_size / 4);
        if (stagnation_count >= stagnation_gate / 2) explore_num = std::max(explore_num, pop_size / 3);
        if (stagnation_count >= stagnation_gate) explore_num = std::max(explore_num, pop_size / 2);

        vector<IMOMAIndividual> explorers;
        explorers.reserve(explore_num);
        for (int i = 0; i < explore_num; i++) {
            IMOMAIndividual cand;
            const double guided_prob = std::clamp(0.80 - 0.60 * intensity, 0.15, 0.80);
            if (has_scalar_best && !population.empty() && Rand01() < guided_prob) {
                const IMOMAIndividual& base = population[RandInt(0, (int)population.size())];
                cand = BestGuidedMutation(base, scalar_best_ind);
                if (Rand01() < 0.45 + 0.25 * intensity) cand = Mutation(cand);
            } else {
                if (Rand01() < std::clamp(0.10 + 0.25 * intensity, 0.0, 0.5)) cand = RandomIndividual();
                else cand = DiversifyFromBest(intensity);
            }
            explorers.push_back(std::move(cand));
        }
        combined.insert(combined.end(), explorers.begin(), explorers.end());
        UpdateArchive(explorers);
    }

    const int intensify_interval = std::max(10, max_generations / 200);
    const bool do_intensify = ((g % intensify_interval) == 0) || stagnation_count >= std::max(8, stagnation_gate / 3);
    if (do_intensify && (!archive.empty() || !population.empty())) {
        IMOMAIndividual base = has_scalar_best ? scalar_best_ind : (!archive.empty() ? archive[0] : population[0]);
        double base_scalar = ScalarFit(base);
        const vector<IMOMAIndividual>& pool = !archive.empty() ? archive : population;
        for (const auto& x : pool) {
            double sx = ScalarFit(x);
            if (sx < base_scalar) {
                base = x;
                base_scalar = sx;
            }
        }

        vector<IMOMAIndividual> refine;
        const int trials = std::max(8, pop_size / 5);
        refine.reserve(trials);
        for (int k = 0; k < trials; k++) {
            IMOMAIndividual cand = base;
            if (!archive.empty() && Rand01() < 0.7) {
                const IMOMAIndividual& mate = archive[RandInt(0, (int)archive.size())];
                cand = BiasedCrossover(base, mate, 0.85);
            }

            const int task_edits = std::max(1, CE_Tnum / 20);
            for (int e = 0; e < task_edits; e++) {
                int t = RandInt(0, CE_Tnum);
                if (Rand01() < 0.65) {
                    int edge = PickNearestAllowedEdgeForTask(t);
                    if (edge >= 0) SetTaskEdge(cand.var, t, edge);
                    else SetTaskCloud(cand.var, t, RandInt(0, solver->Cnum));
                } else {
                    SetTaskCloud(cand.var, t, RandInt(0, solver->Cnum));
                }
            }
            const int gene_edits = std::max(1, Nvar / 80);
            for (int j = 0; j < gene_edits; j++) {
                int idx = RandInt(0, Nvar);
                cand.var[idx] = std::clamp(cand.var[idx] + 0.25 * (Rand01() - 0.5), 0.0, 1.0);
            }
            Evaluate(cand);
            if (ScalarFit(cand) + 1e-12 < base_scalar || Dominates(cand, base)) {
                base = cand;
                base_scalar = ScalarFit(cand);
            }
            refine.push_back(std::move(cand));
        }
        combined.insert(combined.end(), refine.begin(), refine.end());
        UpdateArchive(refine);
    }
    if (stagnation_count >= stagnation_gate) {
        const int immigrant_num = std::max(8, pop_size / 3);
        for (int i = 0; i < immigrant_num; i++) {
            if (has_scalar_best && Rand01() < 0.5 && !population.empty()) {
                const IMOMAIndividual& base = population[RandInt(0, (int)population.size())];
                combined.push_back(BestGuidedMutation(base, scalar_best_ind));
            } else if (!archive.empty() && Rand01() < 0.7) {
                combined.push_back(DiversifyFromBest(0.65 + 0.25 * Rand01()));
            } else {
                combined.push_back(RandomIndividual());
            }
        }
        stagnation_count = stagnation_gate / 2;
    }

    population = SelectNextGeneration(combined, pop_size);

    if (!combined.empty() && !population.empty()) {
        int elite_count = std::max(1, pop_size / 8);
        elite_count = std::min(elite_count, (int)combined.size());

        vector<int> order(combined.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + elite_count, order.end(),
            [&](int a, int b) { return ScalarFit(combined[a]) < ScalarFit(combined[b]); });

        for (int e = 0; e < elite_count; e++) {
            const IMOMAIndividual& elite = combined[order[e]];
            int worst_idx = 0;
            double worst_scalar = ScalarFit(population[0]);
            for (int i = 1; i < (int)population.size(); i++) {
                double sf = ScalarFit(population[i]);
                if (sf > worst_scalar) {
                    worst_scalar = sf;
                    worst_idx = i;
                }
            }
            if (ScalarFit(elite) + 1e-12 < worst_scalar) {
                population[worst_idx] = elite;
            }
        }
    }

    if (has_scalar_best && !population.empty()) {
        int worst_idx = 0;
        double worst_scalar = ScalarFit(population[0]);
        for (int i = 1; i < (int)population.size(); i++) {
            double sf = ScalarFit(population[i]);
            if (sf > worst_scalar) {
                worst_scalar = sf;
                worst_idx = i;
            }
        }
        if (best_scalar_fit + 1e-12 < worst_scalar) {
            population[worst_idx] = scalar_best_ind;
        }
    }

    if (stagnation_count >= stagnation_gate / 2) {
        const int inject_num = std::max(3, pop_size / 6);
        const double inject_intensity = std::clamp(0.35 + 0.60 * stag_ratio, 0.2, 1.0);
        InjectExplorers(population, inject_num, inject_intensity);
    }

    if (stagnation_count >= stagnation_gate * 2 && !population.empty()) {
        int keep = std::max(2, pop_size / 5);
        keep = std::min(keep, (int)population.size());

        vector<int> order(population.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + keep, order.end(),
            [&](int a, int b) { return ScalarFit(population[a]) < ScalarFit(population[b]); });

        vector<IMOMAIndividual> rebuilt;
        rebuilt.reserve(pop_size);
        for (int i = 0; i < keep; i++) rebuilt.push_back(population[order[i]]);
        if (has_scalar_best && !rebuilt.empty()) rebuilt[0] = scalar_best_ind;

        while ((int)rebuilt.size() < pop_size) {
            const double intensity = 0.70 + 0.30 * Rand01();
            IMOMAIndividual cand;
            const double r = Rand01();
            if (r < 0.30) cand = RandomIndividual();
            else cand = DiversifyFromBest(intensity);
            if (has_scalar_best && r >= 0.30 && Rand01() < 0.35) {
                cand = BestGuidedMutation(cand, scalar_best_ind);
            }
            rebuilt.push_back(std::move(cand));
        }

        population.swap(rebuilt);
        UpdateArchive(population);
        stagnation_count = stagnation_gate / 3;
        restart_count++;
    }

    if (best_scalar_fit + 1e-12 < last_best_scalar_fit) {
        stagnation_count = 0;
    } else {
        stagnation_count++;
    }
    last_best_scalar_fit = best_scalar_fit;

    if (!population.empty()) {
        double best_now = std::numeric_limits<double>::infinity();
        for (const auto& ind : population) best_now = std::min(best_now, ScalarFit(ind));
        if (best_now + 1e-12 < best_scalar_fit) {
            best_scalar_fit = best_now;
            stagnation_count = 0;
            last_best_scalar_fit = best_now;
        }
    }
}
