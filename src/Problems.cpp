#include "Problems.h"
#include "Workspace.h"
#include <chrono>

static const double INV_LN2 = 1.4426950408889634074;
static const double D2D_FACTOR = 3600.0 / 100000.0;
static const double QN_GAIN = QN * 1000.0;
static const double RATE_SCALE = 2.5;
static const double COMM_SCALE = 0.01;
static const double CLOUD_COMP_SPEED = 3.7;
static const double EDGE_COMP_SPEED_NO_LOAD = 2.2;

inline double fast_fmax(double a, double b) { return (a > b) ? a : b; }
inline double fast_fabs(double x) { return (x < 0) ? -x : x; }

double randnorm(double miu, double score)
{
	return miu + score * sqrt(-2 * log(rand() / (RAND_MAX + 1.0))) * cos(2 * M_PI * rand() / (RAND_MAX + 1.0));                 //by lyl
}

double CED_Schedule(const double* var, Workspace& ws, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time, double** EtoD_Distance, double** DtoD_Distance, vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices, vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad, vector<int>* CETask_coDevice, double* Edge_Device_comm, double** ST, double** ET, double* CE_ST, double* CE_ET)
{
    if (ws.max_Cnum != Cnum || ws.max_CE_Tnum != CE_Tnum || ws.max_M_Jnum != M_Jnum || ws.max_M_OPTnum != M_OPTnum ||
        ws.max_Enum != Enum || ws.max_Dnum != Dnum) {
        ws.resize(Cnum, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum);
    }
    ws.clear();

    (void)CloudDevices;
    (void)EdgeDevices;
    (void)DeviceLoad;
    (void)CETask_coDevice;
    CloudLoad = ws.cloud_load.data();
    EdgeLoad = ws.edge_load.data();
    Edge_Device_comm = ws.edge_device_comm.data();
    ST = ws.st_rows.data();
    ET = ws.et_rows.data();
    CE_ST = ws.ce_st.data();
    CE_ET = ws.ce_et.data();
    if (!ws.comm_factor_ready) {
        for (int i = 0; i < CE_Tnum; ++i) {
            ws.comm_factor[i] = CETask_Property[i].Communication * 0.01;
        }
        ws.comm_factor_ready = true;
    }

    const int ops = M_Jnum * M_OPTnum;
    const double cloud_capacity_scale = std::max(ws.stress.cloud_capacity_scale, 1e-9);
    const double edge_capacity_scale = std::max(ws.stress.edge_capacity_scale, 1e-9);
    const double device_capacity_scale = std::max(ws.stress.device_capacity_scale, 1e-9);
    const double communication_scale = std::max(ws.stress.communication_scale, 1e-9);
    vector<bool>& ce_sele = ws.ce_sele;
    vector<int>& cevar = ws.cevar;
    vector<int>& mvar = ws.mvar;
    vector<int>& op_order = ws.op_order;
    vector<double>& op_keys = ws.op_keys;
    vector<int>& geneO = ws.geneO;
    vector<int>& last_dev_op = ws.last_dev_op;
    vector<double>& last_dev_end = ws.last_dev_end;
    vector<int>& nearest_device = ws.nearest_device;
    vector<int>& nearest_edge = ws.nearest_edge;
    vector<double>& Edge_smallest_rate = ws.edge_smallest_rate;
    vector<uint32_t>& seen_stamp = ws.seen_stamp;
    vector<int>& task_devs = ws.task_devs;
    vector<int>& task_dev_count = ws.task_dev_count;
    vector<int>& edge_devs = ws.edge_devs;
    vector<int>& edge_dev_count = ws.edge_dev_count;

#ifdef PROFILE_EVAL
    using clock = std::chrono::high_resolution_clock;
    ws.profile.samples++;
    auto t_stage = clock::now();
#endif

    for (int i = 0; i < CE_Tnum; i ++)
    {
        ce_sele[i] = (var[i] > 0.5);
        if (ce_sele[i] == false)
            cevar[i] = (int)(var[CE_Tnum + i] * (Cnum - 1));
        else
        {
            int edge_count = (int)CETask_Property[i].AvailEdgeServerList.size();
            int idx = (edge_count > 1) ? (int)(var[CE_Tnum + i] * (edge_count - 1)) : 0;
            cevar[i] = CETask_Property[i].AvailEdgeServerList[idx];
        }
    }
#ifdef PROFILE_EVAL
    ws.profile.decode_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    for (int i = 0; i < ops; i ++)
    {
        op_order[i] = i;
        op_keys[i] = var[2 * CE_Tnum + i];
    }
    std::sort(op_order.begin(), op_order.end(),
        [&](int a, int b) {
            return (op_keys[a] < op_keys[b]) || ((op_keys[a] == op_keys[b]) && (a < b));
        });
#ifdef PROFILE_EVAL
    ws.profile.sort_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    for (int i = 0; i < ops; i ++)
    {
        int op = op_order[i];
        int avail = (int)AvailDeviceList[op].size();
        int idx = (avail > 1) ? (int)(var[CE_Tnum * 2 + ops + i] * (avail - 1)) : 0;
        if (idx < 0) idx = 0;
        if (idx >= avail) idx = avail - 1;
        mvar[op] = (avail > 0) ? AvailDeviceList[op][idx] : 0;
    }
#ifdef PROFILE_EVAL
    ws.profile.assign_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    const double d2d_factor = D2D_FACTOR;
    for (int i = 0; i < ops; i ++)
    {
        int op = op_order[i];
        int CJ = op >> 3;
        geneO[CJ] ++;
        int CO = geneO[CJ];
        int CM = mvar[op];
        int Cprev = CO - 1;
        double* st_row = ST[CJ];
        double* et_row = ET[CJ];
        // Device-side stress is injected as an equivalent processing slow-down so
        // feasibility is preserved and the existing decoding/simulation logic stays unchanged.
        double mtask_time = MTask_Time[CJ * M_OPTnum + CO] / device_capacity_scale;
        
        if (Cprev < 0)
        {
            if (last_dev_op[CM] < 0)
            {
                st_row[CO] = 0;
                et_row[CO] = mtask_time;
            }
            else
            {
                st_row[CO] = last_dev_end[CM];
                et_row[CO] = st_row[CO] + mtask_time;
            }
        }
        else
        {
            double prev_et = et_row[Cprev];
            if (last_dev_op[CM] < 0)
            {
                st_row[CO] = prev_et;
                int prev_dev = mvar[CJ * M_OPTnum + Cprev];
                int curr_dev = mvar[op];
                et_row[CO] = st_row[CO] + mtask_time + DtoD_Distance[prev_dev][curr_dev] * d2d_factor;
            }
            else
            {
                st_row[CO] = (prev_et > last_dev_end[CM]) ? prev_et : last_dev_end[CM];
                int prev_dev = mvar[CJ * M_OPTnum + Cprev];
                int curr_dev = mvar[op];
                et_row[CO] = st_row[CO] + mtask_time + DtoD_Distance[prev_dev][curr_dev] * d2d_factor;
            }
        }
        last_dev_op[CM] = CJ * M_OPTnum + CO;
        last_dev_end[CM] = et_row[CO];
    }
#ifdef PROFILE_EVAL
    ws.profile.schedule_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    if (!ws.nearest_ready)
    {
        for (int i = 0; i < Enum; i++)
        {
            double min_dis = std::numeric_limits<double>::infinity();
            int min_index = 0;
            for (int j = 0; j < Dnum; j++)
            {
                if (EtoD_Distance[i][j] < min_dis)
                {
                    min_dis = EtoD_Distance[i][j];
                    min_index = j;
                }
            }
            nearest_device[i] = min_index;
        }
        for (int i = 0; i < Dnum; i ++)
        {
            double min_dis = std::numeric_limits<double>::infinity();
            int min_index = 0;
            for (int j = 0; j < Enum; j ++)
            {
                if (min_dis > EtoD_Distance[j][i])
                {
                    min_dis = EtoD_Distance[j][i];
                    min_index = j;
                }
            }
            nearest_edge[i] = min_index;
        }
        ws.nearest_ready = true;
    }

    for (int i = 0; i < Cnum; i ++)
        CloudLoad[i].clear();
    for (int i = 0; i < Enum; i ++)
        EdgeLoad[i].clear();

    for (int i = 0; i < CE_Tnum; i ++)
    {
        uint32_t stamp = ws.next_token();
        int base = i * M_OPTnum;
        int count = 0;
        for (int j = 0; j < M_OPTnum; j ++)
        {
            int dev = mvar[base + j];
            if ((unsigned)dev >= (unsigned)Dnum)
                continue;
            if (seen_stamp[dev] != stamp)
            {
                seen_stamp[dev] = stamp;
                task_devs[base + count] = dev;
                count++;
            }
        }
        task_dev_count[i] = count;
    }

    for (int i = 0; i < CE_Tnum; i ++)
    {
        if (ce_sele[i] == false)  //cloud mode
            CloudLoad[ cevar[i] ].push_back(i);
        else
            EdgeLoad[ cevar[i] ].push_back(i);
    }

    for (int e = 0; e < Enum; e ++)
    {
        uint32_t stamp = ws.next_token();
        int count = 0;
        for (size_t idx = 0; idx < EdgeLoad[e].size(); idx ++)
        {
            int task = EdgeLoad[e][idx];
            int base = task * M_OPTnum;
            int tcount = task_dev_count[task];
            for (int k = 0; k < tcount; k ++)
            {
                int dev = task_devs[base + k];
                if (seen_stamp[dev] != stamp)
                {
                    seen_stamp[dev] = stamp;
                    edge_devs[e * Dnum + count] = dev;
                    count++;
                }
            }
        }
        int nd = nearest_device[e];
        if ((unsigned)nd < (unsigned)Dnum && seen_stamp[nd] != stamp)
        {
            seen_stamp[nd] = stamp;
            edge_devs[e * Dnum + count] = nd;
            count++;
        }
        edge_dev_count[e] = count;
    }
#ifdef PROFILE_EVAL
    ws.profile.devices_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    const double qn_gain = QN_GAIN;
    const double rate_scale = RATE_SCALE;
    for (int i = 0; i < Enum; i ++)
    {
        double min_rate = std::numeric_limits<double>::infinity();
        double bottom_sum = 0.0;
        int base = i * Dnum;
        int count = edge_dev_count[i];
        
        for (int k = 0; k < count; k ++)
        {
            int dev = edge_devs[base + k];
            bottom_sum += qn_gain / EtoD_Distance[i][dev];
        }
        
        for (int k = 0; k < count; k ++)
        {
            int dev = edge_devs[base + k];
            double dist = EtoD_Distance[i][dev];
            double current_gain = qn_gain / dist;
            double snr = current_gain / fast_fabs(bottom_sum - current_gain - 100.0);
            double transmission_rate = rate_scale * std::log1p(snr) * INV_LN2;
            Edge_Device_comm[base + dev] = transmission_rate;
            if (transmission_rate < min_rate)
                min_rate = transmission_rate;
        }
        Edge_smallest_rate[i] = min_rate;
    }
#ifdef PROFILE_EVAL
    ws.profile.comm_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    for (int i = 0; i < CE_Tnum; i ++)
        CE_ST[i] = CE_ET[i] = 0;

    double time_max = 0;
    double energy = 0;
    const double qn_energy_factor = QN / 1000.0;
    const int m_opt_minus_1 = M_OPTnum - 1;
    
    for (int i = 0; i < CE_Tnum; i ++)
    {
        double t_comm = 0, t_comp = 0;
        int base = i * M_OPTnum;
        int tcount = task_dev_count[i];
        double comm_factor = ws.comm_factor[i];
        
        if (!ce_sele[i])
        {
            for (int k = 0; k < tcount; k ++)
            {
                int dev = task_devs[base + k];
                int near_edge = nearest_edge[dev];
                double cur_comm = (comm_factor * communication_scale) / Edge_smallest_rate[near_edge];
                energy += cur_comm * qn_energy_factor;
                if (cur_comm > t_comm)
                    t_comm = cur_comm;
            }
            t_comp = CETask_Property[i].Computation / (CLOUD_COMP_SPEED * cloud_capacity_scale);
        }
        else
        {
            int edge = cevar[i];
            int edge_base = edge * Dnum;
            size_t edge_load_size = EdgeLoad[edge].size();
            
            for (int k = 0; k < tcount; k ++)
            {
                int dev = task_devs[base + k];
                double cur_comm = (comm_factor * communication_scale) / Edge_Device_comm[edge_base + dev];
                energy += cur_comm * qn_energy_factor;
                if (cur_comm > t_comm)
                    t_comm = cur_comm;
            }
            
            if (edge_load_size < 6)
                t_comp = CETask_Property[i].Computation / (EDGE_COMP_SPEED_NO_LOAD * edge_capacity_scale);
            else
                t_comp = CETask_Property[i].Computation / (EDGE_COMP_SPEED_NO_LOAD * edge_capacity_scale) * edge_load_size;
        }

        double max_Prec_EndTime = 0, max_Start_StartTime = 0, max_End_EndTime = 0, max_Iter_EndTime = 0;
        const CETask& task = CETask_Property[i];
        
        if (!task.Precedence.empty())
        {
            for (int idx : task.Precedence)
            {
                double et_val = CE_ET[idx];
                if (et_val > max_Prec_EndTime) max_Prec_EndTime = et_val;
            }
        }
        if (!task.Start_Pre.empty())
        {
            for (int idx : task.Start_Pre)
            {
                double st_val = CE_ST[idx];
                if (st_val > max_Start_StartTime) max_Start_StartTime = st_val;
            }
        }
        if (!task.End_Pre.empty())
        {
            for (int idx : task.End_Pre)
            {
                double et_val = CE_ET[idx];
                if (et_val > max_End_EndTime) max_End_EndTime = et_val;
            }
        }
        if (!task.Interact.empty())
        {
            for (int idx : task.Interact)
            {
                double et_val = CE_ET[idx];
                if (et_val > max_Iter_EndTime) max_Iter_EndTime = et_val;
            }
        }

        double ce_st_val = fast_fmax(max_Start_StartTime, max_Prec_EndTime);
        int job_cons = task.Job_Constraints;
        if (job_cons == 1 || job_cons == 3)
            ce_st_val = fast_fmax(ce_st_val, ST[i][m_opt_minus_1]);
        
        double ce_et_val = ce_st_val + t_comm + t_comp;
        ce_et_val = fast_fmax(ce_et_val, max_End_EndTime);
        ce_et_val = fast_fmax(ce_et_val, max_Iter_EndTime);
        if (job_cons == 2 || job_cons == 3)
            ce_et_val = fast_fmax(ce_et_val, ET[i][m_opt_minus_1]);

        CE_ST[i] = ce_st_val;
        CE_ET[i] = ce_et_val;

        if (!task.Interact.empty())
        {
            for (int idx : task.Interact)
                CE_ET[idx] = ce_et_val;
        }
    }

    for (int i = 0; i < CE_Tnum; i ++)
    {
        double et_val = CE_ET[i];
        if (et_val > time_max)
            time_max = et_val;
    }

    const double energy_scale = 1.0 / 1000.0;
    for (int i = 0; i < Cnum; i ++)
    {
        const auto& load = CloudLoad[i];
        size_t cloud_size = load.size();
        if (cloud_size == 0)
            continue;
        int u_ratio = (int)((cloud_size / 20.0) * 10);
        if (u_ratio > 10) u_ratio = 10;
        int time_expand = 0;
        for (int idx : load)
        {
            int dur = (int)(CE_ET[idx] - CE_ST[idx]);
            if (dur > time_expand) time_expand = dur;
        }
        energy += EnergyList[u_ratio] * time_expand * energy_scale;
    }
    for (int i = 0; i < Enum; i ++)
    {
        const auto& load = EdgeLoad[i];
        size_t edge_size = load.size();
        if (edge_size == 0)
            continue;
        int u_ratio = (int)((edge_size / 6.0) * 10);
        if (u_ratio > 10) u_ratio = 10;
        int time_expand = 0;
        for (int idx : load)
        {
            int dur = (int)(CE_ET[idx] - CE_ST[idx]);
            if (dur > time_expand) time_expand = dur;
        }
        energy += EnergyList[u_ratio] * time_expand * energy_scale;
    }
#ifdef PROFILE_EVAL
    ws.profile.tasks_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
#endif

    const double eps = 1e-6;
    const double f1_ref = (std::abs(ws.f1_ref) > eps) ? ws.f1_ref : 1.0;
    const double f2_ref = (std::abs(ws.f2_ref) > eps) ? ws.f2_ref : 1.0;
    const double alpha = std::clamp(ws.alpha, 0.0, 1.0);

    const double f1_normalized = time_max / f1_ref;
    const double f2_normalized = energy / f2_ref;
    const double fitness = alpha * f1_normalized + (1.0 - alpha) * f2_normalized;
    ws.last_makespan = time_max;
    ws.last_energy = energy;
    ws.last_fitness = fitness;
    return fitness;
}


