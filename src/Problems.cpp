#include "Problems.h"
#include "Workspace.h"
#include <chrono>

double randnorm(double miu, double score)
{
	return miu + score * sqrt(-2 * log(rand() / (RAND_MAX + 1.0))) * cos(2 * M_PI * rand() / (RAND_MAX + 1.0));                 //by lyl
}

double CED_Schedule(const double* var, Workspace& ws, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time, double** EtoD_Distance, double** DtoD_Distance, vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices, vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad, vector<int>* CETask_coDevice, double* Edge_Device_comm, double** ST, double** ET, double* CE_ST, double* CE_ET)
{
    if (ws.max_CE_Tnum != CE_Tnum || ws.max_M_Jnum != M_Jnum || ws.max_M_OPTnum != M_OPTnum ||
        ws.max_Enum != Enum || ws.max_Dnum != Dnum) {
        ws.resize(CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum);
    }
    ws.clear();

    (void)CloudDevices;
    (void)EdgeDevices;
    (void)DeviceLoad;
    (void)CETask_coDevice;

    const int ops = M_Jnum * M_OPTnum;
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
    sort(op_order.begin(), op_order.end(),
        [&](int a, int b) {
            double va = op_keys[a];
            double vb = op_keys[b];
            if (va == vb) return a < b;
            return va < vb;
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

    const double d2d_factor = 3600.0 / 100000.0;
    for (int i = 0; i < ops; i ++)
    {
        int op = op_order[i];
        int CJ = op / M_OPTnum;
        geneO[CJ] ++;
        int CO = geneO[CJ];
        int CM = mvar[op];
        int Cprev = CO - 1;
        if (Cprev < 0)   //no precedence
        {
            if (last_dev_op[CM] < 0)                       //target machine has no operation
            {
                ST[CJ][CO] = 0;
                ET[CJ][CO] = MTask_Time[CJ * M_OPTnum + CO];
            }
            else
            {
                ST[CJ][CO] = last_dev_end[CM];
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO];
            }
        }
        else    //has precedence
        {
            if (last_dev_op[CM] < 0)
            {
                ST[CJ][CO] = ET[CJ][Cprev];
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] +
                             DtoD_Distance[mvar[CJ * M_OPTnum + Cprev]][mvar[CJ * M_OPTnum + CO]] * d2d_factor;
            }
            else
            {
                ST[CJ][CO] = fmax(ET[CJ][Cprev], last_dev_end[CM]);
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] +
                             DtoD_Distance[mvar[CJ * M_OPTnum + Cprev]][mvar[CJ * M_OPTnum + CO]] * d2d_factor;
            }
        }
        last_dev_op[CM] = CJ * M_OPTnum + CO;
        last_dev_end[CM] = ET[CJ][CO];
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

    const double qn_gain = QN * 1000.0;
    const double rate_scale = 2.5;
    for (int i = 0; i < Enum; i ++)
    {
        Edge_smallest_rate[i] = std::numeric_limits<double>::infinity();
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
            double current_gain = qn_gain / EtoD_Distance[i][dev];
            double transmission_rate = rate_scale * log2(1.0 + current_gain / fabs(bottom_sum - current_gain - 100.0));
            Edge_Device_comm[base + dev] = transmission_rate;
            if (transmission_rate < Edge_smallest_rate[i])
                Edge_smallest_rate[i] = transmission_rate;
        }
    }
#ifdef PROFILE_EVAL
    ws.profile.comm_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
    t_stage = clock::now();
#endif

    for (int i = 0; i < CE_Tnum; i ++)
        CE_ST[i] = CE_ET[i] = 0;

    double time_max = 0;
    double energy = 0;
    for (int i = 0; i < CE_Tnum; i ++)
    {
        double t_comm = 0, t_comp = 0;
        int base = i * M_OPTnum;
        int tcount = task_dev_count[i];
        if (ce_sele[i] == false)  //cloud mode
        {
            for (int k = 0; k < tcount; k ++)
            {
                int dev = task_devs[base + k];
                double cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_smallest_rate[ nearest_edge[dev] ]);
                energy += cur_comm * QN / 1000;    //offloading energy qn * bn / rn(a)
                if (cur_comm > t_comm)
                    t_comm = cur_comm;
            }
        }
        else
        {
            int edge = cevar[i];
            int edge_base = edge * Dnum;
            for (int k = 0; k < tcount; k ++)
            {
                int dev = task_devs[base + k];
                double cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_Device_comm[ edge_base + dev ]);
                energy += cur_comm * QN / 1000;    //offloading energy qn * bn / rn(a)
                if (cur_comm > t_comm)
                    t_comm = cur_comm;
            }
        }

        if (ce_sele[i] == false)
            t_comp = CETask_Property[i].Computation / 3.7;
        else
        {
            if (EdgeLoad[ cevar[i] ].size() < 6)
                t_comp = CETask_Property[i].Computation / 2.2;
            else
                t_comp = CETask_Property[i].Computation / 2.2 * EdgeLoad[ cevar[i] ].size();
        }

        double max_Prec_EndTime = 0, max_Start_StartTime = 0, max_End_EndTime = 0, max_Iter_EndTime = 0;
        if (CETask_Property[i].Precedence.size() != 0)
        {
            for (auto iter = CETask_Property[i].Precedence.begin(); iter != CETask_Property[i].Precedence.end(); iter++)
            {
                if (max_Prec_EndTime < CE_ET[*iter])
                    max_Prec_EndTime = CE_ET[*iter];
            }
        }
        if (CETask_Property[i].Start_Pre.size() != 0)
        {
            for (auto iter = CETask_Property[i].Start_Pre.begin(); iter != CETask_Property[i].Start_Pre.end(); iter++)
            {
                if (max_Start_StartTime < CE_ST[*iter])
                    max_Start_StartTime = CE_ST[*iter];
            }
        }
        if (CETask_Property[i].End_Pre.size() != 0)
        {
            for (auto iter = CETask_Property[i].End_Pre.begin(); iter != CETask_Property[i].End_Pre.end(); iter ++)
            {
                if (max_End_EndTime < CE_ET[*iter])
                    max_End_EndTime = CE_ET[*iter];
            }
        }
        if (CETask_Property[i].Interact.size() != 0)
        {
            for (auto iter = CETask_Property[i].Interact.begin(); iter != CETask_Property[i].Interact.end(); iter ++)
            {
                if (max_Iter_EndTime < CE_ET[*iter])
                    max_Iter_EndTime = CE_ET[*iter];
            }
        }

        int theJob = i;
        CE_ST[i] = fmax(max_Start_StartTime, max_Prec_EndTime);
        if (CETask_Property[i].Job_Constraints == 1 || CETask_Property[i].Job_Constraints == 3)
            CE_ST[i] = fmax(CE_ST[i], ST[ theJob ][M_OPTnum - 1]);
        CE_ET[i] = CE_ST[i] + t_comm + t_comp;
        CE_ET[i] = fmax(CE_ET[i], max_End_EndTime);
        CE_ET[i] = fmax(CE_ET[i], max_Iter_EndTime);
        if (CETask_Property[i].Job_Constraints == 2 || CETask_Property[i].Job_Constraints == 3)
            CE_ET[i] = fmax(CE_ET[i], ET[ theJob ][M_OPTnum - 1]);

        //update the endtime of the interact task
        if (CETask_Property[i].Interact.size() != 0)
            for (auto iter = CETask_Property[i].Interact.begin(); iter != CETask_Property[i].Interact.end(); iter ++)
                CE_ET[*iter] = CE_ET[i];
    }

    for (int i = 0; i < CE_Tnum; i ++)
    {
        if (time_max < CE_ET[i])
            time_max = CE_ET[i];
    }

    for (int i = 0; i < Cnum; i ++)
    {
        if (CloudLoad[i].size() == 0)
            continue;
        int u_ratio = (int)((CloudLoad[i].size() / 20.0) * 10);
        if (u_ratio > 10)
            u_ratio = 10;
        int time_expand = 0;
        for (auto iter = CloudLoad[i].begin(); iter != CloudLoad[i].end(); iter ++)
            if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
                time_expand = (int)(CE_ET[*iter] - CE_ST[*iter]);
        energy += EnergyList[u_ratio] * time_expand / 1000.0;
    }
    for (int i = 0; i < Enum; i ++)
    {
        if (EdgeLoad[i].size() == 0)
            continue;
        int u_ratio = (int)((EdgeLoad[i].size() / 6.0) * 10);
        if (u_ratio > 10)
            u_ratio = 10;
        int time_expand = 0;
        for (auto iter = EdgeLoad[i].begin(); iter != EdgeLoad[i].end(); iter ++)
            if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
                time_expand = (int)(CE_ET[*iter] - CE_ST[*iter]);
        energy += EnergyList[u_ratio] * time_expand / 1000.0;
    }
#ifdef PROFILE_EVAL
    ws.profile.tasks_us += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_stage).count();
#endif

    return time_max + energy;
}


