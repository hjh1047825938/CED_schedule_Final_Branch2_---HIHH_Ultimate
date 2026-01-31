#include "Problems.h"
#include "Workspace.h"

double randnorm(double miu, double score)
{
	return miu + score * sqrt(-2 * log(rand() / (RAND_MAX + 1.0))) * cos(2 * M_PI * rand() / (RAND_MAX + 1.0));                 //by lyl
}

double CED_Schedule(const double* var, Workspace& ws, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time, double** EtoD_Distance, double** DtoD_Distance, vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices, vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad, vector<int>* CETask_coDevice, map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST, double* CE_ET)
{
    if (ws.max_CE_Tnum != CE_Tnum || ws.max_M_Jnum != M_Jnum || ws.max_M_OPTnum != M_OPTnum ||
        ws.max_Enum != Enum || ws.max_Dnum != Dnum) {
        ws.resize(CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum);
    }
    ws.clear();

    const int ops = M_Jnum * M_OPTnum;
    vector<bool>& ce_sele = ws.ce_sele;
    vector<int>& cevar = ws.cevar;
    vector<int>& mvar = ws.mvar;
    vector<int>& seq_var = ws.seq_var;
    vector<int>& op_order = ws.op_order;
    vector<int>& geneO = ws.geneO;
    vector<int>& nearest_device = ws.nearest_device;
    vector<int>& nearest_edge = ws.nearest_edge;
    vector<double>& Edge_smallest_rate = ws.edge_smallest_rate;
    vector<char>& seen_device = ws.seen_device;

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

    iota(op_order.begin(), op_order.end(), 0);
    sort(op_order.begin(), op_order.end(),
        [&](int a, int b) {
            double va = var[2 * CE_Tnum + a];
            double vb = var[2 * CE_Tnum + b];
            if (va == vb) return a < b;
            return va < vb;
        });
    for (int i = 0; i < ops; i ++)
        seq_var[i] = op_order[i];

    for (int i = 0; i < ops; i ++)
    {
        int avail = (int)AvailDeviceList[seq_var[i]].size();
        int idx = (avail > 1) ? (int)(var[CE_Tnum * 2 + ops + i] * (avail - 1)) : 0;
        if (idx < 0) idx = 0;
        if (idx >= avail) idx = avail - 1;
        mvar[i] = (avail > 0) ? AvailDeviceList[seq_var[i]][idx] : 0;
    }
    
    for (int i = 0; i < Cnum; i ++)
        CloudDevices[i].clear();
    for (int i = 0; i < Enum; i ++)
        EdgeDevices[i].clear();
    for (int i = 0; i < Cnum; i ++)
        CloudLoad[i].clear();
    for (int i = 0; i < Enum; i ++)
        EdgeLoad[i].clear();
    for (int i = 0; i < Dnum; i ++)
        DeviceLoad[i].clear();
    for (int i = 0; i < CE_Tnum; i ++)
        CETask_coDevice[i].clear();
    for (int i = 0; i < Enum; i ++)
        Edge_Device_comm[i].clear();
    
    for (int i = 0; i < M_Jnum; i ++)
    {
        for (int j = 0; j < M_OPTnum; j ++)
        {
            ST[i][j] = 0;
            ET[i][j] = 0;
        }
    }
    int CJ, CM, CO, Cprev, Qprev;
    for (int i = 0; i < M_Jnum * M_OPTnum; i ++)
    {
        CJ = seq_var[i] / M_OPTnum;
        geneO[CJ] ++;
        CO = geneO[CJ];
        CM = mvar[ seq_var[i] ];
        Cprev = CO - 1;
        if (Cprev < 0)   //no precedence
        {
            if (DeviceLoad[CM].size() == 0)                       //target machine has no operation
            {
                ST[CJ][CO] = 0;
                ET[CJ][CO] = MTask_Time[CJ * M_OPTnum + CO];
                DeviceLoad[CM].push_back(CJ * M_OPTnum + CO);
            }
            else
            {
                Qprev = DeviceLoad[CM][ DeviceLoad[CM].size() - 1];
                ST[CJ][CO] = ET[ Qprev / M_OPTnum ][ Qprev % M_OPTnum ];
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO];
                DeviceLoad[CM].push_back(CJ * M_OPTnum + CO);
            }
        }
        else    //有前驱工序
        {
            if (DeviceLoad[CM].size() == 0)
            {
                ST[CJ][CO] = ET[CJ][Cprev];
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] + DtoD_Distance[ mvar[CJ * M_OPTnum + Cprev] ][ mvar[CJ * M_OPTnum + CO] ] / (100000.0 / 3600.0);
                DeviceLoad[CM].push_back(CJ * M_OPTnum + CO);
            }
            else
            {
                auto iter = DeviceLoad[CM].end();
                iter --;
                Qprev = *iter;
                ST[CJ][CO] = fmax(ET[CJ][Cprev], ET[ Qprev / M_OPTnum ][ Qprev % M_OPTnum ]);
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] + DtoD_Distance[ mvar[CJ * M_OPTnum + Cprev] ][ mvar[CJ * M_OPTnum + CO] ] / (100000.0 / 3600.0);
                DeviceLoad[CM].push_back(CJ * M_OPTnum + CO);
            }
        }
    }
    
    /*for (int i = 0; i < M_Jnum; i ++)
    {
        cout << "The " << i << "th Job :" << endl;
        for (int j = 0; j < M_OPTnum; j ++)
            cout << ST[i][j] << ": " << ET[i][j] << endl;
        cout << endl;
    }*/
    
    for (int i = 0; i < CE_Tnum; i ++)
    {
        CETask_coDevice[i].clear();
        std::fill(seen_device.begin(), seen_device.end(), 0);
        for (int j = 0; j < M_OPTnum; j ++)
        {
            int dev = mvar[i * M_OPTnum + j];
            if (dev < 0 || dev >= Dnum)
                continue;
            if (!seen_device[dev])
            {
                seen_device[dev] = 1;
                CETask_coDevice[i].push_back(dev);
            }
        }
    }
    /*for (int i = 0; i < CE_Tnum; i ++)
    {
        cout << "CETask_coDevice[" << i << "] :" << CETask_coDevice[i].size() << endl;
        for (auto iter = CETask_coDevice[i].begin(); iter != CETask_coDevice[i].end(); iter ++)
            cout << *iter << "  ";
        cout << endl;
    }*/
    
    
    for (int i = 0; i < CE_Tnum; i ++)
    {
        if (ce_sele[i] == false)  //cloud mode
            CloudLoad[ cevar[i] ].push_back(i);
        else
            EdgeLoad[ cevar[i] ].push_back(i);
    }

    for (int c = 0; c < Cnum; c ++)
    {
        CloudDevices[c].clear();
        std::fill(seen_device.begin(), seen_device.end(), 0);
        for (int i = 0; i < CE_Tnum; i ++)
        {
            if (ce_sele[i] == false && cevar[i] == c)
            {
                int theJob = i;
                for (int j = 0; j < M_OPTnum; j ++)
                {
                    int dev = mvar[theJob * M_OPTnum + j];
                    if (dev < 0 || dev >= Dnum)
                        continue;
                    if (!seen_device[dev])
                    {
                        seen_device[dev] = 1;
                        CloudDevices[c].push_back(dev);
                    }
                }
            }
        }
    }
    for (int e = 0; e < Enum; e ++)
    {
        EdgeDevices[e].clear();
        std::fill(seen_device.begin(), seen_device.end(), 0);
        for (int i = 0; i < CE_Tnum; i ++)
        {
            if (ce_sele[i] == true && cevar[i] == e)
            {
                int theJob = i;
                for (int j = 0; j < M_OPTnum; j ++)
                {
                    int dev = mvar[theJob * M_OPTnum + j];
                    if (dev < 0 || dev >= Dnum)
                        continue;
                    if (!seen_device[dev])
                    {
                        seen_device[dev] = 1;
                        EdgeDevices[e].push_back(dev);
                    }
                }
            }
        }
    }
    
    /*for (int i = 0; i < Cnum; i ++)
    {
        cout << "CloudLoad[" << i << "] :" << CloudLoad[i].size() << endl;
        for (auto iter = CloudLoad[i].begin(); iter != CloudLoad[i].end(); iter ++)
            cout << *iter << "  ";
        cout << endl;
    }*/
    
    for (int i = 0; i < Enum; i++)
    {
        // FIX: was min_dis = 0 which caused the condition to never trigger
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
        
#ifdef DEBUG_NEAREST
        if (i == 0) {
            cout << "[DEBUG_NEAREST] Edge server 0: nearest_device = " << min_index 
                 << ", min_dis = " << min_dis << endl;
            cout << "[DEBUG_NEAREST] EtoD_Distance[0][0..4] = ";
            for (int k = 0; k < min(5, Dnum); k++)
                cout << EtoD_Distance[0][k] << " ";
            cout << endl;
        }
#endif
    }
    for (int i = 0; i < Enum; i++)
    {
        std::fill(seen_device.begin(), seen_device.end(), 0);
        for (size_t k = 0; k < EdgeDevices[i].size(); k++)
            seen_device[EdgeDevices[i][k]] = 1;
        if (!seen_device[nearest_device[i]])
            EdgeDevices[i].push_back(nearest_device[i]);  //each device connects the nearest edge for data forwarding...
    }
    
    
    
    for (int i = 0; i < Dnum; i ++)
    {
        double min_dis = 1e10;
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
    
    for (int i = 0; i < Enum; i ++)
    {
        Edge_smallest_rate[i] = 1e10;
        double bottom_sum = 0;
        for (auto iter = EdgeDevices[i].begin(); iter != EdgeDevices[i].end(); iter ++)
            bottom_sum += QN / pow(EtoD_Distance[i][*iter] / 1000, 1.0);
        for (auto iter = EdgeDevices[i].begin(); iter != EdgeDevices[i].end(); iter ++)
        {
            double current_gain = QN / pow(EtoD_Distance[i][*iter] / 1000, 1.0);
            double transmission_rate = 20 * log2(1 + current_gain / abs(bottom_sum - current_gain - 100)) / 8.0;    //Mbps -> MB/s
            Edge_Device_comm[i][*iter] = transmission_rate;
            if (transmission_rate < Edge_smallest_rate[i])
                Edge_smallest_rate[i] = transmission_rate;
        }
        //cout << Edge_smallest_rate[i] << endl;
    }
    
    for (int i = 0; i < CE_Tnum; i ++)
        CE_ST[i] = CE_ET[i] = 0;

    double time_max = 0;
    double energy = 0;
    for (int i = 0; i < CE_Tnum; i ++)
    {
        double t_comm = 0, t_comp = 0;
        if (ce_sele[i] == false)  //cloud mode
        {
            for (auto iter = CETask_coDevice[i].begin(); iter != CETask_coDevice[i].end(); iter++)
            {
                double cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_smallest_rate[ nearest_edge[*iter] ]);
                energy += cur_comm * QN / 1000;    //offloading energy qn * bn / rn(a)
                if (cur_comm > t_comm)
                    t_comm = cur_comm;
            }
        }
        else
        {
            for (auto iter = CETask_coDevice[i].begin(); iter != CETask_coDevice[i].end(); iter++)
            {
                double cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_Device_comm[ cevar[i] ][ *iter ]);
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
        
        //cout << "The " << i << "th computational task time: " << CE_ST[i] << "; " << CE_ET[i] << endl;
        
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
        int u_ratio = (CloudLoad[i].size() / 20.0) * 10;
        if (u_ratio > 10)
            u_ratio = 10;
        int time_expand = 0;
        for (auto iter = CloudLoad[i].begin(); iter != CloudLoad[i].end(); iter ++)
            if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
                time_expand = CE_ET[*iter] - CE_ST[*iter];
        energy += EnergyList[u_ratio] * time_expand / 1000.0;
    }
    for (int i = 0; i < Enum; i ++)
    {
        if (EdgeLoad[i].size() == 0)
            continue;
        int u_ratio = (EdgeLoad[i].size() / 6.0) * 10;
        if (u_ratio > 10)
            u_ratio = 10;
        int time_expand = 0;
        for (auto iter = EdgeLoad[i].begin(); iter != EdgeLoad[i].end(); iter ++)
            if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
                time_expand = CE_ET[*iter] - CE_ST[*iter];
        energy += EnergyList[u_ratio] * time_expand / 1000.0;
    }
    
    return time_max + energy;
}

