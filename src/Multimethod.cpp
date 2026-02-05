#include "Multimethod.h"
#include <limits>
#include <numeric>

#define EVAL_COMPAT(var, Cnum_, Enum_, Dnum_, CE_Tnum_, M_Jnum_, M_OPTnum_, CETask_Property_, MTask_Time_, EtoD_Distance_, DtoD_Distance_, AvailDeviceList_, EnergyList_, CloudDevices_, EdgeDevices_, CloudLoad_, EdgeLoad_, DeviceLoad_, CETask_coDevice_, Edge_Device_comm_, ST_, ET_, CE_ST_, CE_ET_) \
    EvaluFunc((var), workspace, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET)

double MultiMet::randnorm(double miu, double score)
{
    return miu + score * sqrt(abs(-2 * log((rand() + 1) / (RAND_MAX + 1.0)))) * cos(2 * M_PI * rand() / (RAND_MAX + 1.0));                 //by lyl
}

void MultiMet::pop_update(int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i ++)
    {
        for (int j = 0; j < Nvar; j ++)
            pop[i][j] = newpop[i][j];
        pop_fit[i] = newpop_fit[i];

        if (newpop_fit[i] < ibest_fit[i])
        {
            for (int j = 0; j < Nvar; j ++)
                ibest[i][j] = newpop[i][j];
            ibest_fit[i] = newpop_fit[i];
        }
    }
}

void MultiMet::pop_better_update(int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i ++)
    {
        if (newpop_fit[i] < pop_fit[i])
        {
            for (int j = 0; j < Nvar; j ++)
                pop[i][j] = newpop[i][j];
            pop_fit[i] = newpop_fit[i];
        }

        if (newpop_fit[i] < ibest_fit[i])
        {
            for (int j = 0; j < Nvar; j ++)
                ibest[i][j] = newpop[i][j];
            ibest_fit[i] = newpop_fit[i];
        }
    }
}

MultiMet::MultiMet(int psize, int nn, double lb, double ub, int c_num, int e_num, int d_num, int ce_tnum, int m_jnum, int m_optnum, FF evaluate, const std::filesystem::path& data_dir, const std::string& data_file)
    : Population(psize, nn, lb, ub), Cnum(c_num), Enum(e_num), Dnum(d_num), CE_Tnum(ce_tnum), M_Jnum(m_jnum), M_OPTnum(m_optnum), DataDir(data_dir), DataFileName(data_file), Pini(0.4)
{
    EvaluFunc = evaluate;
    eval_count = 0;

    //Prob
    CETask_Property = new CETask[CE_Tnum];
    MTask_Time = new double[M_Jnum * M_OPTnum];
    EtoD_Distance = CreateMatrix(Enum, Dnum);
    DtoD_Distance = CreateMatrix(Dnum, Dnum);
    AvailDeviceList = new vector<int>[M_Jnum * M_OPTnum];
    EnergyList = new double[11]; //0 to 10

    CloudDevices = new vector<int>[Cnum];
    EdgeDevices = new vector<int>[Enum];
    CloudLoad = new vector<int>[Cnum];
    EdgeLoad = new vector<int>[Enum];
    DeviceLoad = new vector<int>[Dnum];
    CETask_coDevice = new vector<int>[CE_Tnum];
    ST = CreateMatrix(M_Jnum, M_OPTnum);
    ET = CreateMatrix(M_Jnum, M_OPTnum);
    CE_ST = new double[CE_Tnum];
    CE_ET = new double[CE_Tnum];

    // Initialize workspace for reusable fitness evaluation buffers
    workspace.resize(CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum);
    Edge_Device_comm = workspace.edge_device_comm.data();

    //PSO
    ibest = CreateMatrix(Popsize, Nvar);
    velocity = CreateMatrix(Popsize, Nvar);
    ibest_fit = new double[Popsize];
    AC1 = new double[Popsize];
    AC2 = new double[Popsize];
    AW = new double[Popsize];
    OArow = OArow = Nvar + 1;
    OA = new int *[OArow];
    for (int i = 0; i < OArow; i ++)
        OA[i] = new int[Nvar];

    //ACO
    ant_tao = CreateMatrix(Popsize, Nvar + 2);

    //ABCA
    trial = new int[Popsize];
    pr = new double[Popsize];

    //VNS
    neigh = new int[Popsize];

    //CMAES

    weights = new double[(int)(Popsize/2)];
    xstart = new double[Nvar];
    stddev = new double[Nvar];

    //PBILc
    PB_center = new double[Nvar];
    PB_sigma = new double[Nvar];

    //BATA
    BAT_r = new double[Popsize];
    BAT_A = new double[Popsize];
    BAT_v = CreateMatrix(Popsize, Nvar);

    //FA
    I = new double[Popsize];
    Index = new int[Popsize];

    //BA
    ngh = new double[Popsize];
    ngh_decay_count = new int[Popsize];

    //meme number
    Ind_meme = new int[Popsize];
    SubDecBase = CreateMatrix(Popsize, Nvar + 2);
}

MultiMet::~MultiMet()
{
    int i;
    //Prob
    for (i = 0; i < CE_Tnum; i ++)
    {
        CETask_Property[i].Precedence.clear();
        CETask_Property[i].Interact.clear();
        CETask_Property[i].Start_Pre.clear();
        CETask_Property[i].End_Pre.clear();
        CETask_Property[i].AvailEdgeServerList.clear();
    }
    delete [] CETask_Property;
    delete [] MTask_Time;
    DeleteMatrix(EtoD_Distance, Enum);
    DeleteMatrix(DtoD_Distance, Dnum);
    delete [] AvailDeviceList;
    delete [] EnergyList;

    delete [] CloudDevices;
    delete [] EdgeDevices;
    delete [] CloudLoad;
    delete [] EdgeLoad;
    delete [] DeviceLoad;
    delete [] CETask_coDevice;
    DeleteMatrix(ST, M_Jnum);
    DeleteMatrix(ET, M_Jnum);
    delete [] CE_ST;
    delete [] CE_ET;

    //PSO
    DeleteMatrix(ibest, Popsize);
    DeleteMatrix(velocity, Popsize);
    delete [] ibest_fit;
    delete [] AC1;
    delete [] AC2;
    delete [] AW;
    for (i = 0; i < OArow; i ++)
        delete [] OA[i];
    delete [] OA;

    //ACO
    DeleteMatrix(ant_tao, Popsize);

    //ABCA
    delete [] trial;
    delete [] pr;

    //VNS
    delete [] neigh;

    //CMAES
    delete [] xstart;
    delete [] stddev;
    delete [] weights;

    //PBILc
    delete [] PB_center;
    delete [] PB_sigma;

    //BATA
    delete [] BAT_r;
    delete [] BAT_A;
    DeleteMatrix(BAT_v, Popsize);

    //BA
    delete [] ngh;
    delete [] ngh_decay_count;

    //meme number
    delete [] Ind_meme;
    DeleteMatrix(SubDecBase, Popsize);
}

void MultiMet::CreateOA()
{
    int u = (int)(log((double)OArow) / log(2.0));
    int b;
    for (int i = 0; i < OArow; i ++)
    {
        for (int j = 0; j < u; j ++)
        {
            b = (1 << j) - 1;
            int denom = 1 << (u - j - 1);
            int tmp = i / denom;
            OA[i][b] = tmp % 2;
        }
    }

    for (int i = 0; i < OArow; i ++)
    {
        for (int j = 0; j < u; j ++)
        {
            b = (1 << j) - 1;
            for (int s = 0; s < b; s ++)
                OA[i][b + s + 1] = (OA[i][s] + OA[i][b]) % 2;
        }
    }
}

void MultiMet::Initial()
{
    //Prob
    fstream fs;

    // Build data file path using DataDir
    std::filesystem::path data_file = DataDir / DataFileName;
    cout << "[Initial] Loading data from: " << data_file << endl;

    fs.open(data_file);
    if (!fs.is_open()) {
        cerr << "[ERROR] Failed to open data file: " << data_file << endl;
        cerr << "        Please specify --data_dir and --data_file with the path to data files." << endl;
        exit(1);
    }

    for (int i = 0; i < Enum; i++)
    {
        for (int j = 0; j < Dnum; j++)
            fs >> EtoD_Distance[i][j];
    }

    for (int i = 0; i < Dnum; i ++)
    {
        for (int j = 0; j < Dnum; j ++)
            fs >> DtoD_Distance[i][j];
    }

    for (int i = 0; i < M_Jnum * M_OPTnum; i ++)
        fs >> MTask_Time[i];

    int vec_num = 0, value;
    for (int i = 0; i < CE_Tnum; i ++)
    {
        fs >> CETask_Property[i].Computation;
        fs >> CETask_Property[i].Communication;

        fs >> vec_num;
        CETask_Property[i].Precedence.clear();
        for (int j = 0; j < vec_num; j ++)
        {
            fs >> value;
            CETask_Property[i].Precedence.push_back(value);
        }

        fs >> vec_num;
        CETask_Property[i].Interact.clear();
        for (int j = 0; j < vec_num; j ++)
        {
            fs >> value;
            CETask_Property[i].Interact.push_back(value);
        }

        fs >> vec_num;
        CETask_Property[i].Start_Pre.clear();
        for (int j = 0; j < vec_num; j ++)
        {
            fs >> value;
            CETask_Property[i].Start_Pre.push_back(value);
        }

        fs >> vec_num;
        CETask_Property[i].End_Pre.clear();
        for (int j = 0; j < vec_num; j ++)
        {
            fs >> value;
            CETask_Property[i].End_Pre.push_back(value);
        }

        fs >> CETask_Property[i].Job_Constraints;
    }

    for (int i = 0; i < M_Jnum; i ++)
    {
        for (int j = 0; j < M_OPTnum; j ++)
        {
            fs >> vec_num;
            AvailDeviceList[i * M_OPTnum + j].clear();
            for (int k = 0; k < vec_num; k ++)
            {
                fs >> value;
                AvailDeviceList[i * M_OPTnum + j].push_back(value);
            }
        }
    }

    for (int i = 0; i < CE_Tnum; i ++)
    {
        fs >> vec_num;
        CETask_Property[i].AvailEdgeServerList.clear();
        for (int j = 0; j < vec_num; j ++)
        {
            fs >> value;
            CETask_Property[i].AvailEdgeServerList.push_back(value);
        }
    }

    for (int i = 0; i < 11; i ++)
        fs >> EnergyList[i];

    fs.close();

    const int ops = M_Jnum * M_OPTnum;
    vector<int> perm(ops);
    vector<int> cloud_load(Cnum, 0);
    vector<int> edge_load(Enum, 0);
    vector<int> device_load(Dnum, 0);
    vector<char> seen_device(Dnum, 0);
    vector<int> task_devices;
    task_devices.reserve(Dnum);
    vector<int> nearest_edge_for_device(Dnum, 0);

    for (int d = 0; d < Dnum; d++)
    {
        double min_dis = std::numeric_limits<double>::infinity();
        int min_index = 0;
        for (int e = 0; e < Enum; e++)
        {
            if (EtoD_Distance[e][d] < min_dis)
            {
                min_dis = EtoD_Distance[e][d];
                min_index = e;
            }
        }
        nearest_edge_for_device[d] = min_index;
    }

    auto bucket_value = [](int idx, int count) -> double {
        if (count <= 1) return 0.0;
        if (idx >= count - 1) return 1.0;
        return (idx + 0.5) / (count - 1);
    };

    auto pick_min_load = [](const vector<int>& loads) -> int {
        int best_idx = 0;
        int best_val = loads[0];
        for (int i = 1; i < (int)loads.size(); i++)
        {
            if (loads[i] < best_val)
            {
                best_val = loads[i];
                best_idx = i;
            }
        }
        return best_idx;
    };

    for (int i = 0; i < Popsize; i ++)
    {
        std::vector<int> op_dev_index(ops, 0);

        std::fill(cloud_load.begin(), cloud_load.end(), 0);
        std::fill(edge_load.begin(), edge_load.end(), 0);
        std::fill(device_load.begin(), device_load.end(), 0);

        bool pure_random = (randval(0.0, 1.0) < Pini);
        int heu = pure_random ? -1 : (rand() % 7); // Heu1..Heu7 -> 0..6

        for (int t = 0; t < CE_Tnum; t ++)
        {
            const vector<int>& edge_list = CETask_Property[t].AvailEdgeServerList;
            bool edge_mode = false;
            int edge_list_idx = 0;
            int cloud_idx = 0;
            int selected_edge = 0;

            task_devices.clear();
            std::fill(seen_device.begin(), seen_device.end(), 0);
            for (int j = 0; j < M_OPTnum; j ++)
            {
                int op_idx = t * M_OPTnum + j;
                for (size_t k = 0; k < AvailDeviceList[op_idx].size(); k ++)
                {
                    int dev = AvailDeviceList[op_idx][k];
                    if (dev >= 0 && dev < Dnum && !seen_device[dev])
                    {
                        seen_device[dev] = 1;
                        task_devices.push_back(dev);
                    }
                }
            }

            if (pure_random)
            {
                edge_mode = (!edge_list.empty() && randval(0.0, 1.0) > 0.5);
                if (edge_mode)
                {
                    edge_list_idx = (int)(rand() % edge_list.size());
                    selected_edge = edge_list[edge_list_idx];
                }
                else
                {
                    cloud_idx = rand() % Cnum;
                }
            }
            else
            {
                if (heu == 0) // Heu1: nearest edge + nearest device
                {
                    if (!edge_list.empty())
                    {
                        edge_mode = true;
                        double best = std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < edge_list.size(); k ++)
                        {
                            int edge = edge_list[k];
                            double sum = 0.0;
                            for (size_t d = 0; d < task_devices.size(); d ++)
                                sum += EtoD_Distance[edge][task_devices[d]];
                            double avg = task_devices.empty() ? 0.0 : (sum / task_devices.size());
                            if (avg < best)
                            {
                                best = avg;
                                edge_list_idx = (int)k;
                                selected_edge = edge;
                            }
                        }
                    }
                }
                else if (heu == 1) // Heu2: cloud min load + device min load
                {
                    edge_mode = false;
                    cloud_idx = pick_min_load(cloud_load);
                }
                else if (heu == 2) // Heu3: edge min load + device min load
                {
                    if (!edge_list.empty())
                    {
                        edge_mode = true;
                        int best_load = INT_MAX;
                        for (size_t k = 0; k < edge_list.size(); k ++)
                        {
                            int edge = edge_list[k];
                            if (edge_load[edge] < best_load)
                            {
                                best_load = edge_load[edge];
                                edge_list_idx = (int)k;
                                selected_edge = edge;
                            }
                        }
                    }
                    else
                    {
                        edge_mode = false;
                        cloud_idx = pick_min_load(cloud_load);
                    }
                }
                else if (heu == 3) // Heu4: communication-aware
                {
                    double ratio = CETask_Property[t].Communication / (CETask_Property[t].Computation + 1e-6);
                    if (!edge_list.empty() && ratio > 0.5)
                    {
                        edge_mode = true;
                        double best = std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < edge_list.size(); k ++)
                        {
                            int edge = edge_list[k];
                            double sum = 0.0;
                            for (size_t d = 0; d < task_devices.size(); d ++)
                                sum += EtoD_Distance[edge][task_devices[d]];
                            double avg = task_devices.empty() ? 0.0 : (sum / task_devices.size());
                            if (avg < best)
                            {
                                best = avg;
                                edge_list_idx = (int)k;
                                selected_edge = edge;
                            }
                        }
                    }
                    else
                    {
                        edge_mode = false;
                        cloud_idx = pick_min_load(cloud_load);
                    }
                }
                else if (heu == 4) // Heu5: energy-aware
                {
                    int cloud_pick = pick_min_load(cloud_load);
                    int edge_pick = 0;
                    int edge_pick_idx = 0;
                    if (!edge_list.empty())
                    {
                        int best_load = INT_MAX;
                        for (size_t k = 0; k < edge_list.size(); k ++)
                        {
                            int edge = edge_list[k];
                            if (edge_load[edge] < best_load)
                            {
                                best_load = edge_load[edge];
                                edge_pick = edge;
                                edge_pick_idx = (int)k;
                            }
                        }
                        int cloud_ratio = (int)((cloud_load[cloud_pick] + 1) / 20.0 * 10.0);
                        if (cloud_ratio > 10) cloud_ratio = 10;
                        int edge_ratio = (int)((edge_load[edge_pick] + 1) / 6.0 * 10.0);
                        if (edge_ratio > 10) edge_ratio = 10;
                        if (EnergyList[edge_ratio] < EnergyList[cloud_ratio])
                        {
                            edge_mode = true;
                            selected_edge = edge_pick;
                            edge_list_idx = edge_pick_idx;
                        }
                        else
                        {
                            edge_mode = false;
                            cloud_idx = cloud_pick;
                        }
                    }
                    else
                    {
                        edge_mode = false;
                        cloud_idx = cloud_pick;
                    }
                }
                else if (heu == 5) // Heu6: random edge/cloud
                {
                    edge_mode = (!edge_list.empty() && randval(0.0, 1.0) > 0.5);
                    if (edge_mode)
                    {
                        edge_list_idx = (int)(rand() % edge_list.size());
                        selected_edge = edge_list[edge_list_idx];
                    }
                    else
                    {
                        cloud_idx = rand() % Cnum;
                    }
                }
                else // Heu7: device-centric
                {
                    int first_op = t * M_OPTnum;
                    int best_dev = 0;
                    if (!AvailDeviceList[first_op].empty())
                        best_dev = AvailDeviceList[first_op][0];
                    double best_score = std::numeric_limits<double>::infinity();
                    for (size_t k = 0; k < AvailDeviceList[first_op].size(); k ++)
                    {
                        int dev = AvailDeviceList[first_op][k];
                        double sum = 0.0;
                        for (size_t kk = 0; kk < AvailDeviceList[first_op].size(); kk ++)
                            sum += DtoD_Distance[dev][AvailDeviceList[first_op][kk]];
                        if (sum < best_score)
                        {
                            best_score = sum;
                            best_dev = dev;
                        }
                    }
                    int near_edge = nearest_edge_for_device[best_dev];
                    auto it = std::find(edge_list.begin(), edge_list.end(), near_edge);
                    if (it != edge_list.end())
                    {
                        edge_mode = true;
                        edge_list_idx = (int)(it - edge_list.begin());
                        selected_edge = near_edge;
                    }
                    else
                    {
                        edge_mode = false;
                        cloud_idx = pick_min_load(cloud_load);
                    }
                }
            }

            pop[i][t] = edge_mode ? 0.75 : 0.25;
            if (edge_mode)
            {
                pop[i][CE_Tnum + t] = bucket_value(edge_list_idx, (int)edge_list.size());
                edge_load[selected_edge]++;
            }
            else
            {
                pop[i][CE_Tnum + t] = bucket_value(cloud_idx, Cnum);
                cloud_load[cloud_idx]++;
            }

            for (int j = 0; j < M_OPTnum; j ++)
            {
                int op_idx = t * M_OPTnum + j;
                const vector<int>& dev_list = AvailDeviceList[op_idx];
                int dev_idx = 0;
                int dev_id = -1;
                if (!dev_list.empty())
                {
                    if (pure_random)
                    {
                        dev_idx = rand() % dev_list.size();
                        dev_id = dev_list[dev_idx];
                    }
                    else if (heu == 0 || heu == 3 || heu == 5 || (heu == 4 && edge_mode))
                    {
                        double best = std::numeric_limits<double>::infinity();
                        int edge = edge_mode ? selected_edge : 0;
                        for (size_t k = 0; k < dev_list.size(); k ++)
                        {
                            int dev = dev_list[k];
                            double d = edge_mode ? EtoD_Distance[edge][dev] : (double)device_load[dev];
                            if (d < best)
                            {
                                best = d;
                                dev_idx = (int)k;
                                dev_id = dev;
                            }
                        }
                    }
                    else if (heu == 1 || heu == 2 || (heu == 4 && !edge_mode))
                    {
                        int best_load = INT_MAX;
                        for (size_t k = 0; k < dev_list.size(); k ++)
                        {
                            int dev = dev_list[k];
                            if (device_load[dev] < best_load)
                            {
                                best_load = device_load[dev];
                                dev_idx = (int)k;
                                dev_id = dev;
                            }
                        }
                    }
                    else
                    {
                        double best_score = std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < dev_list.size(); k ++)
                        {
                            int dev = dev_list[k];
                            double sum = 0.0;
                            for (size_t kk = 0; kk < dev_list.size(); kk ++)
                                sum += DtoD_Distance[dev][dev_list[kk]];
                            if (sum < best_score)
                            {
                                best_score = sum;
                                dev_idx = (int)k;
                                dev_id = dev;
                            }
                        }
                    }
                }

                op_dev_index[op_idx] = dev_idx;
                if (dev_id >= 0 && dev_id < Dnum)
                    device_load[dev_id]++;
            }
        }

        iota(perm.begin(), perm.end(), 0);
        for (int k = ops - 1; k > 0; k --)
        {
            int r = rand() % (k + 1);
            std::swap(perm[k], perm[r]);
        }
        for (int rank = 0; rank < ops; rank ++)
        {
            int op_idx = perm[rank];
            pop[i][2 * CE_Tnum + op_idx] = (rank + 0.5) / ops;
        }

        for (int rank = 0; rank < ops; rank ++)
        {
            int op_idx = perm[rank];
            int avail = (int)AvailDeviceList[op_idx].size();
            pop[i][2 * CE_Tnum + ops + rank] = bucket_value(op_dev_index[op_idx], avail);
        }

        for (int j = 0; j < Nvar; j ++)
            newpop[i][j] = pop[i][j];
    }

    for (int i = 0; i < Popsize; i ++)
        for (int j = 0; j < Nvar; j ++)
            velocity[i][j] = randval(Lbound, Ubound);

    for (int i = 0; i < Popsize; i ++)
    {
        for (int j = 0; j < Nvar; j ++)
            ibest[i][j] = pop[i][j];
        pop_fit[i] = EVAL_COMPAT(pop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
        newpop_fit[i] = pop_fit[i];

        ibest_fit[i] = pop_fit[i];
    }
    worst_and_best();
    for (int j = 0; j < Nvar; j ++)
        gbest[j] = pop[cur_best][j];
    gbest_fit = pop_fit[cur_best];
    CRfit();
    CRold = CRnew;

    //pso
    ac1 = ac2 = 2;
    for (int i = 0; i < Popsize; i ++)
    {
        AC1[i] = AC2[i] = 2;
        AW[i] = 0.85;
    }
    CreateOA();

    //aco
    for (int i = 0; i < Popsize; i ++)
        for (int j = 0; j < Nvar; j ++)
            ant_tao[i][j] = randval(Lbound, Ubound);
    for (int i = 0; i < Popsize; i ++)
        ant_tao[i][Nvar] = EVAL_COMPAT(ant_tao[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    heap_sort(ant_tao, Popsize, Nvar);
    for (int i = 0; i < Popsize; i ++)
    {
        ant_tao[i][Nvar + 1] = 0;
        for (int j = 0; j < Nvar; j ++)
            ant_tao[i][Nvar + 1] += ant_tao[i][j] * ant_tao[i][j];
    }

    //ABCA
    for (int i = 0; i < Popsize; i ++)
        trial[i] = 0;

    //FA
    for (int i = 0; i < Popsize; i ++)
        Index[i] = i;

    //BA
    ne = Popsize / 5 * 2;
    nre = 100;
    stlim = 10;
    ngh_decay = 0.8; ngh_origin = (Ubound - Lbound) * 0.1;
    for (int i = 0; i < Popsize; i ++)
    {
        ngh[i] = ngh_origin;
        ngh_decay_count[i] = 0;
    }

    //meme number
    for (int i = 0; i < Popsize; i ++)
        Ind_meme[i] = rand() % 4;

    for (int i = 0; i < Popsize; i ++)
        for (int j = 0; j < Nvar + 2; j ++)
            SubDecBase[i][j] = 0;
}

void MultiMet::Evaluation(bool s, int p_start, int p_end)
{
    if (s == 0)
    {
        for (int i = p_start; i < p_end; i ++)
        {
            pop_fit[i] = EVAL_COMPAT(pop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            eval_count++;
        }
    }
    else
    {
        for (int i = p_start; i < p_end; i ++)
        {
            newpop_fit[i] = EVAL_COMPAT(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            eval_count++;
        }
    }
}

double MultiMet::Eval(const double* var)
{
    eval_count++;
    return EVAL_COMPAT(var, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
}

void MultiMet::SetPini(double pini)
{
    Pini = pini;
}
void MultiMet::select(int p_start, int p_end)
{
	int i, j;
	double *rfitness, *cfitness;
	double p, sum = 0;
	rfitness = new double[Popsize];
	cfitness = new double[Popsize];
	for (i = 0; i < Popsize; i++)
		sum += 1000.0 / pop_fit[i];                                //适应值总和
	for (i = 0; i < Popsize; i++)
		rfitness[i] = (1000.0 / pop_fit[i]) / sum;                 //适应值所占比??
	cfitness[0] = rfitness[0];
	for (i = 1; i < Popsize; i++)
		cfitness[i] = cfitness[i-1] + rfitness[i];           //轮盘位置

	//srand(unsigned(time(0)));                                 
	for (i = p_start; i < p_end; i++)
	{
		p = randval(0.0, 1.0);
		if (p < cfitness[0])                                 //轮盘赌选择
		{
			for (int k = 0; k < Nvar; k++)
				newpop[i][k] = pop[p_start][k];
		}
		else
		{
			for (j = 0; j < Popsize - 1; j ++)
				if (p >= cfitness[j] && p < cfitness[j+1])
				{
					for (int k = 0; k < Nvar; k++)
						newpop[i][k] = pop[j+1][k];              //选择的新个体存入newpop
				}
		}
	}
	delete [] cfitness;
	delete [] rfitness;
}

void MultiMet::crossover(double pc, int p_start, int p_end)
{
	int mem;
	int pos = 0;
	double p;
	for(mem = p_start; mem < p_end; mem++)
	{
		do 
		{
			pos = rand() % Popsize;
		} while (pos == mem);
		p = randval(0, 1);
		if (p < pc)                           //若概率小于pc，执行交换子xover操作
		{
			xover(pos, mem);
		}
	}
}

void MultiMet::xover(int one, int two)
{
	int i, point;
	double r;
	double temp1, temp2;
	if (Nvar == 2)
		point = 1;
	else
		point = rand() % (Nvar - 1) + 1;

	for (i = 0; i < point ; i++)
	{
		r = randval(0, 1);
		temp1 = newpop[one][i] * r + (1 - r) * newpop[two][i];
		temp2 = newpop[two][i] * r + (1 - r) * newpop[one][i];
		if (temp1 > Ubound)
			temp1 = Ubound;
		else if (temp1 < Lbound)
			temp1 = Lbound;
		newpop[one][i] = temp1;
	}
}

void MultiMet::mutate(double pm, int p_start, int p_end)
{
	double p;
	for (int i = p_start; i < p_end; i++)
	{
		p = randval(0.0, 1.0);                  
		if (p < pm)
		{
			int r = rand() % Nvar;
			newpop[i][r] = randval(Lbound, Ubound);
		}
	}
}

void MultiMet::GA(double pc, double pm, int p_start, int p_end)
{
	select(p_start, p_end);
	crossover(pc, p_start, p_end);
	mutate(pm, p_start, p_end);
}

void MultiMet::NGA(double pc, double pm, double mdis, int p_start, int p_end)
{
	pop_niche(mdis, p_start, p_end);
	select(p_start, p_end);
	crossover(pc, p_start, p_end);
	mutate(pm, p_start, p_end);
}

void MultiMet::LGA(double pc, double pm, double step, int nst, int p_start, int p_end)
{
	select(p_start, p_end);
	crossover(pc, p_start, p_end);
	mutate(pm, p_start, p_end);
	for (int i = p_start; i < p_end; i ++)
        newpop_fit[i] = EVAL_COMPAT(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
	for (int i = p_start; i < p_end; i ++)
		localsearch(newpop[i], nst);
}

void MultiMet::VAGA(int p_start, int p_end)
{
	double vari = pop_random_variance(0, Popsize);
	double p = exp(-vari);
	select(p_start, p_end);
	crossover(1 - p, p_start, p_end);
	mutate(p, p_start, p_end);
}

void MultiMet::EAGA(int p_start, int p_end)
{
	int subn = (int)((Ubound - Lbound) * 0.1 / 2);
	double entr = pop_random_entropy(subn, 0, Popsize);
	double p = exp(-entr);
	select(p_start, p_end);
	crossover(1 - p, p_start, p_end);
	mutate(p, p_start, p_end);
}

void MultiMet::PSO(double w, double c1, double c2, double max_ve, int p_start, int p_end)
{
	int i, j;
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
}

void MultiMet::CPSO(double w, double c1, double c2, double max_ve, int p_start, int p_end)
{
	int i, j;
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = 0.729 * (w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]));                            //constriction factor X = 0.729
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
}

void MultiMet::Subgradient(double *theta, int q, double c_step, double *subgrad)
{
	double *delta = new double[q];
	double **var = new double *[2 * q];
	for (int i = 0; i < 2 * q; i ++)
		var[i] = new double[Nvar];

	for(int i = 0; i < Nvar; i ++)
	{
		for (int j = 0; j < 2 * q; j += 2)
		{
			delta[j / 2] = randval(0, 1);
			for (int k = 0; k < Nvar; k ++)
			{
				if (k == i)
				{
					var[j][k] = theta[k] + c_step * delta[j / 2];
					var[j + 1][k] = theta[k] - c_step * delta[j / 2];
				}
				else
					var[j + 1][k] = var[j][k] = theta[k];
			}
		}
		subgrad[i] = 0;
		for (int j = 0; j < 2 * q; j += 2)
            subgrad[i] += (EVAL_COMPAT(var[j], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET)
                           - EVAL_COMPAT(var[j + 1], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET))
                            / (2 * c_step * delta[j / 2] + 0.001);
		subgrad[i] /= q;
	}
	delete [] delta;
	for (int i = 0; i < 2 * q; i ++)
		delete [] var[i];
	delete [] var;
}

void MultiMet::SPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end)
{
	int i, j;
	double velnorm, subgnorm;
	double *subgrad = new double[Nvar];
	for (i = p_start; i < p_end; i ++)
	{
		velnorm = 0;
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			velnorm += velocity[i][j] * velocity[i][j];
		}
		velnorm = sqrt(velnorm);

		Subgradient(newpop[i], Nvar, velnorm, subgrad);
		subgnorm = 0;
		for (j = 0; j < Nvar; j ++)
			subgnorm += subgrad[j] * subgrad[j];
		subgnorm = sqrt(subgnorm);
		for (j = 0; j < Nvar; j ++)
		{
			newpop[i][j] = newpop[i][j] - (velnorm / (subgnorm + 1e-5)) * subgrad[j];
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Ubound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Lbound;
		}
	}
	delete [] subgrad;
}

void MultiMet::Cauchy_mutation(double* pp, int Gen, int MaxGen)
{
	int bit = rand() % Nvar;
	double p = randval(0, 1);
	if (p < 0.5)
		pp[bit] += (Ubound - pp[bit]) * (1 - pow(randval(0, 1), (1 - (double)Gen / MaxGen)));
	else
		pp[bit] += (pp[bit] - Lbound) * (1 - pow(randval(0, 1), (1 - (double)Gen / MaxGen)));
    if (pp[bit] > Ubound)
		pp[bit] = Ubound;
	else if (pp[bit] < Lbound)
		pp[bit] = Lbound;
}

void MultiMet::CMPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end)
{
	int i, j;
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
        //if (randval(0, 1) < 0.01)
            //Cauchy_mutation(newpop[i], Gen, MaxGen);
	}
}

void MultiMet::APSO_1(double c1, double c2, double max_ve, int Gen, int MaxGen, int p_start, int p_end)
{
	int i, j;
	double w = 0.9 - 0.5 * Gen / MaxGen;
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
}

void MultiMet::APSO_2(double c1, double c2, double max_ve, int p_start, int p_end)
{
	int i, j;
	double w = 0.5 + randval(0, 1) / 2;
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ c1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ c2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
}

void MultiMet::APSO_3(double max_ve, int p_start, int p_end)
{
	int i, j, k;
	double *d = new double[Popsize];
	double dg, dmin, dmax, tmp;
	double f, w;
	double S1, S2, S3, S4;
	dg = 0;
	for (i = 0; i < Popsize; i ++)
	{
		tmp = 0;
		for (j = 0; j < Nvar; j ++)
			tmp += pow(newpop[i][j] - gbest[j], 2);
		dg += sqrt(tmp); 
	}

	dmax = 0; dmin = 1e10;
	for (i = 0; i < Popsize; i ++)
	{
		d[i] = 0;
		for (j = 0; j < Popsize; j ++)
		{
			tmp = 0;
			for (k = 0; k < Nvar; k ++)
				tmp += pow(newpop[i][k] - newpop[j][k], 2);
			d[i] += sqrt(tmp); 
		}
		if (d[i] > dmax)
			dmax = d[i];
		else if (d[i] < dmin)
			dmin = d[i];
	}

	dg /= (Popsize - 1);
	dmin /= (Popsize - 1);
	dmax /= (Popsize - 1);
	f = (dg - dmin) / (dmax - dmin);

	if (f <= 0.4)
		S1 = 0;
	else if (f <= 0.6)
		S1 = 5 * f - 2;
	else if (f <= 0.7)
		S1 = 1;
	else if (f <= 0.8)
		S1 = -10 * f + 8;
	else
		S1 = 0;

	if (f <= 0.2)
		S2 = 0;
	else if (f <= 0.3)
		S2 = 10 * f - 2;
	else if (f <= 0.4)
		S2 = 1;
	else if (f <= 0.6)
		S2 = -5 * f + 3;
	else
		S2 = 0;

	if (f <= 0.1)
		S3 = 1;
	else if (f <= 0.3)
		S3 = -5 * f + 1.5;
	else
		S3 = 0;

	if (f <= 0.7)
		S4 = 0;
	else if (f <= 0.9)
		S4 = 5 * f - 3.5;
	else
		S4 = 1;

	if (S3 > S2)
	{
		ac1 += 0.1; ac2 -= 0.1;
	}
	else if (S2 > S1)
	{
		ac1 += 0.05; ac2 -= 0.05;
	}
	else if (S1 > S4)
	{
		ac1 += 0.05; ac2 += 0.05;
	}
	else
	{
		ac1 -= 0.1; ac2 += 0.1;
	}

	if (ac1 + ac2 > 4)
	{
		ac1 = 4.0 * ac1 / (ac1 + ac2);
		ac2 = 4.0 * ac2 / (ac1 + ac2);
	}
	if (ac1 < 0)
		ac1 = 2;
	if (ac2 < 0)
		ac2 = 2;
	w = 1 / (1 + 1.5 * exp(-2.6 * f));

	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			double r1 = randval(0, 1);
			double r2 = randval(0, 1);
			velocity[i][j] = w * velocity[i][j]
			+ ac1 * r1 * (ibest[i][j] - newpop[i][j]) 
				+ ac2 * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
	delete [] d;
}

void MultiMet::APSO_4(double max_ve, int Gen, int MaxGen, int p_start, int p_end)
 {
	int i, j;
	double gw, gc1, gc2;
	double r1, r2;
	double fdist = 0, alp;

	for (i = p_start; i < p_end; i ++)
	{
		fdist = 0;
		for (j = 0; j < Nvar; j ++)
			fdist += pow(newpop[i][j] - gbest[j], 2);
		fdist = sqrt(fdist);

		gw = gc1 = gc2 = 0;
		for (j = 0; j < Nvar; j ++)
		{
			gw += (newpop[i][j] - gbest[j]) * velocity[i][j];
			gc1 += (newpop[i][j] - gbest[j]) * randval(0, 1) * (ibest[i][j] - newpop[i][j]);
			gc2 += (newpop[i][j] - gbest[j]) * randval(0, 1) * (gbest[j] - newpop[i][j]);
		}
		gw *= 2;
		gc1 *= 2;
		gc2 *= 2;

		alp = fdist / (gw * gw + gc1 * gc1 + gc2 * gc2 + 1);
		AW[i] -= alp * gw;
		AC1[i] -=alp * gc1;
		AC2[i] -=alp * gc2;
		if (AW[i] < 0.4)
			AW[i] = 0.4;
		else if (AW[i] > 0.9)
			AW[i] = 0.9;
		if (AC1[i] < 0.5)
			AC1[i] = 0.5;
		else if (AC1[i] > 2.5)
			AC1[i] = 2.5;
		if (AC2[i] < 0.5)
			AC2[i] = 0.5;
		else if (AC2[i] > 2.5)
			AC2[i] = 2.5;
		
		r1 = randval(0, 1);
		r2 = randval(0, 1);
		for (j = 0; j < Nvar; j ++)
		{
			velocity[i][j] = AW[i] * velocity[i][j]
			+ AC1[i] * r1 * (ibest[i][j] - newpop[i][j]) 
				+ AC2[i] * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
}

void MultiMet::APSO_5(double max_ve, int Gen, int MaxGen, int p_start, int p_end)
{
	int i, j;
	double velnorm, subgnorm;
	double *subgrad = new double[Nvar];
	double gw, gc1, gc2;
	double r1, r2;
	double fdist = 0, alp;

	for (i = p_start; i < p_end; i ++)
	{
		fdist = 0;
		for (j = 0; j < Nvar; j ++)
			fdist += pow(newpop[i][j] - gbest[j], 2);
		fdist = sqrt(fdist);

		gw = gc1 = gc2 = 0;
		for (j = 0; j < Nvar; j ++)
		{
			gw += (newpop[i][j] - gbest[j]) * velocity[i][j];
			gc1 += (newpop[i][j] - gbest[j]) * randval(0, 1) * (ibest[i][j] - newpop[i][j]);
			gc2 += (newpop[i][j] - gbest[j]) * randval(0, 1) * (gbest[j] - newpop[i][j]);
		}
		gw *= 2;
		gc1 *= 2;
		gc2 *= 2;

		alp = fdist / (gw * gw + gc1 * gc1 + gc2 * gc2 + 1);
		AW[i] -= alp * gw;
		AC1[i] -=alp * gc1;
		AC2[i] -=alp * gc2;
		if (AW[i] < 0.4)
			AW[i] = 0.4;
		else if (AW[i] > 0.9)
			AW[i] = 0.9;
		if (AC1[i] < 0.5)
			AC1[i] = 0.5;
		else if (AC1[i] > 2.5)
			AC1[i] = 2.5;
		if (AC2[i] < 0.5)
			AC2[i] = 0.5;
		else if (AC2[i] > 2.5)
			AC2[i] = 2.5;

		r1 = randval(0, 1);
		r2 = randval(0, 1);
		for (j = 0; j < Nvar; j ++)
		{
			velocity[i][j] = AW[i] * velocity[i][j]
			+ AC1[i] * r1 * (ibest[i][j] - newpop[i][j]) 
				+ AC2[i] * r2 * (gbest[j] - newpop[i][j]);
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
		}

		velnorm = 0;
		for (j = 0; j < Nvar; j ++)
			velnorm += velocity[i][j] * velocity[i][j];
		velnorm = sqrt(velnorm);

		Subgradient(newpop[i], Nvar, velnorm / 100, subgrad);
		subgnorm = 0;
		for (j = 0; j < Nvar; j ++)
			subgnorm += subgrad[j] * subgrad[j];
		subgnorm = sqrt(subgnorm);
		for (j = 0; j < Nvar; j ++)
		{
			newpop[i][j] = newpop[i][j] - (velnorm / (100 * subgnorm + 0.001)) * subgrad[j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
	delete [] subgrad;
}

void MultiMet::Orthogonal_P(double *P0, double w, double c, int ppn)
{
	double **Poptmp = new double *[OArow + 1];
	for (int i = 0; i < OArow + 1; i ++)
		Poptmp[i] = new double[Nvar + 1];
	int Pbest_index = 0;
	double Pbest_f = 1e10;
	double LA1, LA2;
	double ve;

	for (int i = 0; i < OArow; i ++)
	{
		for (int j = 0; j < Nvar; j ++)
		{
			double r = randval(0, 1);
			if (OA[i][j] == 0)
				ve = w * velocity[ppn][j] + c * r * (ibest[ppn][j] - newpop[ppn][j]);
			else
				ve = w * velocity[ppn][j] + c * r * (gbest[j] - newpop[ppn][j]);
			Poptmp[i][j] = newpop[ppn][j] + ve; 
			if (Poptmp[i][j] > Ubound)
				Poptmp[i][j] = Lbound;
			else if (Poptmp[i][j] < Lbound)
				Poptmp[i][j] = Ubound;
		}
        Poptmp[i][Nvar] = EVAL_COMPAT(Poptmp[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
		if (Poptmp[i][Nvar] < Pbest_f)
		{
			Pbest_f = Poptmp[i][Nvar];
			Pbest_index = i;
		}
	}

	for (int i = 0; i < Nvar; i ++)
	{
		LA1 = LA2 = 0;
		for (int j = 0; j < OArow; j ++)
		{
			if (OA[j][i] == 0)
				LA1 += Poptmp[j][Nvar];
			else
				LA2 += Poptmp[j][Nvar];
		}
		LA1 /= 2;
		LA2 /= 2;
		if (LA1 < LA2)
			P0[i] = ibest[ppn][i];
		else
			P0[i] = gbest[i];
	}

	for (int j = 0; j < Nvar; j ++)
	{
		double r = randval(0, 1);
		ve = w * velocity[ppn][j] + c * r * (P0[j] - newpop[ppn][j]);
		Poptmp[OArow][j] = newpop[ppn][j] + ve; 
		if (Poptmp[OArow][j] > Ubound)
			Poptmp[OArow][j] = Lbound;
		else if (Poptmp[OArow][j] < Lbound)
			Poptmp[OArow][j] = Ubound;
	}
    Poptmp[OArow][Nvar] = EVAL_COMPAT(Poptmp[OArow], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);

	if (Poptmp[Pbest_index][Nvar] < Poptmp[OArow][Nvar])
	{
		for (int i = 0; i < Nvar; i ++)
		{
			if (OA[OArow] == 0)
				P0[i] = ibest[ppn][i];
			else
				P0[i] = gbest[i];
		}
	}

	for (int i = 0; i < OArow + 1; i ++)
		delete [] Poptmp[i];
	delete [] Poptmp;
}

void MultiMet::OLPSO(double w, double c, double max_ve, int p_start, int p_end)
{
	int i, j;
	double *P0 = new double[Nvar];
	for (i = p_start; i < p_end; i ++)
	{
		for (j = 0; j < Nvar; j ++)
		{
			Orthogonal_P(P0, w, c, i);
			double r = randval(0, 1);
			velocity[i][j] = w * velocity[i][j] + c * r * (P0[j] - newpop[i][j]); 
			if (velocity[i][j] > max_ve)
				velocity[i][j] = max_ve;
			else if (velocity[i][j] < -max_ve)
				velocity[i][j] = -max_ve;
			newpop[i][j] = newpop[i][j] + velocity[i][j]; 
			if (newpop[i][j] > Ubound)
				newpop[i][j] = Lbound;
			else if (newpop[i][j] < Lbound)
				newpop[i][j] = Ubound;
		}
	}
	delete [] P0;
}

void MultiMet::newpop_worst_best(int& w, int& b, int p_start, int p_end)
{
	int i;
	int best = p_start, worst = p_start;
	for (i = p_start + 1; i < p_end; i++)
	{
		if (newpop_fit[i] < newpop_fit[best])
			best = i;
		else if (newpop_fit[i] > newpop_fit[worst])
			worst = i;
	}
	w = worst;
	b = best;
}

void MultiMet::HS(double srate, double trate, double bw, int p_start, int p_end)
{
	int ww = 0, bb = 0;
	double *tmp = new double[Nvar + 1];
	newpop_worst_best(ww, bb, p_start, p_end);
	for (int i = p_start; i < p_end; i ++)
	{
		for (int j = 0; j < Nvar; j ++)
		{
			if (randval(0, 1) < srate)                              //srate:hmcr
				tmp[j] = newpop[rand() % Popsize][j];
			else
				tmp[j] = randval(Lbound, Ubound);
		}

		if (randval(0, 1) < trate)                             //trate:par
		{
			for (int j = 0; j < Nvar; j ++)
			{
				tmp[j] += randval(0, 1) * bw;
				if (tmp[j] > Ubound)
					tmp[j] = Ubound;
				else if (tmp[j] < Lbound)
					tmp[j] = Lbound;
			}
		}

        tmp[Nvar] = EVAL_COMPAT(tmp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
		if (tmp[Nvar] < newpop_fit[ww])
		{
			for (int j = 0; j < Nvar; j ++)
				newpop[ww][j] = tmp[j];
			newpop_fit[ww] = tmp[Nvar];
			newpop_worst_best(ww, bb, p_start, p_end);
		}
	}
	delete [] tmp;
}

void MultiMet::path_finding(double epsl, int p_start, int p_end)
{
	int i, j, l = 0;
	double psum = 0;
	double *rp = new double[Popsize];
	double *cp = new double[Popsize];         //寻找第l位tao的概率密度和累计概率密度
	double *ssco = new double[Nvar];            //standard deviation
	for (i = 0; i < Popsize; i ++)
		psum += ant_tao[i][Nvar + 1];
	for (i = 0; i < Popsize; i ++)
		rp[i] = ant_tao[i][Nvar + 1] / psum;
	cp[0] = rp[0];
	for (i = 1; i < Popsize; i ++)
		cp[i] = cp[i-1] + rp[i];           //轮盘位置

	for (i = p_start; i < p_end; i ++)
	{
        double p = randval(0.0, 1.0);
        if (p < cp[0])
            l = 0;
        else
        {
            for (j = 0; j < Popsize - 1; j ++)
                if (p >= cp[j] && p < cp[j + 1])
                {
                    l = j + 1;
                    break;
                }
        }
        
        for (j = 0; j < Nvar; j ++)
        {
            for (int k = 0; k < Popsize; k ++)
                ssco[j] += abs(ant_tao[k][j] - ant_tao[l][j]) / (Popsize - 1.0);
            ssco[j] *= epsl;                //epsl:speed of convergence
        }
        
		for (j = 0; j < Nvar; j ++)
		{
            if (randval(0, 1) < 0.15)
            {
                newpop[i][j] = ant_tao[l][j] + (gauss() * sqrt(ssco[j]));
                if (newpop[i][j] < Lbound)
                    newpop[i][j] = Lbound;
                else if (newpop[i][j] > Ubound)
                    newpop[i][j] = Ubound;
            }
            else newpop[i][j] = pop[i][j];
		}
	}
	delete [] rp;
	delete [] cp;
	delete [] ssco;
}

void MultiMet::phe_updating(int p_start, int p_end)
{
    double maxTao = -1e5;
    int maxIndex = 0;
    bool sameflag = false;
	for (int i = p_start; i < p_end; i ++)
	{
        sameflag = false;
        for (int j = 0; j < Popsize; j ++)
            if (ant_tao[j][Nvar] == newpop_fit[i])
                sameflag = true;
        if (sameflag == false)
        {
            maxTao = -1e5;
            maxIndex = 0;
            for (int j = 0; j < Popsize; j ++)
            {
                if (ant_tao[j][Nvar] > maxTao)
                {
                    maxTao = ant_tao[j][Nvar];
                    maxIndex = j;
                }
            }
            if (ibest_fit[i] < maxTao)
            {
                for (int j = 0; j < Nvar; j ++)
                    ant_tao[maxIndex][j] = newpop[i][j];
                ant_tao[maxIndex][Nvar] = newpop_fit[i];
            }
        }
	}
    heap_sort(ant_tao, Popsize, Nvar);
    for (int i = 0; i < Popsize; i ++)
    {
        ant_tao[i][Nvar + 1] = exp(-pow(i, 2.0) / (2 * pow(1e-4 * Popsize, 2.0)))
                                / (1e-4 * Popsize * sqrt(2 * M_PI));
    }
}

void MultiMet::ACO(double epsl, int p_start, int p_end)
{
	path_finding(epsl, p_start, p_end);
    phe_updating(p_start, p_end);
}

void MultiMet::chaos(double **incpop, int chaos_n)
{
	int i, j;
	int l, point;
	double *x1 = new double[chaos_n + 1];
	double *x2 = new double[chaos_n + 1];
	x1[0] = randval(0, 1);
	x2[0] = randval(0, 1);
	for (i = 1; i < chaos_n; i ++)
	{
		x1[i] = 4 * x1[i - 1] * (1 - x1[i - 1]);
		x2[i] = 4 * x2[i - 1] * (1 - x2[i - 1]);
	}
	for (i = 0; i < chaos_n; i ++)
	{
		l = rand() % Popsize;
		point = (int)(x1[i] * Nvar) % Nvar;
		for (j = 0; j < Nvar; j ++)
			incpop[l][j] = gbest[j];
		incpop[l][point] = Lbound + x2[i] * (Ubound - Lbound);
	}
	delete [] x1;
	delete [] x2;
}

void MultiMet::COA(int chaos_n, int p_start, int p_end)
{
	double *x1 = new double[chaos_n];
	double *x2 = new double[chaos_n];
	double *tmp = new double[Nvar];
	int point = 0;
	for (int i = p_start; i < p_end; i ++)
	{
		int pp = 0; double MinN = 0, MinF = 1000;
		x1[0] = randval(0, 1);
		x2[0] = randval(0, 1);
		for (int j = 1; j < chaos_n; j ++)
		{
			x1[j] = 4 * x1[j - 1] * (1 - x1[j - 1]);
			x2[j] = 4 * x2[j - 1] * (1 - x2[j - 1]);
		}
		for (int j = 0; j < chaos_n; j ++)
		{
			point = (int)(x1[j] * Nvar) % Nvar;
			for (int k = 0; k < Nvar; k ++)
				tmp[k] = pop[i][k];
			tmp[point] = Lbound + x2[j] * (Ubound - Lbound);

            double ff = EVAL_COMPAT(tmp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
			if (ff < MinF)
			{
				MinF = ff;
				MinN = tmp[point];
				pp = point;
			}
		}
		if (MinF < newpop_fit[i])
		{
			for (int j = 0; j < Nvar; j ++)
				newpop[i][j] = pop[i][j];
			newpop[i][pp] = MinN;
			newpop_fit[i] = MinF;
		}
	}
	delete [] x1;
	delete [] x2;
	delete [] tmp;
}

void MultiMet::differential_mutate(double F, int S, int p_start, int p_end)
{
	int x[5];
	if (S == 1)
	{
		for (int i = p_start; i < p_end; i ++)
		{
			x[0] = rand() % Popsize;
			do 
			{
				x[1] = rand() % Popsize;
			} while (x[1] == x[0]);

			for (int j = 0; j < Nvar; j ++)
			{
				newpop[i][j] = gbest[j] + F * (ibest[x[0]][j] - ibest[x[1]][j]);
				if (newpop[i][j] > Ubound)
					newpop[i][j] = Ubound;
				else if (newpop[i][j] < Lbound)
					newpop[i][j] = Lbound;
			}
		}
	}
	else if (S == 2)
	{
		for (int i = p_start; i < p_end; i ++)
		{
			x[0] = rand() % Popsize;
			do 
			{
				x[1] = rand() % Popsize;
			} while (x[1] == x[0]);
			do 
			{
				x[2] = rand() % Popsize;
			} while (x[2] == x[1] || x[2] == x[0]);
			do 
			{
				x[3] = rand() % Popsize;
			} while (x[3] == x[2] || x[3] == x[1] || x[3] == x[0]);
			do 
			{
				x[4] = rand() % Popsize;
			} while (x[4] == x[3] || x[4] == x[2] || x[4] == x[1] || x[4] == x[0]);

			for (int j = 0; j < Nvar; j ++)
			{
				newpop[i][j] = ibest[x[0]][j] + F * (ibest[x[1]][j] - ibest[x[2]][j])
					+ F * (ibest[x[3]][j] - ibest[x[4]][j]);
				if (newpop[i][j] > Ubound)
					newpop[i][j] = Ubound;
				else if (newpop[i][j] < Lbound)
					newpop[i][j] = Lbound;
			}
		}
	}
	else if (S == 3)
	{
		for (int i = p_start; i < p_end; i ++)
		{
			x[0] = rand() % Popsize;
			do 
			{
				x[1] = rand() % Popsize;
			} while (x[1] == x[0]);
			do 
			{
				x[2] = rand() % Popsize;
			} while (x[2] == x[1] || x[2] == x[0]);
			do 
			{
				x[3] = rand() % Popsize;
			} while (x[3] == x[2] || x[3] == x[1] || x[3] == x[0]);

			for (int j = 0; j < Nvar; j ++)
			{
				newpop[i][j] = gbest[j] + F * (ibest[x[0]][j] - ibest[x[1]][j])
					+ F * (ibest[x[2]][j] - ibest[x[3]][j]);
				if (newpop[i][j] > Ubound)
					newpop[i][j] = Ubound;
				else if (newpop[i][j] < Lbound)
					newpop[i][j] = Lbound;
			}
		}
	}
	else if (S == 4)
	{
		for (int i = p_start; i < p_end; i ++)
		{
			x[0] = rand() % Popsize;
			do 
			{
				x[1] = rand() % Popsize;
			} while (x[1] == x[0]);

			for (int j = 0; j < Nvar; j ++)
			{
				newpop[i][j] = ibest[i][j] + F * (gbest[j] - ibest[i][j])
					+ F * (ibest[x[0]][j] - ibest[x[1]][j]);
				if (newpop[i][j] > Ubound)
					newpop[i][j] = Ubound;
				else if (newpop[i][j] < Lbound)
					newpop[i][j] = Lbound;
			}
		}
	}
	else
	{	
		for (int i = p_start; i < p_end; i ++)
		{
			x[0] = rand() % Popsize;
			do 
			{
				x[1] = rand() % Popsize;
			} while (x[1] == x[0]);
			do 
			{
				x[2] = rand() % Popsize;
			} while (x[2] == x[1] || x[2] == x[0]);

			for (int j = 0; j < Nvar; j ++)
			{
				newpop[i][j] = ibest[x[0]][j] + F * (ibest[x[1]][j] - ibest[x[2]][j]);
				if (newpop[i][j] > Ubound)
					newpop[i][j] = Ubound;
				else if (newpop[i][j] < Lbound)
					newpop[i][j] = Lbound;
			}
		}
	}
}

void MultiMet::differential_crossover(double cr, int p_start, int p_end)
{
	double r;
	int d;
	for (int i = p_start; i < p_end; i ++)
	{
		d = rand() % Nvar;
		for (int j = 0; j < Nvar; j ++)
		{
			r = randval(0, 1);
			if (r > cr && j != d)
				newpop[i][j] = pop[i][j];
		}
	}
}

void MultiMet::DE(double F, int S, double cr, int p_start, int p_end)
{
	differential_mutate(F, S, p_start, p_end);
	differential_crossover(cr, p_start, p_end);
}

/**
 * GDE - Gbest-centric Differential Evolution
 * 
 * Implements Algorithm as specified with:
 * - Eq.29/30 mutation (gbest-centric)
 * - Rollback crossover (Algorithm-2)
 * - Trial vector mutation (Eq.32/33)
 * - Selection with ibest/gbest updates
 * 
 * @param Pmu    Probability of using Eq.29 vs Eq.30 (default 0.5)
 * @param n_centric Number of gbest-centric iterations (default 6)
 */
void MultiMet::GDE(double Pmu, int n_centric, int p_start, int p_end)
{
    for (int i = p_start; i < p_end; i++)
    {
        std::vector<double> trial(Nvar);
        std::vector<double> v(Nvar);
        for (int j = 0; j < Nvar; j++)
            trial[j] = pop[i][j];
        double parent_fit = pop_fit[i];

        int iterations = (i > Popsize / 3) ? 1 : n_centric;
        for (int iter = 0; iter < iterations; iter++)
        {
            double F = randval(0.2, 0.8);

            int r1, r2;
            do { r1 = rand() % Popsize; } while (r1 == i);
            do { r2 = rand() % Popsize; } while (r2 == i || r2 == r1);

            bool use_eq29 = (randval(0.0, 1.0) < Pmu);
            for (int j = 0; j < Nvar; j++)
            {
                if (use_eq29)
                    v[j] = gbest[j] + F * (pop[r1][j] - pop[r2][j]);
                else
                    v[j] = gbest[j] + F * (ibest[i][j] - pop[r2][j]);

                if (v[j] < Lbound || v[j] > Ubound)
                    v[j] = randval(Lbound, Ubound);
            }

            double Pcross = randval(0.01, 0.3);
            for (int j = 0; j < Nvar; j++)
            {
                if (randval(0.0, 1.0) > Pcross)
                    v[j] = gbest[j];
            }

            double v_fit = EVAL_COMPAT(v.data(), Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                Edge_Device_comm, ST, ET, CE_ST, CE_ET);

            if (v_fit < parent_fit)
            {
                for (int j = 0; j < Nvar; j++)
                    trial[j] = v[j];
            }
        }

        double CR = randval(0.1, 0.6);
        int Dr = rand() % Nvar;
        double Pm = randval(0.1, 0.6);
        for (int j = 0; j < Nvar; j++)
        {
            if (randval(0.0, 1.0) <= CR || j == Dr)
            {
                newpop[i][j] = trial[j];
                if (j == Dr && randval(0.0, 1.0) <= Pm)
                    newpop[i][j] = randval(Lbound, Ubound);
            }
            else
            {
                newpop[i][j] = pop[i][j];
            }
        }

        newpop_fit[i] = EVAL_COMPAT(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
            CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
            AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
            CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
            Edge_Device_comm, ST, ET, CE_ST, CE_ET);

        if (newpop_fit[i] >= parent_fit)
        {
            for (int j = 0; j < Nvar; j++)
                newpop[i][j] = pop[i][j];
            newpop_fit[i] = parent_fit;
        }

        if (newpop_fit[i] < ibest_fit[i])
        {
            for (int j = 0; j < Nvar; j++)
                ibest[i][j] = newpop[i][j];
            ibest_fit[i] = newpop_fit[i];

            if (ibest_fit[i] < gbest_fit)
            {
                for (int j = 0; j < Nvar; j++)
                    gbest[j] = ibest[i][j];
                gbest_fit = ibest_fit[i];
            }
        }
    }
}
void MultiMet::levy_cuckoo(int p_start, int p_end)
{
	double beta = 3 / 2;
	double a = randval(0, 1), b = randval(1e-5, 1);
	double sigma = pow(( a * sin(M_PI * beta / 2) / (b * beta * pow(2.0, (beta - 1) / 2.0)) ), (1.0 / beta));

	double u, v, step;
    double stepsize;
	for (int i = p_start; i < p_end; i ++)
	{
		for (int j = 0; j < Nvar; j ++)
		{
			u = randval(0, 1) * sigma;
			v = randval(0, 1);
			step = u / (pow(v, 1.0 / beta) + 1e-5);
            if (step > 1)
                step = 1;
            else if (step < -1)
                step = -1;
			if (randval(0, 1) < 0.3)
            {
                stepsize = 0.1 * step * (ibest[i][j] - pop[i][j]);
                newpop[i][j] = pop[i][j] + stepsize * randval(0, 1);
                if (newpop[i][j] < Lbound)
                    newpop[i][j] = Lbound;
                else if (newpop[i][j] > Ubound)
                    newpop[i][j] = Ubound;
            }
            else
                newpop[i][j] = pop[i][j];
        }
	}
}

void MultiMet::nest_discover(double pa, int p_start, int p_end)
{
	int r1, r2;
	for (int i = p_start; i < p_end; i ++)
	{
		r1 = rand() % Popsize;
		r2 = rand() % Popsize;
		
        if (r1 != r2)
        {
            for (int j = 0; j < Nvar; j ++)
            {
                double p = randval(0, 1);
                if (p < pa)
                {
                    double dis = newpop[r1][j] - newpop[r2][j];
                    newpop[i][j] += randval(0, 1) * dis;
                    if (newpop[i][j] < Lbound)
                        newpop[i][j] = Lbound;
                    else if (newpop[i][j] > Ubound)
                        newpop[i][j] = Ubound;
                }
            }
        }
	}
}

void MultiMet::CSA(double pa, int p_start, int p_end)
{
	levy_cuckoo(p_start, p_end);
	nest_discover(pa, p_start, p_end);
}

void MultiMet::EmployedBee(int pn, double *tmp, double **pp)
{
	int para2change, neighbor;
	for (int i = 0; i < Nvar; i ++)
		tmp[i] = pop[pn][i];
	para2change = rand() % Nvar;
	do 
	{
		neighbor = rand() % Popsize;
	} while (neighbor == pn);
	tmp[para2change] = tmp[para2change] + (pp[neighbor][para2change] - tmp[para2change])
		* randval(-1, 1);
	if (tmp[para2change] < Lbound)
		tmp[para2change] = Lbound;
	if (tmp[para2change] > Ubound)
		tmp[para2change] = Ubound;
}

void MultiMet::OnlookerBee(int p_start, int p_end)
{
	double maxf = 0;
	for (int i = p_start; i < p_end; i ++)
	{
		pr[i] = exp(newpop_fit[i] / 1000);
		if (pr[i] > 1e10)
			pr[i] = 1e10;
		if (pr[i] > maxf)
			maxf = pr[i];
	}
	for (int i = p_start; i < p_end; i ++)
		if (maxf != 0)
			pr[i] = 0.9 * pr[i] / maxf + 0.1;
		else
			pr[i] = 1;
}

void MultiMet::ScoutBee(int limit, int p_start, int p_end)
{
	int maxindex = p_start;
	for (int i = p_start + 1; i < p_end; i ++)
		if (trial[i] > trial[maxindex])
			maxindex = i; 
	if (trial[maxindex] > limit)
	{
		for (int i = 0; i < Nvar; i ++)
			newpop[maxindex][i] = randval(Lbound, Ubound);
        newpop_fit[maxindex] = EVAL_COMPAT(newpop[maxindex], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
	}
}

void MultiMet::ABCA(int limit, int p_start, int p_end)
{
	double *tmp = new double[Nvar];
	double tmp_fit = 0;
	double r;
	int i, j;

	for (i = 0; i < Popsize; i ++)
	{
		EmployedBee(i, tmp, pop);
        tmp_fit = EVAL_COMPAT(tmp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
		if (tmp_fit < pop_fit[i])
		{
			for (j = 0; j < Nvar; j ++)
				newpop[i][j] = tmp[j];
			newpop_fit[i] = tmp_fit;
		}
		else
		{
			for (j = 0; j < Nvar; j ++)
				newpop[i][j] = pop[i][j];
			newpop_fit[i] = pop_fit[i];
			trial[i] ++;
		}
	}
	OnlookerBee(0, Popsize);

	int T = 0;
    i = p_start;
    int totaliter = 0;
	while (T < (p_end - p_start) && totaliter < 2 * Popsize)
	{
		r = randval(0, 1);
		if (r < pr[i])
		{
			T ++;
			EmployedBee(i, tmp, newpop);
            tmp_fit = EVAL_COMPAT(tmp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
			if (tmp_fit < newpop_fit[i])
			{
				for (j = 0; j < Nvar; j ++)
					newpop[i][j] = tmp[j];
				newpop_fit[i] = tmp_fit;
			}
			else
			{
				trial[i] ++;
			}
		}
		i ++;
		if (i >= p_end)
			i = p_start;
        totaliter ++;
	}
	ScoutBee(limit, p_start, p_end);
}

void MultiMet::plant_growth(int p_start, int p_end)     //dis < Ubound - Lbound
{
	double r = 0;
	
	for(int i = p_start; i < p_end; i ++)
	{
		r = randval(0, 1);
		if (r < 0.4)
		{
			for (int j = 0; j < Nvar; j ++)
			{
				if (randval(0, 1) < 0.5)
				{
					newpop[i][j] += (gbest[j] - newpop[i][j]) * randval(0, 1);
					if (newpop[i][j] > Ubound)
						newpop[i][j] = Lbound;
					else if (newpop[i][j] < Lbound)
						newpop[i][j] = Ubound;
				}
			}
		}
		if (r > 0.4 && r < 0.6)
		{
			for (int j = 0; j < Nvar; j ++)
			{
				if (randval(0, 1) < 0.5)
				{
					newpop[i][j] = (pop[cur_best][j] - newpop[i][j]) * randval(0, 1);
					if (newpop[i][j] > Ubound)
						newpop[i][j] = Lbound;
					else if (newpop[i][j] < Lbound)
						newpop[i][j] = Ubound;
				}
			}
		}
		else if (r > 0.6 && r < 0.8)
		{
			for (int j = 0; j < Nvar; j ++)
			{
				if (randval(0, 1) < 0.5)
				{
					newpop[i][j] = (ibest[i][j] - newpop[i][j]) * randval(0, 1);
					if (newpop[i][j] > Ubound)
						newpop[i][j] = Lbound;
					else if (newpop[i][j] < Lbound)
						newpop[i][j] = Ubound;
				}
			}
		}
		else
		{
			for (int j = 0; j < Nvar; j ++)
			{
				if (randval(0, 1) < 0.2)
				{
					newpop[i][j] = randval(Lbound, Ubound);
					if (newpop[i][j] > Ubound)
						newpop[i][j] = Lbound;
					else if (newpop[i][j] < Lbound)
						newpop[i][j] = Ubound;
				}
			}
		}			
	}
}

void MultiMet::POA(int p_start, int p_end)
{
	plant_growth(p_start, p_end);
}

void MultiMet::localsearch(double *pp, int nst)  //same with bit climbing
{
    std::random_device rd;
    std::mt19937_64 g(rd());
	double *temp = new double[Nvar + 1];
	vector<int> permu;
	for (int i = 0; i < Nvar; i ++)
		permu.push_back(i);
	int j = 0;
	while (j < nst)
	{
		if (j % Nvar == 0)
			shuffle(permu.begin(), permu.end(), g);
		for (int k = 0; k < Nvar + 1; k ++)
			temp[k] = pp[k];
		int bit = permu[j % Nvar];
		temp[bit] = randval(Lbound, Ubound);
        temp[Nvar] = EVAL_COMPAT(temp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
		if (temp[Nvar] < pp[Nvar])
		{
			for (int k = 0; k < Nvar + 1; k ++)
				pp[k] = temp[k];
			//break;
		}
		j ++;
	}
	delete [] temp;
}

void MultiMet::ILS(int nst, int p_start, int p_end)
{
	int bit;
	double *tmp1 = new double[Nvar + 1];
	double *tmp2 = new double[Nvar + 1];
	for (int i = p_start; i < p_end; i ++)
	{
		//pertubation
		for (int j = 0; j < Nvar; j ++)
		{
			bit = rand() % Popsize;
			if (randval(0, 1) < 0.5)
			{
				tmp1[j] = pop[i][j] + randval(0, 1) * (pop[bit][j] - pop[i][j]);
				if (tmp1[j] > Ubound)
					tmp1[j] = Lbound;
				else if (tmp1[j] < Lbound)
					tmp1[j] = Ubound;
			}
			else
				tmp1[j] = pop[i][j];
		}
        tmp1[Nvar] = EVAL_COMPAT(tmp1, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);

		//localsearch
		for (int j = 0; j < Nvar + 1; j ++)
			tmp2[j] = tmp1[j];		
		localsearch(tmp2, nst);
		
		if (tmp2[Nvar] < pop_fit[i])
		{
			for (int j = 0; j < Nvar; j ++)
				newpop[i][j] = tmp2[j];
			newpop_fit[i] = tmp2[Nvar];
		}
	}
	delete [] tmp1;
	delete [] tmp2;
}

void MultiMet::VNS(int nst, int p_start, int p_end)
{
	int bit;
	double *tmp1 = new double[Nvar + 1];
	double *tmp2 = new double[Nvar + 1];
	for (int i = p_start; i < p_end; i ++)
	{
		//shaking
		bit = neigh[i];
		for (int j = 0; j < Nvar; j ++)
			tmp1[j] = pop[i][j];
		tmp1[bit] += randval(0, 1) * (pop[rand() % Popsize][bit] - pop[i][bit]);
        tmp1[Nvar] = EVAL_COMPAT(tmp1, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);

		//localsearch
		for (int j = 0; j < Nvar + 1; j ++)
			tmp2[j] = tmp1[j];		
		localsearch(tmp2, nst);

		if (tmp2[Nvar] < pop_fit[i])
		{
			for (int j = 0; j < Nvar; j ++)
				newpop[i][j] = tmp2[j];
			newpop_fit[i] = tmp2[Nvar];
		}
		else
		{
			neigh[i] ++;
			if (neigh[i] > Nvar - 1)
				neigh[i] = 0;
		}
	}
	delete [] tmp1;
	delete [] tmp2;
}

void MultiMet::GRASP(double alfa, int nst, int p_start, int p_end)
{
	int i, j, k, bit;
	double threshold = pop_fit[cur_best] + alfa * (pop_fit[cur_worst] - pop_fit[cur_best]);
	double **RCL = new double *[Popsize];
	for (i = 0; i < Popsize; i ++)
		RCL[i] = new double[Nvar + 1];
	double *tmp = new double[Nvar + 1];
	pop_heap_sort(Popsize);
	k = 0;
	for (i = 0; i < Popsize; i ++)
	{
		if (pop_fit[i] <= threshold)
		{
			for (j = 0; j < Nvar; j ++)
				RCL[k][j] = pop[i][j];
			RCL[k][Nvar] = pop_fit[i];
			k ++;
		}
	}
	if (k == 0)
		k = 1;
	for (i = p_start; i < p_end; i ++)
	{
		bit = rand() % k;
		for (j = 0; j < Nvar + 1; j ++)
			tmp[j] = RCL[bit][j];

		//localsearch
		localsearch(tmp, nst);

		if (tmp[Nvar] < pop_fit[i])
		{
			for (j = 0; j < Nvar; j ++)
				newpop[i][j] = tmp[j];
			newpop_fit[i] = tmp[Nvar];
		}
	}
	for (i = 0; i < Popsize; i ++)
		delete [] RCL[i];
	delete [] RCL;
	delete [] tmp;
}

void MultiMet::PBILC(int p_start, int p_end, double learn_rate)
{
    double *pre_center = new double[Nvar];
    double *pre_sigma = new double[Nvar];
    
    pop_heap_sort(Popsize);
    
    for (int i = 0; i < Nvar; i ++)
    {
        pre_center[i] = PB_center[i];
        pre_sigma[i] = PB_sigma[i];
    }
    
    //update center
    for (int i = 0; i < Nvar; i ++)
        PB_center[i] = pop[0][i] + pop[1][i] - pop[Popsize - 1][i];
    
    for (int i = 0; i < Nvar; i ++)
        PB_center[i] = (1 - learn_rate) * pre_center[i] + learn_rate * PB_center[i];
    
    //update sigma
    for (int i = 0; i < Nvar; i ++)
    {
        double average = 0.0, sum = 0.0;
        for (int j = 0; j < num_of_elite; j ++)
            average += pop[j][i];
        average /= num_of_elite;
        for (int j = 0; j < num_of_elite; j ++)
            sum += pow(pop[j][i] - average, 2);
        sum /= num_of_elite;
        
        PB_sigma[i] = sqrt(sum);
        PB_sigma[i] = (1 - learn_rate) * pre_sigma[i] + learn_rate * PB_sigma[i];
    }
    
    //generate newpop
    for (int i = p_start; i < p_end; i ++)
    {
        for (int j = 0; j < Nvar; j ++)
        {
            newpop[i][j] = PB_center[j] + PB_sigma[i] * gauss();
            if (newpop[i][j] > Ubound)
                newpop[i][j] = Ubound;
            else if (newpop[i][j] < Lbound)
                newpop[i][j] = Lbound;
        }
    }
    
    delete [] pre_center;
    delete [] pre_sigma;
}

void MultiMet::BATA(int gen, int p_start, int p_end)
{
    pop_heap_sort(Popsize);
    for (int i = p_start; i < p_end; i ++)
    {
        for (int j = 0; j < Nvar; j ++)
        {
            BAT_v[i][j] += (pop[i][j] - pop[cur_best][j]) * randval(0.0, 100.);
            newpop[i][j] = pop[i][j] + BAT_v[i][j];
            if (newpop[i][j] > Ubound)
                newpop[i][j] = Ubound;
            else if (newpop[i][j] < Lbound)
                newpop[i][j] = Lbound;
        }
        
        if (randval(0.0, 1.0) > BAT_r[i])
            for (int j = 0; j < Nvar; j ++)
            {
                newpop[i][j] = pop[rand() % (Popsize / 2)][j] + randval(-1.0, 1.0) * BAT_A[i];
                if (newpop[i][j] > Ubound)
                    newpop[i][j] = Ubound;
                else if (newpop[i][j] < Lbound)
                    newpop[i][j] = Lbound;
            }
        
        newpop_fit[i] = EVAL_COMPAT(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);

        if (randval(0.0, 1.0) < BAT_A[i] && newpop_fit[i] < pop_fit[cur_best])
        {
            BAT_A[i] *= 0.95;
            BAT_r[i] *= (1 - exp(-0.95 * gen));
        }
        /*else
            for (int j = 0; j < Nvar; j ++)
                newpop[i][j] = pop[i][j];*/
    }
}

void MultiMet::FA(double gama, double alpha0, double betamin, int MGEN, int p_start, int p_end)
{
    if(alpha == alpha0)
        alpha = pow((pow(10.0, -4.0)/0.9), 1.0/MGEN) * alpha0;
    else
        alpha = pow((pow(10.0, -4.0)/0.9), 1.0/MGEN) * alpha;
    
    if (alpha < 1e-5)
        alpha = 0.5;

	for(int i=0;i<Popsize -1 ;i++)
	{
		for(int j=i+1;j<Popsize;j++)
		{
			if(I[i] > I[j])
			{
				double z = I[i];	// exchange attractiveness
				I[i] = I[j];
				I[j] = z;
				z = pop_fit[i];			// exchange fitness
				pop_fit[i] = pop_fit[j];
				pop_fit[j] = z;
				int k = Index[i];	// exchange indexes
				Index[i] = Index[j];
				Index[j] = k;
			}
		}
	}
	for(int i=0;i<Popsize;i++)
	{
		for(int j=0;j<Nvar;j++)
		{
			newpop[i][j] = pop[Index[i]][j];
		}
	}

	double scale = abs(Ubound - Lbound);
	for(int i=p_start;i<p_end;i++)
	{
		for(int j=0;j<Popsize;j++)
		{
            if (i != j)
            {
                double r = 0.0;
                for(int k=0;k<Nvar;k++)
                {
                    r += (newpop[i][k]-newpop[j][k])*(newpop[i][k]-newpop[j][k]);
                }
                r = sqrt(r);
                if(I[i] > I[j])	// brighter and more attractive
                {
                    double beta0 = 1.0;
                    double beta = (beta0-betamin)*exp(-gama*pow(r, 2.0))+betamin;
                    for(int k=0;k<Nvar;k++)
                    {
                        double r1 = randval(0.0, 2.0);
                        double tmpf = alpha*(r1-1.0) * scale;
                        newpop[i][k] = newpop[i][k]*(1.0-beta) + pop[j][k]*beta + tmpf;
                        if(newpop[i][k] < Lbound)
                            newpop[i][k] = Lbound;
                        if(newpop[i][k] > Ubound)
                            newpop[i][k] = Ubound;
                    }
                }
            }
		}
	}
}

void MultiMet::pop_heap_adjust(int s, int len)
{
	double* temp = pop[s];
	double temp_fit = pop_fit[s];
	for (int i = 2 * s + 1; i < len; i = 2 * i + 1)
	{
		if (i < (len - 1) && pop_fit[i] < pop_fit[i + 1])
			i ++;
		if (temp_fit > pop_fit[i])
			break;
		pop[s] = pop[i];
		pop_fit[s] = pop_fit[i];
		s = i;
	}
	pop[s] = temp;
	pop_fit[s] = temp_fit;
}

void MultiMet::pop_heap_sort(int len)
{
	int i;
	double *temp;
	double temp_fit;
    double p_ngh;
	for (i = len / 2 - 1; i >= 0; i --)
		pop_heap_adjust(i, len);
	for (i = len - 1; i > 0; i --)
	{
		temp = pop[0];
		temp_fit = pop_fit[0];
        p_ngh = ngh[0];
		
        pop[0] = pop[i];
		pop_fit[0] = pop_fit[i];
        ngh[0] = ngh[i];
        
        ngh[i] = p_ngh;
		pop[i] = temp;
		pop_fit[i] = temp_fit;
		pop_heap_adjust(0, i);
	}
}

void MultiMet::newpop_heap_adjust(int s, int len)
{
	double* temp = newpop[s];
	double temp_fit = newpop_fit[s];
	for (int i = 2 * s + 1; i < len; i = 2 * i + 1)
	{
		if (i < (len - 1) && newpop_fit[i] < newpop_fit[i + 1])
			i ++;
		if (temp_fit > newpop_fit[i])
			break;
		newpop[s] = newpop[i];
		newpop_fit[s] = newpop_fit[i];
		s = i;
	}
	newpop[s] = temp;
	newpop_fit[s] = temp_fit;
}

void MultiMet::newpop_heap_sort(int len)
{
	int i;
	double *temp;
	double temp_fit;
	for (i = len / 2 - 1; i >= 0; i --)
		newpop_heap_adjust(i, len);
	for (i = len - 1; i > 0; i --)
	{
		temp = newpop[0];
		temp_fit = newpop_fit[0];
		newpop[0] = newpop[i];
		newpop_fit[0] = newpop_fit[i];
		newpop[i] = temp;
		newpop_fit[i] = temp_fit;
		newpop_heap_adjust(0, i);
	}
}

void MultiMet::pop_niche(double mdis, int p_start, int p_end)
{
	double dis;
	for (int i = p_start; i < p_end; i++)
	{
		for (int j = p_start + 1; j < p_end; j++)
		{
			dis = 0;
			for (int k = 0; k < Nvar; k++)
				dis += pow((double)(pop[i][k] - pop[j][k]), 2);
			dis = sqrt(dis);
			if (dis < mdis)
			{
				if (pop_fit[i] > pop_fit[j])                    //for min prob
					pop_fit[i] = abs(pop_fit[i] * 100);
				else
					pop_fit[j] = abs(pop_fit[i] * 100);
			}
		}
	}
}

void MultiMet::newpop_localstep(double step, int nst, int p_start, int p_end)
{
	double *temp = new double[Nvar];
	double temp_fit = 0;
	double Fave = 0;
	for (int i = p_start; i < p_end; i ++)
		Fave += newpop_fit[i];
	Fave /= Popsize;
	for (int i = p_start; i < p_end; i ++)
	{
		if (randval(0, 1) < 1 / (1 + exp(newpop_fit[i] - Fave)))             //for min prob
		{
			for (int j = 0; j < Nvar; j ++)
				temp[j] = newpop[i][j];
			int bit = rand() % Nvar;
			for (int j = 0; j < nst; j ++)
			{
				temp[bit] -= j * step;
				if (temp[bit] < Lbound)
					temp[bit] = randval(Lbound, Ubound);
                temp_fit = EVAL_COMPAT(temp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
				if (temp_fit < newpop_fit[i])
				{
					newpop[i][bit] = temp[bit];
					newpop_fit[i] = temp_fit;
					break;
				}
			}
		}
	}
	delete [] temp;
}

double MultiMet::pop_random_variance(int p_start, int p_end)
{
	int bit = rand() % Nvar;
	double Bave = 0;
	for (int i = p_start; i < p_end; i ++)
		Bave += pop[i][bit];
	Bave /= Popsize;
	double variance = 0;
	for (int i = p_start; i < p_end; i ++)
		variance += pow((pop[i][bit] - Bave), 2.0);
	variance = sqrt(variance) / Popsize;
	return variance;
}

double MultiMet::pop_random_entropy(int subn, int p_start, int p_end)
{
	double entr = 0;
	int reg = 0;
	int bit = rand() % Nvar;
	double *pp = new double[subn];
	for (int i = 0; i < subn; i ++)
		pp[i] = 0;
	for (int i = p_start; i < p_end; i ++)
	{
		reg = (int)(subn * (pop[i][bit] - Lbound) / (Ubound - Lbound));
		if (reg >= subn)
			reg = subn - 1;
		pp[reg] ++;
	}
	for (int i = 0; i < subn; i ++)
		pp[i] /= subn;
	for (int i = 0; i < subn; i ++)
	{
		if (pp[i] != 0)
			entr -= pp[i] * log(pp[i]);
	}
	delete [] pp;
	return entr;
}

void MultiMet::CMAES(int firstrun, int p_start, int p_end)
{	
	if(!CMAisdone)
	{
		CMAES_parametersetting();
		CMAES_initial();
		CMAisdone = true;
	}
	CMAES_updateDistribution(firstrun);
	CMAES_sampleGenerate(firstrun, p_start, p_end);


}

void MultiMet::CMAES_parametersetting(){
	
	for(int i=0; i<Nvar; i++) xstart[i] = 0.5 * (Lbound + Ubound);
	for(int i=0; i<Nvar; i++) stddev[i] = 0.3 * (Ubound - Lbound);
	lambda = 4 + (int)floor(3 * log(Nvar));
    mu = lambda / 2;
	//setweight
	for(int i = 0; i < mu; ++i) 
		weights[i] = log(mu + 1.) - log(i + 1.);
	double s1=0.,s2=0.;
	for(int i = 0; i < mu; ++i)
    {
      s1 += weights[i];
      s2 += weights[i]*weights[i];
    }
	mueff = s1*s1/s2;
    for(int i = 0; i < mu; ++i)
      weights[i] /= s1;
    cs = (mueff + 2.) / (Nvar + mueff + 3.);
    ccumcov = (4. + mueff / Nvar) / (Nvar + 4 + 2 * mueff / Nvar);
	mucov = mueff;

	double t1 = 2. / ((Nvar + 1.4142)*(Nvar + 1.4142));
    double t2 = (2.* mueff - 1.) / ((Nvar + 2.)*(Nvar + 2.) + mueff);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1. / mucov)* t1 + (1. - 1. / mucov)* t2;
    ccov = t2;
	damps = 1. + 2*(std::max(0.,(std::sqrt((mueff - 1.) / (Nvar + 1.)) - 1.)))+ cs;
    //diagonalCov = 2 + 100. * Nvar / sqrt((double) lambda);
}

void MultiMet::CMAES_initial(){

	double trace=0.;
	for(int i = 0; i < Nvar; ++i)
      trace += stddev[i]*stddev[i];
    sigma = std::sqrt(trace/Nvar);
	//sigma = 0.5;
	chiN = std::sqrt((Nvar) * (1. - 1./(4*Nvar) + 1./(21.*Nvar*Nvar)));

	pcc = new double[Nvar];
    ps = new double[Nvar];
    tempRandom = new double[Nvar+1];
    BDz = new double[Nvar];
    xmean = new double[Nvar];
   /* xmean[0] = Nvar;
    ++xmean;*/
    xold = new double[Nvar];
    /*xold[0] = Nvar;
    ++xold;*/
    rgD = new double[Nvar];
    C = CreateMatrix(Nvar, Nvar);
    B = CreateMatrix(Nvar, Nvar);
	index = new int[lambda];
    for(int i = 0; i < lambda; ++i)
        index[i] = i;
    
	// initialize newed space
    for(int i = 0; i < Nvar; ++i)
      for(int j = 0; j < i; ++j)
        C[i][j] = B[i][j] = B[j][i] = 0.;

    for(int i = 0; i < Nvar; ++i)
    {
      B[i][i] = 1;
      C[i][i] = rgD[i] = stddev[i]*std::sqrt(Nvar/trace);  //can be "C[i][i] = rgD[i] = 1" either
      C[i][i] *= C[i][i];
      pcc[i] = ps[i] = 0;
    }

    for(int i = 0; i < Nvar; ++i)
      xmean[i] = xold[i] = xstart[i];
    
}

void MultiMet::CMAES_sampleGenerate(int firstrun, int p_start, int p_end)
{

	if(firstrun == 0)
	{
		for(int i = 0; i < Nvar; ++i)
			rgD[i] = std::sqrt(C[i][i]);
	}
	else
		updateEigensystem();

	for(int iPop = p_start; iPop < p_end; ++iPop)
	{
		for(int j = 0; j < Nvar; ++j)
			if(firstrun == 0)
				newpop[iPop][j] = xmean[j] + sigma*rgD[j]*gauss();
			else
				tempRandom[j] = rgD[j]*gauss();
		if(firstrun > 0)
			for(int i = 0; i < Nvar; ++i) // add mutation sigma*B*(D*z)
			{
				double sum = 0.0;
				for(int j = 0; j < Nvar; ++j)
					sum += B[i][j]*tempRandom[j];
				newpop[iPop][i] = xmean[i] + sigma*sum;
				if (newpop[iPop][i] > Ubound)
					newpop[iPop][i] = Ubound;
				else if (newpop[iPop][i] < Lbound)
					newpop[iPop][i] = Lbound;
			}
	}
       
}

double* MultiMet::CMAES_updateDistribution(int gen)
{
	if(gen > 0)
	{
	// Generate index
    sortIndex(pop_fit, index, lambda);
	// Test if function values are identical, escape flat fitness
	if(pop_fit[index[0]] == pop_fit[index[(int) lambda / 2]])
		sigma *= std::exp(0.2 + cs / damps);

	 // calculate xmean and rgBDz~N(0,C)
    const double sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    for(int i = 0; i < Nvar; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for(int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*pop[index[iNk]][i];
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

	// calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for(int i = 0; i < Nvar; ++i)
    {
      double sum;
      sum = 0.;
      for(int j = 0; j < Nvar; ++j)
		sum += B[j][i]*BDz[j];
      tempRandom[i] = sum/rgD[i];
    }

	// cumulation for sigma (ps) using B*z
    const double sqrtFactor = std::sqrt(cs*(2.-cs));
    const double invps = 1.-cs;
    for(int i = 0; i < Nvar; ++i)
    {
      double sum;
      sum = 0.;
      double* Bi = B[i];
      for(int j = 0; j < Nvar; ++j)
		sum += Bi[j]*tempRandom[j];
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

	// calculate norm(ps)^2
    double psxps(0);
    for(int i = 0; i < Nvar; ++i)
    {
      const double& rgpsi = ps[i];
      psxps += rgpsi*rgpsi;
    }

	// cumulation for covariance matrix (pcc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(1. - std::pow(1. - cs, 2.* gen))
        / chiN < 1.4 + 2. / (Nvar + 1);
    const double ccumcovinv = 1.-ccumcov;
    const double hsigFactor = hsig*std::sqrt(ccumcov*(2.-ccumcov));
    for(int i = 0; i < Nvar; ++i)
      pcc[i] = ccumcovinv*pcc[i] + hsigFactor*BDz[i];

	// update of C
    adaptC2(hsig,gen);

    // update of sigma
    sigma *= std::exp(((std::sqrt(psxps) / chiN) - 1.)* cs / damps);
	}
	return xmean;

}

/**
   * Dirty index sort.
   */
void MultiMet::sortIndex(const double* rgFunVal, int* iindex, int n)
  {
    int i, j;
    for(i = 1, iindex[0] = 0; i < n; ++i)
    {
      for(j = i; j > 0; --j)
      {
        if(rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1]; // shift up
      }
      iindex[j] = i;
    }
  }
void MultiMet::adaptC2(const int hsig, int gen)
  {
    const int N = Nvar;
    if(ccov != 0.)
    {
      // definitions for speeding up inner-most loop
      const double mucovinv = double(1)/mucov;
      const double commonFactor = ccov *((gen==0) ? (N + double(1.5)) / double(3) : double(1));
      const double ccov1 = std::min(commonFactor*mucovinv, double(1));
      const double ccovmu = std::min(commonFactor*(double(1)-mucovinv), double(1)-ccov1);
      const double sigmasquare = sigma*sigma;
      const double onemccov1ccovmu = double(1)-ccov1-ccovmu;
      const double longFactor = (double(1)-hsig)*ccumcov*(double(2)-ccumcov);


      // update covariance matrix
      for(int i = 0; i < N; ++i)
        for(int j = (gen==0) ? i : 0; j <= i; ++j)
        {
          double& Cij = C[i][j];
          Cij = onemccov1ccovmu*Cij + ccov1 * (pcc[i]*pcc[j] + longFactor*Cij);
          for(int k = 0; k < mu; ++k)
          { // additional rank mu update
            const double* rgrgxindexk = pop[index[k]];
            Cij += ccovmu*weights[k] * (rgrgxindexk[i] - xold[i])
                * (rgrgxindexk[j] - xold[j]) / sigmasquare;
          }
        }
      //// update maximal and minimal diagonal value
      //maxdiagC = mindiagC = C[0][0];
      //for(int i = 1; i < N; ++i)
      //{
      //  const double& Cii = C[i][i];
      //  if(maxdiagC < Cii)
      //    maxdiagC = Cii;
      //  else if(mindiagC > Cii)
      //    mindiagC = Cii;
      //}
    }
  }

void MultiMet::updateEigensystem()
{
    eigen(rgD, B, tempRandom);


    for(int i = 0; i < Nvar; ++i)
      rgD[i] = std::sqrt(rgD[i]);
}

/**
   * Calculating eigenvalues and vectors.
   * @param rgtmp (input) N+1-dimensional vector for temporal use. 
   * @param diag (output) N eigenvalues. 
   * @param Q (output) Columns are normalized eigenvectors.
   */
void MultiMet::eigen(double* diag, double** Q, double* rgtmp)
{
    if(C != Q) // copy C to Q
    {
      for(int i = 0; i < Nvar; ++i)
        for(int j = 0; j <= i; ++j)
          Q[i][j] = Q[j][i] = C[i][j];
    }

    householder(Q, diag, rgtmp);
    ql(diag, rgtmp, Q);
}

/**
   * Symmetric tridiagonal QL algorithm, iterative.
   * Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations
   * code adapted from Java JAMA package, function tql2.
   * @param d input: Diagonale of tridiagonal matrix. output: eigenvalues.
   * @param e input: [1..n-1], off-diagonal, output from Householder
   * @param V input: matrix output of Householder. output: basis of
   *          eigenvectors, according to d
   */
void MultiMet::ql(double* d, double* e, double** V)
  {
    const int n = Nvar;
    double f(0);
    double tst1(0);
    const double eps(2.22e-16); // 2.0^-52.0 = 2.22e-16

    // shift input e
    double* ep1 = e;
    for(double *ep2 = e+1, *const end = e+n; ep2 != end; ep1++, ep2++)
      *ep1 = *ep2;
    *ep1 = double(0); // never changed again

    for(int l = 0; l < n; l++)
    {
      // find small subdiagonal element
      double& el = e[l];
      double& dl = d[l];
      const double smallSDElement = std::fabs(dl) + std::fabs(el);
      if(tst1 < smallSDElement)
        tst1 = smallSDElement;
      const double epsTst1 = eps*tst1;
      int m = l;
      while(m < n)
      {
        if(std::fabs(e[m]) <= epsTst1) break;
        m++;
      }

      // if m == l, d[l] is an eigenvalue, otherwise, iterate.
      if(m > l)
      {
        do {
          double h, g = dl;
          double& dl1r = d[l+1];
          double p = (dl1r - g) / (2.*el);
          double r = myhypot(p, 1.);

          // compute implicit shift
          if(p < 0) r = -r;
          const double pr = p+r;
          dl = el/pr;
          h = g - dl;
          const double dl1 = el*pr;
          dl1r = dl1;
          for(int i = l+2; i < n; i++) d[i] -= h;
          f += h;

          // implicit QL transformation.
          p = d[m];
          double c(1);
          double c2(1);
          double c3(1);
          const double el1 = e[l+1];
          double s(0);
          double s2(0);
          for(int i = m-1; i >= l; i--)
          {
            c3 = c2;
            c2 = c;
            s2 = s;
            const double& ei = e[i];
            g = c*ei;
            h = c*p;
            r = myhypot(p, ei);
            e[i+1] = s*r;
            s = ei/r;
            c = p/r;
            const double& di = d[i];
            p = c*di - s*g;
            d[i+1] = h + s*(c*g + s*di);

            // accumulate transformation.
            for(int k = 0; k < n; k++)
            {
              double& Vki1 = V[k][i+1];
              h = Vki1;
              double& Vki = V[k][i];
              Vki1 = s*Vki + c*h;
              Vki *= c; Vki -= s*h;
            }
          }
          p = -s*s2*c3*el1*el/dl1;
          el = s*p;
          dl = c*p;
        } while(std::fabs(el) > epsTst1);
      }
      dl += f;
      el = 0.0;
    }
  }
  /**
   * Householder transformation of a symmetric matrix V into tridiagonal form.
   * Code slightly adapted from the Java JAMA package, function private tred2().
   * @param V input: symmetric nxn-matrix. output: orthogonal transformation
   *          matrix: tridiag matrix == V* V_in* V^double.
   * @param d output: diagonal
   * @param e output: [0..n-1], off diagonal (elements 1..n-1)
   */
void MultiMet::householder(double** V, double* d, double* e)
  {
    const int n = Nvar;

    for(int j = 0; j < n; j++)
    {
      d[j] = V[n - 1][j];
    }

    // Householder reduction to tridiagonal form

    for(int i = n - 1; i > 0; i--)
    {
      // scale to avoid under/overflow
      double scale = 0.0;
      double h = 0.0;
      for(double *pd = d, *const dend = d+i; pd != dend; pd++)
      {
        scale += std::fabs(*pd);
      }
      if(scale == 0.0)
      {
        e[i] = d[i-1];
        for(int j = 0; j < i; j++)
        {
          d[j] = V[i-1][j];
          V[i][j] = 0.0;
          V[j][i] = 0.0;
        }
      }
      else
      {
        // generate Householder vector
        for(double *pd = d, *const dend = d+i; pd != dend; pd++)
        {
          *pd /= scale;
          h += *pd * *pd;
        }
        double& dim1 = d[i-1];
        double f = dim1;
        double g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
        e[i] = scale*g;
        h = h - f* g;
        dim1 = f - g;
        memset((void *) e, 0, (size_t)i*sizeof(double));

        // apply similarity transformation to remaining columns
        for(int j = 0; j < i; j++)
        {
          f = d[j];
          V[j][i] = f;
          double& ej = e[j];
          g = ej + V[j][j]* f;
          for(int k = j + 1; k <= i - 1; k++)
          {
            double& Vkj = V[k][j];
            g += Vkj*d[k];
            e[k] += Vkj*f;
          }
          ej = g;
        }
        f = 0.0;
        for(int j = 0; j < i; j++)
        {
          double& ej = e[j];
          ej /= h;
          f += ej* d[j];
        }
        double hh = f / (h + h);
        for(int j = 0; j < i; j++)
        {
          e[j] -= hh*d[j];
        }
        for(int j = 0; j < i; j++)
        {
          double& dj = d[j];
          f = dj;
          g = e[j];
          for(int k = j; k <= i - 1; k++)
          {
            V[k][j] -= f*e[k] + g*d[k];
          }
          dj = V[i-1][j];
          V[i][j] = 0.0;
        }
      }
      d[i] = h;
    }

    // accumulate transformations
    const int nm1 = n-1;
    for(int i = 0; i < nm1; i++)
    {
      double h;
      double& Vii = V[i][i];
      V[n-1][i] = Vii;
      Vii = 1.0;
      h = d[i+1];
      if(h != 0.0)
      {
        for(int k = 0; k <= i; k++)
        {
          d[k] = V[k][i+1] / h;
        }
        for(int j = 0; j <= i; j++) {
          double g = 0.0;
          for(int k = 0; k <= i; k++)
          {
            double* Vk = V[k];
            g += Vk[i+1]* Vk[j];
          }
          for(int k = 0; k <= i; k++)
          {
            V[k][j] -= g*d[k];
          }
        }
      }
      for(int k = 0; k <= i; k++)
      {
        V[k][i+1] = 0.0;
      }
    }
    for(int j = 0; j < n; j++)
    {
      double& Vnm1j = V[n-1][j];
      d[j] = Vnm1j;
      Vnm1j = 0.0;
    }
    V[n-1][n-1] = 1.0;
    e[0] = 0.0;
  }

void MultiMet::BA(int p_start, int p_end)
{
    pop_heap_sort(Popsize);
    newpop_heap_sort(Popsize);
    
    for (int i = 0; i < ne; i ++)
        NeighborFlowerPatch(nre, i);
    if (randval(0, 1) < 0.5)
        DE(randval(0.1, 0.9), rand() % 5, randval(0.1, 0.9), ne, Popsize);
    else
        mutate(1, ne, Popsize);
}

void MultiMet::NeighborFlowerPatch(int nr, int point)
{
    meme_selection(point, 0, ngh[point], nr);
    
    if (pop_fit[point] < newpop_fit[point])
    {
        ngh[point] *= ngh_decay;
        ngh_decay_count[point] ++;
        if (ngh_decay_count[point] > stlim)
        {
            for (int j = 0; j < Nvar; j ++)
                newpop[point][j] = randval(Lbound, Ubound);
            newpop_fit[point] = EVAL_COMPAT(newpop[point], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            ngh[point] = ngh_origin;
            ngh_decay_count[point] = 0;
        }
    }
}

void MultiMet::newpop_bit_climbing(int popi, int L, double scale)
{
    std::random_device rd;
    std::mt19937_64 g(rd());
    double *temp = new double[Nvar];
    double temp_fit = 0;
    vector<int> permu;
    for (int i = 0; i < Nvar; i ++)
        permu.push_back(i);
    
    shuffle(permu.begin(), permu.end(), g);
    for (int j = 0; j < L; j ++)
    {
        for (int k = 0; k < Nvar; k ++)
            temp[k] = newpop[popi][k];
        int bit = permu[j % Nvar];
        temp[bit] += scale * randval(Lbound, Ubound);
        temp_fit = EVAL_COMPAT(temp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
        if (temp_fit < newpop_fit[popi])
        {
            newpop[popi][bit] = temp[bit];
            newpop_fit[popi] = temp_fit;
            //break;
        }
    }
    delete [] temp;
}

void MultiMet::newpop_simplex(int popi, int L, int stepn, double scale)
{
    double ***xx = new double **[L + 2];
    for (int i = 0; i < L + 2; i ++)
    {
        xx[i] = new double *[Nvar + 2];
        for (int j = 0; j < Nvar + 2; j ++)
            xx[i][j] = new double[Nvar + 1];
    }
    double *s = new double[Nvar];
    double *x1 = new double[Nvar + 1];
    double *x2 = new double[Nvar + 1];
    double *xav = new double[Nvar + 1];
    int w = 0, b = 0;
    double c = 1;
    
    int j, k, l, miu;
    double tmpW = 0, tmpB = 1000;
    
    for (l = 0; l < Nvar; l ++)
        s[l] = (Ubound - Lbound) * scale;
    
    for (l = 0; l < Nvar; l ++)
        xx[0][0][l] = newpop[popi][l];
    xx[0][0][Nvar] = newpop_fit[popi];
    
    k = 0;
    while (k < L)
    {
        //step1
        for (j = 1; j < Nvar + 1; j ++)
        {
            for (l = 0; l < Nvar; l ++)
            {
                if (l != j - 1)
                    xx[k][j][l] = xx[k][0][l];
                else
                    xx[k][j][l] = xx[k][0][l] + c * s[j - 1];
                if (xx[k][j][l] > Ubound)
                    xx[k][j][l] = Ubound;
                else if (xx[k][j][l] < Lbound)
                    xx[k][j][l] = Lbound;
            }
            xx[k][j][Nvar] = EVAL_COMPAT(xx[k][j], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
        }
        
        while (k < L - 1)
        {
            //step2
            tmpW = xx[k][0][Nvar]; tmpB = xx[k][0][Nvar]; w = 0; b = 0;
            for (j = 1; j < Nvar + 1; j ++)
            {
                if (xx[k][j][Nvar] < tmpB)
                {
                    tmpB = xx[k][j][Nvar];
                    b = j;
                }
                else if (xx[k][j][Nvar] > tmpW)
                {
                    tmpW = xx[k][j][Nvar];
                    w = j;
                }
            }
            
            for (l = 0; l < Nvar; l ++)
            {
                xav[l] = 0;
                for (j = 0; j < Nvar + 1; j ++)
                    if (j != w)
                        xav[l] += xx[k][j][l];
                xav[l] /= Nvar;
                x1[l] = 2 * xav[l] - xx[k][w][l];
                if (x1[l] > Ubound)
                    x1[l] = Ubound;
                else if (x1[l] < Lbound)
                    x1[l] = Lbound;
            }
            x1[Nvar] = EVAL_COMPAT(x1, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            
            if (x1[Nvar] < xx[k][b][Nvar])
            {
                //step4
                for (l = 0; l < Nvar; l ++)
                {
                    x2[l] = 2 * x1[l] - xav[l];
                    if (x2[l] > Ubound)
                        x2[l] = Ubound;
                    else if (x2[l] < Lbound)
                        x2[l] = Lbound;
                }
                x2[Nvar] = EVAL_COMPAT(x2, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                if (x2[Nvar] < xx[k][b][Nvar])
                    for (l = 0; l < Nvar + 1; l ++)
                        xx[k + 1][w][l] = x2[l];
                else
                    for (l = 0; l < Nvar + 1; l ++)
                        xx[k + 1][w][l] = x1[l];
            }
            else
            {
                //step3
                miu = 0;
                for (j = 0; j < Nvar + 1; j ++)
                    if (x1[Nvar] <= xx[k][j][Nvar])
                        miu ++;
                if (miu > 1)
                {
                    for (l = 0; l < Nvar + 1; l ++)
                        xx[k + 1][w][l] = x1[l];
                }
                else if (miu == 1)
                {
                    //step5
                    for (l = 0; l < Nvar; l ++)
                    {
                        x2[l] = 0.5 * (xav[l] + x1[l]);
                        if (x2[l] > Ubound)
                            x2[l] = Ubound;
                        else if (x2[l] < Lbound)
                            x2[l] = Lbound;
                    }
                    x2[Nvar] = EVAL_COMPAT(x2, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                    if (x2[Nvar] <= x1[Nvar])
                        for (l = 0; l < Nvar + 1; l ++)
                            xx[k + 1][w][l] = x2[l];
                    else
                    {
                        //step7
                        for (j = 0; j < Nvar + 1; j ++)
                        {
                            for (l = 0; l < Nvar; l ++)
                            {
                                xx[k + 1][j][l] = 0.5 * (xx[k][b][l] + xx[k][j][l]);
                                if (xx[k + 1][j][l] > Ubound)
                                    xx[k + 1][j][l] = Ubound;
                                else if (xx[k + 1][j][l] < Lbound)
                                    xx[k + 1][j][l] = Lbound;
                            }
                            xx[k + 1][j][Nvar] = EVAL_COMPAT(xx[k + 1][j], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                        }
                    }
                }
                else
                {
                    //step6
                    for (l = 0; l < Nvar; l ++)
                    {
                        x2[l] = 0.5 * (xav[l] + xx[k][w][l]);
                        if (x2[l] > Ubound)
                            x2[l] = Ubound;
                        else if (x2[l] < Lbound)
                            x2[l] = Lbound;
                    }
                    x2[Nvar] = EVAL_COMPAT(x2, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                    if (x2[Nvar] <= x1[Nvar])
                        for (l = 0; l < Nvar + 1; l ++)
                            xx[k + 1][w][l] = x2[l];
                    else
                        for (l = 0; l < Nvar + 1; l ++)
                            xx[k + 1][w][l] = x1[l];
                }
            }
            
            //step8
            for (j = 0; j < Nvar + 1; j ++)
                if (j != w)
                {
                    for (l = 0; l < Nvar + 1; l ++)
                        xx[k + 1][j][l] = xx[k][j][l];
                }
            
            //step9
            k ++;
            double tmp1 = 0, tmp2 = 0;
            for (j = 0; j < Nvar + 1; j ++)
            {
                tmp1 += xx[k][j][Nvar];
                tmp2 += xx[k][j][Nvar] * xx[k][j][Nvar];
            }
            if ((tmp2 - tmp1 * tmp1 / (Nvar + 1)) / Nvar < 0.0001)
                break;
        }
        if (k >= L - 1)
            break;
        //step10
        miu = 0;
        for (j = 0; j < Nvar; j ++)
        {
            if (k - 2 < 0)
                k = rand() % L + 2;
            for (l = 0; l < Nvar; l ++)
                xav[l] = xx[k - 2][w][l];
            xav[j] = xx[k - 2][w][j] + 0.001 * s[j];
            if (xav[j] > Ubound)
                xav[j] = Ubound;
            else if (xav[j] < Lbound)
                xav[j] = Lbound;
            xav[Nvar] = EVAL_COMPAT(xav, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            if (xav[Nvar] < xx[k][w][Nvar])
            {
                for (l = 0; l < Nvar + 1; l ++)
                    xx[k][0][l] = xav[l];
                c = 0.001;
                miu ++;
                break;
            }
            xav[j] = xx[k][w][j] - 0.002 * s[j];
            if (xav[j] > Ubound)
                xav[j] = Ubound;
            else if (xav[j] < Lbound)
                xav[j] = Lbound;
            xav[Nvar] = EVAL_COMPAT(xav, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            if (xav[Nvar] < xx[k][w][Nvar])
            {
                for (l = 0; l < Nvar + 1; l ++)
                    xx[k][0][l] = xav[l];
                c = 0.001;
                miu ++;
                break;
            }
        }
        if (miu == 0)
            break;
    }
    
    if (k < L)
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = xx[k][w][l];
        newpop_fit[popi] = xx[k][w][Nvar];
    }
    else
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = xx[L - 2][w][l];
        newpop_fit[popi] = xx[L - 2][w][Nvar];
    }
    
    delete [] s;
    delete [] x1;
    delete [] x2;
    delete [] xav;
    for (int i = 0; i < L + 2; i ++)
    {
        for (int j = 0; j < Nvar + 2; j ++)
            delete [] xx[i][j];
        delete [] xx[i];
    }
    delete [] xx;
}

void MultiMet::newpop_box_complex(int popi, int L, int stepN, double scale)
{
    double ***xx = new double **[L];
    for (int i = 0; i < L; i ++)
    {
        xx[i] = new double *[stepN];
        for (int j = 0; j < stepN; j ++)
            xx[i][j] = new double[Nvar + 1];
    }
    double *xav = new double[Nvar + 1];
    double *x1 = new double[Nvar + 1];
    
    int w = 0, b = 0, count = 0;
    double tmpW = 0, tmpB = 1000;
    int j, k, l;
    bool miu;
    
    for (l = 0; l < Nvar; l ++)
        xx[0][0][l] = newpop[popi][l];
    xx[0][0][Nvar] = newpop_fit[popi];
    for (j = 1; j < stepN; j ++)
    {
        for (l = 0; l < Nvar; l ++)
        {
            xx[0][j][l] = xx[0][0][l] + (randnorm(0, 1) * (Ubound - Lbound) * scale);
            if (xx[0][j][l] > Ubound)
                xx[0][j][l] = Ubound;
            else if (xx[0][j][l] < Lbound)
                xx[0][j][l] = Lbound;
        }
        xx[0][j][Nvar] = EVAL_COMPAT(xx[0][j], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    }	
    
    k = 0;
    while (k < L - 1)
    {
        //step1
        tmpW = xx[k][0][Nvar];
        w = 0;
        for (j = 1; j < stepN; j ++)
        {
            if (xx[k][j][Nvar] > tmpW)
            {
                tmpW = xx[k][j][Nvar];
                w = j;
            }
        }
        for (l = 0; l < Nvar; l ++)
        {
            xav[l] = 0;
            for (j = 0; j < stepN; j ++)
                if (j != w)
                    xav[l] += xx[k][j][l];
            xav[l] /= (stepN - 1);
            x1[l] = xav[l] + 1.3 * (xav[l] - xx[k][w][l]);
            if (x1[l] > Ubound)
                x1[l] = Ubound;
            else if (x1[l] < Lbound)
                x1[l] = Lbound;
        }
        x1[Nvar] = EVAL_COMPAT(x1, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
        
        while (k < L - 1)
        {
            //step7
            miu = 0;
            for (j = 0; j < stepN; j ++)
            {
                if (x1[Nvar] < xx[k][j][Nvar])
                {
                    miu = 1;
                    break;
                }
            }
            if (miu == 1)
            {
                for (j = 0; j < stepN; j ++)
                {
                    if (j == w)
                        for (l = 0; l < Nvar + 1; l ++)
                            xx[k + 1][j][l] = x1[l];
                    else
                        for (l = 0; l < Nvar + 1; l ++)
                            xx[k + 1][j][l] = xx[k][j][l];
                }
                k ++; count = 0;
                break;
            }
            else
            {
                count ++;
                if (count >= 5)
                    break;
                else
                {
                    for (l = 0; l < Nvar; l ++)
                    {
                        x1[l] = 0.5 * (xav[l] + x1[l]);
                        if (x1[l] > Ubound)
                            x1[l] = Ubound;
                        else if (x1[l] < Lbound)
                            x1[l] = Lbound;
                    }
                    x1[Nvar] = EVAL_COMPAT(x1, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                }
            }
        }
        
        tmpB = xx[k][0][Nvar];
        b = 0;
        for (j = 1; j < stepN; j ++)
        {
            if (xx[k][j][Nvar] < tmpB)
            {
                tmpB = xx[k][j][Nvar];
                b = j;
            }
        }
        if (count >= 5)
            break;
    }
    
    if (k < L)
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = xx[k][b][l];
        newpop_fit[popi] = xx[k][b][Nvar];
    }
    else
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = xx[L - 1][b][l];
        newpop_fit[popi] = xx[L - 1][b][Nvar];
    }
    
    delete [] xav;
    delete [] x1;
    for (int i = 0; i < L; i ++)
    {
        for (int j = 0; j < stepN; j ++)
            delete [] xx[i][j];
        delete [] xx[i];
    }
    delete [] xx;
}

void MultiMet::newpop_powell(int popi, int L, int stepn, double scale)
{
    int i, j, k, l, h;
    
    double s = (Ubound - Lbound) * scale;
    double ***xx = new double **[L + 1];
    for (i = 0; i < L + 1; i ++)
    {
        xx[i] = new double *[Nvar + 1];
        for (j = 0; j < Nvar + 1; j ++)
            xx[i][j] = new double[Nvar + 1];
    }
    double *y1 = new double[Nvar + 1];
    double *y2 = new double[Nvar + 1];
    double *y3 = new double[Nvar + 1];
    double *z = new double[Nvar + 1];
    double *fai = new double[Nvar];
    double **v = new double*[Nvar];
    for (i = 0; i < Nvar; i ++)
        v[i] = new double[Nvar];
    
    int miu = 0, count = 0;
    double tmp = 0;
    for (j = 0; j < Nvar; j ++)
        for (l = 0; l < Nvar; l ++)
            if (j == l)
                v[j][l] = s;
            else
                v[j][l] = 0;
    
    for (l = 0; l < Nvar; l ++)
        xx[0][0][l] = newpop[popi][l];
    xx[0][0][Nvar] = newpop_fit[popi];
    
    k = 0; count = 0;
    while (k < L)
    {
        //step2-step4
        for (j = 1; j < Nvar + 1; j ++)
        {
            for (l = 0; l < Nvar + 1; l ++)
                xx[k][j][l] = z[l] = xx[k][j - 1][l];
            for (h = 0; h < stepn; h ++)
            {
                for (l = 0; l < Nvar; l ++)
                {
                    z[l] += v[j - 1][l] * randnorm(0, 1);
                    if (z[l] > Ubound)
                        z[l] = Ubound;
                    if (z[l] < Lbound)
                        z[l] = Lbound;
                }
                z[Nvar] = EVAL_COMPAT(z, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                if (z[Nvar] < xx[k][j][Nvar])
                {
                    for (l = 0; l < Nvar + 1; l ++)
                        xx[k][j][l] = z[l];
                }
            }
        }
        
        //step5
        k ++;
        for (l = 0; l < Nvar + 1; l ++)
            xx[k][0][l] = xx[k - 1][Nvar][l];
        miu = 0;
        for (l = 0; l < Nvar; l ++)
            if (fabs(xx[k][0][l] - xx[k - 1][0][l]) < 1e-10)
                miu ++;
            else
                break;
        if (miu == Nvar && count == 0)
        {
            //step9
            for (l = 0; l < Nvar + 1; l ++)
                y1[l] = xx[k][0][l];
            for (l = 0; l < Nvar; l ++)
            {
                xx[0][0][l] = y1[l] + s;          //epsl = 0.001;   10 * epsl
                if (xx[0][0][l] > Ubound)
                    xx[0][0][l] = Ubound;
                else if (xx[0][0][l] < Lbound)
                    xx[0][0][l] = Lbound;
            }
            xx[0][0][Nvar] = EVAL_COMPAT(xx[0][0], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            count = 1;
            continue;
        }
        else if (miu == Nvar && count > 0)
        {
            for (l = 0; l < Nvar + 1; l ++)
                y2[l] = xx[k][0][l];
            //determine y3
            for (l = 0; l < Nvar + 1; l ++)
                y3[l] = z[l] = y2[l];
            for (h = 0; h < stepn; h ++)
            {
                for (l = 0; l < Nvar; l ++)
                {
                    z[l] += (y2[l] - y1[l]) * randnorm(0, 1);
                    if (z[l] > Ubound)
                        z[l] = Ubound;
                    if (z[l] < Lbound)
                        z[l] = Lbound;
                }
                z[Nvar] = EVAL_COMPAT(z, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                if (z[Nvar] < y3[Nvar])
                {
                    for (l = 0; l < Nvar + 1; l ++)
                        y3[l] = z[l];
                }
            }
            miu = 0;
            for (l = 0; l < Nvar; l ++)
                if (fabs(y3[l] - y2[l]) < 1e-10 && fabs(y3[l] - y1[l]) < 1e-10)
                    miu ++;
                else
                    break;
            if (miu == Nvar || count > 1)
                break;
            else
            {
                for (l = 0; l < Nvar; l ++)
                    xx[0][0][l] = y3[l];
                xx[0][0][Nvar] = EVAL_COMPAT(xx[0][0], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
                for (l = 0; l < Nvar; l ++)
                    v[0][l] = y3[l] - y1[l];
                k = 0; count ++;
                continue;
            }
        }
        
        //step6
        for (l = 0; l < Nvar; l ++)
        {
            z[l] = 2 * xx[k][0][l] - xx[k - 1][0][l];
            if (z[l] > Ubound)
                z[l] = Ubound;
            else if (z[l] < Lbound)
                z[l] = Lbound;
        }
        z[Nvar] = EVAL_COMPAT(z, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);                          //F~
        if (z[Nvar] >= xx[k - 1][0][Nvar])
            continue;                                       //back to step2
        
        //step7
        miu = 0; tmp = 0;
        for (j = 1; j < Nvar + 1; j ++)
        {
            fai[j - 1] = xx[k - 1][j - 1][Nvar] - xx[k - 1][j][Nvar];
            if (fai[j - 1] > tmp)
            {
                tmp = fai[j - 1];
                miu = j - 1;
            }
        }
        if ((xx[k - 1][0][Nvar] - 2 * xx[k][0][Nvar] + z[Nvar]) * pow(xx[k - 1][0][Nvar] - xx[k][0][Nvar] - fai[miu], 2.0)
            >= fai[miu] * pow(xx[k - 1][0][Nvar] - z[Nvar], 2.0) / 2)
            continue;
        
        //step8
        for (l = 0; l < Nvar; l ++)
            v[miu][l] = xx[k - 1][Nvar][l] - xx[k - 1][0][l];
        //determine a new x(k)
        for (l = 0; l < Nvar + 1; l ++)
            xx[k][0][l] = z[l] = xx[k - 1][Nvar][l];
        for (h = 0; h < stepn; h ++)
        {
            for (l = 0; l < Nvar; l ++)
            {
                z[l] += v[miu][l] * randnorm(0, 1);
                if (z[l] > Ubound)
                    z[l] = Ubound;
                if (z[l] < Lbound)
                    z[l] = Lbound;
            }
            z[Nvar] = EVAL_COMPAT(z, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
            if (z[Nvar] < xx[k][0][Nvar])
            {
                for (l = 0; l < Nvar + 1; l ++)
                    xx[k][0][l] = z[l];
            }
        }
    }//end while
    
    if (k < L)
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = y3[l];
        y3[Nvar] = EVAL_COMPAT(y3, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
        newpop_fit[popi] = y3[Nvar];
    }
    else
    {
        for (l = 0; l < Nvar; l ++)
            newpop[popi][l] = xx[L][0][l];
        newpop_fit[popi] = xx[L][0][Nvar];
    }
    
    delete [] y1;
    delete [] y2;
    delete [] y3;
    delete [] z;
    delete [] fai;
    for (i = 0; i < L + 1; i ++)
    {
        for (j = 0; j < Nvar + 1; j ++)
            delete [] xx[i][j];
        delete [] xx[i];
    }
    delete [] xx;
    for (i = 0; i < Nvar; i ++)
        delete [] v[i];
    delete [] v;
}

void MultiMet::meme_selection(int popi, int X, double scale, int Iter)
{
    switch(X)
    {
        case 0:
        {newpop_bit_climbing(popi, Iter, scale);break;}
        case 1:
        {newpop_simplex(popi, Iter, 10, scale);break;}
        case 2:
        {newpop_box_complex(popi, Iter, 10, scale);break;}
        case 3:
        {newpop_powell(popi, Iter, 10, scale);break;}
        default:
        {newpop_bit_climbing(popi, Iter, scale);break;}
    }
}

void MultiMet::meme_random_walk(double scale)
{
    int step = 0, count = 0;
    double rr;
    for (int i = 0; i < Popsize; i ++)
    {
        meme_selection(i, step, scale, 10);
        rr = randval(0, 1);
        if (count < 5)
        {
            if (rr < 0.5)
                step -= 1;
            else
                step += 1;
        }
        else
        {
            if (rr < 0.2)
                step -= 1;
            else if (rr < 0.4)
                step += 1;
            else if (rr < 0.55)
                step -= 2;
            else if (rr < 0.7)
                step += 2;
            else if (rr < 0.85)
                step -= 3;
            else
                step += 3;
        }
        if (step < 0)
            step = 0;
        else if (step > 4)
            step %= 4;
        
        count ++;
    }
}

void MultiMet::meme_simple_random(double scale)
{
    int rr = 0;
    for (int i = 0; i < Popsize; i ++)
    {
        rr = rand() % 4;
        meme_selection(i, rr, scale, 10);
    }
}

void MultiMet::meme_randperm(double scale)
{
    std::random_device rd;
    std::mt19937_64 g(rd());
    int count = 0;
    vector<int> permu;
    for (int i = 0; i < 4; i ++)
        permu.push_back(i);
    shuffle(permu.begin(), permu.end(), g);
    for (int i = 0; i < Popsize; i ++)
    {
        meme_selection(i, permu[count], scale, 10);
        count ++;
        if (count >= 4)
        {
            count = 0;
            shuffle(permu.begin(), permu.end(), g);
        }
    }
}

void MultiMet::meme_inheritance(double scale)
{
    for (int i = 0; i < Popsize; i ++)
        meme_selection(i, Ind_meme[i], scale, 10);
    
    for (int i = 0; i < Popsize - 1; i += 2)
    {
        if (newpop_fit[i] == newpop_fit[i + 1])
        {
            if (randval(0, 1) < 0.5)
                Ind_meme[i] = Ind_meme[i + 1];
            else
                Ind_meme[i + 1] = Ind_meme[i];
        }
        else if (newpop_fit[i] < newpop_fit[i + 1])
            Ind_meme[i + 1] = Ind_meme[i];
        else
            Ind_meme[i] = Ind_meme[i + 1];
    }
}

void MultiMet::meme_subprob_decomposition(int Gen, int MaxG, int kk, double scale)
{
    int rr, i, j, l;
    double oldfit = 0, reward = 0;
    double **dis;
    dis = new double*[Popsize];      //1:dis, 2:reward, 3:meme number
    for (i = 0; i < Popsize; i ++)
        dis[i] = new double[3];
    
    if (Gen < MaxG)
    {
        if (Gen == 0)
        {
            for (i = 0; i < Popsize; i ++)
            {
                rr = rand() % 4;
                oldfit = newpop_fit[i];
                meme_selection(i, rr, scale, 10);
                for (j = 0; j < Nvar; j ++)
                    SubDecBase[i][j] = newpop[i][j];
                SubDecBase[i][Nvar] = (gbest_fit / newpop_fit[i]) * (oldfit - newpop_fit[i]) / 100;
                SubDecBase[i][Nvar + 1] = rr;
            }
            heap_sort(SubDecBase, Popsize, Nvar);
        }
        else
        {
            for (i = 0; i < Popsize; i ++)
            {
                rr = rand() % 4;
                oldfit = newpop_fit[i];
                meme_selection(i, rr, scale, 10);
                reward = (gbest_fit / newpop_fit[i]) * (oldfit - newpop_fit[i]) / 100;
                if (reward > SubDecBase[0][Nvar])
                {
                    for (j = 0; j < Nvar; j ++)
                        SubDecBase[0][j] = newpop[i][j];
                    SubDecBase[0][Nvar] = reward;
                    SubDecBase[0][Nvar + 1] = rr;
                    heap_sort(SubDecBase, Popsize, Nvar);
                }
            }
        }
    }
    else
    {
        for (i = 0; i < Popsize; i ++)
        {
            for (j = 0; j < Popsize; j ++)
            {
                dis[j][0] = 0;
                for (l = 0; l < Nvar; l ++)
                    dis[j][0] += pow(newpop[i][l] - SubDecBase[j][l], 2.0);
                dis[j][0] = sqrt(dis[j][0]);
                dis[j][1] = SubDecBase[j][Nvar];
                dis[j][2] = SubDecBase[j][Nvar + 1];
            }
            heap_sort(dis, Popsize, 0);    //sort according to distance
            heap_sort(dis, kk, 1);      //sort according to reward
            
            Ind_meme[i] = (int)dis[kk - 1][2];
        }
        for (i = 0; i < Popsize; i ++)
        {
            oldfit = newpop_fit[i];
            meme_selection(i, Ind_meme[i], scale, 10);
            reward = (gbest_fit / newpop_fit[i]) * (oldfit - newpop_fit[i]) / 100;
            if (reward > SubDecBase[Popsize - 1][Nvar])
            {
                for (j = 0; j < Nvar; j ++)
                    SubDecBase[0][j] = newpop[i][j];
                SubDecBase[0][Nvar] = reward;
                SubDecBase[0][Nvar + 1] = Ind_meme[i];
                heap_sort(SubDecBase, Popsize, Nvar);
            }
        }
    }
    
    for (i = 0; i < Popsize; i ++)
        delete [] dis[i];
    delete [] dis;
}

void MultiMet::meme_biasd_roulette(int Gen, int MaxG, double scale)
{
    int rr, i, j;              //use SubDecBase to store global fitness of LS. SubDecBase[i]: ith LS, subDecBase[i][0]: count, subDecBase[i][1]:total fitness
    double oldfit = 0, reward = 0;
    
    double *rfitness, *cfitness;
    rfitness = new double[4];
    cfitness = new double[4];
    double p, sum = 0;
    
    if (Gen < MaxG)
    {
        for (i = 0; i < Popsize; i ++)
        {
            rr = rand() % 4;
            oldfit = newpop_fit[i];
            meme_selection(i, rr, scale, 10);
            reward = (gbest_fit / newpop_fit[i]) * (oldfit - newpop_fit[i]) / 100;
            SubDecBase[rr][0] ++;
            SubDecBase[rr][1] += reward;
        }
    }
    else
    {
        for (i = 0; i < 4; i++)
            sum += SubDecBase[i][1] / (SubDecBase[i][0] + 1);        //适应值总和
        for (i = 0; i < 4; i++)
            rfitness[i] = (SubDecBase[i][1] / (SubDecBase[i][0] + 1)) / sum;                 //适应值所占比??
        cfitness[0] = rfitness[0];
        for (i = 1; i < 4; i++)
            cfitness[i] = cfitness[i-1] + rfitness[i];           //轮盘位置
        
        for (i = 0; i < Popsize; i++)
        {
            p = randval(0.0, 1.0);
            if (p < cfitness[0])                                 //轮盘赌选择
            {
                Ind_meme[i] = 0;
            }
            else
            {
                for (j = 0; j < 3; j ++)
                    if (p >= cfitness[j] && p < cfitness[j+1])
                        Ind_meme[i] = j + 1;
            }
        }
        
        for (i = 0; i < Popsize; i ++)
        {
            rr = Ind_meme[i];
            oldfit = newpop_fit[i];
            meme_selection(i, rr, scale, 10);
            reward = (gbest_fit / newpop_fit[i]) * (oldfit - newpop_fit[i]) / 100;
            SubDecBase[rr][0] ++;
            SubDecBase[rr][1] += reward;
        }
    }
    delete [] rfitness;
    delete [] cfitness;
}

void MultiMet::Direct_Change(int ChangeSize)
{
    double min_dis = 1e10;
    int min_index = -1;
    double tmp_dis = 0;
    if (delta_update_count > ARCHIVE)
    {
        for (int k = 0; k < ChangeSize; k ++)
        {
            min_dis = 1e10; min_index = -1;
            int pop_ind = rand() % Popsize;
            for (int i = 0; i < 10; i ++)
            {
                tmp_dis = Euclidean_dis(pop[pop_ind], delta_pop[i], Nvar);
                if (tmp_dis < min_dis)
                {
                    min_dis = tmp_dis;
                    min_index = i;
                }
            }
            if (min_index >= 0 && min_index < 10)
            {
                for (int i = 0; i < Nvar; i ++)
                {
                    newpop[pop_ind][i] += delta_pop[min_index][Nvar + i];
                }
            }
        }

    }
}

/**
 * InitMigration - Initialize rotated-ring migration parameters
 * 
 * @param nG Number of subpopulations (default 8)
 * @param nC Migration interval in generations (default 5)
 * @param pE Probability of sending gbest vs random (default 0.8)
 */
void MultiMet::InitMigration(int nG, int nC, double pE)
{
    nSubpop = nG;
    nCircle = nC;
    pElitist = pE;
    Dispara = 1;  // Initial neighbor distance
    migrationEnabled = true;
    
    // Allocate subpopulation best arrays
    subpop_gbest = new double*[nSubpop];
    subpop_gbest_fit = new double[nSubpop];
    for (int i = 0; i < nSubpop; i++)
    {
        subpop_gbest[i] = new double[Nvar];
        subpop_gbest_fit[i] = 1e10;
    }
    
    cout << "[Migration] Initialized: " << nSubpop << " subpops, interval=" 
         << nCircle << ", pElitist=" << pElitist << endl;
}

/**
 * UpdateSubpopBest - Update best individual for each subpopulation
 */
void MultiMet::UpdateSubpopBest()
{
    if (!migrationEnabled) return;
    
    int subpop_size = Popsize / nSubpop;
    
    for (int s = 0; s < nSubpop; s++)
    {
        int start = s * subpop_size;
        int end = (s == nSubpop - 1) ? Popsize : (s + 1) * subpop_size;
        
        for (int i = start; i < end; i++)
        {
            if (pop_fit[i] < subpop_gbest_fit[s])
            {
                for (int j = 0; j < Nvar; j++)
                    subpop_gbest[s][j] = pop[i][j];
                subpop_gbest_fit[s] = pop_fit[i];
            }
        }
    }
}

/**
 * GetSubpopWorst - Get index of worst individual in a subpopulation
 */
int MultiMet::GetSubpopWorst(int subpop_idx)
{
    int subpop_size = Popsize / nSubpop;
    int start = subpop_idx * subpop_size;
    int end = (subpop_idx == nSubpop - 1) ? Popsize : (subpop_idx + 1) * subpop_size;
    
    int worst_idx = start;
    double worst_fit = pop_fit[start];
    
    for (int i = start + 1; i < end; i++)
    {
        if (pop_fit[i] > worst_fit)
        {
            worst_fit = pop_fit[i];
            worst_idx = i;
        }
    }
    return worst_idx;
}

/**
 * RingMigration - Perform rotated-ring migration
 * 
 * At every nCircle generations:
 * 1. Each subpop k sends to neighbor (k + Dispara) mod nSubpop
 * 2. With probability pElitist, send gbest; otherwise send random individual
 * 3. Receiving subpop replaces its worst with received individual
 * 4. Update Dispara = (Dispara + 1) mod (nSubpop - 1) for next migration
 */
void MultiMet::RingMigration(int gen)
{
    if (!migrationEnabled) return;
    if (gen % nCircle != 0 || gen == 0) return;
    
    int subpop_size = Popsize / nSubpop;
    
    // Buffer to store migrants (one per subpop)
    double** migrants = new double*[nSubpop];
    double* migrant_fit = new double[nSubpop];
    
    // Step 1: Each subpop prepares its migrant
    for (int k = 0; k < nSubpop; k++)
    {
        migrants[k] = new double[Nvar];
        
        if (randval(0, 1) < pElitist)
        {
            // Send subpop gbest
            for (int j = 0; j < Nvar; j++)
                migrants[k][j] = subpop_gbest[k][j];
            migrant_fit[k] = subpop_gbest_fit[k];
        }
        else
        {
            // Send random individual from subpop
            int start = k * subpop_size;
            int end = (k == nSubpop - 1) ? Popsize : (k + 1) * subpop_size;
            int rand_idx = start + rand() % (end - start);
            
            for (int j = 0; j < Nvar; j++)
                migrants[k][j] = pop[rand_idx][j];
            migrant_fit[k] = pop_fit[rand_idx];
        }
    }
    
    // Step 2: Ring migration - subpop k receives from (k - Dispara + nSubpop) mod nSubpop
    for (int k = 0; k < nSubpop; k++)
    {
        int source = ((k - Dispara) % nSubpop + nSubpop) % nSubpop;
        int worst_idx = GetSubpopWorst(k);
        
        // Replace worst with received migrant
        for (int j = 0; j < Nvar; j++)
            pop[worst_idx][j] = migrants[source][j];
        pop_fit[worst_idx] = migrant_fit[source];
    }
    
    // Step 3: Update Dispara for next migration (rotated ring)
    Dispara = (Dispara % (nSubpop - 1)) + 1;
    
    // Cleanup
    for (int k = 0; k < nSubpop; k++)
        delete[] migrants[k];
    delete[] migrants;
    delete[] migrant_fit;
    
    cout << "[Migration] Gen " << gen << ": ring migration complete, Dispara=" << Dispara << endl;
}












