#ifndef _WORKSPACE_H
#define _WORKSPACE_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>

/**
 * Workspace - Reusable memory buffers for CED_Schedule fitness evaluation.
 * 
 * Purpose: Avoid repeated new/delete allocations during fitness evaluation.
 * Usage: Create one Workspace per thread, call resize() once with max dimensions,
 *        then pass to CED_Schedule_Fast() for each evaluation.
 */
struct Workspace {
    // Normalization and weighting parameters for weighted objective
    double f1_ref = 1.0;   // Reference makespan
    double f2_ref = 1.0;   // Reference energy
    double alpha = 0.5;    // Weight for makespan in [0, 1]
    double last_makespan = 0.0;
    double last_energy = 0.0;
    double last_fitness = 0.0;

    // Decision variable buffers
    std::vector<bool> ce_sele;      // Cloud/Edge selection [CE_Tnum]
    std::vector<int> cevar;          // Cloud/Edge assignment [CE_Tnum]
    std::vector<int> mvar;           // Machine assignment [M_Jnum * M_OPTnum]
    
    // Sorting helper - replaces sort_vec double**
    std::vector<int> op_order;       // Operation order indices [M_Jnum * M_OPTnum]
    std::vector<double> op_keys;     // Cached sort keys [M_Jnum * M_OPTnum]
    
    // Job processing
    std::vector<int> geneO;          // Current operation step per job [M_Jnum]
    std::vector<int> last_dev_op;    // Last operation index per device [Dnum]
    std::vector<double> last_dev_end; // End time of last op per device [Dnum]
    
    // Distance-related
    std::vector<int> nearest_device; // Nearest device to each edge [Enum]
    std::vector<int> nearest_edge;   // Nearest edge to each device [Dnum]
    std::vector<double> edge_smallest_rate; // Smallest rate per edge [Enum]
    std::vector<double> edge_device_comm;   // Flat array [Enum * Dnum]
    bool nearest_ready = false;
    
    // O(1) deduplication - replaces O(n^2) find() calls
    std::vector<uint32_t> seen_stamp; // For deduplication [Dnum]
    uint32_t token = 1;

    // Device sets (flat pools)
    std::vector<int> task_devs;      // [CE_Tnum * M_OPTnum]
    std::vector<int> task_dev_count; // [CE_Tnum]
    std::vector<int> edge_devs;      // [Enum * Dnum]
    std::vector<int> edge_dev_count; // [Enum]

    // Mutable evaluation state (thread-local via workspace)
    std::vector<std::vector<int>> cloud_load; // [Cnum]
    std::vector<std::vector<int>> edge_load;  // [Enum]
    std::vector<double> st_flat;              // [M_Jnum * M_OPTnum]
    std::vector<double> et_flat;              // [M_Jnum * M_OPTnum]
    std::vector<double*> st_rows;             // [M_Jnum]
    std::vector<double*> et_rows;             // [M_Jnum]
    std::vector<double> ce_st;                // [CE_Tnum]
    std::vector<double> ce_et;                // [CE_Tnum]
    std::vector<double> comm_factor;          // [CE_Tnum], Communication * 0.01
    bool comm_factor_ready = false;

    struct EvalProfile {
        uint64_t samples = 0;
        uint64_t decode_us = 0;
        uint64_t sort_us = 0;
        uint64_t assign_us = 0;
        uint64_t schedule_us = 0;
        uint64_t devices_us = 0;
        uint64_t comm_us = 0;
        uint64_t tasks_us = 0;

        void reset() {
            samples = 0;
            decode_us = sort_us = assign_us = schedule_us = 0;
            devices_us = comm_us = tasks_us = 0;
        }
    } profile;
    
    // Dimensions (for validation)
    int max_Cnum = 0;
    int max_CE_Tnum = 0;
    int max_M_Jnum = 0;
    int max_M_OPTnum = 0;
    int max_Enum = 0;
    int max_Dnum = 0;
    
    /**
     * Resize all buffers to accommodate the given dimensions.
     * Call once at initialization, not on every evaluation.
     */
    void resize(int Cnum, int CE_Tnum, int M_Jnum, int M_OPTnum, int Enum, int Dnum) {
        int ops = M_Jnum * M_OPTnum;
        
        ce_sele.resize(CE_Tnum);
        cevar.resize(CE_Tnum);
        mvar.resize(ops);
        op_order.resize(ops);
        op_keys.resize(ops);
        geneO.resize(M_Jnum);
        last_dev_op.resize(Dnum);
        last_dev_end.resize(Dnum);
        nearest_device.resize(Enum);
        nearest_edge.resize(Dnum);
        edge_smallest_rate.resize(Enum);
        edge_device_comm.resize(Enum * Dnum);
        seen_stamp.resize(Dnum);
        task_devs.resize(CE_Tnum * M_OPTnum);
        task_dev_count.resize(CE_Tnum);
        edge_devs.resize(Enum * Dnum);
        edge_dev_count.resize(Enum);
        cloud_load.resize(Cnum);
        edge_load.resize(Enum);
        st_flat.resize(ops);
        et_flat.resize(ops);
        st_rows.resize(M_Jnum);
        et_rows.resize(M_Jnum);
        ce_st.resize(CE_Tnum);
        ce_et.resize(CE_Tnum);
        comm_factor.resize(CE_Tnum);
        comm_factor_ready = false;

        // Prepare row pointers and reserve server task vectors once.
        for (int j = 0; j < M_Jnum; ++j) {
            st_rows[j] = st_flat.data() + j * M_OPTnum;
            et_rows[j] = et_flat.data() + j * M_OPTnum;
        }
        const int cloud_reserve = std::max(1, CE_Tnum / std::max(1, Cnum) + 2);
        const int edge_reserve = std::max(1, CE_Tnum / std::max(1, Enum) + 2);
        for (int c = 0; c < Cnum; ++c) {
            cloud_load[c].clear();
            cloud_load[c].reserve(cloud_reserve);
        }
        for (int e = 0; e < Enum; ++e) {
            edge_load[e].clear();
            edge_load[e].reserve(edge_reserve);
        }

        nearest_ready = false;
        token = 1;
        std::fill(seen_stamp.begin(), seen_stamp.end(), 0);
        
        max_Cnum = Cnum;
        max_CE_Tnum = CE_Tnum;
        max_M_Jnum = M_Jnum;
        max_M_OPTnum = M_OPTnum;
        max_Enum = Enum;
        max_Dnum = Dnum;
    }
    
    /**
     * Clear buffers for reuse (fast - just fills, doesn't reallocate)
     */
    void clear() {
        std::fill(geneO.begin(), geneO.end(), -1);
        std::fill(last_dev_op.begin(), last_dev_op.end(), -1);
        std::fill(last_dev_end.begin(), last_dev_end.end(), 0.0);
        for (auto& v : cloud_load) v.clear();
        for (auto& v : edge_load) v.clear();
    }

    uint32_t next_token() {
        token++;
        if (token == 0) {
            std::fill(seen_stamp.begin(), seen_stamp.end(), 0);
            token = 1;
        }
        return token;
    }

    void reset_profile() {
        profile.reset();
    }

    void set_normalization(double f1_reference, double f2_reference) {
        f1_ref = f1_reference;
        f2_ref = f2_reference;
    }

    void set_alpha(double weight) {
        alpha = std::clamp(weight, 0.0, 1.0);
    }
};

class WorkspacePool {
public:
    void init(int num_threads, int Cnum, int CE_Tnum, int M_Jnum, int M_OPTnum, int Enum, int Dnum) {
        if (num_threads < 1) num_threads = 1;
        pool_.resize(num_threads);
        for (auto& ws : pool_) {
            ws.resize(Cnum, CE_Tnum, M_Jnum, M_OPTnum, Enum, Dnum);
        }
    }

    Workspace& get(int thread_id) {
        if (pool_.empty()) return fallback_;
        if (thread_id < 0) thread_id = 0;
        if (thread_id >= (int)pool_.size()) thread_id = (int)pool_.size() - 1;
        return pool_[thread_id];
    }

    int size() const {
        return (int)pool_.size();
    }

private:
    std::vector<Workspace> pool_;
    Workspace fallback_;
};

#endif // _WORKSPACE_H
