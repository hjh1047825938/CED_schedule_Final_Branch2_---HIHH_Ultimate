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
    int max_CE_Tnum = 0;
    int max_M_Jnum = 0;
    int max_M_OPTnum = 0;
    int max_Enum = 0;
    int max_Dnum = 0;
    
    /**
     * Resize all buffers to accommodate the given dimensions.
     * Call once at initialization, not on every evaluation.
     */
    void resize(int CE_Tnum, int M_Jnum, int M_OPTnum, int Enum, int Dnum) {
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

        nearest_ready = false;
        token = 1;
        std::fill(seen_stamp.begin(), seen_stamp.end(), 0);
        
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
};

#endif // _WORKSPACE_H
