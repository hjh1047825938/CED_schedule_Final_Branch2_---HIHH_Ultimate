#ifndef _WORKSPACE_H
#define _WORKSPACE_H

#include <vector>
#include <numeric>
#include <algorithm>

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
    std::vector<int> seq_var;        // Sequence variable [M_Jnum * M_OPTnum]
    
    // Sorting helper - replaces sort_vec double**
    std::vector<int> op_order;       // Operation order indices [M_Jnum * M_OPTnum]
    
    // Job processing
    std::vector<int> geneO;          // Current operation step per job [M_Jnum]
    
    // Distance-related
    std::vector<int> nearest_device; // Nearest device to each edge [Enum]
    std::vector<int> nearest_edge;   // Nearest edge to each device [Dnum]
    std::vector<double> edge_smallest_rate; // Smallest rate per edge [Enum]
    
    // O(1) deduplication - replaces O(n^2) find() calls
    std::vector<char> seen_device;   // For deduplication [Dnum]
    
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
        seq_var.resize(ops);
        op_order.resize(ops);
        geneO.resize(M_Jnum);
        nearest_device.resize(Enum);
        nearest_edge.resize(Dnum);
        edge_smallest_rate.resize(Enum);
        seen_device.resize(Dnum);
        
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
        std::fill(seen_device.begin(), seen_device.end(), 0);
    }
};

#endif // _WORKSPACE_H
