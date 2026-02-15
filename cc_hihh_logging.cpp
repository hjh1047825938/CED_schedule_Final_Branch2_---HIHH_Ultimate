// cc_hihh_logging.cpp
// Patch reference file for operator logging/selection ablation support.
// The active implementation is integrated into:
//   - include/CC_HIHH.h
//   - src/CC_HIHH_singleisland.cpp
//   - src/main.cpp
//
// Added capabilities:
// 1) Weight logging per block (7D contextual bandit weights + L2 norm)
// 2) Reward logging per operator application
// 3) Global stats logging (best/avg/diversity/epsilon/stagnation/gating)
// 4) Operator selection modes: contextual bandit / random / round-robin
//
// This file is intentionally non-compiled documentation for the patch set.
