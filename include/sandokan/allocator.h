#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

extern "C" {
#include "plad_bridge.h"
#include "incPMAD.h"
}

// ---------- Lifecycle ----------
//
// 8 PMAD size classes — Layer gradients + BatchWorkspace, all float.
// Sorted ascending; percentages sum to 100. Pool = 3 MB.
//
//   Class 0:     104 B — output.db              (26×4)        ×  1  →  1%
//   Class 1:     256 B — hidden.db              (64×4)        ×  2  →  1%
//   Class 2:   6,656 B — output.dW              (26×64×4)     ×  1  →  1%
//   Class 3:  13,312 B — workspace Z3,A3,D3     (26×128×4)    ×  3  →  3%
//   Class 4:  16,384 B — hidden2.dW             (64×64×4)     ×  1  →  1%
//   Class 5:  32,768 B — workspace Z1,A1,Z2,A2,D1,D2 (64×128×4) × 6 → 8%
//   Class 6: 200,704 B — hidden1.dW             (64×784×4)    ×  1  → 10%
//   Class 7: 401,408 B — workspace X0,X1        (784×128×4)   ×  2  → 75%
//
// 17 allocations total: 6 Layer + 11 BatchWorkspace.
// BlockHeader overhead = 16 B; slots per class verified against 3 MB pool.

inline void init_network_pmad() {
    static const size_t class_sizes[8] = { 104, 256, 6656, 13312, 16384, 32768, 200704, 401408 };
    static const size_t percentages[8] = {   1,   1,    1,     3,     1,     8,     10,     75 };
    static const size_t pool_size      = 3 * 1024 * 1024;  // 3 MB

    PmadStatus st = pmad_init(class_sizes, 8, percentages, pool_size);
    if (st != PMAD_OK)
        throw std::runtime_error("pmad_init failed: " + std::to_string(st));

    std::printf("PMAD initialised  pool=3MB  8 classes "
                "[104B 256B 6656B 13312B 16384B 32768B 200704B 401408B]\n\n");
}

inline void destroy_network_pmad() { pmad_destroy(); }

// ---------- Live stats ----------

inline void print_pmad_stats() {
    PmadClassStats stats[8];
    int n = pmad_get_stats(stats, 8);
    std::printf("PMAD live stats:\n");
    for (int i = 0; i < n; ++i)
        std::printf("  class[%d]  block=%6u B  total=%3u  allocated=%u\n",
                    i, stats[i].block_size, stats[i].total_blocks,
                    stats[i].allocated_blocks);
    std::printf("\n");
}
