#pragma once

#include "network.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

// ---------- Deterministic 6-class Layer plan (documentation) ----------
//
// Only dW and db are PMAD-backed in Layer now. a/z/delta removed — the batched
// path uses BatchWorkspace; single-sample path uses local VectorXf.

struct PMADClass {
    std::string name;
    size_t      elements;
    double      percentage;
    bool        pmad_backed;
};

inline std::array<PMADClass, 6> compute_pmad_plan(const Network& net) {
    const Layer* layers[3] = { &net.hidden1, &net.hidden2, &net.output };
    const char*  names[3]  = { "hidden1",    "hidden2",    "output"   };

    std::array<PMADClass, 6> classes;
    int ci = 0;
    for (int li = 0; li < 3; ++li) {
        const Layer& l = *layers[li];
        classes[ci++] = { std::string(names[li]) + ".dW", static_cast<size_t>(l.dW.size()), 0.0, true };
        classes[ci++] = { std::string(names[li]) + ".db", static_cast<size_t>(l.db.size()), 0.0, true };
    }

    size_t total = 0;
    for (auto& c : classes) total += c.elements;
    for (auto& c : classes) c.percentage = 100.0 * c.elements / total;
    return classes;
}

inline void print_pmad_plan(const Network& net) {
    auto classes = compute_pmad_plan(net);
    size_t total = 0;
    for (auto& c : classes) total += c.elements;

    std::printf("PMAD Layer Plan (dW + db only; BatchWorkspace holds 11 more blocks)\n");
    std::printf("Network : %d -> %d -> %d -> %d\n",
                Network::INPUT_SIZE, Network::HIDDEN1,
                Network::HIDDEN2,    Network::OUTPUT_SIZE);
    std::printf("Total   : %zu floats  (%.2f KB)\n\n",
                total, total * sizeof(float) / 1024.0);

    std::printf("  %-3s  %-30s  %10s  %9s\n", "Cls", "Name", "Elements", "Pct");
    std::printf("  %s\n", std::string(58, '-').c_str());
    for (int i = 0; i < 6; ++i) {
        const auto& c = classes[i];
        std::printf("  [%d]  %-30s  %10zu  %8.2f%%\n",
                    i, c.name.c_str(), c.elements, c.percentage);
    }
    std::printf("  %s\n", std::string(58, '-').c_str());
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
