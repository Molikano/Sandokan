#pragma once

#include "network.h"   // already bridges plad_bridge.h + incPMAD.h
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

// ---------- Deterministic 8-class plan (documentation / RL agent input) ----------
//
// dW matrices exceed PMAD's MAX_SIZE_OF_SIZE_CLASS (4096 bytes), so they remain
// Eigen-managed. The plan is still computed for the full budget view.

struct PMADClass {
    std::string name;
    size_t      elements;
    double      percentage;
    bool        pmad_backed;   // false for dW classes (exceed 4096-byte limit)
};

inline std::array<PMADClass, 8> compute_pmad_plan(const Network& net) {
    const Layer* layers[3] = { &net.hidden1, &net.hidden2, &net.output };
    const char*  names[3]  = { "hidden1",    "hidden2",    "output"   };

    std::array<PMADClass, 8> classes;
    int ci = 0;

    for (int li = 0; li < 3; ++li) {
        const Layer& l    = *layers[li];
        bool         last = (li == 2);
        size_t       vec  = static_cast<size_t>(l.db.size());
        size_t       dw   = static_cast<size_t>(l.dW.size());

        // dW: own class, now PMAD-backed (MAX_SIZE_OF_SIZE_CLASS raised to 401408)
        classes[ci++] = { std::string(names[li]) + ".dW", dw, 0.0, true };

        if (last) {
            classes[ci++] = { std::string(names[li]) + ".{db,a,z,d}", vec * 4, 0.0, true };
        } else {
            classes[ci++] = { std::string(names[li]) + ".{db,a}",     vec * 2, 0.0, true };
            classes[ci++] = { std::string(names[li]) + ".{z,d}",      vec * 2, 0.0, true };
        }
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

    std::printf("PMAD Allocation Plan\n");
    std::printf("Network : %d -> %d -> %d -> %d\n",
                Network::INPUT_SIZE, Network::HIDDEN1,
                Network::HIDDEN2,    Network::OUTPUT_SIZE);
    std::printf("Total   : %zu doubles  (%.2f KB)\n\n", total, total * sizeof(double) / 1024.0);

    std::printf("  %-3s  %-26s  %10s  %9s  %s\n",
                "Cls", "Name", "Elements", "Pct", "Backend");
    std::printf("  %s\n", std::string(62, '-').c_str());
    for (int i = 0; i < 8; ++i) {
        const auto& c = classes[i];
        std::printf("  [%d]  %-26s  %10zu  %8.2f%%  %s\n",
                    i, c.name.c_str(), c.elements, c.percentage,
                    c.pmad_backed ? "PMAD" : "Eigen");
    }
    std::printf("  %s\n", std::string(62, '-').c_str());
}

// ---------- Lifecycle ----------
//
// PMAD size classes — Layer buffers + batch workspace, sorted ascending.
// Batch workspace (BatchWorkspace) is the hot path: 9 blocks allocated once,
// reused every batch → eliminates ~440 KB of Eigen new/delete per batch call.
//
//   Class 0:     208 B — output layer vectors (db,a,z,delta)  ×  4  →  1%
//   Class 1:     512 B — hidden layer vectors                 ×  8  →  1%
//   Class 2:  13,312 B — output.dW  (26×64×8)                ×  1  →  1%
//   Class 3:  26,624 B — batch output mats Z3,A3,D3 (26×128) ×  3  →  9%
//   Class 4:  32,768 B — hidden2.dW (64×64×8)                ×  1  →  3%
//   Class 5:  65,536 B — batch hidden mats Z1,A1,Z2,A2,D1,D2 ×  6  → 42%
//   Class 6: 401,408 B — hidden1.dW (64×784×8)               ×  1  → 43%
//
// 24 allocations total: 15 Layer + 9 BatchWorkspace.

inline void init_network_pmad() {
    static const size_t class_sizes[7] = { 208, 512, 13312, 26624, 32768, 65536, 401408 };
    static const size_t percentages[7] = {   1,   1,     1,     9,     3,    42,      43 };
    static const size_t pool_size      = 2 * 1024 * 1024;  // 2 MB

    PmadStatus st = pmad_init(class_sizes, 7, percentages, pool_size);
    if (st != PMAD_OK)
        throw std::runtime_error("pmad_init failed with status " + std::to_string(st));

    std::printf("PMAD initialised  pool=2MB  "
                "classes=[208B@1%% 512B@1%% 13312B@1%% 26624B@9%% "
                "32768B@3%% 65536B@42%% 401408B@43%%]\n\n");
}

inline void destroy_network_pmad() {
    pmad_destroy();
}

// ---------- Live stats ----------

inline void print_pmad_stats() {
    PmadClassStats stats[7];
    int n = pmad_get_stats(stats, 7);

    std::printf("PMAD live stats:\n");
    for (int i = 0; i < n; ++i) {
        std::printf("  class[%d]  block=%4u B  total=%3u  allocated=%u\n",
                    i, stats[i].block_size,
                    stats[i].total_blocks,
                    stats[i].allocated_blocks);
    }
    std::printf("\n");
}
