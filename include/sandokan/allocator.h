#pragma once

#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "plad_bridge.h"
#include "incPMAD.h"
}

// Thread-local registry used by Linear to self-report sizes before PMAD exists.
namespace detail {
    struct PendingLinear {
        int out, in;
        std::function<void()> migrate; // called once by init_pmad_for()
    };
    inline std::vector<PendingLinear>& pending_linears() {
        static thread_local std::vector<PendingLinear> v;
        return v;
    }
    inline void register_pending(int out, int in, std::function<void()> fn) {
        pending_linears().push_back({ out, in, std::move(fn) });
    }

    // Shared PMAD initialisation from a bytes→count map.
    inline void pmad_init_impl(const std::map<size_t, size_t>& bufs) {
        const int n = (int)bufs.size();
        std::vector<size_t> class_sizes, counts;
        class_sizes.reserve(n); counts.reserve(n);
        for (auto& [sz, cnt] : bufs) { class_sizes.push_back(sz); counts.push_back(cnt); }

        size_t total = 0;
        for (int i = 0; i < n; ++i) total += class_sizes[i] * counts[i];

        if (n > 100) throw std::runtime_error("Too many PMAD size classes (> 100)");

        const size_t to_dist = size_t(100 - n);
        std::vector<double> raw(n);
        for (int i = 0; i < n; ++i)
            raw[i] = double(to_dist) * double(class_sizes[i] * counts[i]) / double(total);

        std::vector<size_t> extra(n); size_t extra_sum = 0;
        for (int i = 0; i < n; ++i) { extra[i] = size_t(raw[i]); extra_sum += extra[i]; }

        size_t remainder = to_dist - extra_sum;
        while (remainder > 0) {
            int best = 0; double best_frac = raw[0] - double(extra[0]);
            for (int i = 1; i < n; ++i) {
                double frac = raw[i] - double(extra[i]);
                if (frac > best_frac) { best_frac = frac; best = i; }
            }
            extra[best]++; remainder--;
        }

        std::vector<size_t> pcts(n);
        for (int i = 0; i < n; ++i) pcts[i] = 1 + extra[i];

        const size_t max_class = class_sizes.back();
        if (max_class > MAX_SIZE_OF_SIZE_CLASS)
            throw std::runtime_error(
                "Largest buffer (" + std::to_string(max_class) +
                " B) exceeds PMAD MAX_SIZE_OF_SIZE_CLASS (" +
                std::to_string(MAX_SIZE_OF_SIZE_CLASS) + " B).");

        const size_t pool = std::max(total * 2, size_t(512 * 1024));

        PmadStatus st = pmad_init(class_sizes.data(), n, pcts.data(), pool);
        if (st != PMAD_OK)
            throw std::runtime_error("pmad_init failed: " + std::to_string(st));

        std::printf("PMAD initialised  pool=%zuKB  %d classes\n", pool / 1024, n);
        for (int i = 0; i < n; ++i)
            std::printf("  class[%d]  %7zuB  count=%zu  pct=%zu%%\n",
                        i, class_sizes[i], counts[i], pcts[i]);
        std::printf("\n");
    }
} // namespace detail

// Classic arch-vector variant — for the Network / BatchWorkspace API.
// Sizes the pool for dW, db, Z/A/D activations, and X0/X1 data buffers.
inline void init_pmad(const std::vector<int>& sizes, int batch_size) {
    const int L = (int)sizes.size() - 1;
    std::map<size_t, size_t> bufs;
    for (int i = 0; i < L; ++i) {
        bufs[size_t(sizes[i+1]) * sizes[i]   * sizeof(float)]++;
        bufs[size_t(sizes[i+1])              * sizeof(float)]++;
        bufs[size_t(sizes[i+1]) * batch_size * sizeof(float)] += 3;
    }
    bufs[size_t(sizes[0]) * batch_size * sizeof(float)] += 2;
    detail::pmad_init_impl(bufs);
}

// Module-API variant — derives sizes automatically from registered Linear layers.
// Call AFTER constructing the network, BEFORE training.
// No batch_size needed: the Module API uses plain Eigen for activation caches.
inline void init_pmad_for() {
    auto& pending = detail::pending_linears();
    if (pending.empty())
        throw std::runtime_error(
            "init_pmad_for: no Linear layers registered — construct your network first");

    std::map<size_t, size_t> bufs;
    for (auto& p : pending) {
        bufs[size_t(p.out) * p.in * sizeof(float)]++; // dW
        bufs[size_t(p.out)        * sizeof(float)]++; // db
    }
    detail::pmad_init_impl(bufs);

    for (auto& p : pending) p.migrate();
    pending.clear();
}

inline void destroy_pmad() {
    detail::pending_linears().clear(); // drop any un-migrated registrations
    pmad_destroy();
}

inline void print_pmad_stats() {
    PmadClassStats stats[16];
    int n = pmad_get_stats(stats, 16);
    std::printf("PMAD live stats:\n");
    for (int i = 0; i < n; ++i)
        std::printf("  class[%d]  block=%6u B  total=%3u  allocated=%u\n",
                    i, stats[i].block_size, stats[i].total_blocks,
                    stats[i].allocated_blocks);
    std::printf("\n");
}
