#pragma once

#include "dataset.h"
#include "network.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <numeric>
#include <random>
#include <vector>

inline double compute_accuracy(Network& net, const ImageDataset& ds) {
    int correct = 0, n = ds.cols();
    Eigen::VectorXf x(ds.image_size());
    for (int i = 0; i < n; ++i) {
        ds.get_image_col(i, x);
        Eigen::VectorXf probs = net.forward(x);
        Eigen::Index    pred;
        probs.maxCoeff(&pred);
        if (static_cast<int>(pred) == ds.label(i)) ++correct;
    }
    return 100.0 * correct / n;
}

inline void train_batched(Network& net,
                          const ImageDataset& train_set,
                          const ImageDataset& test_set,
                          int   epochs     = 150,
                          int   batch_size = 128,
                          float lr         = 0.01f) {
    const int n = train_set.cols();

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    // Pre-allocate all workspace buffers (X double-buffered, Z/A/D fixed)
    BatchWorkspace ws(Network::INPUT_SIZE, Network::HIDDEN1, Network::HIDDEN2,
                      Network::OUTPUT_SIZE, batch_size);

    // Gathers batch [start, start+bs) into ws.Xbuf(buf_idx) and returns labels.
    // Called on a background thread while the previous batch is computing.
    auto assemble = [&](int buf_idx, int start, int bs) -> std::vector<int> {
        auto& Xb = ws.Xbuf(buf_idx);
        std::vector<int> lbls(bs);
        for (int i = 0; i < bs; ++i) {
            const int idx = indices[start + i];
            train_set.get_image_col(idx, Xb.col(i));
            lbls[i] = train_set.label(idx);
        }
        return lbls;
    };

    std::printf("Training (float · batched GEMM · parallel assembly): "
                "epochs=%d  batch=%d  lr=%.4f\n\n", epochs, batch_size, lr);
    std::printf("%-10s %-12s %-14s %-12s\n", "Epoch", "Loss", "Train acc", "Test acc");
    std::printf("%s\n", std::string(50, '-').c_str());

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double total_loss = 0.0;

        // comp=0 holds the batch ready to compute; fill=1 is assembled next.
        int fill = 1, comp = 0;
        std::vector<int> curr_labels = assemble(comp, 0, std::min(batch_size, n));

        for (int start = 0; start < n; start += batch_size) {
            const int end        = std::min(start + batch_size, n);
            const int bs         = end - start;
            const int next_start = end;
            const bool has_next  = (next_start + batch_size <= n);

            // Kick off assembly of the next full batch on a background thread
            std::future<std::vector<int>> fut;
            if (has_next)
                fut = std::async(std::launch::async,
                                 assemble, fill, next_start, batch_size);

            if (bs == batch_size) {
                net.zero_grad();
                net.forward_batch(ws.Xbuf(comp), ws);
                net.backward_batch(ws.Xbuf(comp), ws, curr_labels, bs);
                net.update(lr);
                for (int j = 0; j < bs; ++j)
                    total_loss -= std::log(ws.A3(curr_labels[j], j) + 1e-7f);
            } else {
                // Partial last batch: per-sample fallback
                net.zero_grad();
                Eigen::VectorXf x(train_set.image_size());
                for (int i = start; i < end; ++i) {
                    const int idx   = indices[i];
                    const int label = train_set.label(idx);
                    train_set.get_image_col(idx, x);
                    auto probs = net.forward(x);
                    total_loss -= std::log(probs(label) + 1e-7f);
                    net.backward(x, label);
                }
                for (Layer* l : net.layers) { l->dW /= bs; l->db /= bs; }
                net.update(lr);
            }

            // Overlap resolved: swap buffers and grab assembled labels
            if (has_next) {
                curr_labels = fut.get();
                std::swap(fill, comp);
            }
        }

        double avg_loss  = total_loss / n;
        double train_acc = compute_accuracy(net, train_set);
        double test_acc  = compute_accuracy(net, test_set);
        std::printf("%-10d %-12.4f %-14.2f %-12.2f\n",
                    epoch, avg_loss, train_acc, test_acc);
        std::fflush(stdout);
    }
}
