#pragma once

#include "dataset.h"
#include "loss.h"
#include "module.h"
#include "network.h"
#include "optim.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <numeric>
#include <random>
#include <vector>

inline double compute_accuracy(Network& net, const ImageDataset& ds,
                               BatchWorkspace& ws, int batch_size) {
    int correct = 0, n = ds.cols();
    const int L = net.num_layers();
    for (int s = 0; s < n; s += batch_size) {
        const int e  = std::min(s + batch_size, n);
        const int bs = e - s;
        if (bs == batch_size) {
            for (int i = 0; i < bs; ++i)
                ds.get_image_col(s + i, ws.Xbuf(0).col(i));
            net.forward_batch(ws.Xbuf(0), ws);
            for (int j = 0; j < bs; ++j) {
                Eigen::Index pred;
                ws.A(L-1).col(j).maxCoeff(&pred);
                if (static_cast<int>(pred) == ds.label(s + j)) ++correct;
            }
        } else {
            Eigen::VectorXf x(ds.image_size());
            for (int i = s; i < e; ++i) {
                ds.get_image_col(i, x);
                Eigen::Index pred;
                net.forward(x).maxCoeff(&pred);
                if (static_cast<int>(pred) == ds.label(i)) ++correct;
            }
        }
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
    const int L = net.num_layers();
    CrossEntropyLoss criterion;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    BatchWorkspace ws(net.sizes, batch_size);

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

        int fill = 1, comp = 0;
        std::vector<int> curr_labels = assemble(comp, 0, std::min(batch_size, n));

        for (int start = 0; start < n; start += batch_size) {
            const int end        = std::min(start + batch_size, n);
            const int bs         = end - start;
            const int next_start = end;
            const bool has_next  = (next_start + batch_size <= n);

            std::future<std::vector<int>> fut;
            if (has_next)
                fut = std::async(std::launch::async,
                                 assemble, fill, next_start, batch_size);

            if (bs == batch_size) {
                net.zero_grad();
                net.forward_batch(ws.Xbuf(comp), ws);
                auto [batch_loss, delta] = criterion(ws.A(L-1), curr_labels);
                net.backward_batch(ws.Xbuf(comp), ws, delta);
                net.update(lr);
                total_loss += double(batch_loss) * bs;
            } else {
                // Partial last batch: per-sample fallback
                net.zero_grad();
                Eigen::VectorXf x(train_set.image_size());
                for (int i = start; i < end; ++i) {
                    const int idx   = indices[i];
                    const int label = train_set.label(idx);
                    train_set.get_image_col(idx, x);
                    auto a = net.forward(x);
                    auto [sample_loss, delta] = criterion(a, label);
                    total_loss += sample_loss;
                    net.backward(x, delta);
                }
                net.scale_grad(1.0f / bs);
                net.update(lr);
            }

            if (has_next) {
                curr_labels = fut.get();
                std::swap(fill, comp);
            }
        }

        double avg_loss  = total_loss / n;
        double train_acc = compute_accuracy(net, train_set, ws, batch_size);
        double test_acc  = compute_accuracy(net, test_set,  ws, batch_size);
        std::printf("%-10d %-12.4f %-14.2f %-12.2f\n",
                    epoch, avg_loss, train_acc, test_acc);
        std::fflush(stdout);
    }
}

// ---- Module-based API ----

inline double compute_accuracy(Module& net, const ImageDataset& ds,
                               int batch_size = 256) {
    int correct = 0, n = ds.cols();
    Eigen::MatrixXf X(ds.image_size(), batch_size);
    for (int s = 0; s < n; s += batch_size) {
        const int e = std::min(s + batch_size, n), bs = e - s;
        if (bs == batch_size) {
            for (int i = 0; i < bs; ++i) ds.get_image_col(s + i, X.col(i));
            Eigen::MatrixXf probs = net.forward(X);
            for (int j = 0; j < bs; ++j) {
                Eigen::Index pred;
                probs.col(j).maxCoeff(&pred);
                if (static_cast<int>(pred) == ds.label(s + j)) ++correct;
            }
        } else {
            Eigen::MatrixXf x(ds.image_size(), 1);
            for (int i = s; i < e; ++i) {
                ds.get_image_col(i, x.col(0));
                Eigen::Index pred;
                net.forward(x).col(0).maxCoeff(&pred);
                if (static_cast<int>(pred) == ds.label(i)) ++correct;
            }
        }
    }
    return 100.0 * correct / n;
}

template<typename Optim>
inline void train_module(Module& net, Optim& optim,
                         const ImageDataset& train_set,
                         const ImageDataset& test_set,
                         int epochs     = 150,
                         int batch_size = 128) {
    const int n = train_set.cols();
    CrossEntropyLoss criterion;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    Eigen::MatrixXf X(train_set.image_size(), batch_size);
    std::vector<int> labels(batch_size);

    std::printf("Training (Module · batched GEMM): "
                "epochs=%d  batch=%d\n\n", epochs, batch_size);
    std::printf("%-10s %-12s %-14s %-12s\n", "Epoch", "Loss", "Train acc", "Test acc");
    std::printf("%s\n", std::string(50, '-').c_str());

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double total_loss = 0.0;
        int n_full = 0;

        for (int s = 0; s < n; s += batch_size) {
            const int e = std::min(s + batch_size, n), bs = e - s;
            if (bs != batch_size) continue;

            for (int i = 0; i < bs; ++i) {
                train_set.get_image_col(indices[s + i], X.col(i));
                labels[i] = train_set.label(indices[s + i]);
            }

            optim.zero_grad(net);
            Eigen::MatrixXf probs = net.forward(X);
            auto [batch_loss, delta] = criterion(probs, labels);
            net.backward(delta);
            optim.step(net);
            total_loss += double(batch_loss) * bs;
            ++n_full;
        }

        double avg_loss  = total_loss / (n_full * batch_size);
        double train_acc = compute_accuracy(net, train_set, batch_size);
        double test_acc  = compute_accuracy(net, test_set,  batch_size);
        std::printf("%-10d %-12.4f %-14.2f %-12.2f\n",
                    epoch, avg_loss, train_acc, test_acc);
        std::fflush(stdout);
        optim.epoch_end();
    }
}

// Backward-compatible overload — defaults to SGD.
inline void train_module(Module& net,
                         const ImageDataset& train_set,
                         const ImageDataset& test_set,
                         int   epochs     = 150,
                         int   batch_size = 128,
                         float lr         = 0.01f) {
    SGD optim(lr);
    train_module(net, optim, train_set, test_set, epochs, batch_size);
}
