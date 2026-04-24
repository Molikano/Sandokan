#pragma once

#include "dataloader.h"
#include "network.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

inline double cross_entropy(const Eigen::VectorXd& probs, int label) {
    return -std::log(probs(label) + 1e-12);
}

inline double compute_accuracy(Network& net, const ImageDataset& ds) {
    int correct = 0;
    for (size_t i = 0; i < ds.images.size(); ++i) {
        Eigen::VectorXd probs = net.forward(ds.images[i]);
        Eigen::Index pred;
        probs.maxCoeff(&pred);
        if (static_cast<int>(pred) == ds.labels[i]) ++correct;
    }
    return 100.0 * correct / static_cast<double>(ds.images.size());
}

inline void train(Network& net,
                  const ImageDataset& train_set,
                  const ImageDataset& test_set,
                  int    epochs     = 150,
                  int    batch_size = 128,
                  double lr         = 0.01) {
    int n = static_cast<int>(train_set.images.size());

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    std::printf("Training: epochs=%d  batch=%d  lr=%.4f\n\n", epochs, batch_size, lr);
    std::printf("%-10s %-12s %-14s %-12s\n", "Epoch", "Loss", "Train acc", "Test acc");
    std::printf("%s\n", std::string(50, '-').c_str());

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double total_loss = 0.0;

        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int bs  = end - start;

            net.zero_grad();

            for (int i = start; i < end; ++i) {
                int idx = indices[i];
                Eigen::VectorXd probs = net.forward(train_set.images[idx]);
                total_loss += cross_entropy(probs, train_set.labels[idx]);
                net.backward(train_set.images[idx], train_set.labels[idx]);
            }

            // Average gradients over batch before SGD step
            for (Layer* l : net.layers) {
                l->dW /= bs;
                l->db /= bs;
            }

            net.update(lr);
        }

        double avg_loss  = total_loss / n;
        double train_acc = compute_accuracy(net, train_set);
        double test_acc  = compute_accuracy(net, test_set);
        std::printf("%-10d %-12.4f %-14.2f %-12.2f\n",
                    epoch, avg_loss, train_acc, test_acc);
        std::fflush(stdout);
    }
}
