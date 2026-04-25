#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

// Loss functions return { mean_loss_per_sample, output_delta }.
//
// output_delta = dL/dZ_output — the pre-activation gradient of the last layer,
// already normalised by batch size. Pass it directly to net.backward_batch().
//
// Standard pairings:
//   CrossEntropyLoss  ← Softmax  output activation
//   BCELoss           ← Sigmoid  output activation
//   MSELoss           ← Linear   output activation

// ---------- Cross-Entropy ----------

struct CrossEntropyLoss {
    // Batched: probs = softmax output [out × bs], labels = class indices
    std::pair<float, Eigen::MatrixXf>
    operator()(const Eigen::MatrixXf& probs,
               const std::vector<int>& labels) const {
        const int bs = (int)probs.cols();
        Eigen::MatrixXf delta = probs / float(bs);
        float loss = 0.0f;
        for (int j = 0; j < bs; ++j) {
            loss -= std::log(std::max(probs(labels[j], j), 1e-7f));
            delta(labels[j], j) -= 1.0f / float(bs);
        }
        return { loss / bs, std::move(delta) };
    }

    // Single-sample: a = softmax output, label = class index
    std::pair<float, Eigen::VectorXf>
    operator()(const Eigen::VectorXf& a, int label) const {
        float loss = -std::log(std::max(a(label), 1e-7f));
        Eigen::VectorXf delta = a;
        delta(label) -= 1.0f;
        return { loss, std::move(delta) };
    }
};

// ---------- Binary Cross-Entropy ----------

struct BCELoss {
    // Batched: probs = sigmoid output [out × bs], targets ∈ {0, 1} [out × bs]
    std::pair<float, Eigen::MatrixXf>
    operator()(const Eigen::MatrixXf& probs,
               const Eigen::MatrixXf& targets) const {
        const int bs = (int)probs.cols();
        float loss = -(targets.array()       * probs.array().max(1e-7f).log() +
                      (1.0f - targets.array()) * (1.0f - probs.array()).max(1e-7f).log()
                      ).sum() / bs;
        Eigen::MatrixXf delta = (probs - targets) / float(bs);
        return { loss, std::move(delta) };
    }

    // Single-sample: a = sigmoid output, y ∈ {0, 1} per output neuron
    std::pair<float, Eigen::VectorXf>
    operator()(const Eigen::VectorXf& a, const Eigen::VectorXf& y) const {
        float loss = -(y.array()       * a.array().max(1e-7f).log() +
                      (1.0f - y.array()) * (1.0f - a.array()).max(1e-7f).log()
                      ).sum();
        return { loss, (a - y) };
    }
};

// ---------- Mean Squared Error ----------

struct MSELoss {
    // Batched: output = linear output [out × bs], targets = float targets [out × bs]
    std::pair<float, Eigen::MatrixXf>
    operator()(const Eigen::MatrixXf& output,
               const Eigen::MatrixXf& targets) const {
        const int bs = (int)output.cols();
        Eigen::MatrixXf diff = output - targets;
        float loss = diff.squaredNorm() / (2.0f * bs);
        return { loss, diff / float(bs) };
    }

    // Single-sample
    std::pair<float, Eigen::VectorXf>
    operator()(const Eigen::VectorXf& output, const Eigen::VectorXf& target) const {
        Eigen::VectorXf diff = output - target;
        return { 0.5f * diff.squaredNorm(), diff };
    }
};
