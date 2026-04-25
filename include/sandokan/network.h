#pragma once

#include "layer.h"
#include "workspace.h"
#include "ops.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

// ---------- Network: 784 → 64 → 64 → 26 ----------

struct Network {
    static constexpr int INPUT_SIZE  = 784;
    static constexpr int HIDDEN1     = 64;
    static constexpr int HIDDEN2     = 64;
    static constexpr int OUTPUT_SIZE = 26;

    std::mt19937 rng { 42 };

    Layer hidden1 { INPUT_SIZE, HIDDEN1,     rng };
    Layer hidden2 { HIDDEN1,    HIDDEN2,     rng };
    Layer output  { HIDDEN2,    OUTPUT_SIZE, rng };

    std::vector<Layer*> layers { &hidden1, &hidden2, &output };

    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        Eigen::VectorXf a1 = relu(hidden1.W * x  + hidden1.b);
        Eigen::VectorXf a2 = relu(hidden2.W * a1 + hidden2.b);
        return softmax(output.W * a2 + output.b);
    }

    void backward(const Eigen::Ref<const Eigen::VectorXf>& x, int label) {
        Eigen::VectorXf z1 = hidden1.W * x  + hidden1.b;
        Eigen::VectorXf a1 = relu(z1);
        Eigen::VectorXf z2 = hidden2.W * a1 + hidden2.b;
        Eigen::VectorXf a2 = relu(z2);
        Eigen::VectorXf a3 = softmax(output.W * a2 + output.b);

        Eigen::VectorXf d3 = a3; d3(label) -= 1.0f;
        Eigen::VectorXf d2 = (output.W.transpose()  * d3).cwiseProduct(relu_prime(z2));
        Eigen::VectorXf d1 = (hidden2.W.transpose() * d2).cwiseProduct(relu_prime(z1));

        output.dW.noalias()  += d3 * a2.transpose();
        output.db            += d3;
        hidden2.dW.noalias() += d2 * a1.transpose();
        hidden2.db           += d2;
        hidden1.dW.noalias() += d1 * x.transpose();
        hidden1.db           += d1;
    }

    void update(float lr) {
        for (Layer* l : layers) { l->W -= lr * l->dW; l->b -= lr * l->db; }
    }

    void zero_grad() { for (Layer* l : layers) l->zero_grad(); }

    // Batched forward — all GEMMs written directly into PMAD-backed workspace
    void forward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                       BatchWorkspace& ws) {
        ws.Z1.noalias() = hidden1.W * X;
        ws.Z1.colwise() += hidden1.b;
        ws.A1 = ws.Z1.cwiseMax(0.0f);

        ws.Z2.noalias() = hidden2.W * ws.A1;
        ws.Z2.colwise() += hidden2.b;
        ws.A2 = ws.Z2.cwiseMax(0.0f);

        ws.Z3.noalias() = output.W * ws.A2;
        ws.Z3.colwise() += output.b;

        // Scalar column loop: zero allocation, avoids Eigen exp temporary
        const int out = ws.A3.rows(), bs = ws.A3.cols();
        for (int j = 0; j < bs; ++j) {
            float m = ws.Z3.col(j).maxCoeff(), s = 0.0f;
            for (int i = 0; i < out; ++i) { ws.A3(i,j) = std::exp(ws.Z3(i,j)-m); s += ws.A3(i,j); }
            for (int i = 0; i < out; ++i) ws.A3(i,j) /= s;
        }
    }

    // Batched backward — all GEMMs write into PMAD Maps via noalias()
    void backward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        BatchWorkspace& ws,
                        const std::vector<int>& labels, int bs) {
        ws.D3 = ws.A3;
        for (int j = 0; j < bs; ++j) ws.D3(labels[j], j) -= 1.0f;

        ws.D2.noalias() = output.W.transpose()  * ws.D3;
        ws.D2.array()  *= (ws.Z2.array() > 0.0f).cast<float>();

        ws.D1.noalias() = hidden2.W.transpose() * ws.D2;
        ws.D1.array()  *= (ws.Z1.array() > 0.0f).cast<float>();

        const float inv_bs = 1.0f / bs;
        output.dW.noalias()  += inv_bs * (ws.D3 * ws.A2.transpose());
        output.db            += inv_bs *  ws.D3.rowwise().sum();
        hidden2.dW.noalias() += inv_bs * (ws.D2 * ws.A1.transpose());
        hidden2.db           += inv_bs *  ws.D2.rowwise().sum();
        hidden1.dW.noalias() += inv_bs * (ws.D1 * X.transpose());
        hidden1.db           += inv_bs *  ws.D1.rowwise().sum();
    }
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
