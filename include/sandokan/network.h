#pragma once

#include "layer.h"
#include "ops.h"
#include "workspace.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

enum class Activation { Linear, ReLU, Sigmoid, Softmax };

// ---------- Network — arbitrary depth, runtime-configured ----------
//
// sizes: full layer widths including input and output, e.g. {784, 64, 64, 26}.
// acts:  activation per weight layer; defaults to ReLU for hidden, Softmax for output.
//
// backward() and backward_batch() take output_delta = dL/dZ_output, the
// pre-activation gradient produced by the loss function. This decouples the
// loss from the network and allows swapping CrossEntropyLoss / BCELoss / MSELoss
// without touching the network.
//
// Hidden-layer activation derivatives are applied correctly per acts[i]:
//   ReLU    → d = (a > 0)
//   Sigmoid → d = a * (1 - a)
//   Linear  → d = 1  (no-op)
// Softmax at a hidden layer is not supported in the backward pass.

struct Network {
    std::vector<int>                    sizes;
    std::vector<Activation>             acts;
    std::mt19937                        rng { 42 };
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<Eigen::VectorXf>        as_;   // activation cache for single-sample path

    explicit Network(std::vector<int>       layer_sizes,
                     std::vector<Activation> activations = {})
        : sizes(std::move(layer_sizes))
    {
        const int L = (int)sizes.size() - 1;
        if (activations.empty()) {
            acts.assign(L, Activation::ReLU);
            acts.back() = Activation::Softmax;
        } else {
            acts = std::move(activations);
        }
        for (int i = 0; i < L; ++i)
            layers.push_back(std::make_unique<Layer>(sizes[i], sizes[i+1], rng));
        as_.resize(L + 1);
        for (int i = 0; i <= L; ++i)
            as_[i] = Eigen::VectorXf(sizes[i]);
    }

    int input_size()  const { return sizes.front(); }
    int output_size() const { return sizes.back();  }
    int num_layers()  const { return (int)layers.size(); }

    // Single-sample forward — caches activations in as_ for use by backward().
    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        as_[0] = x;
        for (int i = 0; i < num_layers(); ++i) {
            Eigen::VectorXf z = layers[i]->W * as_[i] + layers[i]->b;
            switch (acts[i]) {
                case Activation::Linear:  as_[i+1] = z;          break;
                case Activation::ReLU:    as_[i+1] = relu(z);    break;
                case Activation::Sigmoid: as_[i+1] = sigmoid(z); break;
                case Activation::Softmax: as_[i+1] = softmax(z); break;
            }
        }
        return as_.back();
    }

    // Single-sample backward — uses as_ cached by the preceding forward() call.
    // output_delta = dL/dZ_output from the loss function (not divided by batch size).
    void backward(const Eigen::Ref<const Eigen::VectorXf>& /*x*/,
                  const Eigen::Ref<const Eigen::VectorXf>& output_delta) {
        const int L = num_layers();
        std::vector<Eigen::VectorXf> deltas(L);
        deltas[L-1] = output_delta;
        for (int i = L - 2; i >= 0; --i) {
            deltas[i] = layers[i+1]->W.transpose() * deltas[i+1];
            switch (acts[i]) {
                case Activation::Linear:  break;
                case Activation::ReLU:    deltas[i].array() *= (as_[i+1].array() > 0.0f).cast<float>(); break;
                case Activation::Sigmoid: deltas[i].array() *= as_[i+1].array() * (1.0f - as_[i+1].array()); break;
                case Activation::Softmax: break;
            }
        }
        for (int i = 0; i < L; ++i) {
            layers[i]->dW.noalias() += deltas[i] * as_[i].transpose();
            layers[i]->db           += deltas[i];
        }
    }

    void update(float lr) {
        for (auto& l : layers) { l->W -= lr * l->dW; l->b -= lr * l->db; }
    }

    void zero_grad()    { for (auto& l : layers) l->zero_grad(); }
    void scale_grad(float s) { for (auto& l : layers) { l->dW *= s; l->db *= s; } }

    // Batched forward — all GEMMs write directly into PMAD-backed workspace
    void forward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                       BatchWorkspace& ws) {
        const int L = num_layers();
        for (int i = 0; i < L; ++i) {
            if (i == 0) ws.Z(i).noalias() = layers[i]->W * X;
            else        ws.Z(i).noalias() = layers[i]->W * ws.A(i-1);
            ws.Z(i).colwise() += layers[i]->b;

            switch (acts[i]) {
                case Activation::Linear:
                    ws.A(i) = ws.Z(i);
                    break;
                case Activation::ReLU:
                    ws.A(i) = ws.Z(i).cwiseMax(0.0f);
                    break;
                case Activation::Sigmoid:
                    ws.A(i) = (1.0f / (1.0f + (-ws.Z(i).array()).exp())).matrix();
                    break;
                case Activation::Softmax: {
                    const int rows = ws.A(i).rows(), cols = ws.A(i).cols();
                    for (int j = 0; j < cols; ++j) {
                        float m = ws.Z(i).col(j).maxCoeff(), s = 0.0f;
                        for (int r = 0; r < rows; ++r) { ws.A(i)(r,j) = std::exp(ws.Z(i)(r,j)-m); s += ws.A(i)(r,j); }
                        for (int r = 0; r < rows; ++r) ws.A(i)(r,j) /= s;
                    }
                    break;
                }
            }
        }
    }

    // Batched backward.
    // output_delta = dL/dZ_output [out × bs], already divided by bs, from the loss function.
    void backward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        BatchWorkspace& ws,
                        const Eigen::Ref<const Eigen::MatrixXf>& output_delta) {
        const int L = num_layers();
        ws.D(L-1) = output_delta;

        for (int i = L - 2; i >= 0; --i) {
            ws.D(i).noalias() = layers[i+1]->W.transpose() * ws.D(i+1);
            switch (acts[i]) {
                case Activation::Linear:  break;
                case Activation::ReLU:    ws.D(i).array() *= (ws.A(i).array() > 0.0f).cast<float>(); break;
                case Activation::Sigmoid: ws.D(i).array() *= ws.A(i).array() * (1.0f - ws.A(i).array()); break;
                case Activation::Softmax: break;
            }
        }

        // output_delta is already divided by bs, so sum (not mean) across batch
        for (int i = 0; i < L; ++i) {
            if (i == 0) layers[i]->dW.noalias() += ws.D(i) * X.transpose();
            else        layers[i]->dW.noalias() += ws.D(i) * ws.A(i-1).transpose();
            layers[i]->db += ws.D(i).rowwise().sum();
        }
    }
};

inline void print_pmad_plan(const Network& net) {
    const int L = net.num_layers();
    size_t total = 0;
    for (int i = 0; i < L; ++i)
        total += size_t(net.layers[i]->dW.size() + net.layers[i]->db.size());

    std::printf("PMAD gradient plan — architecture:");
    for (int s : net.sizes) std::printf(" %d", s);
    std::printf("\nGradient buffers: %zu floats  (%.2f KB)\n\n", total, total * sizeof(float) / 1024.0);

    std::printf("  %-4s  %-6s  %10s\n", "Lyr", "Buffer", "Elements");
    std::printf("  %s\n", std::string(28, '-').c_str());
    for (int i = 0; i < L; ++i) {
        std::printf("  [%d]   dW     %10d\n", i, (int)net.layers[i]->dW.size());
        std::printf("  [%d]   db     %10d\n", i, (int)net.layers[i]->db.size());
    }
    std::printf("  %s\n", std::string(28, '-').c_str());
}
