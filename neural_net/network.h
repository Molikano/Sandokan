#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

extern "C" {
#include "plad_bridge.h"
#include "incPMAD.h"
}

// ---------- Activations (float) ----------

inline Eigen::VectorXf relu(const Eigen::VectorXf& z) {
    return z.cwiseMax(0.0f);
}
inline Eigen::VectorXf relu_prime(const Eigen::VectorXf& z) {
    return (z.array() > 0.0f).cast<float>();
}
inline Eigen::VectorXf softmax(const Eigen::VectorXf& z) {
    Eigen::VectorXf e = (z.array() - z.maxCoeff()).exp();
    return e / e.sum();
}

// Column-wise softmax for Eigen-batched path (used by bench NetworkEigen only)
inline Eigen::MatrixXf softmax_cols(const Eigen::MatrixXf& Z) {
    Eigen::RowVectorXf maxv = Z.colwise().maxCoeff();
    Eigen::MatrixXf    e    = (Z.rowwise() - maxv).array().exp();
    return e.array().rowwise() / e.colwise().sum().array();
}

// ---------- BatchCache — plain-Eigen version (bench Eigen path only) ----------

struct BatchCache {
    Eigen::MatrixXf Z1, A1, Z2, A2, Z3, A3;
};

// ---------- BatchWorkspace — PMAD-backed computation + input buffers ----------
//
// All intermediates pre-allocated once, reused every batch — zero heap allocs
// on the hot path. X is double-buffered so assembly and compute can overlap.
//
// float sizes @ bs=128:
//   X0, X1     : [784, 128] × 4 = 401,408 B  → PMAD class 401408  (2 blocks)
//   Z1,A1,D1,
//   Z2,A2,D2   : [ 64, 128] × 4 =  32,768 B  → PMAD class  32768  (6 blocks)
//   Z3,A3,D3   : [ 26, 128] × 4 =  13,312 B  → PMAD class  13312  (3 blocks)

struct BatchWorkspace {
    // Double-buffered input X — assembly thread fills Xbuf(fill) while
    // compute thread reads Xbuf(comp); roles swapped each batch.
    float*                      X0_raw; Eigen::Map<Eigen::MatrixXf> X0;
    float*                      X1_raw; Eigen::Map<Eigen::MatrixXf> X1;

    float*                      Z1_raw; Eigen::Map<Eigen::MatrixXf> Z1;
    float*                      A1_raw; Eigen::Map<Eigen::MatrixXf> A1;
    float*                      Z2_raw; Eigen::Map<Eigen::MatrixXf> Z2;
    float*                      A2_raw; Eigen::Map<Eigen::MatrixXf> A2;
    float*                      Z3_raw; Eigen::Map<Eigen::MatrixXf> Z3;
    float*                      A3_raw; Eigen::Map<Eigen::MatrixXf> A3;
    float*                      D3_raw; Eigen::Map<Eigen::MatrixXf> D3;
    float*                      D2_raw; Eigen::Map<Eigen::MatrixXf> D2;
    float*                      D1_raw; Eigen::Map<Eigen::MatrixXf> D1;

    BatchWorkspace(int in, int h1, int h2, int out, int bs)
        : X0_raw(static_cast<float*>(pmad_alloc(in  * bs * sizeof(float)))),
          X0(X0_raw, in,  bs),
          X1_raw(static_cast<float*>(pmad_alloc(in  * bs * sizeof(float)))),
          X1(X1_raw, in,  bs),
          Z1_raw(static_cast<float*>(pmad_alloc(h1  * bs * sizeof(float)))),
          Z1(Z1_raw, h1,  bs),
          A1_raw(static_cast<float*>(pmad_alloc(h1  * bs * sizeof(float)))),
          A1(A1_raw, h1,  bs),
          Z2_raw(static_cast<float*>(pmad_alloc(h2  * bs * sizeof(float)))),
          Z2(Z2_raw, h2,  bs),
          A2_raw(static_cast<float*>(pmad_alloc(h2  * bs * sizeof(float)))),
          A2(A2_raw, h2,  bs),
          Z3_raw(static_cast<float*>(pmad_alloc(out * bs * sizeof(float)))),
          Z3(Z3_raw, out, bs),
          A3_raw(static_cast<float*>(pmad_alloc(out * bs * sizeof(float)))),
          A3(A3_raw, out, bs),
          D3_raw(static_cast<float*>(pmad_alloc(out * bs * sizeof(float)))),
          D3(D3_raw, out, bs),
          D2_raw(static_cast<float*>(pmad_alloc(h2  * bs * sizeof(float)))),
          D2(D2_raw, h2,  bs),
          D1_raw(static_cast<float*>(pmad_alloc(h1  * bs * sizeof(float)))),
          D1(D1_raw, h1,  bs)
    {
        if (!X0_raw || !X1_raw || !Z1_raw || !A1_raw || !Z2_raw || !A2_raw ||
            !Z3_raw || !A3_raw || !D3_raw || !D2_raw || !D1_raw)
            throw std::runtime_error(
                "PMAD BatchWorkspace alloc failed — check pool size and size classes");
    }

    ~BatchWorkspace() {
        pmad_free(X0_raw); pmad_free(X1_raw);
        pmad_free(Z1_raw); pmad_free(A1_raw);
        pmad_free(Z2_raw); pmad_free(A2_raw);
        pmad_free(Z3_raw); pmad_free(A3_raw);
        pmad_free(D3_raw); pmad_free(D2_raw); pmad_free(D1_raw);
    }

    BatchWorkspace(const BatchWorkspace&)            = delete;
    BatchWorkspace& operator=(const BatchWorkspace&) = delete;

    Eigen::Map<Eigen::MatrixXf>&       Xbuf(int i)       { return i == 0 ? X0 : X1; }
    const Eigen::Map<Eigen::MatrixXf>& Xbuf(int i) const { return i == 0 ? X0 : X1; }
};

// ---------- Layer ----------
//
// PMAD backs only dW and db — the gradient buffers written every batch in
// backward_batch() and read in update(). The a/z/delta single-sample buffers
// are removed: the batched path uses BatchWorkspace; the single-sample path
// (compute_accuracy / partial-batch fallback) uses local VectorXf on the stack.

struct Layer {
    Eigen::MatrixXf W;
    Eigen::VectorXf b;

    float*                      dW_raw;
    float*                      db_raw;
    Eigen::Map<Eigen::MatrixXf> dW;
    Eigen::Map<Eigen::VectorXf> db;

    Layer(int in, int out, std::mt19937& rng)
        : W(Eigen::MatrixXf(out, in)),
          b(Eigen::VectorXf::Zero(out)),
          dW_raw(static_cast<float*>(pmad_alloc(out * in * sizeof(float)))),
          db_raw(static_cast<float*>(pmad_alloc(out      * sizeof(float)))),
          dW(dW_raw, out, in),
          db(db_raw, out)
    {
        if (!dW_raw || !db_raw)
            throw std::runtime_error(
                "PMAD Layer alloc failed — call init_network_pmad() before Network");

        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in));
        for (int r = 0; r < out; ++r)
            for (int c = 0; c < in; ++c)
                W(r, c) = dist(rng);

        dW.setZero(); db.setZero();
    }

    ~Layer() { pmad_free(dW_raw); pmad_free(db_raw); }

    Layer(const Layer&)            = delete;
    Layer& operator=(const Layer&) = delete;

    void zero_grad() { dW.setZero(); db.setZero(); }
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

    // Single-sample forward — local VectorXf, no PMAD maps needed
    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        Eigen::VectorXf a1 = relu(hidden1.W * x  + hidden1.b);
        Eigen::VectorXf a2 = relu(hidden2.W * a1 + hidden2.b);
        return softmax(output.W * a2 + output.b);
    }

    // Single-sample backward — recomputes activations locally.
    // Only called for the partial last batch (when n % batch_size != 0).
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
