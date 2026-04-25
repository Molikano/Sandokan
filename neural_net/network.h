#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

// C bridge: wraps PMAD.h + incPMAD.h (singleton API) for C++ linkage
extern "C" {
#include "plad_bridge.h"
#include "incPMAD.h"
}

// ---------- Activations ----------

inline Eigen::VectorXd relu(const Eigen::VectorXd& z) {
    return z.cwiseMax(0.0);
}

inline Eigen::VectorXd relu_prime(const Eigen::VectorXd& z) {
    return (z.array() > 0.0).cast<double>();
}

inline Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::VectorXd e = (z.array() - z.maxCoeff()).exp();
    return e / e.sum();
}

// Column-wise softmax for batched forward pass
inline Eigen::MatrixXd softmax_cols(const Eigen::MatrixXd& Z) {
    Eigen::RowVectorXd maxv = Z.colwise().maxCoeff();
    Eigen::MatrixXd    e    = (Z.rowwise() - maxv).array().exp();
    return e.array().rowwise() / e.colwise().sum().array();
}

// ---------- BatchCache — plain-Eigen version (used by Eigen bench path) ----------

struct BatchCache {
    Eigen::MatrixXd Z1, A1, Z2, A2, Z3, A3;
};

// ---------- BatchWorkspace — PMAD-backed batch buffers (forward + backward) ----------
//
// Pre-allocated once before the training loop; reused across every batch.
// Eliminates ~440 KB of Eigen new/delete per batch (Z1..A3 + D1..D3).
//
//   Z1,A1,D1,Z2,A2,D2  : [hidden, bs] = 64×128×8 = 65,536 B  → size class 65536
//   Z3,A3,D3            : [out,    bs] = 26×128×8 = 26,624 B  → size class 26624

struct BatchWorkspace {
    double* Z1_raw; Eigen::Map<Eigen::MatrixXd> Z1;
    double* A1_raw; Eigen::Map<Eigen::MatrixXd> A1;
    double* Z2_raw; Eigen::Map<Eigen::MatrixXd> Z2;
    double* A2_raw; Eigen::Map<Eigen::MatrixXd> A2;
    double* Z3_raw; Eigen::Map<Eigen::MatrixXd> Z3;
    double* A3_raw; Eigen::Map<Eigen::MatrixXd> A3;
    double* D3_raw; Eigen::Map<Eigen::MatrixXd> D3;
    double* D2_raw; Eigen::Map<Eigen::MatrixXd> D2;
    double* D1_raw; Eigen::Map<Eigen::MatrixXd> D1;

    BatchWorkspace(int h1, int h2, int out, int bs)
        : Z1_raw(static_cast<double*>(pmad_alloc(h1  * bs * sizeof(double)))),
          Z1(Z1_raw, h1,  bs),
          A1_raw(static_cast<double*>(pmad_alloc(h1  * bs * sizeof(double)))),
          A1(A1_raw, h1,  bs),
          Z2_raw(static_cast<double*>(pmad_alloc(h2  * bs * sizeof(double)))),
          Z2(Z2_raw, h2,  bs),
          A2_raw(static_cast<double*>(pmad_alloc(h2  * bs * sizeof(double)))),
          A2(A2_raw, h2,  bs),
          Z3_raw(static_cast<double*>(pmad_alloc(out * bs * sizeof(double)))),
          Z3(Z3_raw, out, bs),
          A3_raw(static_cast<double*>(pmad_alloc(out * bs * sizeof(double)))),
          A3(A3_raw, out, bs),
          D3_raw(static_cast<double*>(pmad_alloc(out * bs * sizeof(double)))),
          D3(D3_raw, out, bs),
          D2_raw(static_cast<double*>(pmad_alloc(h2  * bs * sizeof(double)))),
          D2(D2_raw, h2,  bs),
          D1_raw(static_cast<double*>(pmad_alloc(h1  * bs * sizeof(double)))),
          D1(D1_raw, h1,  bs)
    {
        if (!Z1_raw || !A1_raw || !Z2_raw || !A2_raw || !Z3_raw || !A3_raw ||
            !D3_raw || !D2_raw || !D1_raw)
            throw std::runtime_error(
                "PMAD BatchWorkspace alloc failed — check pool size and size classes");
    }

    ~BatchWorkspace() {
        pmad_free(Z1_raw); pmad_free(A1_raw);
        pmad_free(Z2_raw); pmad_free(A2_raw);
        pmad_free(Z3_raw); pmad_free(A3_raw);
        pmad_free(D3_raw); pmad_free(D2_raw); pmad_free(D1_raw);
    }

    BatchWorkspace(const BatchWorkspace&)            = delete;
    BatchWorkspace& operator=(const BatchWorkspace&) = delete;
};

// ---------- Layer ----------
//
// All 5 per-layer buffer types are now PMAD-backed.
// Raw pointers must be declared before their Maps (init order = declaration order).
//   W       — weight matrix           (Eigen; read-only during forward, updated via SGD)
//   b       — bias vector             (Eigen; small, updated via SGD)
//   dW      — gradient matrix         (PMAD Map<MatrixXd>)
//   db      — gradient bias vector    (PMAD Map<VectorXd>)
//   a       — activations             (PMAD Map<VectorXd>)
//   z       — pre-activation buffer   (PMAD Map<VectorXd>)
//   delta   — backprop error signal   (PMAD Map<VectorXd>)

struct Layer {
    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    double* dW_raw;
    double* db_raw;
    double* a_raw;
    double* z_raw;
    double* delta_raw;

    Eigen::Map<Eigen::MatrixXd> dW;
    Eigen::Map<Eigen::VectorXd> db;
    Eigen::Map<Eigen::VectorXd> a;
    Eigen::Map<Eigen::VectorXd> z;
    Eigen::Map<Eigen::VectorXd> delta;

    Layer(int in, int out, std::mt19937& rng)
        : W(Eigen::MatrixXd(out, in)),
          b(Eigen::VectorXd::Zero(out)),
          dW_raw   (static_cast<double*>(pmad_alloc(out * in  * sizeof(double)))),
          db_raw   (static_cast<double*>(pmad_alloc(out       * sizeof(double)))),
          a_raw    (static_cast<double*>(pmad_alloc(out       * sizeof(double)))),
          z_raw    (static_cast<double*>(pmad_alloc(out       * sizeof(double)))),
          delta_raw(static_cast<double*>(pmad_alloc(out       * sizeof(double)))),
          dW   (dW_raw,    out, in),
          db   (db_raw,    out),
          a    (a_raw,     out),
          z    (z_raw,     out),
          delta(delta_raw, out)
    {
        if (!dW_raw || !db_raw || !a_raw || !z_raw || !delta_raw)
            throw std::runtime_error(
                "PMAD alloc failed — call init_network_pmad() before constructing Network");

        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in));
        for (int r = 0; r < out; ++r)
            for (int c = 0; c < in; ++c)
                W(r, c) = dist(rng);

        dW.setZero(); db.setZero(); a.setZero(); z.setZero(); delta.setZero();
    }

    ~Layer() {
        pmad_free(dW_raw);
        pmad_free(db_raw);
        pmad_free(a_raw);
        pmad_free(z_raw);
        pmad_free(delta_raw);
    }

    Layer(const Layer&)            = delete;
    Layer& operator=(const Layer&) = delete;

    void zero_grad() { dW.setZero(); db.setZero(); }
};

// ---------- Network: 784 -> 64 -> 64 -> 26 ----------

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

    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        hidden1.z = hidden1.W * x         + hidden1.b;
        hidden1.a = relu(hidden1.z);

        hidden2.z = hidden2.W * hidden1.a + hidden2.b;
        hidden2.a = relu(hidden2.z);

        output.z  = output.W  * hidden2.a + output.b;
        output.a  = softmax(output.z);

        return output.a;
    }

    void backward(const Eigen::VectorXd& x, int label) {
        output.delta = output.a;
        output.delta(label) -= 1.0;

        hidden2.delta = (output.W.transpose() * output.delta)
                            .cwiseProduct(relu_prime(hidden2.z));

        hidden1.delta = (hidden2.W.transpose() * hidden2.delta)
                            .cwiseProduct(relu_prime(hidden1.z));

        output.dW  += output.delta  * hidden2.a.transpose();
        output.db  += output.delta;
        hidden2.dW += hidden2.delta * hidden1.a.transpose();
        hidden2.db += hidden2.delta;
        hidden1.dW += hidden1.delta * x.transpose();
        hidden1.db += hidden1.delta;
    }

    void update(double lr) {
        for (Layer* l : layers) {
            l->W -= lr * l->dW;
            l->b -= lr * l->db;
        }
    }

    void zero_grad() {
        for (Layer* l : layers) l->zero_grad();
    }

    // Batched forward: writes into PMAD-backed workspace, zero heap allocation
    // (one ~26 KB exp intermediate inside softmax is unavoidable with Eigen SIMD)
    void forward_batch(const Eigen::MatrixXd& X, BatchWorkspace& ws) {
        ws.Z1.noalias() = hidden1.W * X;
        ws.Z1.colwise() += hidden1.b;
        ws.A1 = ws.Z1.cwiseMax(0.0);

        ws.Z2.noalias() = hidden2.W * ws.A1;
        ws.Z2.colwise() += hidden2.b;
        ws.A2 = ws.Z2.cwiseMax(0.0);

        ws.Z3.noalias() = output.W * ws.A2;
        ws.Z3.colwise() += output.b;

        // Scalar column loop: zero allocation, avoids Eigen exp temporary
        const int out = ws.A3.rows(), bs = ws.A3.cols();
        for (int j = 0; j < bs; ++j) {
            double m = ws.Z3.col(j).maxCoeff();
            double s = 0.0;
            for (int i = 0; i < out; ++i) { ws.A3(i,j) = std::exp(ws.Z3(i,j) - m); s += ws.A3(i,j); }
            for (int i = 0; i < out; ++i) ws.A3(i,j) /= s;
        }
    }

    // Batched backward: all GEMMs write directly into PMAD Maps via noalias()
    void backward_batch(const Eigen::MatrixXd& X, BatchWorkspace& ws,
                        const std::vector<int>& labels, int bs) {
        ws.D3 = ws.A3;
        for (int j = 0; j < bs; ++j) ws.D3(labels[j], j) -= 1.0;

        ws.D2.noalias() = output.W.transpose()  * ws.D3;
        ws.D2.array() *= (ws.Z2.array() > 0.0).cast<double>();

        ws.D1.noalias() = hidden2.W.transpose() * ws.D2;
        ws.D1.array() *= (ws.Z1.array() > 0.0).cast<double>();

        const double inv_bs = 1.0 / bs;
        output.dW.noalias()  += inv_bs * (ws.D3 * ws.A2.transpose());
        output.db            += inv_bs *  ws.D3.rowwise().sum();
        hidden2.dW.noalias() += inv_bs * (ws.D2 * ws.A1.transpose());
        hidden2.db           += inv_bs *  ws.D2.rowwise().sum();
        hidden1.dW.noalias() += inv_bs * (ws.D1 * X.transpose());
        hidden1.db           += inv_bs *  ws.D1.rowwise().sum();
    }
};
