#pragma once

#include "allocator.h"
#include <Eigen/Dense>
#include <random>
#include <stdexcept>

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
