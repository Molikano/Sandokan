#pragma once

#include "allocator.h"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <vector>

// BatchCache — plain-Eigen version (bench Eigen path only)
struct BatchCache {
    Eigen::MatrixXf Z1, A1, Z2, A2, Z3, A3;
};

// BatchWorkspace — PMAD-backed computation + input buffers
//
// Parameterized by the full architecture sizes vector and batch_size.
// Allocates:
//   X0, X1      : [sizes[0],    bs] × 2  — double-buffered input
//   Z[i], A[i], D[i] : [sizes[i+1], bs] × 3L — one triple per layer

struct BatchWorkspace {
    struct LayerBufs {
        float* Z_raw; Eigen::Map<Eigen::MatrixXf> Z;
        float* A_raw; Eigen::Map<Eigen::MatrixXf> A;
        float* D_raw; Eigen::Map<Eigen::MatrixXf> D;

        LayerBufs(int rows, int bs)
            : Z_raw(static_cast<float*>(pmad_alloc(rows * bs * sizeof(float)))),
              Z(Z_raw, rows, bs),
              A_raw(static_cast<float*>(pmad_alloc(rows * bs * sizeof(float)))),
              A(A_raw, rows, bs),
              D_raw(static_cast<float*>(pmad_alloc(rows * bs * sizeof(float)))),
              D(D_raw, rows, bs)
        {
            if (!Z_raw || !A_raw || !D_raw)
                throw std::runtime_error("PMAD BatchWorkspace layer alloc failed");
        }
        ~LayerBufs() { pmad_free(Z_raw); pmad_free(A_raw); pmad_free(D_raw); }
        LayerBufs(const LayerBufs&)            = delete;
        LayerBufs& operator=(const LayerBufs&) = delete;
    };

    float*                      X0_raw; Eigen::Map<Eigen::MatrixXf> X0;
    float*                      X1_raw; Eigen::Map<Eigen::MatrixXf> X1;
    std::vector<std::unique_ptr<LayerBufs>> layer_bufs;

    BatchWorkspace(const std::vector<int>& sizes, int bs)
        : X0_raw(static_cast<float*>(pmad_alloc(sizes[0] * bs * sizeof(float)))),
          X0(X0_raw, sizes[0], bs),
          X1_raw(static_cast<float*>(pmad_alloc(sizes[0] * bs * sizeof(float)))),
          X1(X1_raw, sizes[0], bs)
    {
        if (!X0_raw || !X1_raw)
            throw std::runtime_error("PMAD BatchWorkspace X alloc failed");
        for (int i = 1; i < (int)sizes.size(); ++i)
            layer_bufs.push_back(std::make_unique<LayerBufs>(sizes[i], bs));
    }

    ~BatchWorkspace() { pmad_free(X0_raw); pmad_free(X1_raw); }

    BatchWorkspace(const BatchWorkspace&)            = delete;
    BatchWorkspace& operator=(const BatchWorkspace&) = delete;

    Eigen::Map<Eigen::MatrixXf>&       Xbuf(int i)       { return i == 0 ? X0 : X1; }
    const Eigen::Map<Eigen::MatrixXf>& Xbuf(int i) const { return i == 0 ? X0 : X1; }

    Eigen::Map<Eigen::MatrixXf>& Z(int i) { return layer_bufs[i]->Z; }
    Eigen::Map<Eigen::MatrixXf>& A(int i) { return layer_bufs[i]->A; }
    Eigen::Map<Eigen::MatrixXf>& D(int i) { return layer_bufs[i]->D; }
};
