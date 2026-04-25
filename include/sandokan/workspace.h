#pragma once

#include "allocator.h"
#include <Eigen/Dense>
#include <stdexcept>

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
