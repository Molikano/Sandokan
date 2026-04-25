#pragma once

#include <Eigen/Dense>
#include <cmath>

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
