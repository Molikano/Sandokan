#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// General-purpose in-memory dataset for tabular data (CSV, Eigen matrices, etc.)
//
// Satisfies the Dataset concept expected by train_module / compute_accuracy:
//   cols()            — number of samples
//   image_size()      — number of input features
//   get_image_col()   — fill a VectorXf with (optionally normalized) features
//   label()           — integer class label  (classification)
//   target()          — float target value   (regression)
//
// For regression tasks, compute_accuracy() is meaningless — implement your
// own evaluation using target() and the model's raw output.

struct TabularDataset {
    Eigen::MatrixXf X;          // features × samples (column-major)
    Eigen::VectorXf targets;    // float targets: class index for classif, value for regress
    Eigen::VectorXf mean;
    Eigen::VectorXf inv_sigma;
    bool            normalize = false;

    // Target normalization (regression) — set by compute_target_normalization().
    float target_mean_  = 0.0f;
    float target_scale_ = 1.0f;  // 1 / std
    bool  normalize_targets_ = false;

    int cols()       const { return (int)X.cols(); }
    int image_size() const { return (int)X.rows(); }

    // For classification: return the target as an integer class index.
    int   label(int idx)  const { return static_cast<int>(std::round(targets(idx))); }
    // For regression: raw float target in original units.
    float target(int idx) const { return targets(idx); }
    // For regression training: normalized target (zero mean, unit variance).
    float normalized_target(int idx) const {
        return normalize_targets_
            ? (targets(idx) - target_mean_) * target_scale_
            : targets(idx);
    }
    // Convert a normalized prediction back to original units.
    float denormalize_target(float y) const {
        return normalize_targets_ ? y / target_scale_ + target_mean_ : y;
    }

    void get_image_col(int idx, Eigen::Ref<Eigen::VectorXf> out) const {
        out = X.col(idx);
        if (normalize)
            out.array() = (out.array() - mean.array()) * inv_sigma.array();
    }

    void get_raw_image_col(int idx, Eigen::Ref<Eigen::VectorXf> out) const {
        out = X.col(idx);
    }

    // One-pass streaming normalization over all feature columns.
    void compute_normalization() {
        if (mean.size() == X.rows()) { normalize = true; return; }
        const int F = X.rows(), N = X.cols();
        Eigen::ArrayXd sum   = Eigen::ArrayXd::Zero(F);
        Eigen::ArrayXd sumsq = Eigen::ArrayXd::Zero(F);
        for (int i = 0; i < N; ++i) {
            Eigen::ArrayXd v = X.col(i).cast<double>().array();
            sum   += v;
            sumsq += v.square();
        }
        Eigen::ArrayXd mu  = sum / N;
        Eigen::ArrayXd var = (sumsq / N - mu.square()).max(0.0);
        mean      = mu.cast<float>().matrix();
        inv_sigma = var.sqrt().unaryExpr([](double s) {
                        return s < 1e-8 ? 1.0 : 1.0 / s;
                    }).cast<float>().matrix();
        normalize = true;
    }

    void apply_normalization_from(const TabularDataset& src) {
        mean = src.mean; inv_sigma = src.inv_sigma; normalize = true;
    }

    // Compute per-target mean and std from training targets.
    void compute_target_normalization() {
        const int N = (int)targets.size();
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < N; ++i) { sum += targets(i); sumsq += double(targets(i)) * targets(i); }
        double mu  = sum / N;
        double sig = std::sqrt(std::max(sumsq / N - mu * mu, 0.0));
        target_mean_  = float(mu);
        target_scale_ = float(sig < 1e-8 ? 1.0 : 1.0 / sig);
        normalize_targets_ = true;
    }

    void apply_target_normalization_from(const TabularDataset& src) {
        target_mean_       = src.target_mean_;
        target_scale_      = src.target_scale_;
        normalize_targets_ = src.normalize_targets_;
    }

    // Build directly from Eigen matrices — for programmatic construction.
    static TabularDataset from_matrices(const Eigen::MatrixXf& features,
                                        const Eigen::VectorXf& tgts) {
        if (features.cols() != tgts.size())
            throw std::runtime_error("TabularDataset: sample count mismatch");
        TabularDataset ds;
        ds.X       = features;
        ds.targets = tgts;
        return ds;
    }
};

// Load a CSV file into a TabularDataset.
//
// Parameters:
//   path      — path to the CSV file
//   label_col — column index of the target (-1 = last column)
//   header    — true if the first row is a header and should be skipped
//   delim     — field separator character
//
// All fields must be numeric (pre-encode categoricals before loading).
// Missing values are not handled — clean your data first.
//
// Example — Boston housing (13 features, price in last column):
//   auto ds = load_csv("boston.csv");
//   ds.compute_normalization();
inline TabularDataset load_csv(const std::string& path,
                               int  label_col = -1,
                               bool header    = true,
                               char delim     = ',') {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_csv: cannot open '" + path + "'");

    std::vector<std::vector<float>> rows;
    std::string line;
    int ncols = -1;

    if (header) std::getline(f, line); // skip header row

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<float> row;
        std::stringstream  ss(line);
        std::string        field;
        while (std::getline(ss, field, delim)) {
            if (field.empty())
                throw std::runtime_error(
                    "load_csv: empty field at row " + std::to_string(rows.size() + 1) +
                    " — clean missing values before loading");
            try { row.push_back(std::stof(field)); }
            catch (...) {
                throw std::runtime_error(
                    "load_csv: non-numeric field '" + field + "' at row " +
                    std::to_string(rows.size() + 1) +
                    " — encode categorical features before loading");
            }
        }
        if (row.empty()) continue;
        if (ncols < 0) ncols = (int)row.size();
        if ((int)row.size() != ncols)
            throw std::runtime_error(
                "load_csv: expected " + std::to_string(ncols) +
                " columns but got " + std::to_string(row.size()) +
                " at row " + std::to_string(rows.size() + 1));
        rows.push_back(std::move(row));
    }

    if (rows.empty()) throw std::runtime_error("load_csv: no data rows in '" + path + "'");

    const int n = (int)rows.size();
    if (label_col < 0) label_col = ncols - 1;
    if (label_col >= ncols)
        throw std::runtime_error("load_csv: label_col " + std::to_string(label_col) +
                                 " out of range (" + std::to_string(ncols) + " columns)");

    const int feat_count = ncols - 1;
    TabularDataset ds;
    ds.X.resize(feat_count, n);
    ds.targets.resize(n);

    for (int i = 0; i < n; ++i) {
        int fi = 0;
        for (int c = 0; c < ncols; ++c) {
            if (c == label_col) ds.targets(i) = rows[i][c];
            else                ds.X(fi++, i) = rows[i][c];
        }
    }

    return ds;
}
