#pragma once

#include "allocator.h"
#include "module.h"
#include "ops.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// Linear layer with PMAD-backed gradient buffers.
// Construct the network freely — call init_pmad_for() afterwards to migrate
// gradient buffers from the initial Eigen storage into the PMAD pool.
// The activation cache (in_) is always plain Eigen.
struct Linear : Module {
    int                         out_feat_, in_feat_;
    Eigen::MatrixXf             W;
    Eigen::VectorXf             b;

    // Gradient buffers — Eigen-owned until migrate_to_pmad() is called.
    std::vector<float>          dW_buf_, db_buf_;
    float*                      dW_raw;
    float*                      db_raw;
    bool                        pmad_owned_ = false;
    Eigen::Map<Eigen::MatrixXf> dW;
    Eigen::Map<Eigen::VectorXf> db;

    Eigen::MatrixXf             in_;

    Linear(int in_features, int out_features)
        : out_feat_(out_features), in_feat_(in_features),
          W(out_features, in_features),
          b(Eigen::VectorXf::Zero(out_features)),
          dW_buf_(size_t(out_features) * in_features, 0.0f),
          db_buf_(size_t(out_features), 0.0f),
          dW_raw(dW_buf_.data()),
          db_raw(db_buf_.data()),
          dW(dW_raw, out_features, in_features),
          db(db_raw, out_features)
    {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features));
        for (int r = 0; r < out_features; ++r)
            for (int c = 0; c < in_features; ++c)
                W(r, c) = dist(nn::global_rng());
        // Register for deferred PMAD migration by init_pmad_for().
        detail::register_pending(out_features, in_features,
                                 [this] { migrate_to_pmad(); });
    }

    ~Linear() { if (pmad_owned_) { pmad_free(dW_raw); pmad_free(db_raw); } }
    Linear(const Linear&)            = delete;
    Linear& operator=(const Linear&) = delete;

    // Called by init_pmad_for() after the pool is ready.
    void migrate_to_pmad() {
        auto* pw = static_cast<float*>(pmad_alloc(out_feat_ * in_feat_ * sizeof(float)));
        auto* pb = static_cast<float*>(pmad_alloc(out_feat_ * sizeof(float)));
        if (!pw || !pb) return; // stay on Eigen heap if pool is full
        std::copy(dW_buf_.begin(), dW_buf_.end(), pw);
        std::copy(db_buf_.begin(), db_buf_.end(), pb);
        dW_buf_.clear(); dW_buf_.shrink_to_fit();
        db_buf_.clear(); db_buf_.shrink_to_fit();
        dW_raw = pw; db_raw = pb; pmad_owned_ = true;
        // Rewire the Maps to point at the new PMAD storage.
        new (&dW) Eigen::Map<Eigen::MatrixXf>(dW_raw, out_feat_, in_feat_);
        new (&db) Eigen::Map<Eigen::VectorXf>(db_raw, out_feat_);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        in_ = x;
        return (W * x).colwise() + b;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        dW.noalias() += dy * in_.transpose();
        db            += dy.rowwise().sum();
        return W.transpose() * dy;
    }

    void zero_grad() override { dW.setZero(); db.setZero(); }
    void update(float lr) override { W -= lr * dW; b -= lr * db; }

    void update_adam(float lr, float b1, float b2, float eps, int t) override {
        if (m_W_.size() == 0) {
            m_W_ = Eigen::MatrixXf::Zero(W.rows(), W.cols());
            v_W_ = Eigen::MatrixXf::Zero(W.rows(), W.cols());
            m_b_ = Eigen::VectorXf::Zero(b.size());
            v_b_ = Eigen::VectorXf::Zero(b.size());
        }
        const float bc1 = 1.0f - std::pow(b1, t);
        const float bc2 = 1.0f - std::pow(b2, t);

        m_W_ = b1 * m_W_ + (1.0f - b1) * dW;
        v_W_ = b2 * v_W_ + (1.0f - b2) * dW.cwiseProduct(dW);
        W.array() -= lr / bc1 * m_W_.array() / ((v_W_.array() / bc2).sqrt() + eps);

        m_b_ = b1 * m_b_ + (1.0f - b1) * db;
        v_b_ = b2 * v_b_ + (1.0f - b2) * db.cwiseProduct(db);
        b.array() -= lr / bc1 * m_b_.array() / ((v_b_.array() / bc2).sqrt() + eps);
    }

    void save(std::FILE* f) const override {
        uint32_t r = static_cast<uint32_t>(W.rows());
        uint32_t c = static_cast<uint32_t>(W.cols());
        std::fwrite(&r, sizeof(uint32_t), 1, f);
        std::fwrite(&c, sizeof(uint32_t), 1, f);
        std::fwrite(W.data(), sizeof(float), r * c, f);
        uint32_t bn = static_cast<uint32_t>(b.size());
        std::fwrite(&bn, sizeof(uint32_t), 1, f);
        std::fwrite(b.data(), sizeof(float), bn, f);
    }

    void load(std::FILE* f) override {
        uint32_t r, c;
        std::fread(&r, sizeof(uint32_t), 1, f);
        std::fread(&c, sizeof(uint32_t), 1, f);
        if (r != static_cast<uint32_t>(W.rows()) || c != static_cast<uint32_t>(W.cols()))
            throw std::runtime_error("Linear weight shape mismatch in .sand file");
        std::fread(W.data(), sizeof(float), r * c, f);
        uint32_t bn;
        std::fread(&bn, sizeof(uint32_t), 1, f);
        if (bn != static_cast<uint32_t>(b.size()))
            throw std::runtime_error("Linear bias shape mismatch in .sand file");
        std::fread(b.data(), sizeof(float), bn, f);
    }

private:
    Eigen::MatrixXf m_W_, v_W_;
    Eigen::VectorXf m_b_, v_b_;
};

struct ReLU : Module {
    Eigen::MatrixXf out_;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        out_ = x.cwiseMax(0.0f);
        return out_;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return dy.cwiseProduct((out_.array() > 0.0f).cast<float>().matrix());
    }
};

struct Sigmoid : Module {
    Eigen::MatrixXf out_;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        out_ = (1.0f / (1.0f + (-x.array()).exp())).matrix();
        return out_;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return dy.cwiseProduct((out_.array() * (1.0f - out_.array())).matrix());
    }
};

struct Softmax : Module {
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return softmax_cols(x);
    }
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override { return dy; }
};

// Sequential container — chains modules in order, handles backward automatically.
//
// Usage:
//   Sequential net;
//   net.add<Linear>(784, 128).add<ReLU>().add<Linear>(128, 26).add<Softmax>();
struct Sequential : Module {
    std::vector<std::unique_ptr<Module>> mods_;

    template<typename M, typename... Args>
    Sequential& add(Args&&... args) {
        mods_.push_back(std::make_unique<M>(std::forward<Args>(args)...));
        return *this;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        Eigen::MatrixXf out = x;
        for (auto& m : mods_) out = m->forward(out);
        return out;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        Eigen::MatrixXf grad = dy;
        for (int i = (int)mods_.size() - 1; i >= 0; --i)
            grad = mods_[i]->backward(grad);
        return grad;
    }

    void zero_grad() override { for (auto& m : mods_) m->zero_grad(); }
    void update(float lr) override { for (auto& m : mods_) m->update(lr); }
    void update_adam(float lr, float b1, float b2, float eps, int t) override {
        for (auto& m : mods_) m->update_adam(lr, b1, b2, eps, t);
    }
};
