#pragma once

#include <Eigen/Dense>
#include <cstdio>
#include <random>
#include <vector>

namespace nn {
    inline std::mt19937& global_rng() {
        static thread_local std::mt19937 g { 42 };
        return g;
    }
    inline void manual_seed(unsigned s) { global_rng().seed(s); }
}

// Base class for all network modules.
//
// Subclass and implement forward() and backward(). Call register_module()
// for each sub-module in the constructor — this wires zero_grad() and
// update() to propagate automatically without overriding them.
//
// backward() receives the upstream gradient and returns the downstream
// gradient (dL/dx), accumulating parameter gradients as a side effect.
//
// Gradient scaling convention: CrossEntropyLoss returns output_delta
// already divided by batch size, so all accumulated gradients are means.
// Softmax::backward is a passthrough because CrossEntropyLoss already
// folds in the Softmax Jacobian (returns dL/dZ, not dL/dA).

struct Module {
protected:
    std::vector<Module*> children_;

public:
    virtual ~Module() = default;

    void register_module(Module& m) { children_.push_back(&m); }

    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& x) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) = 0;

    virtual void zero_grad() {
        for (auto* m : children_) m->zero_grad();
    }

    virtual void update(float lr) {
        for (auto* m : children_) m->update(lr);
    }

    virtual void update_adam(float lr, float b1, float b2, float eps, int t) {
        for (auto* m : children_) m->update_adam(lr, b1, b2, eps, t);
    }

    virtual void save(std::FILE* f) const { for (auto* m : children_) m->save(f); }
    virtual void load(std::FILE* f)       { for (auto* m : children_) m->load(f); }
};
