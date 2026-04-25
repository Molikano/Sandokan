#pragma once

#include "module.h"
#include <algorithm>
#include <cmath>

struct SGD {
    float lr;

    explicit SGD(float lr) : lr(lr) {}

    void zero_grad(Module& net) { net.zero_grad(); }
    void step(Module& net)      { net.update(lr);  }
    void epoch_end()            {}
};

struct Adam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int   t = 0;

    explicit Adam(float lr    = 1e-3f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float eps   = 1e-8f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

    void zero_grad(Module& net) { net.zero_grad(); }
    void step(Module& net)      { net.update_adam(lr, beta1, beta2, eps, ++t); }
    void epoch_end()            {}
};

// LinearLR — wraps any optimizer, decays lr linearly from start_lr to end_lr
// over total_epochs. Pass it directly to train_module in place of the optimizer.
//
// Usage:
//   Adam optim(1e-3f);
//   LinearLR sched(optim, 150, 1e-5f);   // decay to 1e-5 over 150 epochs
//   train_module(net, sched, train_set, test_set, 150);
template<typename Optim>
struct LinearLR {
    Optim& optim;
    float  start_lr;
    float  end_lr;
    int    total_epochs;
    int    epoch = 0;

    LinearLR(Optim& optim, int total_epochs, float end_lr = 0.0f)
        : optim(optim), start_lr(optim.lr), end_lr(end_lr), total_epochs(total_epochs) {}

    void zero_grad(Module& net) { optim.zero_grad(net); }
    void step(Module& net)      { optim.step(net); }

    void epoch_end() {
        ++epoch;
        float t  = std::min(float(epoch) / float(total_epochs), 1.0f);
        optim.lr = start_lr + t * (end_lr - start_lr);
    }

    float current_lr() const { return optim.lr; }
};
