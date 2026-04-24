#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

// ---------- Activations ----------

inline Eigen::VectorXd relu(const Eigen::VectorXd& z) {
    return z.cwiseMax(0.0);
}

inline Eigen::VectorXd relu_prime(const Eigen::VectorXd& z) {
    return (z.array() > 0.0).cast<double>();
}

// Numerically stable: subtract max before exp
inline Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::VectorXd e = (z.array() - z.maxCoeff()).exp();
    return e / e.sum();
}

// ---------- Layer ----------

struct Layer {
    // 1. Weights
    Eigen::MatrixXd W;
    Eigen::VectorXd b;

    // 2. Gradients
    Eigen::MatrixXd dW;
    Eigen::VectorXd db;

    // 3. Activations  (post-activation output, filled by forward pass)
    Eigen::VectorXd a;

    // 4. Temp buffers (filled during forward/backward, reused each sample)
    Eigen::VectorXd z;      // pre-activation: W*input + b
    Eigen::VectorXd delta;  // error signal propagated by backprop

    // He init: std = sqrt(2 / fan_in), good for ReLU
    Layer(int in, int out, std::mt19937& rng) {
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in));

        W.resize(out, in);
        for (int r = 0; r < out; ++r)
            for (int c = 0; c < in; ++c)
                W(r, c) = dist(rng);

        b     = Eigen::VectorXd::Zero(out);
        dW    = Eigen::MatrixXd::Zero(out, in);
        db    = Eigen::VectorXd::Zero(out);
        a     = Eigen::VectorXd::Zero(out);
        z     = Eigen::VectorXd::Zero(out);
        delta = Eigen::VectorXd::Zero(out);
    }

    void zero_grad() { dW.setZero(); db.setZero(); }
};

// ---------- Network: 784 -> 64 -> 64 -> 26 ----------

struct Network {
    static constexpr int INPUT_SIZE  = 784;
    static constexpr int HIDDEN1     = 64;
    static constexpr int HIDDEN2     = 64;
    static constexpr int OUTPUT_SIZE = 26;  // A-Z

    // rng must be declared before the Layer members (init order = declaration order)
    std::mt19937 rng { 42 };

    Layer hidden1 { INPUT_SIZE, HIDDEN1,     rng };
    Layer hidden2 { HIDDEN1,    HIDDEN2,     rng };
    Layer output  { HIDDEN2,    OUTPUT_SIZE, rng };

    std::vector<Layer*> layers { &hidden1, &hidden2, &output };

    // Returns softmax probabilities for input x
    Eigen::VectorXd forward(const Eigen::VectorXd& x) {
        hidden1.z = hidden1.W * x         + hidden1.b;
        hidden1.a = relu(hidden1.z);

        hidden2.z = hidden2.W * hidden1.a + hidden2.b;
        hidden2.a = relu(hidden2.z);

        output.z  = output.W  * hidden2.a + output.b;
        output.a  = softmax(output.z);

        return output.a;
    }

    // Accumulates dW/db — call zero_grad() before each batch
    void backward(const Eigen::VectorXd& x, int label) {
        // Softmax + cross-entropy combined gradient: p - one_hot(label)
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

    // Vanilla SGD step (call after averaging gradients over the batch)
    void update(double lr) {
        for (Layer* l : layers) {
            l->W -= lr * l->dW;
            l->b -= lr * l->db;
        }
    }

    void zero_grad() {
        for (Layer* l : layers) l->zero_grad();
    }
};
