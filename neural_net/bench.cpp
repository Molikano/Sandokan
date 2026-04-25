#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dataloader.h"
#include "network.h"   // PMAD-backed Network + BatchCache + BatchWorkspace + softmax_cols
#include "pmad.h"

// ============================================================
// Plain-Eigen Network — identical math, all plain MatrixXd/VectorXd
// ============================================================

struct LayerEigen {
    Eigen::MatrixXd W;
    Eigen::VectorXd b;
    Eigen::MatrixXd dW;
    Eigen::VectorXd db;
    Eigen::VectorXd a;
    Eigen::VectorXd z;
    Eigen::VectorXd delta;

    LayerEigen(int in, int out, std::mt19937& rng)
        : W(out, in), b(Eigen::VectorXd::Zero(out)),
          dW(Eigen::MatrixXd::Zero(out, in)),
          db(Eigen::VectorXd::Zero(out)),
          a(Eigen::VectorXd::Zero(out)),
          z(Eigen::VectorXd::Zero(out)),
          delta(Eigen::VectorXd::Zero(out))
    {
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in));
        for (int r = 0; r < out; ++r)
            for (int c = 0; c < in; ++c)
                W(r, c) = dist(rng);
    }

    void zero_grad() { dW.setZero(); db.setZero(); }
};

struct NetworkEigen {
    static constexpr int INPUT_SIZE  = 784;
    static constexpr int HIDDEN1     = 64;
    static constexpr int HIDDEN2     = 64;
    static constexpr int OUTPUT_SIZE = 26;

    std::mt19937 rng { 42 };

    LayerEigen hidden1 { INPUT_SIZE, HIDDEN1,     rng };
    LayerEigen hidden2 { HIDDEN1,    HIDDEN2,     rng };
    LayerEigen output  { HIDDEN2,    OUTPUT_SIZE, rng };

    std::vector<LayerEigen*> layers { &hidden1, &hidden2, &output };

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
        for (LayerEigen* l : layers) { l->W -= lr * l->dW; l->b -= lr * l->db; }
    }

    void zero_grad() { for (LayerEigen* l : layers) l->zero_grad(); }

    // Batched: returns BatchCache (plain Eigen heap allocs per call)
    BatchCache forward_batch(const Eigen::MatrixXd& X) {
        BatchCache c;
        c.Z1 = (hidden1.W * X).colwise()    + hidden1.b;
        c.A1 = c.Z1.cwiseMax(0.0);
        c.Z2 = (hidden2.W * c.A1).colwise() + hidden2.b;
        c.A2 = c.Z2.cwiseMax(0.0);
        c.Z3 = (output.W  * c.A2).colwise() + output.b;
        c.A3 = softmax_cols(c.Z3);
        return c;
    }

    void backward_batch(const Eigen::MatrixXd& X, const BatchCache& c,
                        const std::vector<int>& labels, int bs) {
        Eigen::MatrixXd D3 = c.A3;
        for (int j = 0; j < bs; ++j) D3(labels[j], j) -= 1.0;
        Eigen::MatrixXd D2 = ((output.W.transpose()  * D3).array()
                                 * (c.Z2.array() > 0.0).cast<double>()).matrix();
        Eigen::MatrixXd D1 = ((hidden2.W.transpose() * D2).array()
                                 * (c.Z1.array() > 0.0).cast<double>()).matrix();
        const double inv_bs = 1.0 / bs;
        output.dW  += inv_bs * (D3 * c.A2.transpose());
        output.db  += inv_bs *  D3.rowwise().sum();
        hidden2.dW += inv_bs * (D2 * c.A1.transpose());
        hidden2.db += inv_bs *  D2.rowwise().sum();
        hidden1.dW += inv_bs * (D1 * X.transpose());
        hidden1.db += inv_bs *  D1.rowwise().sum();
    }
};

// ============================================================
// Benchmark harnesses
// ============================================================

struct BenchResult {
    double total_ms;
    double ms_per_epoch;
    double ms_per_sample;
    double samples_per_sec;
};

// Single-sample: BLAS L2 (matrix × vector) per sample
template<typename Net, typename LayerPtr>
BenchResult run_bench_single(Net& net, std::vector<LayerPtr*>& layers,
                             const ImageDataset& data,
                             int epochs, int batch_size, double lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = static_cast<int>(data.images.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e  = std::min(s + batch_size, n);
            int bs = e - s;
            net.zero_grad();
            for (int i = s; i < e; ++i) {
                net.forward(data.images[idx[i]]);
                net.backward(data.images[idx[i]], data.labels[idx[i]]);
            }
            for (auto* l : layers) { l->dW /= bs; l->db /= bs; }
            net.update(lr);
        }
    };

    run_epoch();  // warm-up
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double total_ms    = Ms(Clock::now() - t0).count();
    int    total_samples = epochs * n;
    return { total_ms, total_ms/epochs, total_ms/total_samples,
             total_samples/(total_ms/1000.0) };
}

// Eigen batched: BLAS L3 GEMM, BatchCache allocated by Eigen per batch
BenchResult run_bench_eigen_batched(NetworkEigen& net, const ImageDataset& data,
                                    int epochs, int batch_size, double lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n          = static_cast<int>(data.images.size());
    int input_size = static_cast<int>(data.images[0].size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e  = std::min(s + batch_size, n);
            int bs = e - s;
            Eigen::MatrixXd  X(input_size, bs);
            std::vector<int> labels(bs);
            for (int i = 0; i < bs; ++i) {
                X.col(i)  = data.images[idx[s + i]];
                labels[i] = data.labels[idx[s + i]];
            }
            net.zero_grad();
            BatchCache c = net.forward_batch(X);
            net.backward_batch(X, c, labels, bs);
            net.update(lr);
        }
    };

    run_epoch();  // warm-up
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double total_ms    = Ms(Clock::now() - t0).count();
    int    total_samples = epochs * n;
    return { total_ms, total_ms/epochs, total_ms/total_samples,
             total_samples/(total_ms/1000.0) };
}

// PMAD batched: BLAS L3 GEMM + PMAD workspace (pre-allocated, reused every batch)
BenchResult run_bench_pmad_batched(Network& net, const ImageDataset& data,
                                   int epochs, int batch_size, double lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n          = static_cast<int>(data.images.size());
    int input_size = static_cast<int>(data.images[0].size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    // Allocate workspace once — reused across all batches and all epochs
    BatchWorkspace ws(Network::HIDDEN1, Network::HIDDEN2, Network::OUTPUT_SIZE, batch_size);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e  = std::min(s + batch_size, n);
            int bs = e - s;
            if (bs < batch_size) continue;  // skip partial last batch in bench
            Eigen::MatrixXd  X(input_size, bs);
            std::vector<int> labels(bs);
            for (int i = 0; i < bs; ++i) {
                X.col(i)  = data.images[idx[s + i]];
                labels[i] = data.labels[idx[s + i]];
            }
            net.zero_grad();
            net.forward_batch(X, ws);
            net.backward_batch(X, ws, labels, bs);
            net.update(lr);
        }
    };

    run_epoch();  // warm-up
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double total_ms    = Ms(Clock::now() - t0).count();
    int    total_samples = epochs * n;
    return { total_ms, total_ms/epochs, total_ms/total_samples,
             total_samples/(total_ms/1000.0) };
}

// ============================================================
// Main
// ============================================================

int main() {
    const std::string data_dir    = "data/Emnist Letters";
    const int         bench_epochs = 5;
    const int         batch_size   = 128;
    const double      lr           = 0.01;

    std::printf("Loading dataset...\n");
    ImageDataset data = load_emnist_letters(data_dir, /*train=*/true);
    std::printf("  %zu images loaded and normalised\n\n", data.images.size());

    std::printf("Benchmark: %d epochs  batch=%d  lr=%.4f\n", bench_epochs, batch_size, lr);
    std::printf("(1 warm-up epoch excluded from timing)\n\n");

    // ---- PMAD single-sample ----
    init_network_pmad();
    BenchResult pmad_single;
    { Network net; pmad_single = run_bench_single(net, net.layers, data, bench_epochs, batch_size, lr); }
    destroy_network_pmad();

    // ---- Eigen single-sample ----
    BenchResult eigen_single;
    { NetworkEigen net; eigen_single = run_bench_single(net, net.layers, data, bench_epochs, batch_size, lr); }

    // ---- PMAD batched (workspace — pre-allocated, no per-batch alloc) ----
    init_network_pmad();
    BenchResult pmad_batched;
    { Network net; pmad_batched = run_bench_pmad_batched(net, data, bench_epochs, batch_size, lr); }
    destroy_network_pmad();

    // ---- Eigen batched (BatchCache — Eigen allocs per batch) ----
    BenchResult eigen_batched;
    { NetworkEigen net; eigen_batched = run_bench_eigen_batched(net, data, bench_epochs, batch_size, lr); }

    // ---- Report ----
    std::printf("%-28s  %10s  %12s  %14s  %14s\n",
                "Backend", "Total (ms)", "ms/epoch", "ms/sample", "samples/sec");
    std::printf("%s\n", std::string(82, '-').c_str());

    auto print_row = [](const char* name, const BenchResult& r) {
        std::printf("%-28s  %10.1f  %12.1f  %14.4f  %14.0f\n",
                    name, r.total_ms, r.ms_per_epoch, r.ms_per_sample, r.samples_per_sec);
    };

    print_row("PMAD single-sample",   pmad_single);
    print_row("Eigen single-sample",  eigen_single);
    print_row("PMAD batched+workspace", pmad_batched);
    print_row("Eigen batched",         eigen_batched);

    std::printf("%s\n", std::string(82, '-').c_str());
    std::printf("Batched speedup over single-sample:  PMAD %.2fx   Eigen %.2fx\n",
                pmad_single.total_ms / pmad_batched.total_ms,
                eigen_single.total_ms / eigen_batched.total_ms);
    std::printf("PMAD workspace vs Eigen batched:     %.3fx %s\n",
                eigen_batched.total_ms / pmad_batched.total_ms,
                eigen_batched.total_ms >= pmad_batched.total_ms ? "(PMAD wins)" : "(Eigen wins)");

    return 0;
}
