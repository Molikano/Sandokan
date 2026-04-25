#include <chrono>
#include <cstdio>
#include <future>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <sandokan/allocator.h>
#include <sandokan/dataset.h>
#include <sandokan/network.h>

// ============================================================
// Plain-Eigen Network — identical math, all plain MatrixXf/VectorXf (no PMAD)
// ============================================================

struct LayerEigen {
    Eigen::MatrixXf W, dW;
    Eigen::VectorXf b, db, a, z, delta;

    LayerEigen(int in, int out, std::mt19937& rng)
        : W(out, in), dW(Eigen::MatrixXf::Zero(out, in)),
          b(Eigen::VectorXf::Zero(out)), db(Eigen::VectorXf::Zero(out)),
          a(Eigen::VectorXf::Zero(out)), z(Eigen::VectorXf::Zero(out)),
          delta(Eigen::VectorXf::Zero(out))
    {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in));
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

    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        hidden1.z = hidden1.W * x         + hidden1.b; hidden1.a = relu(hidden1.z);
        hidden2.z = hidden2.W * hidden1.a + hidden2.b; hidden2.a = relu(hidden2.z);
        output.z  = output.W  * hidden2.a + output.b;
        return softmax(output.z);
    }

    void backward(const Eigen::Ref<const Eigen::VectorXf>& x, int label) {
        output.delta = softmax(output.z); output.delta(label) -= 1.0f;
        hidden2.delta = (output.W.transpose() * output.delta).cwiseProduct(relu_prime(hidden2.z));
        hidden1.delta = (hidden2.W.transpose() * hidden2.delta).cwiseProduct(relu_prime(hidden1.z));
        output.dW  += output.delta  * hidden2.a.transpose(); output.db  += output.delta;
        hidden2.dW += hidden2.delta * hidden1.a.transpose(); hidden2.db += hidden2.delta;
        hidden1.dW += hidden1.delta * x.transpose();         hidden1.db += hidden1.delta;
    }

    void update(float lr) {
        for (auto* l : layers) { l->W -= lr * l->dW; l->b -= lr * l->db; }
    }
    void zero_grad() { for (auto* l : layers) l->zero_grad(); }

    // Batched — BatchCache allocated by Eigen per call
    BatchCache forward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X) {
        BatchCache c;
        c.Z1 = (hidden1.W * X).colwise()    + hidden1.b; c.A1 = c.Z1.cwiseMax(0.0f);
        c.Z2 = (hidden2.W * c.A1).colwise() + hidden2.b; c.A2 = c.Z2.cwiseMax(0.0f);
        c.Z3 = (output.W  * c.A2).colwise() + output.b;  c.A3 = softmax_cols(c.Z3);
        return c;
    }

    void backward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        const BatchCache& c,
                        const std::vector<int>& labels, int bs) {
        Eigen::MatrixXf D3 = c.A3;
        for (int j = 0; j < bs; ++j) D3(labels[j], j) -= 1.0f;
        Eigen::MatrixXf D2 = ((output.W.transpose()  * D3).array()
                              * (c.Z2.array() > 0.0f).cast<float>()).matrix();
        Eigen::MatrixXf D1 = ((hidden2.W.transpose() * D2).array()
                              * (c.Z1.array() > 0.0f).cast<float>()).matrix();
        const float inv = 1.0f / bs;
        output.dW  += inv * (D3 * c.A2.transpose()); output.db  += inv * D3.rowwise().sum();
        hidden2.dW += inv * (D2 * c.A1.transpose()); hidden2.db += inv * D2.rowwise().sum();
        hidden1.dW += inv * (D1 * X.transpose());    hidden1.db += inv * D1.rowwise().sum();
    }
};

// ============================================================
// Benchmark harnesses
// ============================================================

struct BenchResult {
    double total_ms, ms_per_epoch, ms_per_sample, samples_per_sec;
};

// Single-sample: BLAS L2 (matrix × vector) per sample
template<typename Net, typename LayerPtr>
BenchResult run_bench_single(Net& net, std::vector<LayerPtr*>& layers,
                             const ImageDataset& data,
                             int epochs, int batch_size, double lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    Eigen::VectorXf x(data.image_size());
    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            net.zero_grad();
            for (int i = s; i < e; ++i) {
                data.get_image_col(idx[i], x);
                net.forward(x);
                net.backward(x, data.label(idx[i]));
            }
            for (auto* l : layers) { l->dW /= bs; l->db /= bs; }
            net.update(static_cast<float>(lr));
        }
    };

    run_epoch();
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double ms = Ms(Clock::now() - t0).count();
    int    total = epochs * n;
    return { ms, ms/epochs, ms/total, total/(ms/1000.0) };
}

// Eigen batched: GEMM, BatchCache heap-allocated by Eigen per batch
BenchResult run_bench_eigen_batched(NetworkEigen& net, const ImageDataset& data,
                                    int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols(), in = data.image_size();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            Eigen::MatrixXf X(in, bs); std::vector<int> labels(bs);
            for (int i = 0; i < bs; ++i) {
                data.get_image_col(idx[s+i], X.col(i));
                labels[i] = data.label(idx[s+i]);
            }
            net.zero_grad();
            BatchCache c = net.forward_batch(X);
            net.backward_batch(X, c, labels, bs);
            net.update(lr);
        }
    };

    run_epoch();
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double ms = Ms(Clock::now() - t0).count();
    int    total = epochs * n;
    return { ms, ms/epochs, ms/total, total/(ms/1000.0) };
}

// PMAD batched+workspace: GEMM + pre-allocated PMAD Maps + parallel assembly
BenchResult run_bench_pmad_batched(Network& net, const ImageDataset& data,
                                   int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    BatchWorkspace ws(Network::INPUT_SIZE, Network::HIDDEN1, Network::HIDDEN2,
                      Network::OUTPUT_SIZE, batch_size);

    auto assemble = [&](int buf, int start, int bs) -> std::vector<int> {
        auto& Xb = ws.Xbuf(buf); std::vector<int> lbls(bs);
        for (int i = 0; i < bs; ++i) {
            data.get_image_col(idx[start+i], Xb.col(i));
            lbls[i] = data.label(idx[start+i]);
        }
        return lbls;
    };

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        int fill = 1, comp = 0;
        std::vector<int> curr = assemble(comp, 0, std::min(batch_size, n));

        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            bool has_next = (e + batch_size <= n);
            std::future<std::vector<int>> fut;
            if (has_next)
                fut = std::async(std::launch::async, assemble, fill, e, batch_size);

            if (bs == batch_size) {
                net.zero_grad();
                net.forward_batch(ws.Xbuf(comp), ws);
                net.backward_batch(ws.Xbuf(comp), ws, curr, bs);
                net.update(lr);
            }
            if (has_next) { curr = fut.get(); std::swap(fill, comp); }
        }
    };

    run_epoch();
    auto t0 = Clock::now();
    for (int ep = 0; ep < epochs; ++ep) run_epoch();
    double ms = Ms(Clock::now() - t0).count();
    int    total = epochs * n;
    return { ms, ms/epochs, ms/total, total/(ms/1000.0) };
}

// ============================================================
// Main
// ============================================================

int main() {
    const std::string data_dir     = "data/Emnist Letters";
    const int         bench_epochs = 5;
    const int         batch_size   = 128;
    const float       lr           = 0.01f;

    std::printf("Loading dataset...\n");
    ImageDataset data = load_emnist_letters(data_dir, true);
    std::printf("  %d images loaded (%.0f MB raw, mmap-resident)\n\n",
                data.cols(),
                double(data.cols()) * data.image_size() / 1048576.0);

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

    // ---- PMAD batched+workspace+parallel ----
    init_network_pmad();
    BenchResult pmad_batched;
    { Network net; pmad_batched = run_bench_pmad_batched(net, data, bench_epochs, batch_size, lr); }
    destroy_network_pmad();

    // ---- Eigen batched ----
    BenchResult eigen_batched;
    { NetworkEigen net; eigen_batched = run_bench_eigen_batched(net, data, bench_epochs, batch_size, lr); }

    // ---- Report ----
    std::printf("%-30s  %10s  %12s  %14s  %14s\n",
                "Backend", "Total (ms)", "ms/epoch", "ms/sample", "samples/sec");
    std::printf("%s\n", std::string(84, '-').c_str());
    auto row = [](const char* n, const BenchResult& r) {
        std::printf("%-30s  %10.1f  %12.1f  %14.4f  %14.0f\n",
                    n, r.total_ms, r.ms_per_epoch, r.ms_per_sample, r.samples_per_sec);
    };
    row("PMAD single-sample",          pmad_single);
    row("Eigen single-sample",         eigen_single);
    row("PMAD batched+ws+parallel",    pmad_batched);
    row("Eigen batched",               eigen_batched);
    std::printf("%s\n", std::string(84, '-').c_str());
    std::printf("Batched speedup:  PMAD %.2fx   Eigen %.2fx\n",
                pmad_single.total_ms / pmad_batched.total_ms,
                eigen_single.total_ms / eigen_batched.total_ms);
    std::printf("PMAD vs Eigen batched: %.3fx %s\n",
                eigen_batched.total_ms / pmad_batched.total_ms,
                pmad_batched.total_ms <= eigen_batched.total_ms ? "(PMAD wins)" : "(Eigen wins)");
    return 0;
}
