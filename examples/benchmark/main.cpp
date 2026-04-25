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
#include <sandokan/layers.h>
#include <sandokan/loss.h>
#include <sandokan/network.h>

// ============================================================
// Plain-Eigen Network — identical math, all plain MatrixXf/VectorXf (no PMAD)
// Generic over any architecture.
// ============================================================

struct LayerEigen {
    Eigen::MatrixXf W, dW;
    Eigen::VectorXf b, db, a, z;

    LayerEigen(int in, int out, std::mt19937& rng)
        : W(out, in), dW(Eigen::MatrixXf::Zero(out, in)),
          b(Eigen::VectorXf::Zero(out)), db(Eigen::VectorXf::Zero(out)),
          a(Eigen::VectorXf::Zero(out)), z(Eigen::VectorXf::Zero(out))
    {
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in));
        for (int r = 0; r < out; ++r)
            for (int c = 0; c < in; ++c)
                W(r, c) = dist(rng);
    }
    void zero_grad() { dW.setZero(); db.setZero(); }
};

struct NetworkEigen {
    std::vector<LayerEigen> layers;

    explicit NetworkEigen(const std::vector<int>& sizes) {
        std::mt19937 rng(42);
        for (int i = 0; i + 1 < (int)sizes.size(); ++i)
            layers.emplace_back(sizes[i], sizes[i+1], rng);
    }

    int num_layers() const { return (int)layers.size(); }

    Eigen::VectorXf forward(const Eigen::Ref<const Eigen::VectorXf>& x) {
        const int L = num_layers();
        Eigen::VectorXf a = x;
        for (int i = 0; i < L; ++i) {
            layers[i].z = layers[i].W * a + layers[i].b;
            layers[i].a = (i < L - 1) ? relu(layers[i].z) : softmax(layers[i].z);
            a = layers[i].a;
        }
        return a;
    }

    void backward(const Eigen::Ref<const Eigen::VectorXf>& x,
                  const Eigen::Ref<const Eigen::VectorXf>& output_delta) {
        const int L = num_layers();
        std::vector<Eigen::VectorXf> deltas(L);
        deltas[L-1] = output_delta;
        for (int i = L - 2; i >= 0; --i)
            deltas[i] = (layers[i+1].W.transpose() * deltas[i+1])
                            .cwiseProduct(relu_prime(layers[i].z));
        for (int i = 0; i < L; ++i) {
            if (i == 0) layers[0].dW.noalias() += deltas[0] * x.transpose();
            else        layers[i].dW.noalias() += deltas[i] * layers[i-1].a.transpose();
            layers[i].db += deltas[i];
        }
    }

    using ZA = std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>>;

    ZA forward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X) {
        const int L = num_layers();
        std::vector<Eigen::MatrixXf> Z(L), A(L);
        for (int i = 0; i < L; ++i) {
            if (i == 0) Z[i] = (layers[i].W * X).colwise()    + layers[i].b;
            else        Z[i] = (layers[i].W * A[i-1]).colwise() + layers[i].b;
            A[i] = (i < L - 1) ? Z[i].cwiseMax(0.0f) : softmax_cols(Z[i]);
        }
        return {std::move(Z), std::move(A)};
    }

    void backward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        const std::vector<Eigen::MatrixXf>& Z,
                        const std::vector<Eigen::MatrixXf>& A,
                        const Eigen::Ref<const Eigen::MatrixXf>& output_delta) {
        const int L = num_layers();
        std::vector<Eigen::MatrixXf> D(L);
        D[L-1] = output_delta;
        for (int i = L - 2; i >= 0; --i)
            D[i] = ((layers[i+1].W.transpose() * D[i+1]).array()
                    * (Z[i].array() > 0.0f).cast<float>()).matrix();
        for (int i = 0; i < L; ++i) {
            if (i == 0) layers[0].dW.noalias() += D[0] * X.transpose();
            else        layers[i].dW.noalias() += D[i] * A[i-1].transpose();
            layers[i].db += D[i].rowwise().sum();
        }
    }

    void update(float lr) {
        for (auto& l : layers) { l.W -= lr * l.dW; l.b -= lr * l.db; }
    }
    void zero_grad() { for (auto& l : layers) l.zero_grad(); }
    void scale_grad(float s) { for (auto& l : layers) { l.dW *= s; l.db *= s; } }
};

// ============================================================
// Benchmark harnesses
// ============================================================

struct BenchResult {
    double total_ms, ms_per_epoch, ms_per_sample, samples_per_sec;
};

template<typename Net>
BenchResult run_bench_single(Net& net, const ImageDataset& data,
                             int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    CrossEntropyLoss criterion;
    Eigen::VectorXf x(data.image_size());
    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            net.zero_grad();
            for (int i = s; i < e; ++i) {
                data.get_image_col(idx[i], x);
                auto a = net.forward(x);
                auto [_, delta] = criterion(a, data.label(idx[i]));
                net.backward(x, delta);
            }
            net.scale_grad(1.0f / bs);
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

BenchResult run_bench_eigen_batched(NetworkEigen& net, const ImageDataset& data,
                                    int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols(), in = data.image_size();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    Eigen::MatrixXf X(in, batch_size);
    std::vector<int> labels(batch_size);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            if (bs != batch_size) continue;
            for (int i = 0; i < bs; ++i) {
                data.get_image_col(idx[s+i], X.col(i));
                labels[i] = data.label(idx[s+i]);
            }
            net.zero_grad();
            auto [Z, A] = net.forward_batch(X);
            auto [_, delta] = CrossEntropyLoss{}(A.back(), labels);
            net.backward_batch(X, Z, A, delta);
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

BenchResult run_bench_pmad_batched(Network& net, const ImageDataset& data,
                                   int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    const int L = net.num_layers();
    BatchWorkspace ws(net.sizes, batch_size);

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
                auto [_, delta] = CrossEntropyLoss{}(ws.A(L-1), curr);
                net.backward_batch(ws.Xbuf(comp), ws, delta);
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
// Module network with residual block (Module API benchmark)
// Architecture: 784 → 64(ReLU) → [64→64 ResBlock] → 26(Softmax)
// ============================================================

struct BenchResBlock : Module {
    Submodule<Linear> fc1 { *this, 64, 64 };
    Submodule<Linear> fc2 { *this, 64, 64 };
    ReLU relu1, relu2;

    BenchResBlock() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return relu2.forward(fc2.forward(relu1.forward(fc1.forward(x)))) + x;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return fc1.backward(relu1.backward(fc2.backward(relu2.backward(dy)))) + dy;
    }
};

struct SmallResNet : Module {
    Submodule<Linear>        proj { *this, 784, 64 };
    ReLU                     relu0;
    Submodule<BenchResBlock> res  { *this };
    Submodule<Linear>        head { *this, 64, 26 };
    Softmax                  sm;

    SmallResNet() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return sm.forward(head.forward(res.forward(relu0.forward(proj.forward(x)))));
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        auto d = head.backward(sm.backward(dy));
        d = res.backward(d);
        return proj.backward(relu0.backward(d));
    }
};

BenchResult run_bench_module_batched(Module& net, const ImageDataset& data,
                                     int epochs, int batch_size, float lr) {
    using Clock = std::chrono::high_resolution_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    int n = data.cols();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    Eigen::MatrixXf  X(data.image_size(), batch_size);
    std::vector<int> labels(batch_size);

    auto run_epoch = [&]() {
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int s = 0; s < n; s += batch_size) {
            int e = std::min(s + batch_size, n), bs = e - s;
            if (bs != batch_size) continue;
            for (int i = 0; i < bs; ++i) {
                data.get_image_col(idx[s+i], X.col(i));
                labels[i] = data.label(idx[s+i]);
            }
            net.zero_grad();
            auto probs = net.forward(X);
            auto [_, delta] = CrossEntropyLoss{}(probs, labels);
            net.backward(delta);
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

// ============================================================
// Main
// ============================================================

int main() {
    const std::string data_dir     = "data/Emnist Letters";
    const std::vector<int> arch_seq = { 784, 64, 64, 26 };
    const int   bench_epochs = 5;
    const int   batch_size   = 128;
    const float lr           = 0.01f;

    std::printf("Loading dataset...\n");
    ImageDataset data = load_emnist_letters(data_dir, true);
    std::printf("  %d images loaded (%.0f MB raw, mmap-resident)\n\n",
                data.cols(),
                double(data.cols()) * data.image_size() / 1048576.0);

    std::printf("Sequential arch:"); for (int s : arch_seq) std::printf(" %d", s);
    std::printf("\nModule arch: derived automatically via init_pmad_for()");
    std::printf("\nBenchmark: %d epochs  batch=%d  lr=%.4f\n", bench_epochs, batch_size, lr);
    std::printf("(1 warm-up epoch excluded from timing)\n\n");

    // Batched paths run first — CPU cold, no thermal throttle from prior load.
    // Single-sample paths run after — slower and hotter, but that's expected.

    // ---- PMAD batched+workspace+parallel ----
    init_pmad(arch_seq, batch_size);
    BenchResult pmad_batched;
    { Network net(arch_seq); pmad_batched = run_bench_pmad_batched(net, data, bench_epochs, batch_size, lr); }
    destroy_pmad();

    // ---- Eigen batched ----
    BenchResult eigen_batched;
    { NetworkEigen net(arch_seq); eigen_batched = run_bench_eigen_batched(net, data, bench_epochs, batch_size, lr); }

    // ---- Module+ResBlock batched ----
    BenchResult module_batched;
    { nn::manual_seed(42); SmallResNet net; init_pmad_for(); module_batched = run_bench_module_batched(net, data, bench_epochs, batch_size, lr); }
    destroy_pmad();

    // ---- PMAD single-sample ----
    init_pmad(arch_seq, batch_size);
    BenchResult pmad_single;
    { Network net(arch_seq); pmad_single = run_bench_single(net, data, bench_epochs, batch_size, lr); }
    destroy_pmad();

    // ---- Eigen single-sample ----
    BenchResult eigen_single;
    { NetworkEigen net(arch_seq); eigen_single = run_bench_single(net, data, bench_epochs, batch_size, lr); }

    // ---- Report ----
    std::printf("\n%-32s  %10s  %12s  %14s  %14s\n",
                "Backend", "Total (ms)", "ms/epoch", "ms/sample", "samples/sec");
    std::printf("%s\n", std::string(88, '-').c_str());
    auto row = [](const char* n, const BenchResult& r) {
        std::printf("%-32s  %10.1f  %12.1f  %14.4f  %14.0f\n",
                    n, r.total_ms, r.ms_per_epoch, r.ms_per_sample, r.samples_per_sec);
    };
    row("PMAD single-sample (seq)",      pmad_single);
    row("Eigen single-sample (seq)",     eigen_single);
    row("PMAD batched+ws+parallel (seq)",pmad_batched);
    row("Eigen batched (seq)",           eigen_batched);
    row("Module+ResBlock batched",       module_batched);
    std::printf("%s\n", std::string(88, '-').c_str());
    std::printf("Batched speedup:  PMAD %.2fx   Eigen %.2fx\n",
                pmad_single.total_ms / pmad_batched.total_ms,
                eigen_single.total_ms / eigen_batched.total_ms);
    std::printf("PMAD vs Eigen batched (seq): %.3fx %s\n",
                eigen_batched.total_ms / pmad_batched.total_ms,
                pmad_batched.total_ms <= eigen_batched.total_ms ? "(PMAD wins)" : "(Eigen wins)");
    std::printf("Module overhead vs PMAD batched: %.3fx\n",
                module_batched.total_ms / pmad_batched.total_ms);
    return 0;
}
