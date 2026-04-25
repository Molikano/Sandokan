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

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ============================================================
// Plain-Eigen network — identical math, no PMAD
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

    using ZA = std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>>;

    ZA forward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X) {
        const int L = num_layers();
        std::vector<Eigen::MatrixXf> Z(L), A(L);
        for (int i = 0; i < L; ++i) {
            if (i == 0) Z[i] = (layers[i].W * X).colwise()     + layers[i].b;
            else        Z[i] = (layers[i].W * A[i-1]).colwise() + layers[i].b;
            A[i] = (i < L - 1) ? Z[i].cwiseMax(0.0f) : softmax_cols(Z[i]);
        }
        return {std::move(Z), std::move(A)};
    }

    void backward_batch(const Eigen::Ref<const Eigen::MatrixXf>& X,
                        const std::vector<Eigen::MatrixXf>& Z,
                        const std::vector<Eigen::MatrixXf>& A,
                        const Eigen::Ref<const Eigen::MatrixXf>& delta) {
        const int L = num_layers();
        std::vector<Eigen::MatrixXf> D(L);
        D[L-1] = delta;
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
};

// ============================================================
// Harnesses
// ============================================================
std::vector<double> bench_pmad_batched(Network& net, const ImageDataset& data,
                                        int epochs, int batch_size, float lr) {
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

    run_epoch(); // warmup
    std::vector<double> per_epoch;
    per_epoch.reserve(epochs);
    for (int ep = 0; ep < epochs; ++ep) {
        auto t0 = Clock::now();
        run_epoch();
        per_epoch.push_back(Ms(Clock::now() - t0).count());
    }
    return per_epoch;
}

std::vector<double> bench_eigen_batched(NetworkEigen& net, const ImageDataset& data,
                                         int epochs, int batch_size, float lr) {
    int n = data.cols(), in = data.image_size();
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);

    Eigen::MatrixXf  X(in, batch_size);
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

    run_epoch(); // warmup
    std::vector<double> per_epoch;
    per_epoch.reserve(epochs);
    for (int ep = 0; ep < epochs; ++ep) {
        auto t0 = Clock::now();
        run_epoch();
        per_epoch.push_back(Ms(Clock::now() - t0).count());
    }
    return per_epoch;
}

// ============================================================
// Main
// ============================================================
int main() {
    const std::string      data_dir   = "data/Fashion MNIST";
    const std::vector<int> arch       = { 784, 64, 64, 10 };
    const int              epochs     = 20;
    const int              batch_size = 128;
    const float            lr         = 0.01f;
    const std::string      out_csv    = "fashion_mnist_bench.csv";

    std::printf("Loading Fashion MNIST (train)...\n");
    ImageDataset data = load_fashion_mnist(data_dir, true);
    std::printf("  %d images  |  %d px/image\n", data.cols(), data.image_size());

    std::printf("\nArch:"); for (int s : arch) std::printf(" %d", s);
    std::printf("\n%d epochs  batch=%d  lr=%.4f  (1 warmup excluded)\n\n",
                epochs, batch_size, lr);

    // ---- PMAD batched + workspace + parallel assembly ----
    std::printf("[ 1/2 ] PMAD batched+ws+parallel ...\n");
    init_pmad(arch, batch_size);
    std::vector<double> pmad_ms;
    { Network net(arch); pmad_ms = bench_pmad_batched(net, data, epochs, batch_size, lr); }
    destroy_pmad();

    // ---- Eigen batched ----
    std::printf("[ 2/2 ] Eigen batched ...\n");
    std::vector<double> eigen_ms;
    { NetworkEigen net(arch); eigen_ms = bench_eigen_batched(net, data, epochs, batch_size, lr); }

    // ---- Write CSV ----
    {
        std::FILE* f = std::fopen(out_csv.c_str(), "w");
        std::fprintf(f, "backend,epoch,ms\n");
        for (int i = 0; i < epochs; ++i) {
            std::fprintf(f, "PMAD,%d,%.4f\n",  i+1, pmad_ms[i]);
            std::fprintf(f, "Eigen,%d,%.4f\n", i+1, eigen_ms[i]);
        }
        std::fclose(f);
    }

    // ---- Summary ----
    auto avg = [](const std::vector<double>& v) {
        double s = 0; for (double x : v) s += x; return s / v.size();
    };
    double p_avg = avg(pmad_ms), e_avg = avg(eigen_ms);
    int spe = (data.cols() / batch_size) * batch_size;

    std::printf("\n%-32s  %10s  %14s\n", "Backend", "ms/epoch", "samples/sec");
    std::printf("%s\n", std::string(60, '-').c_str());
    std::printf("%-32s  %10.1f  %14.0f\n", "PMAD batched+ws+parallel",
                p_avg, spe / (p_avg / 1000.0));
    std::printf("%-32s  %10.1f  %14.0f\n", "Eigen batched",
                e_avg, spe / (e_avg / 1000.0));
    std::printf("%s\n", std::string(60, '-').c_str());
    std::printf("PMAD speedup: %.3fx\n", e_avg / p_avg);
    std::printf("\nResults written to: %s\n", out_csv.c_str());
    std::printf("Run  python3 scripts/plot_fashion_mnist_bench.py  to generate figures.\n");
    return 0;
}
