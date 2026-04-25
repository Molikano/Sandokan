#include <cstdio>
#include <sandokan/allocator.h>
#include <sandokan/layers.h>
#include <sandokan/optim.h>
#include <sandokan/tabular.h>
#include <sandokan/train.h>

// Iris classification (4 features → 3 classes) as a demo.
// Replace load_csv("iris.csv") with your own dataset (housing, etc.).
//
// Expected CSV format — no header, comma-separated:
//   5.1,3.5,1.4,0.2,0  ← 4 features, class label in last column
//
// For regression (e.g. house prices):
//   1. Change head to Linear { features, 1 } with no Softmax
//   2. Use MSE loss instead of CrossEntropyLoss (not yet in library)
//   3. Use target(idx) instead of label(idx) for evaluation

struct IrisNet : Module {
    Submodule<Linear>  fc1 { *this, 4, 16 };
    ReLU               relu;
    Submodule<Linear>  head { *this, 16, 3 };
    Softmax            sm;

    IrisNet() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return sm.forward(head.forward(relu.forward(fc1.forward(x))));
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return fc1.backward(relu.backward(head.backward(sm.backward(dy))));
    }
};

// Generate synthetic Iris-like data so the demo runs without a real file.
// 150 samples, 4 features, 3 classes (50 each) — linearly separable blobs.
TabularDataset make_synthetic_iris() {
    const int N = 150, F = 4, C = 3;
    Eigen::MatrixXf X(F, N);
    Eigen::VectorXf y(N);

    std::mt19937 rng(0);
    std::normal_distribution<float> noise(0.0f, 0.3f);

    const float centers[3][4] = {
        { 5.0f, 3.4f, 1.5f, 0.3f },
        { 5.9f, 2.8f, 4.3f, 1.3f },
        { 6.6f, 3.0f, 5.6f, 2.0f },
    };

    for (int i = 0; i < N; ++i) {
        int cls = i / (N / C);
        y(i) = float(cls);
        for (int f = 0; f < F; ++f)
            X(f, i) = centers[cls][f] + noise(rng);
    }

    return TabularDataset::from_matrices(X, y);
}

int main() {
    // ---- Load data ----
    // To use a real CSV:  auto full = load_csv("iris.csv", /*label_col=*/-1, /*header=*/true);
    auto full = make_synthetic_iris();
    std::printf("Dataset: %d samples  %d features  label range [%d, %d]\n\n",
                full.cols(), full.image_size(),
                full.label(0),
                full.label(full.cols() - 1));

    // Split 80/20 train/test
    const int n_train = (full.cols() * 4) / 5;
    const int n_test  = full.cols() - n_train;

    TabularDataset train_set, test_set;
    train_set.X       = full.X.leftCols(n_train);
    train_set.targets = full.targets.head(n_train);
    test_set.X        = full.X.rightCols(n_test);
    test_set.targets  = full.targets.tail(n_test);

    train_set.compute_normalization();
    test_set.apply_normalization_from(train_set);

    // ---- Train ----
    {
        nn::manual_seed(42);
        IrisNet net;
        init_pmad_for();

        Adam     optim(1e-2f);
        LinearLR sched(optim, 50, 1e-4f);

        std::printf("Architecture : 4 → 16(ReLU) → 3(Softmax)\n");
        std::printf("Optimizer    : Adam  lr 1e-2 → 1e-4  (50 epochs)\n\n");

        train_module(net, sched, train_set, test_set, 50, 16);
    }

    destroy_pmad();
    return 0;
}
