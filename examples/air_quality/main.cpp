#include <cstdio>
#include <fstream>
#include <sandokan/allocator.h>
#include <sandokan/layers.h>
#include <sandokan/optim.h>
#include <sandokan/tabular.h>
#include <sandokan/train.h>

// Load the UCI Air Quality dataset.
//
// Format quirks handled here:
//   - Semicolon delimiter
//   - European decimal separator (comma → dot)
//   - -200 sentinel for missing values → rows dropped
//   - Trailing empty columns (;;) → ignored
//
// Columns used:
//   Features (11): PT08.S1(CO), C6H6(GT), PT08.S2(NMHC), NOx(GT),
//                  PT08.S3(NOx), NO2(GT), PT08.S4(NO2), PT08.S5(O3), T, RH, AH
//   Target        : CO(GT)  — ground truth CO concentration (mg/m³)
//   Skipped       : Date, Time, NMHC(GT) (mostly missing)
TabularDataset load_air_quality(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    // Column indices in the raw CSV (0-based)
    const int TARGET_COL = 2;
    const std::vector<int> FEAT_COLS = { 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };

    auto to_float = [](std::string s) -> float {
        for (char& c : s) if (c == ',') c = '.';
        return std::stof(s);
    };

    std::vector<std::vector<float>> feat_rows;
    std::vector<float>              target_vals;

    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == ';') continue;

        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ';')) fields.push_back(tok);
        if (fields.size() < 15) continue;

        // Parse target
        float tgt = to_float(fields[TARGET_COL]);
        if (tgt <= -200.0f) continue; // missing target

        // Parse features — skip row if any are missing
        std::vector<float> feat(FEAT_COLS.size());
        bool valid = true;
        for (int i = 0; i < (int)FEAT_COLS.size(); ++i) {
            feat[i] = to_float(fields[FEAT_COLS[i]]);
            if (feat[i] <= -200.0f) { valid = false; break; }
        }
        if (!valid) continue;

        feat_rows.push_back(std::move(feat));
        target_vals.push_back(tgt);
    }

    const int N = (int)feat_rows.size();
    const int F = (int)FEAT_COLS.size();
    std::printf("Loaded %d usable rows  %d features\n", N, F);

    Eigen::MatrixXf X(F, N);
    Eigen::VectorXf y(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < F; ++j) X(j, i) = feat_rows[i][j];
        y(i) = target_vals[i];
    }
    return TabularDataset::from_matrices(X, y);
}

// 11 → 64 → 32 → 1  (linear output, no activation — regression)
struct AirNet : Module {
    Submodule<Linear> fc1  { *this, 11, 64 };
    ReLU              relu1;
    Submodule<Linear> fc2  { *this, 64, 32 };
    ReLU              relu2;
    Submodule<Linear> head { *this, 32,  1 };

    AirNet() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return head.forward(relu2.forward(fc2.forward(relu1.forward(fc1.forward(x)))));
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return fc1.backward(relu1.backward(fc2.backward(relu2.backward(head.backward(dy)))));
    }
};

int main() {
    const std::string data_path = "data/Air Quality/AirQualityUCI.csv";
    const int   epochs     = 60;
    const int   batch_size = 64;

    // ---- Load ----
    std::printf("Loading Air Quality dataset...\n");
    TabularDataset full = load_air_quality(data_path);

    // Shuffle before split (data is time-ordered)
    {
        std::mt19937 rng(0);
        const int N = full.cols();
        std::vector<int> perm(N); std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        Eigen::MatrixXf Xs(full.X.rows(), N);
        Eigen::VectorXf ys(N);
        for (int i = 0; i < N; ++i) { Xs.col(i) = full.X.col(perm[i]); ys(i) = full.targets(perm[i]); }
        full.X = std::move(Xs); full.targets = std::move(ys);
    }

    // 80/20 split
    const int n_train = (full.cols() * 4) / 5;
    TabularDataset train_set, test_set;
    train_set.X       = full.X.leftCols(n_train);
    train_set.targets = full.targets.head(n_train);
    test_set.X        = full.X.rightCols(full.cols() - n_train);
    test_set.targets  = full.targets.tail(full.cols() - n_train);

    train_set.compute_normalization();
    test_set.apply_normalization_from(train_set);

    train_set.compute_target_normalization();
    test_set.apply_target_normalization_from(train_set);

    std::printf("  train: %d  test: %d\n", train_set.cols(), test_set.cols());
    std::printf("  CO(GT) mean=%.3f  std=%.3f mg/m³\n\n",
                train_set.target_mean_, 1.0f / train_set.target_scale_);

    // ---- Train ----
    {
        nn::manual_seed(42);
        AirNet net;
        init_pmad_for();

        Adam     optim(1e-3f);
        LinearLR sched(optim, epochs, 1e-5f);

        std::printf("Architecture : 11 → 64(ReLU) → 32(ReLU) → 1\n");
        std::printf("Optimizer    : Adam  lr 1e-3 → 1e-5  (%d epochs, LinearLR)\n\n", epochs);

        train_regression(net, sched, train_set, test_set, epochs, batch_size);
    }

    destroy_pmad();
    return 0;
}
