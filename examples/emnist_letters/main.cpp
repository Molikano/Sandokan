#include <cstdio>
#include <sandokan/allocator.h>
#include <sandokan/dataset.h>
#include <sandokan/inference.h>
#include <sandokan/io.h>
#include <sandokan/layers.h>
#include <sandokan/optim.h>
#include <sandokan/train.h>

// Residual block: output = ReLU(fc2(ReLU(fc1(x)))) + x
struct ResBlock : Module {
    Submodule<Linear> fc1 { *this, 64, 64 };
    Submodule<Linear> fc2 { *this, 64, 64 };
    ReLU relu1, relu2;

    ResBlock() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return relu2.forward(fc2.forward(relu1.forward(fc1.forward(x)))) + x;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return fc1.backward(relu1.backward(fc2.backward(relu2.backward(dy)))) + dy;
    }
};

// 784 → 64 → ResBlock(64) → 26
struct LetterNet : Module {
    Submodule<Linear>   proj { *this, 784, 64 };
    ReLU                relu0;
    Submodule<ResBlock> res  { *this };
    Submodule<Linear>   head { *this, 64, 26 };
    Softmax             sm;

    LetterNet() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        auto h = relu0.forward(proj.forward(x));
        h = res.forward(h);
        return sm.forward(head.forward(h));
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        auto d = head.backward(sm.backward(dy));
        d = res.backward(d);
        return proj.backward(relu0.backward(d));
    }
};

static const std::string LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int main() {
    const std::string data_dir   = "data/Emnist Letters";
    const int         batch_size = 128;
    const int         epochs     = 30;
    const std::string model_path = "letternet.sand";

    // ---- Load data ----
    std::printf("Loading datasets...\n");
    ImageDataset train_set = load_emnist_letters(data_dir, true,  false);
    ImageDataset test_set  = load_emnist_letters(data_dir, false, false);
    std::printf("  train: %d images  |  test: %d images\n\n",
                train_set.cols(), test_set.cols());

    train_set.compute_normalization();
    test_set.apply_normalization_from(train_set);

    // ---- Train ----
    {
        nn::manual_seed(42);
        LetterNet net;
        init_pmad_for();

        Adam     optim(1e-3f);
        LinearLR sched(optim, epochs, 1e-5f);

        std::printf("Architecture : 784 → 64 → ResBlock(64) → 26\n");
        std::printf("Optimizer    : Adam  lr 1e-3 → 1e-5  (%d epochs, LinearLR)\n\n", epochs);

        train_module(net, sched, train_set, test_set, epochs, batch_size);

        std::printf("\nSaving '%s' (weights + normalization)...\n", model_path.c_str());
        save_model(net, model_path, train_set);
        std::printf("Done.\n\n");
    }

    // ---- Inference ----
    std::printf("=== Inference pipeline ===\n\n");
    {
        ImageDataset infer_set = load_emnist_letters(data_dir, false, false);
        LetterNet    net2;
        load_model(net2, model_path, infer_set);  // restores weights + norm params

        double acc = predict_accuracy(net2, infer_set, batch_size);
        std::printf("Test accuracy (loaded model): %.2f%%\n\n", acc);

        std::printf("--- Sample predictions (top-3) ---\n\n");
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dist(0, infer_set.cols() - 1);

        Eigen::VectorXf raw(infer_set.image_size());
        Eigen::VectorXf norm(infer_set.image_size());

        for (int i = 0; i < 8; ++i) {
            const int idx = dist(rng);
            infer_set.get_raw_image_col(idx, raw);   // no flag toggling
            infer_set.get_image_col(idx, norm);
            auto top3 = predict_topk(net2, norm, 3);
            show_prediction(raw, infer_set.label(idx), top3, LETTERS);
        }
    }

    destroy_pmad();
    return 0;
}
