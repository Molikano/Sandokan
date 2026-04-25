#include <cstdio>
#include <sandokan/allocator.h>
#include <sandokan/dataset.h>
#include <sandokan/io.h>
#include <sandokan/layers.h>
#include <sandokan/optim.h>
#include <sandokan/train.h>

// Residual block: output = ReLU(fc2(ReLU(fc1(x)))) + x
// in_features must equal out_features for the skip connection.
struct ResBlock : Module {
    Linear fc1, fc2;
    ReLU   relu1, relu2;

    ResBlock(int features) : fc1(features, features), fc2(features, features) {
        register_module(fc1);
        register_module(fc2);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return relu2.forward(fc2.forward(relu1.forward(fc1.forward(x)))) + x;
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        Eigen::MatrixXf d = fc1.backward(relu1.backward(fc2.backward(relu2.backward(dy))));
        return d + dy;
    }
};

// 784 → 64 → ResBlock(64) → 26
struct LetterNet : Module {
    Linear   proj  { 784, 64 };
    ReLU     relu0;
    ResBlock res   { 64 };
    Linear   head  { 64, 26 };
    Softmax  sm;

    LetterNet() {
        register_module(proj);
        register_module(res);
        register_module(head);
    }

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

int main() {
    const std::string      data_dir   = "data/Emnist Letters";
    // proj(784→64) + res.fc1(64→64) + res.fc2(64→64) + head(64→26)
    const std::vector<int> arch       = { 784, 64, 64, 64, 26 };
    const int              batch_size = 128;
    const int              epochs     = 30;
    const std::string      model_path = "letternet.sand";

    std::printf("Loading datasets...\n");
    ImageDataset train_set = load_emnist_letters(data_dir, true,  false);
    ImageDataset test_set  = load_emnist_letters(data_dir, false);
    std::printf("  train: %d images (%.0f MB raw, mmap-resident)\n",
                train_set.cols(),
                double(train_set.cols()) * train_set.image_size() / 1048576.0);
    std::printf("  test : %d images\n\n", test_set.cols());

    Eigen::VectorXf sample(train_set.image_size());
    train_set.get_image_col(0, sample);
    const int first_label = train_set.label(0);
    visualize(sample, first_label, static_cast<char>('A' + first_label));
    std::printf("\n");

    train_set.compute_normalization();

    init_pmad(arch, batch_size);

    // ---- Train and save ----
    {
        nn::manual_seed(42);
        LetterNet net;

        Adam     optim(1e-3f);
        LinearLR sched(optim, epochs, 1e-5f);

        std::printf("Optimizer : Adam (lr=1e-3 → 1e-5 over %d epochs via LinearLR)\n\n",
                    epochs);

        train_module(net, sched, train_set, test_set, epochs, batch_size);

        std::printf("\nSaving model to '%s'...\n", model_path.c_str());
        save_model(net, model_path);
        std::printf("Saved.\n");
    } // net destroyed here — PMAD gradient buffers returned to pool

    // ---- Load into a fresh network and verify ----
    {
        std::printf("Loading model into fresh LetterNet...\n");
        LetterNet net2;
        load_model(net2, model_path);
        double acc = compute_accuracy(net2, test_set, batch_size);
        std::printf("Loaded model test accuracy: %.2f%%\n", acc);
    }

    destroy_pmad();
    return 0;
}
