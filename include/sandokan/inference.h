#pragma once

#include "dataset.h"
#include "module.h"
#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

struct Prediction {
    int   label;
    float confidence;
};

// Run a single already-normalized sample through the network.
inline Prediction predict(Module& net, const Eigen::VectorXf& x) {
    Eigen::MatrixXf col = x;           // (features, 1)
    Eigen::MatrixXf out = net.forward(col);
    Eigen::Index pred;
    float conf = out.col(0).maxCoeff(&pred);
    return { static_cast<int>(pred), conf };
}

// Normalize a raw pixel vector (0-255 floats) using stored params, then predict.
inline Prediction predict_raw(Module& net, const Eigen::VectorXf& raw,
                              const ImageDataset& ds) {
    Eigen::VectorXf x = (raw.array() - ds.mean.array()) * ds.inv_sigma.array();
    return predict(net, x);
}

// Top-k predictions for an already-normalized sample.
inline std::vector<Prediction> predict_topk(Module& net,
                                            const Eigen::VectorXf& x,
                                            int k) {
    Eigen::MatrixXf col = x;
    Eigen::VectorXf probs = net.forward(col).col(0);

    const int n = static_cast<int>(probs.size());
    k = std::min(k, n);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs(a) > probs(b); });

    std::vector<Prediction> out(k);
    for (int i = 0; i < k; ++i)
        out[i] = { idx[i], probs(idx[i]) };
    return out;
}

// Full test-set accuracy using batched forward (same path as training eval).
inline double predict_accuracy(Module& net, const ImageDataset& ds,
                               int batch_size = 256) {
    int correct = 0, n = ds.cols();
    Eigen::MatrixXf X(ds.image_size(), batch_size);
    for (int s = 0; s < n; s += batch_size) {
        const int e = std::min(s + batch_size, n), bs = e - s;
        if (bs == batch_size) {
            for (int i = 0; i < bs; ++i) ds.get_image_col(s + i, X.col(i));
            Eigen::MatrixXf probs = net.forward(X);
            for (int j = 0; j < bs; ++j) {
                Eigen::Index pred;
                probs.col(j).maxCoeff(&pred);
                if (static_cast<int>(pred) == ds.label(s + j)) ++correct;
            }
        } else {
            Eigen::VectorXf x(ds.image_size());
            for (int i = s; i < e; ++i) {
                ds.get_image_col(i, x);
                if (predict(net, x).label == ds.label(i)) ++correct;
            }
        }
    }
    return 100.0 * correct / n;
}

// Visualize a raw image + top-k predictions side by side.
// letter_names: e.g. {'A', 'B', ...} for EMNIST letters.
inline void show_prediction(const Eigen::VectorXf& raw_pixels,
                            int true_label,
                            const std::vector<Prediction>& topk,
                            const std::string& label_names) {
    static const char* shades = " .:-=+*#%@";
    const int H = 28, W = 28;

    const bool correct = (!topk.empty() && topk[0].label == true_label);
    std::printf("Ground truth: %c   Prediction: %c (%.1f%%)  %s\n",
                label_names[true_label], label_names[topk[0].label],
                topk[0].confidence * 100.0f,
                correct ? "OK" : "WRONG");

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = int(raw_pixels(r * W + c) / 256.0f * 10);
            idx = std::max(0, std::min(9, idx));
            std::printf("%c%c", shades[idx], shades[idx]);
        }
        if (r < (int)topk.size()) {
            std::printf("   #%d  %c  %.2f%%",
                        r + 1,
                        label_names[topk[r].label],
                        topk[r].confidence * 100.0f);
        }
        std::printf("\n");
    }
    std::printf("\n");
}
