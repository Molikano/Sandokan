#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Images stored as [pixels, N] column-major — images.col(i) = sample i.
// One contiguous allocation vs 124,800 separate VectorXf heap allocs.
struct ImageDataset {
    Eigen::MatrixXf  images;
    std::vector<int> labels;
    int cols() const { return static_cast<int>(images.cols()); }
};

static uint32_t read_be_uint32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

// Per-pixel (row-wise) zero-mean unit-std normalization across the dataset
inline void normalize(Eigen::MatrixXf& images) {
    Eigen::VectorXf mean  = images.rowwise().mean();
    images.colwise() -= mean;
    Eigen::VectorXf sigma = (images.array().square().rowwise().mean()).sqrt();
    sigma = sigma.unaryExpr([](float v) { return v < 1e-8f ? 1.0f : v; });
    images.array().colwise() /= sigma.array();
}

// ASCII-art visualiser — expects raw 0–255 float values per pixel
inline void visualize(const Eigen::Ref<const Eigen::VectorXf>& img,
                      int label, char letter = 0) {
    static const char* shades = " .:-=+*#%@";
    if (letter) std::printf("Label: %d ('%c')\n", label, letter);
    else        std::printf("Label: %d\n", label);
    for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
            int idx = static_cast<int>(img(row * 28 + col) / 256.0f * 10);
            std::printf("%c%c", shades[idx], shades[idx]);
        }
        std::printf("\n");
    }
}

// Loads IDX image file directly into a [pixels, N] MatrixXf.
// EMNIST orientation fix: output[r*28+c] = raw[(27-c)*28+r]
static Eigen::MatrixXf load_images_matrix(const std::string& path,
                                           bool fix_orientation) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    if (read_be_uint32(f) != 0x00000803u)
        throw std::runtime_error("Bad magic in image file: " + path);

    uint32_t n      = read_be_uint32(f);
    uint32_t rows   = read_be_uint32(f);
    uint32_t cols_  = read_be_uint32(f);
    uint32_t pixels = rows * cols_;

    std::vector<uint8_t> buf(pixels);
    Eigen::MatrixXf      mat(pixels, n);

    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buf.data()), pixels);
        if (fix_orientation) {
            for (uint32_t r = 0; r < rows; ++r)
                for (uint32_t c = 0; c < cols_; ++c)
                    mat(r * cols_ + c, i) =
                        static_cast<float>(buf[(cols_ - 1 - c) * rows + r]);
        } else {
            for (uint32_t p = 0; p < pixels; ++p)
                mat(p, i) = static_cast<float>(buf[p]);
        }
    }
    return mat;
}

static std::vector<int> load_labels(const std::string& path, int offset = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    if (read_be_uint32(f) != 0x00000801u)
        throw std::runtime_error("Bad magic in label file: " + path);

    uint32_t n = read_be_uint32(f);
    std::vector<int> labels(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t b;
        f.read(reinterpret_cast<char*>(&b), 1);
        labels[i] = static_cast<int>(b) + offset;
    }
    return labels;
}

inline ImageDataset load_emnist_letters(const std::string& data_dir, bool train,
                                         bool do_normalize = true) {
    std::string split = train ? "train" : "test";
    ImageDataset ds;
    ds.images = load_images_matrix(
        data_dir + "/emnist-letters-" + split + "-images-idx3-ubyte", true);
    ds.labels = load_labels(
        data_dir + "/emnist-letters-" + split + "-labels-idx1-ubyte", -1);

    if (static_cast<size_t>(ds.cols()) != ds.labels.size())
        throw std::runtime_error("Image/label count mismatch");

    if (do_normalize) normalize(ds.images);
    return ds;
}
