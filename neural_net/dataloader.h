#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ImageDataset {
    std::vector<Eigen::VectorXd> images;
    std::vector<int>             labels;
};

// IDX files store multi-byte integers in big-endian order
static uint32_t read_be_uint32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

static std::vector<Eigen::VectorXd> load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic = read_be_uint32(f);
    if (magic != 0x00000803)
        throw std::runtime_error("Bad magic number in image file: " + path);

    uint32_t n      = read_be_uint32(f);
    uint32_t rows   = read_be_uint32(f);
    uint32_t cols   = read_be_uint32(f);
    uint32_t pixels = rows * cols;

    std::vector<Eigen::VectorXd> images(n);
    for (uint32_t i = 0; i < n; ++i) {
        Eigen::VectorXd img(pixels);
        for (uint32_t p = 0; p < pixels; ++p) {
            uint8_t byte;
            f.read(reinterpret_cast<char*>(&byte), 1);
            img(p) = static_cast<double>(byte);
        }
        images[i] = std::move(img);
    }
    return images;
}

static std::vector<int> load_labels(const std::string& path, int label_offset = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint32_t magic = read_be_uint32(f);
    if (magic != 0x00000801)
        throw std::runtime_error("Bad magic number in label file: " + path);

    uint32_t n = read_be_uint32(f);
    std::vector<int> labels(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t byte;
        f.read(reinterpret_cast<char*>(&byte), 1);
        labels[i] = static_cast<int>(byte) + label_offset;
    }
    return labels;
}

// Computes per-pixel mean and std across all images, then normalises in-place
static void normalize(std::vector<Eigen::VectorXd>& images) {
    int n      = static_cast<int>(images.size());
    int pixels = static_cast<int>(images[0].size());

    Eigen::VectorXd mu = Eigen::VectorXd::Zero(pixels);
    for (const auto& img : images) mu += img;
    mu /= n;

    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(pixels);
    for (const auto& img : images) {
        Eigen::VectorXd diff = img - mu;
        sigma += diff.cwiseProduct(diff);
    }
    sigma = (sigma / n).cwiseSqrt();
    sigma = sigma.unaryExpr([](double v) { return v < 1e-8 ? 1.0 : v; });

    for (auto& img : images)
        img = (img - mu).cwiseQuotient(sigma);
}

// EMNIST images are stored transposed relative to their correct orientation.
// Fix: for each pixel at (row, col), read from (27-col, row) in the stored data.
static void fix_emnist_orientation(std::vector<Eigen::VectorXd>& images) {
    for (auto& img : images) {
        Eigen::VectorXd fixed(784);
        for (int r = 0; r < 28; ++r)
            for (int c = 0; c < 28; ++c)
                fixed(r * 28 + c) = img((27 - c) * 28 + r);
        img = fixed;
    }
}

// Prints a 28x28 image (raw 0-255 pixel values) as ASCII art
inline void visualize(const Eigen::VectorXd& img, int label, char letter = 0) {
    static const char* shades = " .:-=+*#%@";
    if (letter)
        std::printf("Label: %d ('%c')\n", label, letter);
    else
        std::printf("Label: %d\n", label);

    for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
            double v = img(row * 28 + col);
            int idx  = static_cast<int>(v / 256.0 * 10);
            char c   = shades[idx];
            std::printf("%c%c", c, c);
        }
        std::printf("\n");
    }
}

// ---------- Loaders ----------

inline ImageDataset load_mnist(const std::string& data_dir, bool train,
                                bool do_normalize = true) {
    std::string img_file = train ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
    std::string lbl_file = train ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";

    ImageDataset ds;
    ds.images = load_images(data_dir + "/" + img_file);
    ds.labels = load_labels(data_dir + "/" + lbl_file);

    if (ds.images.size() != ds.labels.size())
        throw std::runtime_error("Image/label count mismatch");

    if (do_normalize) normalize(ds.images);
    return ds;
}

inline ImageDataset load_emnist_letters(const std::string& data_dir, bool train,
                                         bool do_normalize = true) {
    std::string split    = train ? "train" : "test";
    std::string img_file = "emnist-letters-" + split + "-images-idx3-ubyte";
    std::string lbl_file = "emnist-letters-" + split + "-labels-idx1-ubyte";

    ImageDataset ds;
    ds.images = load_images(data_dir + "/" + img_file);
    // Labels are 1-26; subtract 1 to make them 0-indexed (0=A … 25=Z)
    ds.labels = load_labels(data_dir + "/" + lbl_file, /*label_offset=*/-1);

    if (ds.images.size() != ds.labels.size())
        throw std::runtime_error("Image/label count mismatch");

    fix_emnist_orientation(ds.images);
    if (do_normalize) normalize(ds.images);
    return ds;
}
