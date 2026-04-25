#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// IDX file layout (big-endian header, raw uint8 payload):
//   images: [magic=0x00000803][n][rows][cols] then n*rows*cols bytes
//   labels: [magic=0x00000801][n] then n bytes

namespace detail {
inline uint32_t be_u32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) <<  8) |  uint32_t(p[3]);
}
}

// RAII read-only POSIX mmap.
class MappedFile {
  public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path) { open(path); }
    ~MappedFile() { close(); }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept { steal(o); }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) { close(); steal(o); }
        return *this;
    }

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open failed: " + path);

        struct stat st{};
        if (::fstat(fd_, &st) < 0) {
            ::close(fd_); throw std::runtime_error("fstat failed: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);

        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            ::close(fd_); throw std::runtime_error("mmap failed: " + path);
        }
        data_ = static_cast<const uint8_t*>(p);

        // Shuffled training touches images in random order — disable the
        // kernel's readahead heuristic so we don't pull in pages we'll never use.
        ::madvise(p, size_, MADV_RANDOM);
    }

    void close() {
        if (data_) { ::munmap(const_cast<uint8_t*>(data_), size_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        size_ = 0;
    }

    const uint8_t* data() const { return data_; }
    size_t         size() const { return size_; }

  private:
    void steal(MappedFile& o) {
        data_ = o.data_; size_ = o.size_; fd_ = o.fd_;
        o.data_ = nullptr; o.size_ = 0; o.fd_ = -1;
    }

    const uint8_t* data_ = nullptr;
    size_t         size_ = 0;
    int            fd_   = -1;
};

// mmap-backed IDX dataset. No pixel data on the heap — get_image_col() reads
// straight from the OS-mapped pages, page-faulting from SSD only when an
// image isn't already resident. RSS stays bounded regardless of file size.
struct ImageDataset {
    MappedFile      img_file;
    MappedFile      lbl_file;
    const uint8_t*  pixels = nullptr;       // points inside img_file mapping
    const uint8_t*  lbls   = nullptr;       // points inside lbl_file mapping
    int             n      = 0;
    int             rows   = 0;
    int             cols_  = 0;             // image width (28 for EMNIST)
    int             pixels_per_image = 0;
    int             label_offset     = 0;
    bool            fix_orientation  = false;
    bool            normalize        = false;

    Eigen::VectorXf mean;                   // empty until compute_normalization()
    Eigen::VectorXf inv_sigma;              // 1/sigma, precomputed for speed

    int cols()       const { return n; }
    int image_size() const { return pixels_per_image; }
    int label(int idx) const { return int(lbls[idx]) + label_offset; }

    // Always returns raw 0-255 pixels regardless of the normalize flag.
    // Use this for visualization — no flag toggling needed.
    void get_raw_image_col(int idx, Eigen::Ref<Eigen::VectorXf> out) const {
        const bool was = normalize;
        const_cast<ImageDataset*>(this)->normalize = false;
        get_image_col(idx, out);
        const_cast<ImageDataset*>(this)->normalize = was;
    }

    // Copy normalization params from a source dataset (e.g. training set)
    // and activate normalization. Avoids repeating mean/inv_sigma manually.
    void apply_normalization_from(const ImageDataset& src) {
        mean      = src.mean;
        inv_sigma = src.inv_sigma;
        normalize = true;
    }

    // Hot path: writes one image (orientation-fixed, optionally normalized)
    // into `out`. `out` must have size == pixels_per_image. Called from the
    // assembly thread; this is where the page-fault magic happens.
    void get_image_col(int idx, Eigen::Ref<Eigen::VectorXf> out) const {
        const uint8_t* src = pixels + size_t(idx) * pixels_per_image;

        if (fix_orientation) {
            // EMNIST quirk: out[r*W + c] = src[(W-1-c)*H + r]
            const int H = rows, W = cols_;
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c)
                    out(r * W + c) = float(src[(W - 1 - c) * H + r]);
        } else {
            for (int p = 0; p < pixels_per_image; ++p)
                out(p) = float(src[p]);
        }

        if (normalize)
            out.array() = (out.array() - mean.array()) * inv_sigma.array();
    }

    // One-shot streaming pass: page-fault every image in once, accumulate
    // per-pixel sum and sum-of-squares, then derive mean / inv_sigma.
    // Computed against the orientation-fixed view so it lines up with
    // get_image_col() output.
    void compute_normalization() {
        if (mean.size() == pixels_per_image) { normalize = true; return; }

        Eigen::ArrayXd sum   = Eigen::ArrayXd::Zero(pixels_per_image);
        Eigen::ArrayXd sumsq = Eigen::ArrayXd::Zero(pixels_per_image);
        Eigen::VectorXf tmp(pixels_per_image);

        const bool was_norm = normalize;
        normalize = false;                  // raw pixels for this pass
        for (int i = 0; i < n; ++i) {
            get_image_col(i, tmp);
            Eigen::ArrayXd v = tmp.cast<double>().array();
            sum   += v;
            sumsq += v.square();
        }
        normalize = was_norm;

        Eigen::ArrayXd mean_d  = sum   / double(n);
        Eigen::ArrayXd var_d   = (sumsq / double(n) - mean_d.square()).max(0.0);
        Eigen::ArrayXd sigma_d = var_d.sqrt();

        mean      = mean_d.cast<float>().matrix();
        inv_sigma = sigma_d.unaryExpr([](double s) { return s < 1e-8 ? 1.0 : 1.0 / s; })
                          .cast<float>().matrix();
        normalize = true;
    }
};

// ASCII-art visualizer — expects raw 0–255 floats (i.e. unnormalized).
// Pull a sample with get_image_col() into a temp VectorXf before calling.
inline void visualize(const Eigen::Ref<const Eigen::VectorXf>& img,
                      int label, char letter = 0) {
    static const char* shades = " .:-=+*#%@";
    if (letter) std::printf("Label: %d ('%c')\n", label, letter);
    else        std::printf("Label: %d\n", label);
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            int idx = int(img(r * 28 + c) / 256.0f * 10);
            if (idx < 0) idx = 0; if (idx > 9) idx = 9;
            std::printf("%c%c", shades[idx], shades[idx]);
        }
        std::printf("\n");
    }
}

inline ImageDataset load_emnist_letters(const std::string& data_dir, bool train,
                                        bool do_normalize = true) {
    const std::string split = train ? "train" : "test";
    ImageDataset ds;
    ds.img_file.open(data_dir + "/emnist-letters-" + split + "-images-idx3-ubyte");
    ds.lbl_file.open(data_dir + "/emnist-letters-" + split + "-labels-idx1-ubyte");

    if (ds.img_file.size() < 16 ||
        detail::be_u32(ds.img_file.data()) != 0x00000803u)
        throw std::runtime_error("Bad IDX image magic / size");
    if (ds.lbl_file.size() < 8 ||
        detail::be_u32(ds.lbl_file.data()) != 0x00000801u)
        throw std::runtime_error("Bad IDX label magic / size");

    const uint32_t img_n = detail::be_u32(ds.img_file.data() + 4);
    const uint32_t rows  = detail::be_u32(ds.img_file.data() + 8);
    const uint32_t cols  = detail::be_u32(ds.img_file.data() + 12);
    const uint32_t lbl_n = detail::be_u32(ds.lbl_file.data() + 4);

    if (img_n != lbl_n) throw std::runtime_error("Image/label count mismatch");
    if (size_t(16) + size_t(img_n) * rows * cols != ds.img_file.size())
        throw std::runtime_error("IDX image file size mismatch");
    if (size_t(8) + size_t(lbl_n) != ds.lbl_file.size())
        throw std::runtime_error("IDX label file size mismatch");

    ds.pixels           = ds.img_file.data() + 16;
    ds.lbls             = ds.lbl_file.data() + 8;
    ds.n                = int(img_n);
    ds.rows             = int(rows);
    ds.cols_            = int(cols);
    ds.pixels_per_image = int(rows * cols);
    ds.label_offset     = -1;               // EMNIST letters are 1..26 → 0..25
    ds.fix_orientation  = true;

    if (do_normalize) ds.compute_normalization();
    return ds;
}

inline ImageDataset load_fashion_mnist(const std::string& data_dir, bool train,
                                       bool do_normalize = true) {
    const std::string img_path = data_dir + (train ? "/train-images-idx3-ubyte"
                                                    : "/t10k-images-idx3-ubyte");
    const std::string lbl_path = data_dir + (train ? "/train-labels-idx1-ubyte"
                                                    : "/t10k-labels-idx1-ubyte");
    ImageDataset ds;
    ds.img_file.open(img_path);
    ds.lbl_file.open(lbl_path);

    if (ds.img_file.size() < 16 ||
        detail::be_u32(ds.img_file.data()) != 0x00000803u)
        throw std::runtime_error("Bad IDX image magic / size: " + img_path);
    if (ds.lbl_file.size() < 8 ||
        detail::be_u32(ds.lbl_file.data()) != 0x00000801u)
        throw std::runtime_error("Bad IDX label magic / size: " + lbl_path);

    const uint32_t img_n = detail::be_u32(ds.img_file.data() + 4);
    const uint32_t rows  = detail::be_u32(ds.img_file.data() + 8);
    const uint32_t cols  = detail::be_u32(ds.img_file.data() + 12);
    const uint32_t lbl_n = detail::be_u32(ds.lbl_file.data() + 4);

    if (img_n != lbl_n) throw std::runtime_error("Image/label count mismatch");
    if (size_t(16) + size_t(img_n) * rows * cols != ds.img_file.size())
        throw std::runtime_error("IDX image file size mismatch");
    if (size_t(8) + size_t(lbl_n) != ds.lbl_file.size())
        throw std::runtime_error("IDX label file size mismatch");

    ds.pixels           = ds.img_file.data() + 16;
    ds.lbls             = ds.lbl_file.data() + 8;
    ds.n                = int(img_n);
    ds.rows             = int(rows);
    ds.cols_            = int(cols);
    ds.pixels_per_image = int(rows * cols);
    ds.label_offset     = 0;
    ds.fix_orientation  = false;

    if (do_normalize) ds.compute_normalization();
    return ds;
}
