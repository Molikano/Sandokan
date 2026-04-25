#pragma once

#include "dataset.h"
#include "module.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// .sand file format:
//   [4B] magic    = 0x444E4153  ("SAND")
//   [4B] version  = 1
//   [4B] endian   = 0x01020304  (little-endian sentinel)
//   [4B] norm_size (0 = no normalization block)
//   if norm_size > 0:
//     [norm_size * 4B] mean
//     [norm_size * 4B] inv_sigma
//   --- for each Linear layer in DFS (register_module) order: ---
//   [4B] rows  (uint32)
//   [4B] cols  (uint32)
//   [rows*cols * 4B] W  (column-major floats)
//   [4B] bias_size (uint32)
//   [bias_size * 4B] b

static constexpr uint32_t SAND_MAGIC   = 0x444E4153u;
static constexpr uint32_t SAND_VERSION = 1u;
static constexpr uint32_t SAND_ENDIAN  = 0x01020304u;

// Save weights only.
inline void save_model(const Module& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_model: cannot open '" + path + "' for writing");
    const uint32_t hdr[4] = { SAND_MAGIC, SAND_VERSION, SAND_ENDIAN, 0u };
    std::fwrite(hdr, sizeof(uint32_t), 4, f);
    net.save(f);
    std::fclose(f);
}

// Save weights + normalization params from any dataset with mean/inv_sigma fields.
// Enables self-contained inference without the original training set.
template<typename Dataset>
inline void save_model(const Module& net, const std::string& path,
                       const Dataset& ds) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_model: cannot open '" + path + "' for writing");

    const uint32_t norm_size = static_cast<uint32_t>(ds.mean.size());
    const uint32_t hdr[4] = { SAND_MAGIC, SAND_VERSION, SAND_ENDIAN, norm_size };
    std::fwrite(hdr, sizeof(uint32_t), 4, f);

    if (norm_size > 0) {
        std::fwrite(ds.mean.data(),      sizeof(float), norm_size, f);
        std::fwrite(ds.inv_sigma.data(), sizeof(float), norm_size, f);
    }

    net.save(f);
    std::fclose(f);
}

// Load weights only (normalization block is skipped if present).
inline void load_model(Module& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_model: cannot open '" + path + "'");

    uint32_t hdr[4];
    std::fread(hdr, sizeof(uint32_t), 4, f);
    if (hdr[0] != SAND_MAGIC)   { std::fclose(f); throw std::runtime_error("Not a .sand file"); }
    if (hdr[1] != SAND_VERSION) { std::fclose(f); throw std::runtime_error("Unsupported .sand version"); }
    if (hdr[2] != SAND_ENDIAN)  { std::fclose(f); throw std::runtime_error(".sand: endianness mismatch"); }

    const uint32_t norm_size = hdr[3];
    if (norm_size > 0)
        std::fseek(f, long(norm_size * 2 * sizeof(float)), SEEK_CUR);

    net.load(f);
    std::fclose(f);
}

// Load weights and restore normalization params into any dataset with mean/inv_sigma fields.
template<typename Dataset>
inline void load_model(Module& net, const std::string& path, Dataset& ds) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_model: cannot open '" + path + "'");

    uint32_t hdr[4];
    std::fread(hdr, sizeof(uint32_t), 4, f);
    if (hdr[0] != SAND_MAGIC)   { std::fclose(f); throw std::runtime_error("Not a .sand file"); }
    if (hdr[1] != SAND_VERSION) { std::fclose(f); throw std::runtime_error("Unsupported .sand version"); }
    if (hdr[2] != SAND_ENDIAN)  { std::fclose(f); throw std::runtime_error(".sand: endianness mismatch"); }

    const uint32_t norm_size = hdr[3];
    if (norm_size > 0) {
        ds.mean.resize(norm_size);
        ds.inv_sigma.resize(norm_size);
        std::fread(ds.mean.data(),      sizeof(float), norm_size, f);
        std::fread(ds.inv_sigma.data(), sizeof(float), norm_size, f);
        ds.normalize = true;
    }

    net.load(f);
    std::fclose(f);
}
