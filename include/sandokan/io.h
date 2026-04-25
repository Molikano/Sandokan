#pragma once

#include "module.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// .sand file format:
//   [4B] magic    = 0x444E4153  ("SAND")
//   [4B] version  = 1
//   [4B] endian   = 0x01020304  (little-endian sentinel)
//   --- for each Linear layer in DFS (register_module) order: ---
//   [4B] rows  (uint32)
//   [4B] cols  (uint32)
//   [rows*cols * 4B] W  (column-major floats)
//   [4B] bias_size (uint32)
//   [bias_size * 4B] b

inline void save_model(const Module& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("save_model: cannot open '" + path + "' for writing");
    const uint32_t hdr[3] = { 0x444E4153u, 1u, 0x01020304u };
    std::fwrite(hdr, sizeof(uint32_t), 3, f);
    net.save(f);
    std::fclose(f);
}

inline void load_model(Module& net, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("load_model: cannot open '" + path + "'");
    uint32_t hdr[3];
    std::fread(hdr, sizeof(uint32_t), 3, f);
    if (hdr[0] != 0x444E4153u) { std::fclose(f); throw std::runtime_error("Not a .sand file"); }
    if (hdr[1] != 1u)          { std::fclose(f); throw std::runtime_error("Unsupported .sand version"); }
    if (hdr[2] != 0x01020304u) { std::fclose(f); throw std::runtime_error(".sand: endianness mismatch"); }
    net.load(f);
    std::fclose(f);
}
