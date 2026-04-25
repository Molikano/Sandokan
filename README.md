# Sandokan

A from-scratch C++ neural network training engine. No Python. No PyTorch dependency. Drop a single header into any C++ project and get a complete training pipeline — classification, regression, custom datasets, optimizers, learning rate schedules, model persistence — backed by a custom slab allocator and Apple AMX acceleration.

---

## Why

Training neural networks in C++ today means dragging in LibTorch (a ~1 GB dependency with a Python runtime assumption) or writing raw BLAS calls with no abstraction. Sandokan fills the gap: a PyTorch-style API at native C++ speed, designed for environments where Python is not an option — embedded systems, edge devices, game engines, trading systems, or any latency-sensitive C++ codebase that needs on-device learning.

---

## Core Features

### Module System

Define networks by composing typed submodules. `Submodule<T>` auto-registers with the parent on construction — you cannot accidentally forget a `register_module` call.

```cpp
struct LetterNet : Module {
    Submodule<Linear>   proj { *this, 784, 64 };
    ReLU                relu;
    Submodule<Linear>   head { *this, 64,  26 };

    LetterNet() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return head.forward(relu.forward(proj.forward(x)));
    }
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return proj.backward(relu.backward(head.backward(dy)));
    }
};
```

Residual blocks are first-class:

```cpp
struct ResBlock : Module {
    Submodule<Linear> fc1 { *this, 64, 64 };
    ReLU              relu1;
    Submodule<Linear> fc2 { *this, 64, 64 };
    ReLU              relu2;

    ResBlock() = default;

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        return relu2.forward(fc2.forward(relu1.forward(fc1.forward(x)))) + x;
    }
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dy) override {
        return fc1.backward(relu1.backward(fc2.backward(relu2.backward(dy)))) + dy;
    }
};
```

### PMAD Slab Allocator

All gradient buffers are served from a pre-allocated contiguous slab. Size classes are derived automatically from the network topology — zero `malloc`/`free` during training, no fragmentation over long runs. Combined with Apple Accelerate/AMX for batched GEMM, this is the engine's primary performance lever.

```cpp
LetterNet net;
init_pmad_for();   // reads topology, allocates slab, migrates gradients in one pass
```

### Optimizers and Learning Rate Schedulers

```cpp
Adam     optim(1e-3f);
LinearLR sched(optim, 150 /*total epochs*/, 1e-5f /*end lr*/);

train_module(net, sched, train_set, test_set, 150, 128);
```

| Optimizer | Notes |
|-----------|-------|
| `SGD` | Stochastic gradient descent with fixed lr |
| `Adam` | Adaptive moments, bias-corrected |

| Scheduler | Notes |
|-----------|-------|
| `LinearLR` | Linearly decays lr from start to end over N epochs |

### Loss Functions

| Loss | Output activation | Use case |
|------|------------------|----------|
| `CrossEntropyLoss` | `Softmax` | Multi-class classification |
| `BCELoss` | `Sigmoid` | Binary / multi-label classification |
| `MSELoss` | Linear (none) | Regression |

### Dataset Abstractions

**`ImageDataset`** — mmap-backed IDX loader. Images are page-faulted on demand; RSS stays bounded regardless of dataset size.

```cpp
ImageDataset train = load_emnist_letters("data/Emnist Letters", /*train=*/true);
ImageDataset test  = load_fashion_mnist("data/Fashion MNIST",   /*train=*/false);
train.compute_normalization();
test.apply_normalization_from(train);
```

**`TabularDataset`** — in-memory column-major store for numeric CSVs or Eigen matrices.

```cpp
// From CSV — last column is the target by default
TabularDataset ds = load_csv("boston.csv");
ds.compute_normalization();
ds.compute_target_normalization();

// From Eigen matrices
TabularDataset ds = TabularDataset::from_matrices(X_features, y_targets);
```

### Training Loops

```cpp
// Classification — reports cross-entropy loss + accuracy each epoch
train_module(net, optim, train_set, test_set, epochs, batch_size);

// Regression — normalises targets during training, reports RMSE in original units
train_regression(net, optim, train_set, test_set, epochs, batch_size);
```

Both loops shuffle each epoch, skip partial batches, and call `optim.epoch_end()` for scheduler stepping.

### Model Persistence

Custom `.sand` binary format — 4-word header, optional normalisation block, then Linear weight blocks in DFS traversal order.

```cpp
#include <sandokan/io.h>

save_model(net, "letter_net.sand");                        // weights only
save_model<TabularDataset>(net, "model.sand", ds);         // weights + normalization

load_model(net, "letter_net.sand");
load_model<TabularDataset>(net, "model.sand", ds);
```

### Inference

```cpp
#include <sandokan/inference.h>

auto pred = predict(net, x);              // {label, confidence}
auto topk = predict_topk(net, x, 5);     // top-5 predictions
show_prediction(raw_image, true_label, topk, label_names);  // ASCII art + ranked list
```

---

## Performance

Benchmarks run on Apple Silicon (M-series) with BLAS via `EIGEN_USE_BLAS` (Apple Accelerate / AMX).  
Architecture `784 → 64 → 64 → 26` &nbsp;|&nbsp; batch = 128 &nbsp;|&nbsp; lr = 0.01

### EMNIST Letters — 124 800 training samples

| Backend | Total (ms) | ms / epoch | ms / sample | samples / sec |
|---------|-----------|------------|-------------|---------------|
| Sandokan single-sample | 7 540 | 1 508 | 0.0121 | 82 757 |
| Eigen single-sample | 9 257 | 1 851 | 0.0148 | 67 408 |
| **Sandokan batched + parallel** | **386** | **77** | **0.0006** | **1 615 666** |
| Eigen batched | 614 | 123 | 0.0010 | 1 015 951 |

Sandokan's batched path is **19.5× faster than single-sample** and **1.5× faster than plain Eigen batched**.

![EMNIST benchmark](emnist_bench.png)

### Fashion MNIST — 60 000 training samples

![Fashion MNIST benchmark](fashion_mnist_bench.png)

| Backend | ms / epoch | samples / sec |
|---------|-----------|---------------|
| **Sandokan batched + parallel** | **34.4** | **1 742 000** |
| Eigen batched | 40.9 | 1 464 000 |

**Speedup: 1.19×**

---

## Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected — He-initialised weights, PMAD-backed gradient buffers |
| `ReLU` | Element-wise rectifier, stores pre-activation for backward |
| `Softmax` | Numerically stable column-wise softmax, passthrough backward (CE loss folds in Jacobian) |
| `Sigmoid` | Element-wise sigmoid |

---

## Build

**Requirements:** C++17, CMake ≥ 3.15, Eigen 3.

```bash
cmake -B build .
cmake --build build -j
```

For Apple AMX acceleration (strongly recommended on Apple Silicon):

```cmake
target_compile_definitions(sandokan INTERFACE EIGEN_USE_BLAS)
target_link_libraries(sandokan INTERFACE "-framework Accelerate")
```

---

## Examples

| Example | Task | Dataset |
|---------|------|---------|
| `examples/emnist_letters` | 26-class letter recognition | EMNIST Letters |
| `examples/tabular_demo` | Generic CSV classification | any numeric CSV |
| `examples/benchmark` | Full timing sweep (single / batched / Module) | EMNIST Letters |
| `examples/emnist_bench` | Sandokan vs Eigen — per-epoch CSV + figure | EMNIST Letters |
| `examples/fashion_mnist_bench` | Sandokan vs Eigen — per-epoch CSV + figure | Fashion MNIST |

Run examples from the **project root** so relative `data/` paths resolve:

```bash
./build/examples/emnist_letters/emnist_letters
./build/examples/emnist_bench/emnist_bench
./build/examples/fashion_mnist_bench/fashion_mnist_bench
```

---

## Accuracy

| Dataset | Architecture | Optimizer | Result |
|---------|-------------|-----------|--------|
| EMNIST Letters | 784 → 64 → ResBlock(64) → 26 | Adam + LinearLR | **88.25% test accuracy** |
| Fashion MNIST | 784 → 64 → 64 → 10 | SGD | converges to ~85% |
