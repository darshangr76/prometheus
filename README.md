# Prometheus: Bare-Metal AI Engine
# Prometheus: Bare-Metal AI Engine 🧠

A complete Transformer LLM built from scratch in C++. No PyTorch. No TensorFlow. Just raw math and memory management.

## Architecture

*   **C++ Core:** Custom Tensor class with memory strides, Autograd engine, and Adam optimizer.
*   **CUDA:** Custom matrix multiplication kernel using Shared Memory Tiling.
*   **Transformer:** Multi-Head Attention, Causal Mask, LayerNorm, and Feed-Forward Networks.
*   **Rust:** Distributed TCP Scheduler for data-parallel training.

## Benchmarks

*   **Loss:** 4.5 → 0.0000005 (on a character-level dataset)
*   **Prediction:** Successfully predicts the next character (`"hello"`)
*   **CUDA Speedup:** 235x faster than the CPU implementation

## Tech Stack

*   C++ (Autograd, Tensor Engine)
*   CUDA (Shared Memory Tiling)
*   Rust (Distributed Network)
*   Adam Optimizer (Custom Implementation)

## How to Run

```bash
g++ -std=c++17 src/*.cpp -o llm_train -Iinclude
./llm_train