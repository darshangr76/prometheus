# Prometheus: Bare-Metal AI Engine

A complete Transformer LLM built from scratch in C++. No PyTorch. No TensorFlow.

## Architecture

* **C++ Core:** Custom Tensor, Autograd engine, Adam optimizer.
* **CUDA:** Matrix multiplication with Shared Memory Tiling.
* **Transformer:** Multi-Head Attention, Causal Mask, LayerNorm, Feed-Forward.
* **Rust:** Distributed TCP Scheduler for data-parallel training.

## Benchmarks

* **Loss:** 4.5 → 0.0000005
* **Prediction:** Generates "hello"
* **CUDA Speedup:** 235x over CPU

## Build & Run

```bash
g++ -std=c++17 src/tensor.cpp src/tokenizer.cpp src/embedding.cpp src/attention.cpp src/ffn.cpp src/layernorm.cpp src/transformer_block.cpp src/lm_head.cpp src/llm.cpp src/llm_train.cpp -o llm_train -Iinclude
./llm_train