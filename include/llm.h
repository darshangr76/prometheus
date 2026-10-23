#pragma once
#include <vector>
#include <memory>
#include "tensor.h"
#include "transformer_block.h"

class LLM {
public:
    std::vector<TransformerBlock> blocks;
    int num_layers;

    LLM(int num_layers, int embed_dim, int num_heads, int hidden_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out);
    void zero_grad();
    void adam_step(float lr);
};
