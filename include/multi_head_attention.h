#pragma once
#include <memory>
#include "tensor.h"
#include "attention.h"

class MultiHeadAttention {
public:
    Attention attn;
    MultiHeadAttention(int num_heads, int embed_dim) : attn(embed_dim) {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) { return attn.forward(x, true); }
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out) { return attn.backward(d_out, true); }
    void zero_grad() { attn.zero_grad(); }
    void sgd_step(float lr) { attn.sgd_step(lr); }
    void adam_step(float lr) { attn.adam_step(lr); }
};