#pragma once
#include <memory>
#include "tensor.h"
#include "attention.h"

class MultiHeadAttention {
public:
    Attention attn;

    MultiHeadAttention(int num_heads, int embed_dim) : attn(embed_dim) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
        return attn.forward(x, false);
    }

    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out) {
        return attn.backward(d_out, false);
    }

    void zero_grad() {
        attn.zero_grad();
    }

    void sgd_step(float lr) {
        attn.sgd_step(lr);
    }
};
