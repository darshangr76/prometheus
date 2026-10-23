#pragma once
#include <memory>
#include "tensor.h"

class Attention {
public:
    int embed_dim;
    std::shared_ptr<Tensor> W_q, W_k, W_v, W_o;
    std::shared_ptr<Tensor> saved_Q, saved_K, saved_V, saved_attn_weights, saved_input;

    Attention(int embed_dim);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x, bool causal_mask);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out, bool causal_mask);
    void zero_grad();
    void sgd_step(float lr);
    void adam_step(float lr);  // <--- Declaration only
};