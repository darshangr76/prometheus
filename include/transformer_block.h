#pragma once
#include <memory>
#include "tensor.h"
#include "multi_head_attention.h"
#include "ffn.h"
#include "layernorm.h"

class TransformerBlock {
public:
    MultiHeadAttention mha;
    FeedForward ffn;
    LayerNorm ln1, ln2;
    
    std::shared_ptr<Tensor> saved_x, saved_norm_x, saved_attn_out, saved_x2, saved_norm_x2, saved_ffn_out;

    TransformerBlock(int embed_dim, int num_heads, int hidden_dim)
        : mha(num_heads, embed_dim), ffn(embed_dim, hidden_dim), ln1(embed_dim), ln2(embed_dim) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> d_out);
    void zero_grad() { mha.zero_grad(); ffn.zero_grad(); ln1.zero_grad(); ln2.zero_grad(); }
    void sgd_step(float lr) { mha.sgd_step(lr); ffn.sgd_step(lr); ln1.sgd_step(lr); ln2.sgd_step(lr); }
};
