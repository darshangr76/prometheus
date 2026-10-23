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
    LayerNorm ln1;
    LayerNorm ln2;

    TransformerBlock(int embed_dim, int num_heads, int hidden_dim);
    
    std::shared_ptr<Tensor> forward(
        std::shared_ptr<Tensor> x, 
        std::shared_ptr<Tensor> Q, 
        std::shared_ptr<Tensor> K, 
        std::shared_ptr<Tensor> V
    );
};